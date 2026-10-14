/* nvapi_QueryInterface forwarder for kernel-dispatch tracing.
 *
 * Deployed by the probe harness: copied over the system32 nvapi64.dll that
 * Proton redeploys at startup (copy runs inside the Wine session, first
 * command line of the probe, only when DLSS_NVAPI_LABEL is set). Loads the
 * real DXVK-NVAPI (nvapi64_dxvk.dll next to the probe executable) and
 * forwards its single export, nvapi_QueryInterface. Two observability
 * modes:
 *
 *   resolve log  - every QueryInterface(id) with result pointer and caller.
 *   call log     - resolved entry points are replaced by per-ID trampolines
 *                  that log (id, caller module+RVA) before tail-jumping to
 *                  the real implementation. Argument values are NOT read;
 *                  the wrapper is behaviour-preserving for the callee.
 *   args log     - DLSS_NVAPI_ARGS adds a second record per dispatch call,
 *                  reading the argument slots of the two cubin interfaces
 *                  only (see below). Default mode still reads nothing.
 *   boundary cut - DLSS_NVAPI_CUT=1 (requires ARGS level 2) turns the launch
 *                  trampoline into a truncating gate: the launch named
 *                  cc_dec_input_upsample_1024_512_tilesync_fp8 is forwarded
 *                  normally and arms the window, subsequent launches are
 *                  logged but NOT forwarded (the stub returns NVAPI_OK
 *                  without calling the real entry point), and the window
 *                  closes at the next cc_cb_clear launch or at a bounded
 *                  budget of 56 withheld launches, whichever comes first.
 *                  Each evaluation therefore executes only slots 0..99 of
 *                  the recovered 156-launch sequence, so nothing overwrites
 *                  the decoder boundary buffers and the harness can read
 *                  them back after the frame completes. Each armed decoder
 *                  launch also appends one `boundary eval=.. psize=..
 *                  pmem=<hex>` line to DLSS_NVAPI_CUT_FILE, giving the
 *                  harness the boundary word values directly. See the state
 *                  machine's block comment for the exact fail-closed
 *                  semantics.
 *
 * The caller return address is captured per event so a trace line attributes
 * each resolve/call to its calling module and RVA (the snippet's dispatch
 * code lives in nvngx_dlssnr.dll; attribution pins which of its routines
 * touches NvAPI around CreateFeature/EvaluateFeature).
 *
 * Environment (read once on first use):
 *   DLSS_NVAPI_LOG   trace file path (relative paths resolve against the
 *                    process CWD); absent/empty = pure passthrough: no
 *                    logging, no trampolines, QueryInterface results are
 *                    returned untouched.
 *   DLSS_NVAPI_WRAP  0 = log resolves only; 1 (default when logging) =
 *                    also wrap resolved entry points with call trampolines.
 *   DLSS_NVAPI_ARGS  0/absent = no argument reads (default). 1 = decode the
 *                    documented NvAPI_D3D12_LaunchCubinShader ABI for every
 *                    launch and log raw argument slots for
 *                    NvAPI_D3D12_CreateCubinComputeShaderExV2 (whose ABI is
 *                    not published). 2 = additionally hex-dump the first
 *                    512 bytes of each launch's parameter block. Invalid
 *                    values, or ARGS set together with WRAP=0 or without a
 *                    trace, fail closed (QueryInterface returns NULL).
 *
 * Argument reads are bounded and guarded: pointers are dereferenced only
 * after VirtualQuery reports a committed, readable region, reads stop at the
 * first unreadable byte, and the dumps are capped (128-byte name, 512-byte
 * parameter block). Parameter-block contents are recorded as hex for offline
 * analysis; the pointers inside them are never dereferenced, so no tensor or
 * weight content is captured. Stack argument slots are always readable by
 * construction (they are the caller's own frame).
 *
 * Everything is bounded and best-effort: failures fall back to transparent
 * forwarding and are logged where possible. Without DLSS_NVAPI_ARGS the trace
 * contains interface IDs, pointers and code addresses only. ARGS mode adds
 * kernel names and, at level 2, hex of the kernel parameter block (scalars and
 * device pointers) - never tensor or weight contents.
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>

#define REAL_DLL_NAME L"nvapi64_dxvk.dll"
#define MAX_TRAMPS 256
#define STUB_SIZE 384
#define ARENA_SIZE (MAX_TRAMPS * STUB_SIZE)

/* Interfaces whose argument slots are read in ARGS mode. Launch carries the
 * documented 12-argument ABI; the create call's V2 ABI is not published, so
 * its slots are logged undecoded, in ABI argument order. */
#define ID_CREATE_CUBIN_V2 0x299F5FDCu
#define ID_LAUNCH_CUBIN 0x5C52BB86u
#define ARGS_NAME_MAX 128
#define ARGS_PMEM_MAX 512

typedef void*(__cdecl* PFN_QI)(unsigned int);

static PFN_QI g_real_qi;
static HMODULE g_self;
static SRWLOCK g_lock = SRWLOCK_INIT;
static HANDLE g_log = INVALID_HANDLE_VALUE;
static int g_wrap;
static int g_args;
static int g_init_done;
static uint64_t g_seq;

static unsigned char* g_arena;
static size_t g_used;

struct Tramp {
  unsigned int id;
  void* stub;
  void* real;
};
static struct Tramp g_tramps[MAX_TRAMPS];
static int g_ntramps;

static void flush_pending_create(void);

static void log_line(const char* fmt, ...) {
  char buf[512];
  va_list ap;
  int n;
  DWORD written = 0;
  if (g_log == INVALID_HANDLE_VALUE) return;
  n = snprintf(buf, sizeof(buf), "%06llu %llu %lu ",
               (unsigned long long)g_seq++, (unsigned long long)GetTickCount64(),
               (unsigned long)GetCurrentThreadId());
  if (n < 0 || n >= (int)sizeof(buf)) return;
  va_start(ap, fmt);
  n += vsnprintf(buf + n, sizeof(buf) - (size_t)n, fmt, ap);
  va_end(ap);
  if (n < 0) return;
  if (n >= (int)sizeof(buf)) {
    /* Never drop a record: the sequence prefix must stay contiguous or a
     * validator would read a lost line as tampering. */
    n = (int)sizeof(buf) - 1;
    memcpy(buf + n - 8, " trunc=1", 8);
  }
  buf[n++] = '\n';
  WriteFile(g_log, buf, (DWORD)n, &written, NULL);
}

/* As log_line, but with a larger buffer and truncation-safe writes: an
 * over-long record is emitted up to the buffer limit and flagged `trunc=1`
 * rather than dropped, so argument evidence is never silently lost. */
static void log_line_big(const char* fmt, ...) {
  char buf[2048];
  va_list ap;
  int n;
  DWORD written = 0;
  if (g_log == INVALID_HANDLE_VALUE) return;
  n = snprintf(buf, sizeof(buf), "%06llu %llu %lu ",
               (unsigned long long)g_seq++, (unsigned long long)GetTickCount64(),
               (unsigned long)GetCurrentThreadId());
  if (n < 0 || n >= (int)sizeof(buf)) return;
  va_start(ap, fmt);
  n += vsnprintf(buf + n, sizeof(buf) - (size_t)n, fmt, ap);
  va_end(ap);
  if (n < 0) return;
  if (n >= (int)sizeof(buf)) {
    n = (int)sizeof(buf) - 1;
    memcpy(buf + n - 8, " trunc=1", 8);
  }
  buf[n++] = '\n';
  WriteFile(g_log, buf, (DWORD)n, &written, NULL);
}

/* Copy at most `want` bytes out of caller-owned memory, stopping at the first
 * unreadable byte. Returns the number of bytes copied. VirtualQuery validates
 * every region before it is touched, so no uncommitted or write-only page is
 * ever read. */
static size_t safe_read(const void* src, void* dst, size_t want) {
  const unsigned char* in = (const unsigned char*)src;
  unsigned char* out = (unsigned char*)dst;
  size_t done = 0;
  if (src == NULL || dst == NULL) return 0;
  while (done < want) {
    MEMORY_BASIC_INFORMATION mbi;
    unsigned char* end;
    size_t chunk;
    DWORD prot;
    if (VirtualQuery(in + done, &mbi, sizeof(mbi)) != sizeof(mbi)) break;
    if (mbi.State != MEM_COMMIT) break;
    if (mbi.Protect & (PAGE_GUARD | PAGE_NOACCESS)) break;
    prot = mbi.Protect & 0xffu;
    if (prot != PAGE_READONLY && prot != PAGE_READWRITE && prot != PAGE_WRITECOPY &&
        prot != PAGE_EXECUTE_READ && prot != PAGE_EXECUTE_READWRITE &&
        prot != PAGE_EXECUTE_WRITECOPY) {
      break;
    }
    end = (unsigned char*)mbi.BaseAddress + mbi.RegionSize;
    chunk = (size_t)(end - (in + done));
    if (chunk > want - done) chunk = want - done;
    memcpy(out + done, in + done, chunk);
    done += chunk;
  }
  return done;
}

/* Render a bounded raw string as a single whitespace-free log field:
 * printable non-space bytes pass through, everything else becomes \xNN.
 * Returns the number of bytes written (excluding the terminator). */
static size_t fmt_name(char* out, size_t cap, const char* raw, size_t n) {
  static const char hex[] = "0123456789abcdef";
  size_t o = 0;
  size_t i;
  if (cap == 0) return 0;
  for (i = 0; i < n; i++) {
    unsigned char c = (unsigned char)raw[i];
    if (o + 5 >= cap) break;
    if (c > 0x20 && c < 0x7f) {
      out[o++] = (char)c;
    } else {
      out[o++] = '\\';
      out[o++] = 'x';
      out[o++] = hex[c >> 4];
      out[o++] = hex[c & 15];
    }
  }
  out[o] = 0;
  return o;
}

/* Module + RVA attribution for a code address (best-effort). */
static void log_site(const char* kind, unsigned int id, void* ret) {
  HMODULE hm = NULL;
  wchar_t wpath[MAX_PATH];
  char path[MAX_PATH * 4];
  uintptr_t rva = 0;
  const char* name = "<unknown>";
  if (ret != NULL &&
      GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         (LPCWSTR)ret, &hm) &&
      hm != NULL) {
    rva = (uintptr_t)ret - (uintptr_t)hm;
    if (GetModuleFileNameW(hm, wpath, MAX_PATH) > 0) {
      size_t i, j;
      for (i = 0; wpath[i] != 0 && i < MAX_PATH; i++) {
        /* keep only the file name */
      }
      j = i;
      while (j > 0 && wpath[j - 1] != L'\\' && wpath[j - 1] != L'/') j--;
      {
        size_t k = 0;
        for (; wpath[j] != 0 && k + 1 < sizeof(path); j++) {
          wchar_t c = wpath[j];
          path[k++] = (c < 0x80) ? (char)c : '?';
        }
        path[k] = 0;
        name = path;
      }
    }
  }
  log_line("%s id=0x%08x ret=%p site=%s+0x%llx", kind, id, ret, name,
           (unsigned long long)rva);
}

/* Trampoline: log (id, return address) then tail-jump to the real entry
 * point. All volatile registers (rax, rcx, rdx, r8-r11, xmm0-xmm5) and the
 * flags are saved around the logger call; the tail jump preserves the
 * original return address so the real function returns straight to its
 * caller with its own return value. */
static void __cdecl tramp_logger(unsigned int id, void* ret) {
  AcquireSRWLockExclusive(&g_lock);
  log_site("call", id, ret);
  ReleaseSRWLockExclusive(&g_lock);
}

/* Argument-capturing trampoline target. `saved` points at the trampoline's
 * saved-register block: [0]=r11 [1]=r10 [2]=r9 [3]=r8 [4]=rdx [5]=rcx [6]=rax
 * [7]=flags [8]=dup return address [9]=return address [10..21]=the caller's
 * shadow space and stack slots. The block layout is fixed by the stub's push
 * sequence; the generic stub does not pass it and still reads nothing.
 *
 * LaunchCubinShader ABI - seven arguments, pinned twice over: by DXVK-NVAPI's
 * implementation (NvAPI_D3D12_LaunchCubinShader(pCmdList, pShader, blockX,
 * blockY, blockZ, params, paramSize)) and by the caller's own marshalling at
 * nvngx_dlssnr.dll+0x2404 (rcx/rdx/r8/r9 then three stack slots at
 * [rsp+0x20..0x30]). There is no shader-name argument at launch: the name,
 * block dimensions and shared-memory size are fixed at creation time in
 * NVAPI_D3D12_CREATE_CUBIN_SHADER_PARAMS (a single pointer in rcx).
 *
 * Creation is captured with a call-style stub that reads the parameter struct
 * after the real call returns, because hShader is an output field written by
 * the callee; at that point the struct holds the name, cubin pointer/size,
 * block dims and the resulting handle. The struct read is bounded and
 * VirtualQuery-guarded; the cubin pointer is recorded but never dereferenced.
 *
 * Every captured call also logs its full raw slot array, so the mapping above
 * stays checkable offline. The slots include the caller's shadow space, which
 * is not part of the ABI and is deliberately left uninterpreted. */

struct PendingCreate {
  uint64_t struct_ptr;
  int active; /* a create call registered and awaits its post-call read */
};
static struct PendingCreate g_pending;
/* Return address of the call being processed by a call-style (post-logging)
 * stub. Single-threaded, non-reentrant dispatch is observed and required (the
 * report pins "all create/launch calls on one thread"); a nested create call on
 * the same thread would corrupt the outer call's saved return. The call-style
 * stub also cannot propagate a structured exception out of the callee: it
 * returns status codes in this call pattern, and 97/97 creations passed the
 * probe's own validation. See NVAPI-ARGS.md. */
static void* g_saved_ret;

/* Boundary-cut mode (DLSS_NVAPI_CUT). Launches are forwarded until the
 * decoder kernel launch, withheld from there until the window closes, then
 * forwarded again. The handle names come from the create records captured by
 * flush_pending_create below. Implemented semantics, precisely:
 *   - Outside a window a launch with no readable name is forwarded and can
 *     never trigger a cut. The decoder name arms the window only when the
 *     parameter dump is complete, a clear-kernel name is currently mapped
 *     (cut_clear_known), and the sidecar line was written in full; every
 *     other case logs and leaves the run uncut.
 *   - Inside a window everything is withheld except a recognised clear
 *     launch, which closes the window and is forwarded. A decoder-named
 *     launch inside the window is withheld and counted as an anomaly; it
 *     never re-arms. A clear that closes a window is also recovered slot 0
 *     of the next evaluation, so in index mode at slot 0 the same launch
 *     immediately arms the next window (without that fall-through every
 *     evaluation after the first ran uncut and the capture failed on
 *     missing sidecar lines - the s0 ladder cut). Name mode is unaffected:
 *     a clear is never the decoder boundary.
 *   - The window withholds at most CUT_DROP_BUDGET launches (the recovered
 *     inventory's 56 post-decoder launches, anomalies included): at the
 *     budget the window force-closes and the current launch is forwarded.
 *     No failure mode can withhold more than one budget per window; a
 *     repeated failure yields repeated bounded windows, which the report's
 *     exact cut counts reject. In the pinned 156-launch inventory the
 *     withheld set is exactly slots 100-155, and the next evaluation's
 *     clear - named or not - arrives at the exhaustion position and is
 *     forwarded either way.
 * Requires ARGS level 2 and a cut file; fail closed otherwise. */
#define CUT_DECODER_NAME "cc_dec_input_upsample_1024_512_tilesync_fp8"
#define CUT_CLEAR_NAME "cc_cb_clear"
#define CUT_NAME_ENTRIES 128
#define CUT_NAME_MAX 64
#define CUT_DROP_BUDGET 56
struct CutNameEntry {
  uint64_t handle;
  char name[CUT_NAME_MAX];
};
static struct CutNameEntry g_names[CUT_NAME_ENTRIES];
static int g_nnames;
static int g_cut;          /* cut requested and armed */
/* Index-based cut (DLSS_NVAPI_CUT_AT): arm at recovered slot N of the
 * evaluation instead of at a kernel name, so any stage boundary can be
 * captured. The evaluation window opens with the clear kernel (recovered
 * slot 0 of the 156-launch inventory, NVAPI-ARGS.md), so slots are counted
 * from it. g_cut_at stays -1 for the original name-anchored mode, whose
 * behaviour is unchanged: it arms only on the decoder name, withholds the
 * recovered 56 post-decoder launches and treats a decoder name inside the
 * window as an anomaly. In index mode those three rules become "arm at
 * slot N", "budget 155 - N" and "no name-based anomaly", respectively. */
static long g_cut_at = -1;
static unsigned long long g_cut_slot; /* slot of the current launch */
static int g_cut_started;             /* a clear kernel has been seen */
static int g_cut_dropping; /* in the dropped window after the decoder */
static unsigned long long g_cut_window; /* launches withheld in this window */
static unsigned long long g_cut_evals; /* decoder launches seen */
static unsigned long long g_cut_drops; /* launches withheld */
static HANDLE g_cut_file = INVALID_HANDLE_VALUE;

/* Called under g_lock from the create post-call read. The most recent create
 * record wins: a repeated handle whose new name is unreadable invalidates any
 * stale name (an entry with an empty name is looked up as unknown), so a stale
 * name can never misattribute a later launch. */
static void cut_note_name(uint64_t handle, const char* name) {
  int i;
  int usable = name != NULL && name[0] != 0 && strcmp(name, "<none>") != 0;
  for (i = 0; i < g_nnames; i++) {
    if (g_names[i].handle == handle) break;
  }
  if (i == g_nnames) {
    if (!usable) return; /* nothing known about this handle */
    if (g_nnames >= CUT_NAME_ENTRIES) {
      log_line("cut-name-table-full handle=0x%016llx",
               (unsigned long long)handle);
      return;
    }
    g_names[g_nnames].handle = handle;
    g_nnames++;
  }
  strncpy(g_names[i].name, usable ? name : "", CUT_NAME_MAX - 1);
  g_names[i].name[CUT_NAME_MAX - 1] = 0;
}

/* Called under g_lock. Returns NULL when the handle has no readable name. */
static const char* cut_lookup(uint64_t handle) {
  int i;
  for (i = 0; i < g_nnames; i++) {
    if (g_names[i].handle == handle) {
      return g_names[i].name[0] ? g_names[i].name : NULL;
    }
  }
  return NULL;
}

/* Called under g_lock: does any currently mapped handle belong to the clear
 * kernel? Arming requires this: without it the window cannot be closed by
 * name, only by the budget, and the run must stay uncut. */
static int cut_clear_known(void) {
  int i;
  for (i = 0; i < g_nnames; i++) {
    if (g_names[i].name[0] && strcmp(g_names[i].name, CUT_CLEAR_NAME) == 0) {
      return 1;
    }
  }
  return 0;
}

/* Called under g_lock: one sidecar line per decoder launch. Returns 1 only
 * when the line was written in full: the cut never arms without it. */
static int cut_write_boundary(uint64_t eval, unsigned psize, const char* pmem) {
  char buf[2048];
  DWORD written = 0;
  int n;
  if (g_cut_file == INVALID_HANDLE_VALUE || pmem == NULL) return 0;
  n = snprintf(buf, sizeof(buf), "boundary eval=%llu psize=%u pmem=%s\n",
               (unsigned long long)eval, psize, pmem);
  if (n <= 0 || n >= (int)sizeof(buf)) return 0;
  return WriteFile(g_cut_file, buf, (DWORD)n, &written, NULL) && written == (DWORD)n;
}

/* Bounded, guarded read of NVAPI_D3D12_CREATE_CUBIN_SHADER_PARAMS (0x50). */
static void flush_pending_create(void) {
  unsigned char buf[0x50];
  char namebuf[ARGS_NAME_MAX * 4];
  uint64_t ptr;
  size_t got, k;
  uint64_t struct_size, device, cubin, namep, handle;
  unsigned size, bx, by, bz, smem, flags;
  if (!g_pending.active) return; /* no create awaiting a read-back */
  g_pending.active = 0;
  ptr = g_pending.struct_ptr;
  if (ptr == 0) {
    log_line_big("created-null");
    return;
  }
  memset(buf, 0, sizeof(buf));
  got = safe_read((const void*)(uintptr_t)ptr, buf, sizeof(buf));
  if (got != sizeof(buf)) {
    log_line_big("created-invalid struct=0x%016llx read=%llu",
                 (unsigned long long)ptr, (unsigned long long)got);
    return;
  }
  memcpy(&struct_size, buf + 0x00, 8);
  memcpy(&device, buf + 0x10, 8);
  memcpy(&cubin, buf + 0x18, 8);
  memcpy(&size, buf + 0x20, 4);
  memcpy(&bx, buf + 0x24, 4);
  memcpy(&by, buf + 0x28, 4);
  memcpy(&bz, buf + 0x2C, 4);
  memcpy(&smem, buf + 0x30, 4);
  memcpy(&namep, buf + 0x38, 8);
  memcpy(&flags, buf + 0x40, 4);
  memcpy(&handle, buf + 0x48, 8);
  namebuf[0] = 0;
  if (namep != 0) {
    unsigned char rawname[ARGS_NAME_MAX + 1];
    size_t n = safe_read((const void*)(uintptr_t)namep, rawname, ARGS_NAME_MAX);
    k = 0;
    while (k < n && rawname[k] != 0) k++;
    fmt_name(namebuf, sizeof(namebuf), (const char*)rawname, k);
  }
  cut_note_name(handle, namebuf[0] ? namebuf : NULL);
  log_line_big(
      "created struct=0x%016llx struct_size=%llu device=0x%016llx "
      "cubin=0x%016llx cubin_size=%u block=%u,%u,%u smem=%u flags=%u "
      "name=%s handle=0x%016llx",
      (unsigned long long)ptr, (unsigned long long)struct_size,
      (unsigned long long)device, (unsigned long long)cubin, size, bx, by, bz,
      smem, flags, namebuf[0] ? namebuf : "<none>", (unsigned long long)handle);
}

/* Post-call target for the create path: the parameters are complete only now
 * (hShader is written by the callee), so the deferred read runs here. */
static void __cdecl tramp_post_logger(unsigned int id) {
  AcquireSRWLockExclusive(&g_lock);
  flush_pending_create();
  ReleaseSRWLockExclusive(&g_lock);
  (void)id;
}

static int __cdecl tramp_args_logger(unsigned int id, void* ret, void* saved) {
  char pmem[ARGS_PMEM_MAX * 2 + 4];
  char raw[22 * 19 + 8];
  const uint64_t* s = (const uint64_t*)saved;
  size_t plen = 0;
  int pcap = 3; /* 3 = parameter dump not requested (ARGS level 1) */
  int drop = 0; /* boundary cut: 1 = withhold this launch from the callee */
  unsigned psize = 0;
  unsigned grid[3] = {0, 0, 0};
  uint64_t handle = 0, params = 0, cmd = 0;
  size_t i, o = 0;

  pmem[0] = 0;
  raw[0] = 0;
  if (saved != NULL) {
    for (i = 0; i < 22; i++) {
      int n = snprintf(raw + o, sizeof(raw) - o, "%s0x%016llx", i ? "," : "",
                       (unsigned long long)s[i]);
      if (n <= 0 || (size_t)n >= sizeof(raw) - o) break;
      o += (size_t)n;
    }
    if (id == ID_LAUNCH_CUBIN) {
      /* Seven-argument launch ABI (see the block comment above). */
      cmd = s[5];
      handle = s[4];
      grid[0] = (unsigned)s[3];
      grid[1] = (unsigned)s[2];
      grid[2] = (unsigned)s[14];
      params = s[15];
      psize = (unsigned)s[16];
      if (g_args >= 2 && params != 0 && psize > 0) {
        unsigned char dump[ARGS_PMEM_MAX];
        size_t want = (size_t)psize;
        pcap = 0; /* complete dump */
        if (want > ARGS_PMEM_MAX) {
          want = ARGS_PMEM_MAX;
          pcap = 1; /* capped */
        }
        plen = safe_read((const void*)(uintptr_t)params, dump, want);
        if (plen < want) pcap = 2; /* short read: unreadable tail */
        for (i = 0; i < plen; i++) {
          static const char hex[] = "0123456789abcdef";
          pmem[i * 2] = hex[dump[i] >> 4];
          pmem[i * 2 + 1] = hex[dump[i] & 15];
        }
        pmem[plen * 2] = 0;
      }
    }
  }

  AcquireSRWLockExclusive(&g_lock);
  log_site("call", id, ret);
  if (saved != NULL && id == ID_LAUNCH_CUBIN) {
    log_line_big(
        "launch id=0x%08x cmd=0x%016llx handle=0x%016llx grid=%u,%u,%u "
        "params=0x%016llx psize=%u pn=%llu pcap=%d pmem=%s s=%s",
        id, (unsigned long long)cmd, (unsigned long long)handle, grid[0],
        grid[1], grid[2], (unsigned long long)params, psize,
        (unsigned long long)plen, pcap, pmem, raw);
    if (g_cut) {
      const char* kname = cut_lookup(handle);
      int complete = (pcap == 0 && plen == psize && psize > 0);
      int is_decoder = kname != NULL && strcmp(kname, CUT_DECODER_NAME) == 0;
      int is_clear = kname != NULL && strcmp(kname, CUT_CLEAR_NAME) == 0;
      unsigned long long budget;
      int at_boundary;
      int was_dropping = g_cut_dropping;
      int released = 0;
      /* Slot accounting: the clear kernel opens an evaluation at slot 0.
       * Without a seen clear there is no evaluation boundary to index from,
       * so index mode never arms (the run stays uncut). */
      if (is_clear) {
        g_cut_slot = 0;
        g_cut_started = 1;
      } else if (g_cut_started) {
        g_cut_slot++;
      }
      /* The recovered inventory has 156 launches (slots 0..155): everything
       * after slot g_cut_at is withheld, and no window can exceed that. */
      budget = (g_cut_at >= 0)
                   ? (155ull - (unsigned long long)g_cut_at)
                   : (unsigned long long)CUT_DROP_BUDGET;
      at_boundary = (g_cut_at >= 0) ? (g_cut_started &&
                                       (unsigned long long)g_cut_at == g_cut_slot)
                                    : is_decoder;
      if (g_cut_dropping) {
        if (is_clear) {
          log_line("cut-release eval=%llu window=%llu",
                   (unsigned long long)g_cut_evals,
                   (unsigned long long)g_cut_window);
          g_cut_dropping = 0;
          g_cut_window = 0;
          /* The clear kernel is recovered slot 0, so when the cut point is
           * slot 0 this same launch is also the next window's boundary. Fall
           * through to the boundary check below instead of skipping it:
           * otherwise every evaluation after the first runs uncut and the
           * capture fails on missing sidecar lines. */
          released = 1;
        } else if (is_decoder && g_cut_at < 0) {
          /* Anomalous repeat inside the window: never re-arm mid-window.
           * Counted toward the budget so the window bound stays exact. */
          drop = 1;
          g_cut_window++;
          g_cut_drops++;
          log_line("cut-anomaly decoder-inside-window handle=0x%016llx",
                   (unsigned long long)handle);
        } else if (g_cut_window >= budget) {
          /* Bounded window: without this a lost clear name would truncate
           * the rest of the run. Force a release and forward again. */
          g_cut_dropping = 0;
          g_cut_window = 0;
          log_line("cut-window-exhausted handle=0x%016llx",
                   (unsigned long long)handle);
        } else {
          drop = 1;
          g_cut_window++;
          g_cut_drops++;
          log_line("cut-drop handle=0x%016llx name=%s",
                   (unsigned long long)handle, kname ? kname : "<unknown>");
        }
      }
      if ((released || !was_dropping) && at_boundary) {
        if (!complete) {
          /* Never cut without the boundary words: the truncated run would be
           * unusable and silently misleading. The run stays uncut instead. */
          log_line("cut-pmem-incomplete pcap=%d psize=%u plen=%llu", pcap,
                   psize, (unsigned long long)plen);
        } else if (!cut_clear_known()) {
          /* Without a mapped clear kernel the window cannot be closed by
           * name; never arm (the run stays uncut and the report rejects it). */
          log_line("cut-clear-unknown handle=0x%016llx",
                   (unsigned long long)handle);
        } else if (cut_write_boundary(g_cut_evals, psize, pmem)) {
          g_cut_dropping = 1;
          g_cut_window = 0;
          log_line("cut-boundary eval=%llu slot=%llu psize=%u",
                   (unsigned long long)g_cut_evals,
                   (unsigned long long)g_cut_slot, psize);
          log_line("cut-armed eval=%llu handle=0x%016llx",
                   (unsigned long long)g_cut_evals,
                   (unsigned long long)handle);
          g_cut_evals++;
        } else {
          log_line("cut-sidecar-write-failed handle=0x%016llx",
                   (unsigned long long)handle);
        }
      } else if (!was_dropping && kname == NULL && g_cut_at < 0) {
        /* Unchanged from the original chain: only launches processed outside
         * a dropped window are reported as unknown. */
        log_line("cut-unknown handle=0x%016llx", (unsigned long long)handle);
      }
    }
  } else if (saved != NULL && id == ID_CREATE_CUBIN_V2) {
    /* s[9] is the return-address slot the call-style stub drops; keep it so
     * the post-call stub can return to the original caller. */
    g_saved_ret = (void*)(uintptr_t)s[9];
    g_pending.struct_ptr = s[5];
    g_pending.active = 1;
    log_line_big("createargs id=0x%08x struct=0x%016llx s=%s", id,
                 (unsigned long long)s[5], raw);
    if (s[5] == 0) {
      g_pending.struct_ptr = 0; /* nothing to read back */
    }
  } else if (saved != NULL) {
    log_line_big("args id=0x%08x s=%s", id, raw);
  }
  ReleaseSRWLockExclusive(&g_lock);
  return drop;
}

/* Build a trampoline stub. `logger` receives (id, return address) and, when
 * `pass_saved` is set, a third argument pointing at the saved-register block
 * (see tramp_args_logger). With pass_saved clear the emitted stub is
 * byte-identical to the pre-ARGS tracer: no argument is ever touched. */
static void* build_stub(unsigned int id, void* real, uintptr_t logger,
                        int pass_saved, uintptr_t post_logger) {
  unsigned char* p;
  unsigned char* base;
  size_t off_id, off_logger, off_target, off_id2 = 0, off_post = 0, off_saved = 0;
  size_t off_drop_jmp = 0, off_drop_start = 0;
  if (g_arena == NULL) {
    g_arena = (unsigned char*)VirtualAlloc(NULL, ARENA_SIZE,
                                           MEM_COMMIT | MEM_RESERVE,
                                           PAGE_EXECUTE_READWRITE);
    if (g_arena == NULL) return NULL;
    g_used = 0;
  }
  if (g_used + STUB_SIZE > ARENA_SIZE) return NULL;
  base = g_arena + g_used;
  p = base;
#define EMIT(...)                                                              \
  do {                                                                         \
    unsigned char bytes__[] = {__VA_ARGS__};                                   \
    for (size_t i__ = 0; i__ < sizeof(bytes__); i__++) *p++ = bytes__[i__];    \
  } while (0)
  EMIT(0xFF, 0x34, 0x24);             /* push qword [rsp]   (dup retaddr) */
  EMIT(0x9C);                         /* pushfq */
  EMIT(0x50);                         /* push rax */
  EMIT(0x51);                         /* push rcx */
  EMIT(0x52);                         /* push rdx */
  EMIT(0x41, 0x50);                   /* push r8 */
  EMIT(0x41, 0x51);                   /* push r9 */
  EMIT(0x41, 0x52);                   /* push r10 */
  EMIT(0x41, 0x53);                   /* push r11 */
  EMIT(0x48, 0x81, 0xEC, 0x80, 0x00, 0x00, 0x00); /* sub rsp, 0x80 */
  EMIT(0x0F, 0x11, 0x44, 0x24, 0x20); /* movdqu [rsp+0x20], xmm0 */
  EMIT(0x0F, 0x11, 0x4C, 0x24, 0x30); /* movdqu [rsp+0x30], xmm1 */
  EMIT(0x0F, 0x11, 0x54, 0x24, 0x40); /* movdqu [rsp+0x40], xmm2 */
  EMIT(0x0F, 0x11, 0x5C, 0x24, 0x50); /* movdqu [rsp+0x50], xmm3 */
  EMIT(0x0F, 0x11, 0x64, 0x24, 0x60); /* movdqu [rsp+0x60], xmm4 */
  EMIT(0x0F, 0x11, 0x6C, 0x24, 0x70); /* movdqu [rsp+0x70], xmm5 */
  EMIT(0xB9);                         /* mov ecx, id */
  off_id = (size_t)(p - base);
  EMIT(0, 0, 0, 0);
  EMIT(0x48, 0x8B, 0x94, 0x24, 0xC0, 0x00, 0x00, 0x00); /* mov rdx, [rsp+0xC0] */
  if (pass_saved) {
    EMIT(0x4C, 0x8D, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00); /* lea r8, [rsp+0x80] */
  }
  EMIT(0x48, 0xB8);                   /* mov rax, logger */
  off_logger = (size_t)(p - base);
  EMIT(0, 0, 0, 0, 0, 0, 0, 0);
  EMIT(0xFF, 0xD0);                   /* call rax */
  if (pass_saved && post_logger == 0) {
    /* Logger return value decides forwarding (boundary cut): zero forwards
     * to the real entry point, nonzero drops the call below. */
    EMIT(0x85, 0xC0);                 /* test eax, eax */
    EMIT(0x75, 0x00);                 /* jnz drop path (rel8 patched) */
    off_drop_jmp = (size_t)(p - base) - 1;
  }
  EMIT(0x0F, 0x10, 0x44, 0x24, 0x20); /* movdqu xmm0, [rsp+0x20] */
  EMIT(0x0F, 0x10, 0x4C, 0x24, 0x30); /* movdqu xmm1, [rsp+0x30] */
  EMIT(0x0F, 0x10, 0x54, 0x24, 0x40); /* movdqu xmm2, [rsp+0x40] */
  EMIT(0x0F, 0x10, 0x5C, 0x24, 0x50); /* movdqu xmm3, [rsp+0x50] */
  EMIT(0x0F, 0x10, 0x64, 0x24, 0x60); /* movdqu xmm4, [rsp+0x60] */
  EMIT(0x0F, 0x10, 0x6C, 0x24, 0x70); /* movdqu xmm5, [rsp+0x70] */
  EMIT(0x48, 0x81, 0xC4, 0x80, 0x00, 0x00, 0x00); /* add rsp, 0x80 */
  EMIT(0x41, 0x5B);                   /* pop r11 */
  EMIT(0x41, 0x5A);                   /* pop r10 */
  EMIT(0x41, 0x59);                   /* pop r9 */
  EMIT(0x41, 0x58);                   /* pop r8 */
  EMIT(0x5A);                         /* pop rdx */
  EMIT(0x59);                         /* pop rcx */
  EMIT(0x58);                         /* pop rax */
  EMIT(0x9D);                         /* popfq */
  if (post_logger == 0) {
    EMIT(0x48, 0x83, 0xC4, 0x08);     /* add rsp, 8 (drop dup retaddr) */
    EMIT(0x48, 0xB8);                 /* mov rax, real */
    off_target = (size_t)(p - base);
    EMIT(0, 0, 0, 0, 0, 0, 0, 0);
    EMIT(0xFF, 0xE0);                 /* jmp rax */
    if (pass_saved) {
      /* Drop path (boundary cut): the same volatile-register restore as the
       * forward path, then return NVAPI_OK to the original caller without
       * ever reaching the real entry point. The caller observes a successful
       * launch that was never dispatched. */
      off_drop_start = (size_t)(p - base);
      EMIT(0x0F, 0x10, 0x44, 0x24, 0x20); /* movdqu xmm0, [rsp+0x20] */
      EMIT(0x0F, 0x10, 0x4C, 0x24, 0x30);
      EMIT(0x0F, 0x10, 0x54, 0x24, 0x40);
      EMIT(0x0F, 0x10, 0x5C, 0x24, 0x50);
      EMIT(0x0F, 0x10, 0x64, 0x24, 0x60);
      EMIT(0x0F, 0x10, 0x6C, 0x24, 0x70);
      EMIT(0x48, 0x81, 0xC4, 0x80, 0x00, 0x00, 0x00); /* add rsp, 0x80 */
      EMIT(0x41, 0x5B);               /* pop r11 */
      EMIT(0x41, 0x5A);               /* pop r10 */
      EMIT(0x41, 0x59);               /* pop r9 */
      EMIT(0x41, 0x58);               /* pop r8 */
      EMIT(0x5A);                     /* pop rdx */
      EMIT(0x59);                     /* pop rcx */
      EMIT(0x58);                     /* pop rax */
      EMIT(0x9D);                     /* popfq */
      EMIT(0x48, 0x83, 0xC4, 0x08);   /* add rsp, 8 (drop dup retaddr) */
      EMIT(0x31, 0xC0);               /* xor eax, eax (NVAPI_OK) */
      EMIT(0xC3);                     /* ret */
    }
  } else {
    /* Call-style stub: drop the saved return addresses (the original one was
     * captured into g_saved_ret by the pre-call logger) so rsp is exactly the
     * caller's pre-call value, then CALL the real entry point. Its arguments
     * line up again because the call re-creates the same frame the original
     * caller set up (stack arguments at [rsp_entry+0x28] and below). After it
     * returns, run the post-call logger - creation writes hShader as an
     * output, so the parameters can only be read at that point - and finally
     * return to the original caller with the real return value. */
    EMIT(0x48, 0x83, 0xC4, 0x10);     /* add rsp, 16 (drop dup + orig retaddr) */
    EMIT(0x48, 0xB8);                 /* mov rax, real */
    off_target = (size_t)(p - base);
    EMIT(0, 0, 0, 0, 0, 0, 0, 0);
    EMIT(0xFF, 0xD0);                 /* call rax */
    EMIT(0x50);                       /* push rax (return value) */
    EMIT(0x9C);                       /* pushfq */
    EMIT(0x51);                       /* push rcx */
    EMIT(0x52);                       /* push rdx */
    EMIT(0x41, 0x50);                 /* push r8 */
    EMIT(0x41, 0x51);                 /* push r9 */
    EMIT(0x41, 0x52);                 /* push r10 */
    EMIT(0x41, 0x53);                 /* push r11 */
    EMIT(0x48, 0x81, 0xEC, 0x80, 0x00, 0x00, 0x00); /* sub rsp, 0x80 */
    EMIT(0x0F, 0x11, 0x44, 0x24, 0x20); /* movdqu [rsp+0x20], xmm0 */
    EMIT(0x0F, 0x11, 0x4C, 0x24, 0x30); /* movdqu [rsp+0x30], xmm1 */
    EMIT(0x0F, 0x11, 0x54, 0x24, 0x40); /* movdqu [rsp+0x40], xmm2 */
    EMIT(0x0F, 0x11, 0x5C, 0x24, 0x50); /* movdqu [rsp+0x50], xmm3 */
    EMIT(0x0F, 0x11, 0x64, 0x24, 0x60); /* movdqu [rsp+0x60], xmm4 */
    EMIT(0x0F, 0x11, 0x6C, 0x24, 0x70); /* movdqu [rsp+0x70], xmm5 */
    EMIT(0xB9);                       /* mov ecx, id */
    off_id2 = (size_t)(p - base);
    EMIT(0, 0, 0, 0);
    EMIT(0x31, 0xD2);                 /* xor edx, edx */
    EMIT(0x4C, 0x8D, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00); /* lea r8, [rsp+0x80] */
    EMIT(0x48, 0xB8);                 /* mov rax, post_logger */
    off_post = (size_t)(p - base);
    EMIT(0, 0, 0, 0, 0, 0, 0, 0);
    EMIT(0xFF, 0xD0);                 /* call rax */
    EMIT(0x0F, 0x10, 0x44, 0x24, 0x20); /* movdqu xmm0, [rsp+0x20] */
    EMIT(0x0F, 0x10, 0x4C, 0x24, 0x30); /* movdqu xmm1, [rsp+0x30] */
    EMIT(0x0F, 0x10, 0x54, 0x24, 0x40); /* movdqu xmm2, [rsp+0x40] */
    EMIT(0x0F, 0x10, 0x5C, 0x24, 0x50); /* movdqu xmm3, [rsp+0x50] */
    EMIT(0x0F, 0x10, 0x64, 0x24, 0x60); /* movdqu xmm4, [rsp+0x60] */
    EMIT(0x0F, 0x10, 0x6C, 0x24, 0x70); /* movdqu xmm5, [rsp+0x70] */
    EMIT(0x48, 0x81, 0xC4, 0x80, 0x00, 0x00, 0x00); /* add rsp, 0x80 */
    EMIT(0x41, 0x5B);                 /* pop r11 */
    EMIT(0x41, 0x5A);                 /* pop r10 */
    EMIT(0x41, 0x59);                 /* pop r9 */
    EMIT(0x41, 0x58);                 /* pop r8 */
    EMIT(0x5A);                       /* pop rdx */
    EMIT(0x59);                       /* pop rcx */
    EMIT(0x9D);                       /* popfq */
    EMIT(0x58);                       /* pop rax (real return value) */
    EMIT(0x49, 0xBB);                 /* mov r11, &g_saved_ret */
    off_saved = (size_t)(p - base);
    EMIT(0, 0, 0, 0, 0, 0, 0, 0);
    EMIT(0x4D, 0x8B, 0x1B);           /* mov r11, [r11] */
    EMIT(0x41, 0x53);                 /* push r11 */
    EMIT(0xC3);                       /* ret */
  }
#undef EMIT
  if ((size_t)(p - base) > STUB_SIZE) return NULL;
  if (off_drop_jmp != 0) {
    size_t rel = off_drop_start - (off_drop_jmp + 1);
    if (off_drop_start == 0 || rel > 127) return NULL;
    base[off_drop_jmp] = (unsigned char)rel;
  }
  memcpy(base + off_id, &id, sizeof(id));
  {
    uintptr_t v = logger;
    memcpy(base + off_logger, &v, sizeof(v));
    v = (uintptr_t)real;
    memcpy(base + off_target, &v, sizeof(v));
    if (post_logger != 0) {
      v = (uintptr_t)&tramp_post_logger;
      memcpy(base + off_post, &v, sizeof(v));
      v = (uintptr_t)&g_saved_ret;
      memcpy(base + off_saved, &v, sizeof(v));
      memcpy(base + off_id2, &id, sizeof(id));
    }
  }
  FlushInstructionCache(GetCurrentProcess(), base, STUB_SIZE);
  g_used += STUB_SIZE;
  return base;
}

/* ARGS capture policy for one interface: 0 = generic trampoline (arguments
 * never read), 1 = decoded launch ABI, 2 = raw slots (create ABI unpublish-
 * ed). Only the two cubin interfaces are ever read. */
static int capture_policy(unsigned int id) {
  if (g_args == 0) return 0;
  if (id == ID_LAUNCH_CUBIN) return 1;      /* decoded launch ABI */
  if (id == ID_CREATE_CUBIN_V2) return 2;   /* raw slots + post-call struct */
  return 0;
}

static void* wrap_entry(unsigned int id, void* real) {
  void* stub;
  int policy = capture_policy(id);
  int i;
  for (i = 0; i < g_ntramps; i++) {
    if (g_tramps[i].id == id) {
      if (g_tramps[i].real == real) return g_tramps[i].stub;
      /* Unexpected: re-resolution returned a different pointer. Do not
       * wrap a stale target; forward untouched and record the divergence. */
      log_line("wrap-mismatch id=0x%08x kept=%p new=%p", id, g_tramps[i].real,
               real);
      return real;
    }
  }
  if (g_ntramps >= MAX_TRAMPS) {
    log_line("wrap-exhausted id=0x%08x", id);
    return real;
  }
  stub = build_stub(id, real,
                    policy ? (uintptr_t)&tramp_args_logger
                           : (uintptr_t)&tramp_logger,
                    policy != 0,
                    policy == 2 ? (uintptr_t)&tramp_post_logger : 0);
  if (stub == NULL) {
    log_line("wrap-failed id=0x%08x", id);
    return real;
  }
  g_tramps[g_ntramps].id = id;
  g_tramps[g_ntramps].stub = stub;
  g_tramps[g_ntramps].real = real;
  g_ntramps++;
  return stub;
}

/* Load the real DXVK-NVAPI and open the trace file. Called lazily: doing
 * LoadLibrary from DllMain would run under the loader lock. The real DLL is
 * located through DLSS_NVAPI_REAL (absolute path), then next to this module,
 * then next to the process executable (where the probe harness keeps a
 * pristine copy, nvapi64_dxvk.dll).
 *
 * Threading: the g_init_done double-check reads are unsynchronised (benign
 * on x86-64 TSO; all writes happen under g_lock before the flag). A wrapped
 * NvAPI call made while another module holds the loader lock (inside its
 * DllMain) would invert lock order and could deadlock the tracer; that is
 * outside the probe's call pattern and unobserved in practice. */
static int ensure_init(void) {
  wchar_t path[MAX_PATH];
  wchar_t* slash;
  char env[MAX_PATH];
  DWORD n;
  int fail = 0;
  if (g_init_done) return g_real_qi != NULL;
  AcquireSRWLockExclusive(&g_lock);
  if (g_init_done) {
    ReleaseSRWLockExclusive(&g_lock);
    return g_real_qi != NULL;
  }
  /* Precedence: explicit DLSS_NVAPI_REAL override, then next to this
   * module, then next to the process executable. */
  n = GetEnvironmentVariableA("DLSS_NVAPI_REAL", env, sizeof(env));
  if (n > 0 && n < sizeof(env)) {
    HMODULE real_mod = LoadLibraryExA(env, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (real_mod != NULL) {
      g_real_qi = (PFN_QI)(void*)GetProcAddress(real_mod, "nvapi_QueryInterface");
    }
  } else if (n >= sizeof(env)) {
    fail = 1; /* truncated path: never guess */
  }
  if (!fail && GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                 GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                             (LPCWSTR)&ensure_init, &g_self) &&
      g_self != NULL) {
    n = GetModuleFileNameW(g_self, path, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
      slash = path;
      for (wchar_t* q = path; *q != 0; q++) {
        if (*q == L'\\' || *q == L'/') slash = q + 1;
      }
      if ((size_t)(slash - path) + wcslen(REAL_DLL_NAME) + 1 < MAX_PATH) {
        HMODULE real_mod;
        wcscpy(slash, REAL_DLL_NAME);
        real_mod = LoadLibraryExW(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (real_mod != NULL) {
          g_real_qi =
              (PFN_QI)(void*)GetProcAddress(real_mod, "nvapi_QueryInterface");
        }
      }
    }
  }
  if (g_real_qi == NULL && !fail) {
    n = GetModuleFileNameW(NULL, path, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
      slash = path;
      for (wchar_t* q = path; *q != 0; q++) {
        if (*q == L'\\' || *q == L'/') slash = q + 1;
      }
      if ((size_t)(slash - path) + wcslen(REAL_DLL_NAME) + 1 < MAX_PATH) {
        HMODULE real_mod;
        wcscpy(slash, REAL_DLL_NAME);
        real_mod = LoadLibraryExW(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (real_mod != NULL) {
          g_real_qi =
              (PFN_QI)(void*)GetProcAddress(real_mod, "nvapi_QueryInterface");
        }
      }
    }
  }
  /* Argument-capture level. Parsed even when no trace is configured, so a
   * requested capture fails closed instead of silently passing through. */
  {
    char argsmode[8];
    DWORD a =
        GetEnvironmentVariableA("DLSS_NVAPI_ARGS", argsmode, sizeof(argsmode));
    if (a > 0) {
      if (a < sizeof(argsmode) && argsmode[1] == 0 &&
          (argsmode[0] == '0' || argsmode[0] == '1' || argsmode[0] == '2')) {
        g_args = argsmode[0] - '0'; /* 0 = no argument reads, as documented */
      } else {
        fail = 1; /* unknown value: never silently degrade */
      }
    }
  }
  n = GetEnvironmentVariableA("DLSS_NVAPI_LOG", env, sizeof(env));
  if (n > 0 && n < sizeof(env)) {
    g_log = CreateFileA(env, GENERIC_WRITE, FILE_SHARE_READ, NULL,
                        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (g_log == INVALID_HANDLE_VALUE) {
      fail = 1; /* fail closed: never forward without a trace */
    } else {
      char wrap[8];
      g_wrap = 1;
      if (GetEnvironmentVariableA("DLSS_NVAPI_WRAP", wrap, sizeof(wrap)) > 0 &&
          wrap[0] == '0') {
        g_wrap = 0;
      }
      log_line("init real=%p wrap=%d args=%d", (void*)g_real_qi, g_wrap, g_args);
    }
  } else if (n >= sizeof(env)) {
    fail = 1; /* truncated path: never forward without a trace */
  }
  if (g_args > 0 && (g_log == INVALID_HANDLE_VALUE || !g_wrap)) {
    fail = 1; /* argument capture needs a trace and trampolines */
  }
  /* Boundary-cut mode: needs full parameter dumps (ARGS level 2) so the
   * sidecar can carry the boundary words, plus the trace and trampolines. */
  {
    /* Index-based cut point: decimal slot 0..155 (the whole recovered
     * inventory). Anything else fails the whole run closed rather than
     * silently degrading to a wrong boundary. */
    char atbuf[16];
    DWORD a = GetEnvironmentVariableA("DLSS_NVAPI_CUT_AT", atbuf, sizeof(atbuf));
    if (a > 0) {
      char* end = NULL;
      long v;
      if (a >= sizeof(atbuf)) {
        fail = 1;
      } else {
        v = strtol(atbuf, &end, 10);
        if (end == atbuf || *end != 0 || v < 0 || v > 155) {
          fail = 1;
        } else {
          g_cut_at = v;
        }
      }
    }
    if (g_cut_at >= 0) { g_cut = 1; } /* index mode is a cut mode */
    char cutmode[8];
    DWORD c =
        GetEnvironmentVariableA("DLSS_NVAPI_CUT", cutmode, sizeof(cutmode));
    if (c > 0) {
      if (c < sizeof(cutmode) && cutmode[1] == 0 &&
          (cutmode[0] == '0' || cutmode[0] == '1')) {
        /* Both selectors at once is ambiguous: never silently degrade to one
         * of them (the stated policy of this file). Fail the run closed. */
        if (g_cut_at >= 0) {
          fail = 1;
        } else {
          g_cut = cutmode[0] - '0'; /* 0 = no truncation, like ARGS level 0 */
        }
      } else {
        fail = 1; /* unknown value: never silently degrade */
      }
    }
    if (g_cut) {
      char path[MAX_PATH];
      DWORD f =
          GetEnvironmentVariableA("DLSS_NVAPI_CUT_FILE", path, sizeof(path));
      if (f == 0 || f >= sizeof(path)) {
        fail = 1; /* never cut without recording the boundary words */
      } else {
        g_cut_file = CreateFileA(path, GENERIC_WRITE, FILE_SHARE_READ, NULL,
                                 CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (g_cut_file == INVALID_HANDLE_VALUE) fail = 1;
      }
      if (g_args < 2 || g_log == INVALID_HANDLE_VALUE || !g_wrap) fail = 1;
    }
    if (g_log != INVALID_HANDLE_VALUE) {
      log_line("cut-init armed=%d evals=0 drops=0", g_cut);
    }
  }
  if (fail) g_real_qi = NULL;
  g_init_done = 1;
  ReleaseSRWLockExclusive(&g_lock);
  return g_real_qi != NULL;
}

__declspec(dllexport) void* __cdecl nvapi_QueryInterface(unsigned int id) {
  void* real;
  void* caller = __builtin_return_address(0);
  if (!ensure_init()) return NULL;
  real = g_real_qi(id);
  AcquireSRWLockExclusive(&g_lock);
  log_line("result id=0x%08x ptr=%p", id, real);
  log_site("resolve", id, caller);
  ReleaseSRWLockExclusive(&g_lock);
  if (real == NULL) return NULL;
  if (!g_wrap) return real;
  AcquireSRWLockExclusive(&g_lock);
  real = wrap_entry(id, real);
  ReleaseSRWLockExclusive(&g_lock);
  return real;
}

BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID reserved) {
  (void)inst;
  (void)reserved;
  if (reason == DLL_PROCESS_ATTACH) {
    DisableThreadLibraryCalls((HMODULE)inst);
  }
  return TRUE;
}
