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
 *
 * Everything is bounded and best-effort: failures fall back to transparent
 * forwarding and are logged where possible. The trace contains interface
 * IDs, pointers and code addresses only - no tensor or weight contents.
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
#define STUB_SIZE 256
#define ARENA_SIZE (MAX_TRAMPS * STUB_SIZE)

typedef void*(__cdecl* PFN_QI)(unsigned int);

static PFN_QI g_real_qi;
static HMODULE g_self;
static SRWLOCK g_lock = SRWLOCK_INIT;
static HANDLE g_log = INVALID_HANDLE_VALUE;
static int g_wrap;
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
  if (n < 0 || n >= (int)sizeof(buf)) return;
  buf[n++] = '\n';
  WriteFile(g_log, buf, (DWORD)n, &written, NULL);
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

static void* build_stub(unsigned int id, void* real) {
  unsigned char* p;
  unsigned char* base;
  size_t off_id, off_logger, off_target;
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
  EMIT(0x48, 0xB8);                   /* mov rax, tramp_logger */
  off_logger = (size_t)(p - base);
  EMIT(0, 0, 0, 0, 0, 0, 0, 0);
  EMIT(0xFF, 0xD0);                   /* call rax */
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
  EMIT(0x48, 0x83, 0xC4, 0x08);       /* add rsp, 8 (drop dup retaddr) */
  EMIT(0x48, 0xB8);                   /* mov rax, real */
  off_target = (size_t)(p - base);
  EMIT(0, 0, 0, 0, 0, 0, 0, 0);
  EMIT(0xFF, 0xE0);                   /* jmp rax */
#undef EMIT
  if ((size_t)(p - base) > STUB_SIZE) return NULL;
  memcpy(base + off_id, &id, sizeof(id));
  {
    uintptr_t v = (uintptr_t)&tramp_logger;
    memcpy(base + off_logger, &v, sizeof(v));
    v = (uintptr_t)real;
    memcpy(base + off_target, &v, sizeof(v));
  }
  FlushInstructionCache(GetCurrentProcess(), base, STUB_SIZE);
  g_used += STUB_SIZE;
  return base;
}

static void* wrap_entry(unsigned int id, void* real) {
  void* stub;
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
  stub = build_stub(id, real);
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
      log_line("init real=%p wrap=%d", (void*)g_real_qi, g_wrap);
    }
  } else if (n >= sizeof(env)) {
    fail = 1; /* truncated path: never forward without a trace */
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
