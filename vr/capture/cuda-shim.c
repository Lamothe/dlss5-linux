/* Host-side CUDA call tracer for reference capture.
 *
 * Interposes the CUDA driver entry points used by the DLSSNR runtime inside the
 * Wine/Proton process (the runtime resolves its CUDA function table at run time
 * and the Wine process maps host libcuda.so). Forwarding is transparent; the
 * trace log records module/function identity, launch geometry, argument blocks,
 * allocations and (optionally) tensor contents.
 *
 * The runtime looks up unsuffixed entry-point names (cuMemAlloc, cuMemcpy2D,
 * ...). cuda.h aliases several of these to *_v2 macros, so this shim undefines
 * the aliases and exports BOTH spellings, each forwarding to the same-named
 * real entry point first. Note the historical ABI caveat: the unsuffixed
 * driver symbols predate 64-bit sizes on some platforms, so logged sizes for
 * those entry points can carry garbage upper register bits; forwarding is
 * behaviour-preserving because the real callee reads the same width.
 *
 * Environment:
 *   DLSS_SHIM_REAL_LIBCUDA  absolute path of the real libcuda (auto-detected)
 *   DLSS_SHIM_LOG           trace file path (required to enable logging)
 *   DLSS_SHIM_DUMP_MEMCPY   1 = dump deduplicated host->device tensor payloads
 *   DLSS_SHIM_SNAPSHOT_FUNCS substring; after matching launches, device
 *                           allocations referenced by the argument block are
 *                           read back once each
 *   DLSS_SHIM_SNAPSHOT_DIR  directory for dumps (default: dirname of log)
 *   DLSS_SHIM_SNAPSHOT_MAX  max bytes per snapshot window (default 33554432)
 *
 * Everything is best-effort and bounded: failures are logged and the call is
 * forwarded unchanged. Dumped payloads are raw memory for offline comparison;
 * this tracer does not interpret model semantics.
 */
#define _GNU_SOURCE
#include <cuda.h>
/* Remove cuda.h's legacy-name aliases so our exports keep exact call names. */
#undef cuCtxCreate
#undef cuMemAlloc
#undef cuMemAllocHost
#undef cuMemFree
#undef cuMemcpyHtoD
#undef cuMemcpyHtoDAsync
#undef cuMemcpyDtoH
#undef cuMemcpyDtoHAsync
#undef cuMemcpy2D
#undef cuMemcpy2DAsync
#undef cuMemsetD32
#undef cuMemsetD32Async
#include <dlfcn.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

typedef CUresult (*fn_cuInit)(unsigned int);
typedef CUresult (*fn_cuCtxCreate)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*fn_cuCtxSetCurrent)(CUcontext);
typedef CUresult (*fn_cuCtxGetDevice)(CUdevice*);
typedef CUresult (*fn_cuCtxSynchronize)(void);
typedef CUresult (*fn_cuModuleLoadData)(CUmodule*, const void*);
typedef CUresult (*fn_cuModuleLoadDataEx)(CUmodule*, const void*, unsigned int, CUjit_option*, void**);
typedef CUresult (*fn_cuModuleUnload)(CUmodule);
typedef CUresult (*fn_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
typedef CUresult (*fn_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                                      unsigned int, unsigned int, unsigned int, unsigned int,
                                      CUstream, void**, void**);
typedef CUresult (*fn_cuMemAlloc)(CUdeviceptr*, size_t);
typedef CUresult (*fn_cuMemAllocHost)(void**, size_t);
typedef CUresult (*fn_cuMemFree)(CUdeviceptr);
typedef CUresult (*fn_cuMemFreeHost)(void*);
typedef CUresult (*fn_cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
typedef CUresult (*fn_cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream);
typedef CUresult (*fn_cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
typedef CUresult (*fn_cuMemcpyDtoHAsync)(void*, CUdeviceptr, size_t, CUstream);
typedef CUresult (*fn_cuMemcpy2D)(const CUDA_MEMCPY2D*);
typedef CUresult (*fn_cuMemcpy2DAsync)(const CUDA_MEMCPY2D*, CUstream);
typedef CUresult (*fn_cuMemsetD32)(CUdeviceptr, unsigned int, size_t);
typedef CUresult (*fn_cuMemsetD32Async)(CUdeviceptr, unsigned int, size_t, CUstream);
typedef CUresult (*fn_cuStreamSynchronize)(CUstream);

static struct {
    fn_cuInit cuInit;
    fn_cuCtxCreate cuCtxCreate;
    fn_cuCtxSetCurrent cuCtxSetCurrent;
    fn_cuCtxGetDevice cuCtxGetDevice;
    fn_cuCtxSynchronize cuCtxSynchronize;
    fn_cuModuleLoadData cuModuleLoadData;
    fn_cuModuleLoadDataEx cuModuleLoadDataEx;
    fn_cuModuleUnload cuModuleUnload;
    fn_cuModuleGetFunction cuModuleGetFunction;
    fn_cuLaunchKernel cuLaunchKernel;
    fn_cuMemAlloc cuMemAlloc;
    fn_cuMemAllocHost cuMemAllocHost;
    fn_cuMemFree cuMemFree;
    fn_cuMemFreeHost cuMemFreeHost;
    fn_cuMemcpyHtoD cuMemcpyHtoD;
    fn_cuMemcpyHtoDAsync cuMemcpyHtoDAsync;
    fn_cuMemcpyDtoH cuMemcpyDtoH;
    fn_cuMemcpyDtoHAsync cuMemcpyDtoHAsync;
    fn_cuMemcpy2D cuMemcpy2D;
    fn_cuMemcpy2DAsync cuMemcpy2DAsync;
    fn_cuMemsetD32 cuMemsetD32;
    fn_cuMemsetD32Async cuMemsetD32Async;
    fn_cuStreamSynchronize cuStreamSynchronize;
} real;

static void* (*real_dlsym)(void*, const char*);
static void* real_handle;
static FILE* log_file;
static pthread_mutex_t log_lock = PTHREAD_MUTEX_INITIALIZER;
static unsigned long long seq_counter;
static char snapshot_dir[512];
static const char* snapshot_funcs;
static size_t snapshot_max = 33554432u;
static int dump_memcpy;

/* ---- allocation bookkeeping (device and host) ---- */
#define MAX_ALLOCS 4096
typedef struct { uint64_t base; size_t size; int is_host; unsigned long long id; } AllocRec;
static AllocRec allocs[MAX_ALLOCS];
static unsigned int alloc_count;
static unsigned long long alloc_id_counter;
static pthread_mutex_t alloc_lock = PTHREAD_MUTEX_INITIALIZER;

/* ---- module/function bookkeeping ---- */
#define MAX_FUNCS 4096
typedef struct { void* handle; char name[96]; unsigned long long id; } FuncRec;
static FuncRec funcs[MAX_FUNCS];
static unsigned int func_count;
static pthread_mutex_t func_lock = PTHREAD_MUTEX_INITIALIZER;

/* ---- dedupe for memcpy dumps and snapshots ---- */
#define MAX_DUMPS 1024
typedef struct { uint64_t id; uint64_t off; } SnapshotKey;
static SnapshotKey snapshot_seen[MAX_DUMPS];
static unsigned int snapshot_seen_count;
static uint64_t memcpy_seen_digest[MAX_DUMPS];
static unsigned int memcpy_seen_count;
static pthread_mutex_t dump_lock = PTHREAD_MUTEX_INITIALIZER;

static void log_line(const char* fmt, ...) {
    va_list ap;
    if (!log_file) { return; }
    pthread_mutex_lock(&log_lock);
    fprintf(log_file, "%llu|%llu|", ++seq_counter, (unsigned long long)pthread_self());
    va_start(ap, fmt);
    vfprintf(log_file, fmt, ap);
    va_end(ap);
    fputc('\n', log_file);
    /* Bounded flush cadence: every 64 lines keeps data safe without hammering. */
    if ((seq_counter & 63u) == 0u) { fflush(log_file); }
    pthread_mutex_unlock(&log_lock);
}

static uint64_t fnv1a(const void* data, size_t size) {
    const unsigned char* p = (const unsigned char*)data;
    uint64_t hash = 14695981039346656037ull;
    for (size_t i = 0; i < size; ++i) { hash ^= p[i]; hash *= 1099511628211ull; }
    return hash;
}

/* Returns 1 = seen before, -1 = newly recorded, 0 = table full. */
static int snapshot_key_seen(uint64_t id, uint64_t off) {
    int found = 0;
    pthread_mutex_lock(&dump_lock);
    for (unsigned int i = 0; i < snapshot_seen_count; ++i) {
        if (snapshot_seen[i].id == id && snapshot_seen[i].off == off) { found = 1; break; }
    }
    if (!found && snapshot_seen_count < MAX_DUMPS) {
        snapshot_seen[snapshot_seen_count].id = id;
        snapshot_seen[snapshot_seen_count].off = off;
        ++snapshot_seen_count;
        found = -1;
    }
    pthread_mutex_unlock(&dump_lock);
    return found;
}

static int memcpy_digest_seen(uint64_t digest) {
    int found = 0;
    pthread_mutex_lock(&dump_lock);
    for (unsigned int i = 0; i < memcpy_seen_count; ++i) {
        if (memcpy_seen_digest[i] == digest) { found = 1; break; }
    }
    if (!found && memcpy_seen_count < MAX_DUMPS) {
        memcpy_seen_digest[memcpy_seen_count++] = digest;
        found = -1;
    }
    pthread_mutex_unlock(&dump_lock);
    return found;
}

static void write_blob(const char* tag, unsigned long long id, const void* data, size_t size) {
    char path[768];
    FILE* out;
    int written;
    if (!snapshot_dir[0]) { return; }
    written = snprintf(path, sizeof(path), "%s/%s-%llu.bin", snapshot_dir, tag, id);
    if (written < 0 || (size_t)written >= sizeof(path)) {
        log_line("error|write_blob_path_too_long|%s", tag);
        return;
    }
    out = fopen(path, "wb");
    if (!out) { log_line("error|write_blob|%s", path); return; }
    if (size) { fwrite(data, 1, size, out); }
    fclose(out);
    log_line("blob|%s|%llu|%zu", tag, id, size);
}

static void record_alloc(uint64_t base, size_t size, int is_host, const char* kind) {
    unsigned long long id;
    if (!base || !size) { return; }
    pthread_mutex_lock(&alloc_lock);
    if (alloc_count < MAX_ALLOCS) {
        id = ++alloc_id_counter;
        allocs[alloc_count].base = base;
        allocs[alloc_count].size = size;
        allocs[alloc_count].is_host = is_host;
        allocs[alloc_count].id = id;
        ++alloc_count;
        log_line("alloc|%s|%llu|0x%llx|%zu", kind, id, (unsigned long long)base, size);
    } else {
        log_line("error|alloc_table_full|%s|0x%llx|%zu", kind, (unsigned long long)base, size);
    }
    pthread_mutex_unlock(&alloc_lock);
}

static void forget_alloc(uint64_t base, const char* kind) {
    pthread_mutex_lock(&alloc_lock);
    for (unsigned int i = 0; i < alloc_count; ++i) {
        if (allocs[i].base == base) {
            log_line("free|%s|%llu|0x%llx", kind, allocs[i].id, (unsigned long long)base);
            allocs[i] = allocs[alloc_count - 1];
            --alloc_count;
            break;
        }
    }
    pthread_mutex_unlock(&alloc_lock);
}

/* Find the allocation containing addr; returns id and fills offset/size. */
static unsigned long long lookup_alloc(uint64_t addr, int is_host, size_t* offset, size_t* size) {
    unsigned long long id = 0;
    pthread_mutex_lock(&alloc_lock);
    for (unsigned int i = 0; i < alloc_count; ++i) {
        if (allocs[i].is_host != is_host) { continue; }
        if (addr >= allocs[i].base && addr < allocs[i].base + allocs[i].size) {
            id = allocs[i].id;
            if (offset) { *offset = (size_t)(addr - allocs[i].base); }
            if (size) { *size = allocs[i].size; }
            break;
        }
    }
    pthread_mutex_unlock(&alloc_lock);
    return id;
}

static void snapshot_referenced(const char* fn_name, unsigned long long launch_id,
                                const unsigned char* arg_bytes, size_t arg_size) {
    char tag[256];
    if (!snapshot_funcs || !real.cuMemcpyDtoH || !real.cuCtxSynchronize) { return; }
    if (!strstr(fn_name, snapshot_funcs)) { return; }
    log_line("snapshot|begin|%s|%llu|arg_bytes=%zu", fn_name, launch_id, arg_size);
    if (real.cuCtxSynchronize() != CUDA_SUCCESS) {
        log_line("error|snapshot_sync|%s|%llu", fn_name, launch_id);
        return;
    }
    for (size_t off = 0; off + 8 <= arg_size; off += 8) {
        uint64_t value;
        size_t alloc_size = 0, alloc_offset = 0;
        unsigned long long id;
        void* buffer;
        size_t window;
        memcpy(&value, arg_bytes + off, 8);
        if (value < 0x10000ull) { continue; }
        id = lookup_alloc(value, 0, &alloc_offset, &alloc_size);
        if (!id) { continue; }
        if (snapshot_key_seen(id, (uint64_t)off) != -1) { continue; }
        window = alloc_size - alloc_offset;
        if (window > snapshot_max) { window = snapshot_max; }
        buffer = malloc(window ? window : 1);
        if (!buffer) { continue; }
        if (real.cuMemcpyDtoH(buffer, (CUdeviceptr)value, window) == CUDA_SUCCESS) {
            snprintf(tag, sizeof(tag), "snap-%s-l%llu-a%llu-o%zu", fn_name, launch_id, id, alloc_offset);
            write_blob(tag, (unsigned long long)off, buffer, window);
        } else {
            log_line("error|snapshot_copy|%s|%llu|alloc=%llu", fn_name, launch_id, id);
        }
        free(buffer);
    }
    log_line("snapshot|end|%s|%llu", fn_name, launch_id);
}

static void resolve_real(void) {
    const char* candidates[5];
    const char* env_path = getenv("DLSS_SHIM_REAL_LIBCUDA");
    candidates[0] = env_path;
    candidates[1] = "/run/host/usr/lib64/libcuda.so.1";
    candidates[2] = "/usr/lib64/libcuda.so.1";
    candidates[3] = "/lib64/libcuda.so.1";
    candidates[4] = "/usr/lib/x86_64-linux-gnu/libcuda.so.1";
    for (unsigned int i = 0; i < 5; ++i) {
        if (!candidates[i] || !candidates[i][0]) { continue; }
        real_handle = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (real_handle) {
            log_line("real_lib|%s", candidates[i]);
            return;
        }
    }
    log_line("error|real_lib_missing");
}

/* Prefer the same-named real entry point; fall back to the sibling spelling. */
#define RESOLVE_PAIR(field, name, alt) \
    do { *(void**)&real.field = real_dlsym(real_handle, name); \
         if (!real.field && (alt)) { *(void**)&real.field = real_dlsym(real_handle, alt); } \
         log_line("resolve|%s|%s", name, real.field ? "ok" : "missing"); } while (0)

static void init_shim(void) {
    const char* log_path = getenv("DLSS_SHIM_LOG");
    const char* dir = getenv("DLSS_SHIM_SNAPSHOT_DIR");
    const char* max = getenv("DLSS_SHIM_SNAPSHOT_MAX");
    snapshot_funcs = getenv("DLSS_SHIM_SNAPSHOT_FUNCS");
    dump_memcpy = getenv("DLSS_SHIM_DUMP_MEMCPY") && getenv("DLSS_SHIM_DUMP_MEMCPY")[0] == '1';
    if (max && max[0]) {
        size_t parsed = (size_t)strtoull(max, NULL, 10);
        if (parsed > 0) { snapshot_max = parsed; }
    }
    if (log_path && log_path[0]) {
        log_file = fopen(log_path, "a");
        if (!log_file) { return; }
    } else {
        return; /* no log configured: act as a transparent forwarder */
    }
    if (dir && dir[0]) {
        snprintf(snapshot_dir, sizeof(snapshot_dir), "%s", dir);
    } else {
        snprintf(snapshot_dir, sizeof(snapshot_dir), "%s", log_path);
        char* slash = strrchr(snapshot_dir, '/');
        if (slash) { *slash = '\0'; }
    }
    mkdir(snapshot_dir, 0755);
    log_line("shim|start|pid=%d|snapshot_funcs=%s|snapshot_max=%zu|dump_memcpy=%d",
             (int)getpid(), snapshot_funcs ? snapshot_funcs : "-", snapshot_max, dump_memcpy);
    resolve_real();
    if (!real_handle) { return; }
    RESOLVE_PAIR(cuInit, "cuInit", NULL);
    RESOLVE_PAIR(cuCtxCreate, "cuCtxCreate", "cuCtxCreate_v2");
    RESOLVE_PAIR(cuCtxSetCurrent, "cuCtxSetCurrent", NULL);
    RESOLVE_PAIR(cuCtxGetDevice, "cuCtxGetDevice", NULL);
    RESOLVE_PAIR(cuCtxSynchronize, "cuCtxSynchronize", NULL);
    RESOLVE_PAIR(cuModuleLoadData, "cuModuleLoadData", NULL);
    RESOLVE_PAIR(cuModuleLoadDataEx, "cuModuleLoadDataEx", NULL);
    RESOLVE_PAIR(cuModuleUnload, "cuModuleUnload", NULL);
    RESOLVE_PAIR(cuModuleGetFunction, "cuModuleGetFunction", NULL);
    RESOLVE_PAIR(cuLaunchKernel, "cuLaunchKernel", NULL);
    RESOLVE_PAIR(cuMemAlloc, "cuMemAlloc", "cuMemAlloc_v2");
    RESOLVE_PAIR(cuMemAllocHost, "cuMemAllocHost", "cuMemAllocHost_v2");
    RESOLVE_PAIR(cuMemFree, "cuMemFree", "cuMemFree_v2");
    RESOLVE_PAIR(cuMemFreeHost, "cuMemFreeHost", NULL);
    RESOLVE_PAIR(cuMemcpyHtoD, "cuMemcpyHtoD", "cuMemcpyHtoD_v2");
    RESOLVE_PAIR(cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync", "cuMemcpyHtoDAsync_v2");
    RESOLVE_PAIR(cuMemcpyDtoH, "cuMemcpyDtoH", "cuMemcpyDtoH_v2");
    RESOLVE_PAIR(cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync", "cuMemcpyDtoHAsync_v2");
    RESOLVE_PAIR(cuMemcpy2D, "cuMemcpy2D", "cuMemcpy2D_v2");
    RESOLVE_PAIR(cuMemcpy2DAsync, "cuMemcpy2DAsync", "cuMemcpy2DAsync_v2");
    RESOLVE_PAIR(cuMemsetD32, "cuMemsetD32", "cuMemsetD32_v2");
    RESOLVE_PAIR(cuMemsetD32Async, "cuMemsetD32Async", "cuMemsetD32Async_v2");
    RESOLVE_PAIR(cuStreamSynchronize, "cuStreamSynchronize", NULL);
}

__attribute__((constructor)) static void shim_ctor(void) {
    /* Resolve glibc's dlsym via dlvsym so the dlsym hook below cannot recurse. */
    real_dlsym = (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    init_shim();
}

/* ---- dlsym hook: covers runtimes that resolve entry points by name ---- */
void* dlsym(void* handle, const char* symbol) {
    void* result;
    if (!real_dlsym) {
        real_dlsym = (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }
    result = real_dlsym ? real_dlsym(handle, symbol) : NULL;
    /* glibc declares `symbol` nonnull; the lookup name is trusted here. */
    if (result && log_file && strncmp(symbol, "cu", 2) == 0) {
        void* ours = real_dlsym ? real_dlsym(RTLD_DEFAULT, symbol) : NULL;
        if (ours && ours != result) {
            log_line("dlsym_hook|%s|intercepted", symbol);
            result = ours;
        }
    }
    return result;
}

static void hex_arg_bytes(const unsigned char* bytes, size_t size, char* out, size_t out_size) {
    static const char digits[] = "0123456789abcdef";
    size_t limit = size < 256u ? size : 256u;
    size_t used = 0;
    for (size_t i = 0; i < limit && used + 3 < out_size; ++i) {
        out[used++] = digits[bytes[i] >> 4];
        out[used++] = digits[bytes[i] & 15u];
    }
    out[used < out_size ? used : out_size - 1] = '\0';
}

static void parse_extra_launch(void** extra, const unsigned char** bytes, size_t* size) {
    *bytes = NULL;
    *size = 0;
    if (!extra) { return; }
    for (unsigned int i = 0; i < 63u; ++i) {
        void* token = extra[i];
        void* value;
        if (token == CU_LAUNCH_PARAM_END) { return; }
        if (token != CU_LAUNCH_PARAM_BUFFER_POINTER && token != CU_LAUNCH_PARAM_BUFFER_SIZE) {
            return; /* unknown token: do not walk past the end */
        }
        value = extra[i + 1];
        /* Malformed value slots (NULL, token, END) are never dereferenced. */
        if (value == NULL || value == CU_LAUNCH_PARAM_END ||
            value == CU_LAUNCH_PARAM_BUFFER_POINTER || value == CU_LAUNCH_PARAM_BUFFER_SIZE) {
            return;
        }
        if (token == CU_LAUNCH_PARAM_BUFFER_POINTER) {
            *bytes = (const unsigned char*)value;
        } else {
            *size = *(size_t*)value;
        }
        ++i;
    }
}

/* ---- wrapped entry points ---- */
CUresult cuInit(unsigned int flags) {
    CUresult result = real.cuInit ? real.cuInit(flags) : CUDA_ERROR_NOT_INITIALIZED;
    log_line("cuInit|0x%x|%d", flags, (int)result);
    return result;
}

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    CUresult result = real.cuCtxCreate ? real.cuCtxCreate(pctx, flags, dev) : CUDA_ERROR_NOT_INITIALIZED;
    log_line("cuCtxCreate|%d|0x%x|ctx=%p|%d", (int)dev, flags, (void*)(pctx ? *pctx : NULL), (int)result);
    return result;
}
/* Export the v2 spelling with identical semantics. */
CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    return cuCtxCreate(pctx, flags, dev);
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    CUresult result = real.cuCtxSetCurrent ? real.cuCtxSetCurrent(ctx) : CUDA_ERROR_NOT_INITIALIZED;
    log_line("cuCtxSetCurrent|%p|%d", (void*)ctx, (int)result);
    return result;
}

CUresult cuCtxGetDevice(CUdevice* device) {
    CUresult result = real.cuCtxGetDevice ? real.cuCtxGetDevice(device) : CUDA_ERROR_NOT_INITIALIZED;
    log_line("cuCtxGetDevice|%d|%d", device ? (int)*device : -1, (int)result);
    return result;
}

CUresult cuCtxSynchronize(void) {
    CUresult result = real.cuCtxSynchronize ? real.cuCtxSynchronize() : CUDA_ERROR_NOT_INITIALIZED;
    log_line("cuCtxSynchronize|%d", (int)result);
    return result;
}

CUresult cuModuleLoadData(CUmodule* module, const void* image) {
    CUresult result = real.cuModuleLoadData ? real.cuModuleLoadData(module, image) : CUDA_ERROR_NOT_INITIALIZED;
    const unsigned char* p = (const unsigned char*)image;
    size_t size = 0;
    size_t digest_len = 256u;
    if (p && p[0] == 0x7f && p[1] == 'E') {
        uint64_t shoff;
        uint16_t shentsize, shnum;
        memcpy(&shoff, p + 0x28, 8);
        memcpy(&shentsize, p + 0x3A, 2);
        memcpy(&shnum, p + 0x3C, 2);
        size = (size_t)(shoff + (uint64_t)shentsize * shnum);
        if (size && size < digest_len) { digest_len = size; }
    } else if (p && p[0] == 0x50 && p[1] == 0xED && p[2] == 0x55 && p[3] == 0xBA) {
        size = 0; /* CUDA fatbin magic 0xBA55ED50: extent not parsed, prefix digest only */
    }
    /* Heuristic bound: the driver itself reads the image, so the first 256
     * bytes of a driver-accepted image are assumed mapped. */
    log_line("cuModuleLoadData|mod=%p|image=%p|size=%zu|prefix=%02x%02x%02x%02x|prefix_fnv=0x%llx|%d",
             (void*)(module ? *module : NULL), image, size,
             p ? p[0] : 0, p ? p[1] : 0, p ? p[2] : 0, p ? p[3] : 0,
             p ? (unsigned long long)fnv1a(p, digest_len) : 0ull, (int)result);
    return result;
}

CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int num_options,
                            CUjit_option* options, void** option_values) {
    CUresult result = real.cuModuleLoadDataEx ? real.cuModuleLoadDataEx(module, image, num_options, options, option_values)
                                              : CUDA_ERROR_NOT_INITIALIZED;
    log_line("cuModuleLoadDataEx|mod=%p|image=%p|options=%u|%d",
             (void*)(module ? *module : NULL), image, num_options, (int)result);
    return result;
}

CUresult cuModuleUnload(CUmodule module) {
    CUresult result = real.cuModuleUnload ? real.cuModuleUnload(module) : CUDA_ERROR_NOT_INITIALIZED;
    log_line("cuModuleUnload|%p|%d", (void*)module, (int)result);
    return result;
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    CUresult result = real.cuModuleGetFunction ? real.cuModuleGetFunction(hfunc, hmod, name)
                                               : CUDA_ERROR_NOT_INITIALIZED;
    if (result == CUDA_SUCCESS && hfunc) {
        pthread_mutex_lock(&func_lock);
        if (func_count < MAX_FUNCS) {
            unsigned int index = func_count;
            funcs[index].handle = (void*)*hfunc;
            snprintf(funcs[index].name, sizeof(funcs[index].name), "%s", name ? name : "?");
            funcs[index].id = index + 1u;
            func_count = index + 1u;
        }
        pthread_mutex_unlock(&func_lock);
    }
    log_line("cuModuleGetFunction|mod=%p|fn=%p|name=%s|%d",
             (void*)hmod, (void*)(hfunc ? *hfunc : NULL), name ? name : "?", (int)result);
    return result;
}

static void function_name(void* handle, char* out, size_t out_size) {
    snprintf(out, out_size, "%p", handle);
    pthread_mutex_lock(&func_lock);
    for (unsigned int i = 0; i < func_count; ++i) {
        if (funcs[i].handle == handle) {
            snprintf(out, out_size, "%s", funcs[i].name);
            break;
        }
    }
    pthread_mutex_unlock(&func_lock);
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz,
                        unsigned int bx, unsigned int by, unsigned int bz,
                        unsigned int shared_mem_bytes, CUstream stream,
                        void** kernel_params, void** extra) {
    const unsigned char* arg_bytes = NULL;
    size_t arg_size = 0;
    char name[96];
    char hex[520];
    unsigned long long launch_id;
    CUresult result;

    if (extra) {
        parse_extra_launch(extra, &arg_bytes, &arg_size);
        if (arg_bytes && arg_size > (1u << 20)) { arg_size = 0; } /* defensive bound */
    } else if (kernel_params && kernel_params[0]) {
        arg_bytes = (const unsigned char*)kernel_params[0];
        arg_size = 256u; /* unknown contract: log a bounded prefix only */
    }
    function_name((void*)f, name, sizeof(name));
    hex[0] = '\0';
    if (arg_bytes && arg_size) { hex_arg_bytes(arg_bytes, arg_size, hex, sizeof(hex)); }
    result = real.cuLaunchKernel ? real.cuLaunchKernel(f, gx, gy, gz, bx, by, bz,
                                                      shared_mem_bytes, stream,
                                                      kernel_params, extra)
                                 : CUDA_ERROR_NOT_INITIALIZED;
    if (log_file) {
        pthread_mutex_lock(&log_lock);
        launch_id = ++seq_counter;
        fprintf(log_file, "%llu|%llu|cuLaunchKernel|fn=%s|grid=%u,%u,%u|block=%u,%u,%u|shared=%u|stream=%p|args=%zu|argc=%s|fnv=0x%llx|%d\n",
                launch_id, (unsigned long long)pthread_self(), name, gx, gy, gz, bx, by, bz,
                shared_mem_bytes, (void*)stream, arg_size, hex,
                arg_bytes ? (unsigned long long)fnv1a(arg_bytes, arg_size) : 0ull, (int)result);
        fflush(log_file);
        pthread_mutex_unlock(&log_lock);
    } else {
        launch_id = 0; /* transparent-forwarder mode: no logging, no snapshots */
    }
    if (log_file && result == CUDA_SUCCESS && arg_bytes && arg_size) {
        snapshot_referenced(name, launch_id, arg_bytes, arg_size);
    }
    return result;
}

CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    CUresult result = real.cuMemAlloc ? real.cuMemAlloc(dptr, bytesize) : CUDA_ERROR_NOT_INITIALIZED;
    if (result == CUDA_SUCCESS && dptr) { record_alloc((uint64_t)*dptr, bytesize, 0, "device"); }
    log_line("cuMemAlloc|0x%llx|%zu|%d", (unsigned long long)(dptr ? *dptr : 0), bytesize, (int)result);
    return result;
}
CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) { return cuMemAlloc(dptr, bytesize); }

CUresult cuMemAllocHost(void** pp, size_t bytesize) {
    CUresult result = real.cuMemAllocHost ? real.cuMemAllocHost(pp, bytesize) : CUDA_ERROR_NOT_INITIALIZED;
    if (result == CUDA_SUCCESS && pp) { record_alloc((uint64_t)(uintptr_t)*pp, bytesize, 1, "host"); }
    log_line("cuMemAllocHost|%p|%zu|%d", pp ? *pp : NULL, bytesize, (int)result);
    return result;
}
CUresult cuMemAllocHost_v2(void** pp, size_t bytesize) { return cuMemAllocHost(pp, bytesize); }

CUresult cuMemFree(CUdeviceptr dptr) {
    CUresult result = real.cuMemFree ? real.cuMemFree(dptr) : CUDA_ERROR_NOT_INITIALIZED;
    forget_alloc((uint64_t)dptr, "device");
    return result;
}
CUresult cuMemFree_v2(CUdeviceptr dptr) { return cuMemFree(dptr); }

CUresult cuMemFreeHost(void* p) {
    CUresult result = real.cuMemFreeHost ? real.cuMemFreeHost(p) : CUDA_ERROR_NOT_INITIALIZED;
    forget_alloc((uint64_t)(uintptr_t)p, "host");
    return result;
}

static void log_memcpy(const char* direction, uint64_t dst, uint64_t src, size_t size,
                       const void* host_bytes) {
    uint64_t digest = host_bytes && size ? fnv1a(host_bytes, size) : 0ull;
    unsigned long long src_id = lookup_alloc(src, 1, NULL, NULL);
    unsigned long long dst_id = lookup_alloc(dst, 0, NULL, NULL);
    log_line("memcpy|%s|dst=0x%llx(dev=%llu)|src=0x%llx(host=%llu)|size=%zu|fnv=0x%llx",
             direction, (unsigned long long)dst, dst_id, (unsigned long long)src, src_id, size, digest);
    if (dump_memcpy && host_bytes && size && strncmp(direction, "HtoD", 4) == 0) {
        if (memcpy_digest_seen(digest) == -1) {
            write_blob("memcpy-h2d", (unsigned long long)digest, host_bytes, size);
        }
    }
}

CUresult cuMemcpyHtoD(CUdeviceptr dst, const void* src, size_t size) {
    CUresult result = real.cuMemcpyHtoD ? real.cuMemcpyHtoD(dst, src, size) : CUDA_ERROR_NOT_INITIALIZED;
    log_memcpy("HtoD", (uint64_t)dst, (uint64_t)(uintptr_t)src, size, src);
    return result;
}
CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void* src, size_t size) { return cuMemcpyHtoD(dst, src, size); }

CUresult cuMemcpyHtoDAsync(CUdeviceptr dst, const void* src, size_t size, CUstream stream) {
    CUresult result = real.cuMemcpyHtoDAsync ? real.cuMemcpyHtoDAsync(dst, src, size, stream)
                                             : CUDA_ERROR_NOT_INITIALIZED;
    log_memcpy("HtoDAsync", (uint64_t)dst, (uint64_t)(uintptr_t)src, size, src);
    return result;
}
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void* src, size_t size, CUstream stream) {
    return cuMemcpyHtoDAsync(dst, src, size, stream);
}

CUresult cuMemcpyDtoH(void* dst, CUdeviceptr src, size_t size) {
    CUresult result = real.cuMemcpyDtoH ? real.cuMemcpyDtoH(dst, src, size) : CUDA_ERROR_NOT_INITIALIZED;
    log_memcpy("DtoH", (uint64_t)(uintptr_t)dst, (uint64_t)src, size, NULL);
    return result;
}
CUresult cuMemcpyDtoH_v2(void* dst, CUdeviceptr src, size_t size) { return cuMemcpyDtoH(dst, src, size); }

CUresult cuMemcpyDtoHAsync(void* dst, CUdeviceptr src, size_t size, CUstream stream) {
    CUresult result = real.cuMemcpyDtoHAsync ? real.cuMemcpyDtoHAsync(dst, src, size, stream)
                                             : CUDA_ERROR_NOT_INITIALIZED;
    log_memcpy("DtoHAsync", (uint64_t)(uintptr_t)dst, (uint64_t)src, size, NULL);
    return result;
}
CUresult cuMemcpyDtoHAsync_v2(void* dst, CUdeviceptr src, size_t size, CUstream stream) {
    return cuMemcpyDtoHAsync(dst, src, size, stream);
}

CUresult cuMemcpy2D(const CUDA_MEMCPY2D* copy) {
    CUresult result = real.cuMemcpy2D ? real.cuMemcpy2D(copy) : CUDA_ERROR_NOT_INITIALIZED;
    if (copy) {
        const void* host_bytes = copy->srcMemoryType == CU_MEMORYTYPE_HOST ? copy->srcHost : NULL;
        log_memcpy("2D", (uint64_t)(uintptr_t)copy->dstDevice, (uint64_t)(uintptr_t)copy->srcHost,
                   (size_t)copy->WidthInBytes * copy->Height, host_bytes);
        log_line("memcpy2d|src_pitch=%zu|dst_pitch=%zu|wbytes=%zu|h=%u|src_type=%d|dst_type=%d",
                 (size_t)copy->srcPitch, (size_t)copy->dstPitch, (size_t)copy->WidthInBytes,
                 (unsigned int)copy->Height, (int)copy->srcMemoryType, (int)copy->dstMemoryType);
    }
    return result;
}
CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D* copy) { return cuMemcpy2D(copy); }

CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D* copy, CUstream stream) {
    CUresult result = real.cuMemcpy2DAsync ? real.cuMemcpy2DAsync(copy, stream) : CUDA_ERROR_NOT_INITIALIZED;
    if (copy) {
        const void* host_bytes = copy->srcMemoryType == CU_MEMORYTYPE_HOST ? copy->srcHost : NULL;
        log_memcpy("2DAsync", (uint64_t)(uintptr_t)copy->dstDevice, (uint64_t)(uintptr_t)copy->srcHost,
                   (size_t)copy->WidthInBytes * copy->Height, host_bytes);
        log_line("memcpy2d|async=1|src_pitch=%zu|dst_pitch=%zu|wbytes=%zu|h=%u|src_type=%d|dst_type=%d|stream=%p",
                 (size_t)copy->srcPitch, (size_t)copy->dstPitch, (size_t)copy->WidthInBytes,
                 (unsigned int)copy->Height, (int)copy->srcMemoryType, (int)copy->dstMemoryType, (void*)stream);
    }
    return result;
}
CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D* copy, CUstream stream) { return cuMemcpy2DAsync(copy, stream); }

CUresult cuMemsetD32(CUdeviceptr dst, unsigned int value, size_t count) {
    CUresult result = real.cuMemsetD32 ? real.cuMemsetD32(dst, value, count) : CUDA_ERROR_NOT_INITIALIZED;
    log_line("memsetD32|dst=0x%llx|value=0x%08x|count=%zu|%d", (unsigned long long)dst, value, count, (int)result);
    return result;
}
CUresult cuMemsetD32_v2(CUdeviceptr dst, unsigned int value, size_t count) { return cuMemsetD32(dst, value, count); }

CUresult cuMemsetD32Async(CUdeviceptr dst, unsigned int value, size_t count, CUstream stream) {
    CUresult result = real.cuMemsetD32Async ? real.cuMemsetD32Async(dst, value, count, stream)
                                            : CUDA_ERROR_NOT_INITIALIZED;
    log_line("memsetD32Async|dst=0x%llx|value=0x%08x|count=%zu|stream=%p|%d",
             (unsigned long long)dst, value, count, (void*)stream, (int)result);
    return result;
}
CUresult cuMemsetD32Async_v2(CUdeviceptr dst, unsigned int value, size_t count, CUstream stream) {
    return cuMemsetD32Async(dst, value, count, stream);
}

CUresult cuStreamSynchronize(CUstream stream) {
    CUresult result = real.cuStreamSynchronize ? real.cuStreamSynchronize(stream) : CUDA_ERROR_NOT_INITIALIZED;
    log_line("cuStreamSynchronize|%p|%d", (void*)stream, (int)result);
    return result;
}
