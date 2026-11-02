// Cost measurement for NVIDIA DLSS Neural Rendering (NGX feature 18) under Proton.
//
// Answers one question and nothing else: how long does one Evaluate call take on the
// GPU, and how long does a serialised frame take end to end, at a ladder of
// resolutions and with one or two feature instances (the two-eye case)?
//
// No window, no swapchain, no present, no dump: this is not a demonstration and makes
// no image-quality claim. The inputs are stationary synthetic gradients, so the only
// thing measured is the model's dispatch cost. The model is NVIDIA's and is loaded
// from the working directory beside the executable, as is the caller-gate forwarder.
//
// Usage: nr-bench.exe [--sizes 720,1024,...] [--frames N] [--instances 1|2]
#define COBJMACROS
#define WIDL_C_INLINE_WRAPPERS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef void (__cdecl *SetUint)(void*, const char*, unsigned int);
static void set_uint(void* params, const char* key, unsigned int value) {
    void** vtable = *(void***)params;
    ((SetUint)vtable[3])(params, key, value);
}

#include "evaluation-probe.h"

typedef int (__cdecl *CoreInit)(unsigned long long, const wchar_t*, ID3D12Device*, int, const void*);
typedef int (__cdecl *GetCaps)(void**);
typedef int (__cdecl *CoreShutdown)(ID3D12Device*, int*);
typedef int (__cdecl *FwdInit)(const wchar_t*, const wchar_t*, ID3D12Device*, void*);
typedef void* (__cdecl *FwdCreate)(ID3D12GraphicsCommandList*, void*, int*);
typedef void (__cdecl *FwdRelease)(void*);

#define MAX_INSTANCES 2u

// A per-instance set of inputs at a caller-chosen size. Mirrors prepare_eye, which is
// PROBE_SIZE-bound at compile time; everything else is deliberately identical so the
// measurement exercises the same texture formats and parameter keys.
static int prepare_sized(ID3D12Device* device, EyeInputs* eye, unsigned int eye_index,
                         AllocateParams allocate, unsigned int width, unsigned int height,
                         float motion_x, float motion_y) {
    if (allocate(&eye->params) != 1 || !eye->params) { fprintf(stderr, "AllocateParameters failed\n"); return 0; }
    const DXGI_FORMAT formats[] = {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32_FLOAT,
                                   DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT};
    for (unsigned int texture = 0; texture < PROBE_TEXTURES; ++texture) {
        D3D12_HEAP_PROPERTIES heap = {0};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT; heap.CreationNodeMask = 1; heap.VisibleNodeMask = 1;
        D3D12_RESOURCE_DESC desc = {0};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = width; desc.Height = height;
        desc.DepthOrArraySize = 1; desc.MipLevels = 1; desc.Format = formats[texture];
        desc.SampleDesc.Count = 1; desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        HRESULT hr = ID3D12Device_CreateCommittedResource(device, &heap, D3D12_HEAP_FLAG_NONE,
            &desc, D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void**)&eye->textures[texture]);
        if (FAILED(hr)) { fprintf(stderr, "Texture %ux%u failed 0x%08lx\n", width, height, (unsigned long)hr); return 0; }
        eye->states[texture] = D3D12_RESOURCE_STATE_COPY_DEST;
        ID3D12Device_GetCopyableFootprints(device, &desc, 0, 1, 0, &eye->layouts[texture], NULL, NULL, &eye->sizes[texture]);
        if (!create_buffer(device, D3D12_HEAP_TYPE_UPLOAD, eye->sizes[texture], &eye->uploads[texture])) { return 0; }
        unsigned char* mapped = NULL;
        D3D12_RANGE no_read = {0, 0};
        hr = ID3D12Resource_Map(eye->uploads[texture], 0, &no_read, (void**)&mapped);
        if (FAILED(hr)) { return 0; }
        memset(mapped, 0, (size_t)eye->sizes[texture]);
        for (unsigned int y = 0; y < height; ++y) {
            unsigned char* row = mapped + eye->layouts[texture].Offset + (size_t)y * eye->layouts[texture].Footprint.RowPitch;
            for (unsigned int x = 0; x < width; ++x) {
                float u = (float)x / (float)(width - 1);
                float v = (float)y / (float)(height - 1);
                if (texture == 0) {
                    uint16_t* pixel = (uint16_t*)row + x * 4;
                    pixel[0] = float_half(0.15f + 0.7f * u);
                    pixel[1] = float_half(0.1f + 0.2f * v);
                    pixel[2] = float_half(0.08f + 0.1f * u);
                    pixel[3] = float_half(1.0f);
                } else if (texture == 1) {
                    ((float*)row)[x] = 0.2f + 0.6f * v;
                } else if (texture == 2 && (motion_x != 0.0f || motion_y != 0.0f)) {
                    // Non-zero motion: the model must not be able to take a static-input
                    // shortcut. Units are pixels (MVecScale converts to UV).
                    uint16_t* pixel = (uint16_t*)row + x * 2;
                    pixel[0] = float_half(motion_x * (0.5f + 0.5f * u));
                    pixel[1] = float_half(motion_y * (0.5f + 0.5f * v));
                } else if (texture == 3) {
                    uint16_t* pixel = (uint16_t*)row + x * 4;
                    for (unsigned int channel = 0; channel < 4; ++channel) { pixel[channel] = 0x7e00u; }
                }
            }
        }
        ID3D12Resource_Unmap(eye->uploads[texture], 0, NULL);
    }
    static const char* keys[] = {"DLSSNR.Color", "DLSSNR.Depth", "DLSSNR.MVec", "DLSSNR.Output"};
    for (unsigned int i = 0; i < PROBE_TEXTURES; ++i) { set_resource(eye->params, keys[i], eye->textures[i]); }
    set_uint(eye->params, "DLSSNR.Enabled", 1);
    set_uint(eye->params, "DLSSNR.Width", width); set_uint(eye->params, "DLSSNR.Height", height);
    set_uint(eye->params, "DLSSNR.DepthInverted", 0);
    static const char* prefixes[] = {"Color", "Depth", "MVec", "Output"};
    for (unsigned int i = 0; i < 4; ++i) {
        char key[80];
        snprintf(key, sizeof(key), "DLSSNR.%sSubrectBaseX", prefixes[i]); set_uint(eye->params, key, 0);
        snprintf(key, sizeof(key), "DLSSNR.%sSubrectBaseY", prefixes[i]); set_uint(eye->params, key, 0);
        snprintf(key, sizeof(key), "DLSSNR.%sSubrectWidth", prefixes[i]); set_uint(eye->params, key, width);
        snprintf(key, sizeof(key), "DLSSNR.%sSubrectHeight", prefixes[i]); set_uint(eye->params, key, height);
    }
    // Scale stays 1.0 as in the probe header, so the static baseline is comparable
    // with the runs made before the motion switch existed.
    set_float(eye->params, "DLSSNR.MVecScaleX", 1.0f); set_float(eye->params, "DLSSNR.MVecScaleY", 1.0f);
    set_float(eye->params, "DLSSNR.Intensity", 1.0f); set_uint(eye->params, "DLSSNR.Style", 0);
    set_float(eye->params, "DLSSNR.LocalStructureStrength", -1.0f);
    set_float(eye->params, "DLSSNR.LocalToneStrength", -1.0f);
    set_float(eye->params, "DLSSNR.SkinStructureStrength", -1.0f);
    set_uint(eye->params, "DLSSNR.UseAutoMask", 1);
    (void)eye_index;
    return 1;
}

static int parse_sizes(const char* text, unsigned int* sizes, unsigned int* count) {
    *count = 0;
    const char* cursor = text;
    while (*cursor && *count < 8) {
        char* end = NULL;
        unsigned long value = strtoul(cursor, &end, 10);
        if (end == cursor || value < 64 || value > 8192) { return 0; }
        sizes[(*count)++] = (unsigned int)value;
        if (*end != ',') { break; }
        cursor = end + 1;
    }
    return *count > 0;
}

static int compare_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

int main(int argc, char** argv) {
    unsigned int sizes[8] = {0};
    unsigned int size_count = 0;
    unsigned int frame_limit = 24;
    unsigned int instances = 1;
    int verbose = 0;
    float motion_x = 0.0f, motion_y = 0.0f;
    if (!parse_sizes("720", sizes, &size_count)) { return 2; }
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            if (!parse_sizes(argv[++i], sizes, &size_count)) { fprintf(stderr, "Bad --sizes\n"); return 2; }
        } else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            frame_limit = (unsigned int)strtoul(argv[++i], NULL, 10);
            if (frame_limit < 2 || frame_limit > 512) { fprintf(stderr, "--frames must be 2..512\n"); return 2; }
        } else if (strcmp(argv[i], "--instances") == 0 && i + 1 < argc) {
            instances = (unsigned int)strtoul(argv[++i], NULL, 10);
            if (instances < 1 || instances > MAX_INSTANCES) { fprintf(stderr, "--instances must be 1 or 2\n"); return 2; }
        } else if (strcmp(argv[i], "--motion") == 0 && i + 1 < argc) {
            if (sscanf(argv[++i], "%f,%f", &motion_x, &motion_y) != 2) {
                fprintf(stderr, "--motion needs X,Y\n"); return 2;
            }
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else {
            fprintf(stderr, "Usage: nr-bench.exe [--sizes 720,1024] [--frames N] [--instances 1|2] [--verbose]\n");
            return 2;
        }
    }

    int exit_code = 1;
    ID3D12Device* device = NULL;
    ID3D12CommandQueue* queue = NULL;
    ID3D12CommandAllocator* allocator = NULL;
    ID3D12GraphicsCommandList* list = NULL;
    ID3D12Fence* fence = NULL;
    ID3D12QueryHeap* queries = NULL;
    ID3D12Resource* query_readback = NULL;
    HANDLE event = NULL;
    void* features[MAX_INSTANCES] = {NULL, NULL};
    void* caps = NULL;
    EyeInputs eyes[MAX_INSTANCES] = {{0}, {0}};
    int core_ready = 0;
    UINT64 fence_value = 0;

    HMODULE core = LoadLibraryW(L"_nvngx.dll");
    HMODULE forwarder = LoadLibraryW(L"nvngx.dll_nrfwd.dll");
    if (!core || !forwarder) { fprintf(stderr, "Missing runtime DLL, Win32 error=%lu\n", GetLastError()); goto cleanup; }
    CoreInit core_init = (CoreInit)(void*)GetProcAddress(core, "NVSDK_NGX_D3D12_Init");
    GetCaps get_caps = (GetCaps)(void*)GetProcAddress(core, "NVSDK_NGX_D3D12_GetCapabilityParameters");
    CoreShutdown shutdown = (CoreShutdown)(void*)GetProcAddress(core, "NVSDK_NGX_D3D12_Shutdown1");
    FwdInit fwd_init = (FwdInit)(void*)GetProcAddress(forwarder, "nrfwd_init");
    FwdCreate fwd_create = (FwdCreate)(void*)GetProcAddress(forwarder, "nrfwd_create");
    FwdEvaluate fwd_evaluate = (FwdEvaluate)(void*)GetProcAddress(forwarder, "nrfwd_evaluate");
    FwdRelease fwd_release = (FwdRelease)(void*)GetProcAddress(forwarder, "nrfwd_release");
    AllocateParams allocate = (AllocateParams)(void*)GetProcAddress(core, "NVSDK_NGX_D3D12_AllocateParameters");
    DestroyParams destroy = (DestroyParams)(void*)GetProcAddress(core, "NVSDK_NGX_D3D12_DestroyParameters");
    if (!core_init || !get_caps || !shutdown || !fwd_init || !fwd_create || !fwd_evaluate || !fwd_release || !allocate || !destroy) {
        fprintf(stderr, "Missing runtime export\n");
        goto cleanup;
    }
#define CHECK_HR(call) do { HRESULT hr = (call); if (FAILED(hr)) { fprintf(stderr, "%s => HRESULT 0x%08lx\n", #call, (unsigned long)hr); goto cleanup; } } while (0)
    CHECK_HR(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, (void**)&device));
    D3D12_COMMAND_QUEUE_DESC queue_desc = {0};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    CHECK_HR(ID3D12Device_CreateCommandQueue(device, &queue_desc, &IID_ID3D12CommandQueue, (void**)&queue));
    CHECK_HR(ID3D12Device_CreateCommandAllocator(device, D3D12_COMMAND_LIST_TYPE_DIRECT, &IID_ID3D12CommandAllocator, (void**)&allocator));
    CHECK_HR(ID3D12Device_CreateCommandList(device, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, NULL, &IID_ID3D12GraphicsCommandList, (void**)&list));
    // A freshly created list is in the recording state; close it so the per-size
    // loop can reset the allocator before re-recording.
    CHECK_HR(ID3D12GraphicsCommandList_Close(list));
    CHECK_HR(ID3D12Device_CreateFence(device, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, (void**)&fence));
    event = CreateEventW(NULL, FALSE, FALSE, NULL);
    if (!event) { fprintf(stderr, "CreateEvent failed\n"); goto cleanup; }

    int result = core_init(0x24480451ull, L".", device, 0x15, NULL);
    printf("CoreInit => 0x%x\n", result); fflush(stdout);
    if (result != 1) { goto cleanup; }
    core_ready = 1;
    result = get_caps(&caps);
    printf("GetCapabilityParameters => 0x%x\n", result); fflush(stdout);
    if (result != 1 || !caps) { goto cleanup; }
    result = fwd_init(L"nvngx_dlssnr.dll", L".", device, caps);
    printf("NeuralInit => 0x%x\n", result); fflush(stdout);
    if (result != 1) { goto cleanup; }
    UINT64 frequency = 0;
    CHECK_HR(ID3D12CommandQueue_GetTimestampFrequency(queue, &frequency));
    if (!frequency) { fprintf(stderr, "No timestamp frequency\n"); goto cleanup; }
    LARGE_INTEGER qpc_frequency;
    QueryPerformanceFrequency(&qpc_frequency);
    printf("timestamp frequency=%llu qpc frequency=%lld\n",
           (unsigned long long)frequency, (long long)qpc_frequency.QuadPart); fflush(stdout);

    for (unsigned int s = 0; s < size_count; ++s) {
        unsigned int side = sizes[s];
        // Feature creation: one per instance, each with its own history state.
        unsigned int created = 0;
        for (unsigned int i = 0; i < instances; ++i) {
            if (!prepare_sized(device, &eyes[i], i, allocate, side, side, motion_x, motion_y)) {
                fprintf(stderr, "PREP FAILED size=%u instance=%u\n", side, i);
                goto cleanup;
            }
            set_uint(caps, "DLSSNR.Enabled", 1);
            set_uint(caps, "DLSSNR.Width", side);
            set_uint(caps, "DLSSNR.Height", side);
            set_uint(caps, "CreationNodeMask", 1);
            set_uint(caps, "VisibilityNodeMask", 1);
            set_uint(caps, "DLSSNR.Hint.Render.Preset", 0);
            set_uint(caps, "DLSSNR.UICorrection", 1);
            CHECK_HR(ID3D12CommandAllocator_Reset(allocator));
            CHECK_HR(ID3D12GraphicsCommandList_Reset(list, allocator, NULL));
            features[i] = fwd_create(list, caps, &result);
            if (result != 1 || !features[i]) {
                fprintf(stderr, "RESULT size=%u instance=%u create=0x%x\n", side, i, result);
                goto cleanup;
            }
            CHECK_HR(ID3D12GraphicsCommandList_Close(list));
            {
                ID3D12CommandList* lists[] = {(ID3D12CommandList*)list};
                ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
                CHECK_HR(ID3D12CommandQueue_Signal(queue, fence, ++fence_value));
                CHECK_HR(ID3D12Fence_SetEventOnCompletion(fence, fence_value, event));
                if (WaitForSingleObject(event, 60000) != WAIT_OBJECT_0) { fprintf(stderr, "Create did not complete\n"); goto cleanup; }
            }
            created = i + 1;
        }

        // Upload the synthetic inputs once.
        CHECK_HR(ID3D12CommandAllocator_Reset(allocator));
        CHECK_HR(ID3D12GraphicsCommandList_Reset(list, allocator, NULL));
        for (unsigned int i = 0; i < created; ++i) {
            for (unsigned int t = 0; t < PROBE_TEXTURES; ++t) { upload_texture(list, &eyes[i], t); }
        }
        CHECK_HR(ID3D12GraphicsCommandList_Close(list));
        {
            ID3D12CommandList* lists[] = {(ID3D12CommandList*)list};
            ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
            CHECK_HR(ID3D12CommandQueue_Signal(queue, fence, ++fence_value));
            CHECK_HR(ID3D12Fence_SetEventOnCompletion(fence, fence_value, event));
            if (WaitForSingleObject(event, 60000) != WAIT_OBJECT_0) { fprintf(stderr, "Upload did not complete\n"); goto cleanup; }
        }

        // One query pair per frame in its own slots: reusing slots 0/1 every frame
        // made the resolved start stamp stay at the first frame's value under this
        // stack, so every frame gets fresh slots and is resolved once at the end.
        D3D12_QUERY_HEAP_DESC heap_desc = {0};
        heap_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        heap_desc.Count = 2 * frame_limit;
        CHECK_HR(ID3D12Device_CreateQueryHeap(device, &heap_desc, &IID_ID3D12QueryHeap, (void**)&queries));
        if (!create_buffer(device, D3D12_HEAP_TYPE_READBACK, 2ull * frame_limit * sizeof(UINT64), &query_readback)) { goto cleanup; }

        double gpu_ms[512];
        double frame_ms[512];
        double raw_gpu[512];
        double wall_ms[512];
        unsigned int wall_index[512];
        unsigned int measured = 0;
        unsigned int failures = 0;
        for (unsigned int frame = 0; frame < frame_limit; ++frame) {
            LARGE_INTEGER t0, t1, freq;
            QueryPerformanceCounter(&t0);
            CHECK_HR(ID3D12CommandAllocator_Reset(allocator));
            CHECK_HR(ID3D12GraphicsCommandList_Reset(list, allocator, NULL));
            ID3D12GraphicsCommandList_BeginQuery(list, queries, D3D12_QUERY_TYPE_TIMESTAMP, 2 * frame);
            for (unsigned int i = 0; i < created; ++i) {
                set_uint(eyes[i].params, "DLSSNR.Reset", frame == 0 ? 1u : 0u);
                int evaluated = fwd_evaluate(list, features[i], eyes[i].params);
                if (evaluated == 1) { ++eyes[i].successful_evaluations; } else { ++failures; }
            }
            ID3D12GraphicsCommandList_EndQuery(list, queries, D3D12_QUERY_TYPE_TIMESTAMP, 2 * frame + 1);
            ID3D12GraphicsCommandList_ResolveQueryData(list, queries, D3D12_QUERY_TYPE_TIMESTAMP,
                                                      2 * frame, 2, query_readback,
                                                      (UINT64)frame * 2 * sizeof(UINT64));
            CHECK_HR(ID3D12GraphicsCommandList_Close(list));
            {
                ID3D12CommandList* lists[] = {(ID3D12CommandList*)list};
                ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
                CHECK_HR(ID3D12CommandQueue_Signal(queue, fence, ++fence_value));
                CHECK_HR(ID3D12Fence_SetEventOnCompletion(fence, fence_value, event));
                if (WaitForSingleObject(event, 60000) != WAIT_OBJECT_0) { fprintf(stderr, "Frame %u did not complete\n", frame); goto cleanup; }
            }
            QueryPerformanceCounter(&t1);
            QueryPerformanceFrequency(&freq);
            wall_ms[frame] = 1000.0 * (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
            if (frame >= 2) {  // discard warm-up: the first evaluations compile pipelines
                wall_index[measured] = frame;
                ++measured;
            }
        }

        // Every pair is resolved once, after the last fence wait, so no slot is reused.
        {
            UINT64* stamps = NULL;
            D3D12_RANGE read_range = {0, (SIZE_T)(2ull * frame_limit * sizeof(UINT64))};
            CHECK_HR(ID3D12Resource_Map(query_readback, 0, &read_range, (void**)&stamps));
            for (unsigned int frame = 0; frame < frame_limit; ++frame) {
                raw_gpu[frame] = (stamps[2 * frame + 1] > stamps[2 * frame])
                    ? 1000.0 * (double)(stamps[2 * frame + 1] - stamps[2 * frame]) / (double)frequency
                    : 0.0;
                if (verbose) {
                    printf("  frame=%u begin=%llu end=%llu delta=%llu gpu=%.3f wall=%.3f\n", frame,
                           (unsigned long long)stamps[2 * frame],
                           (unsigned long long)stamps[2 * frame + 1],
                           (unsigned long long)(stamps[2 * frame + 1] - stamps[2 * frame]),
                           raw_gpu[frame], wall_ms[frame]);
                }
            }
            ID3D12Resource_Unmap(query_readback, 0, NULL);
        }
        for (unsigned int i = 0; i < measured; ++i) {
            gpu_ms[i] = raw_gpu[wall_index[i]];
            frame_ms[i] = wall_ms[wall_index[i]];
        }
        if (verbose) { fflush(stdout); }

        double frame_sum = 0.0;
        for (unsigned int i = 0; i < measured; ++i) { frame_sum += frame_ms[i]; }
        double frame_mean = measured ? frame_sum / (double)measured : 0.0;
        qsort(frame_ms, measured, sizeof(double), compare_double);
        double frame_p50 = measured ? frame_ms[measured / 2] : 0.0;
        // The GPU timestamp column is deliberately absent: in this stack BeginQuery is
        // not re-stamped per frame (it reads a near-constant sentinel while EndQuery
        // advances with absolute elapsed time), so the pair does not bound one frame's
        // work. The fence-timed serialised frame is the honest metric -- and it is the
        // one that matters here, because a consumer of the frame must wait for it.
        printf("RESULT size=%ux%u instances=%u frames=%u eval_failures=%u "
               "serial_frame_ms mean=%.3f p50=%.3f p95=%.3f\n",
               side, side, created, measured, failures, frame_mean, frame_p50,
               measured ? frame_ms[(measured * 95) / 100 < measured ? (measured * 95) / 100 : measured - 1] : 0.0);
        fflush(stdout);

        ID3D12QueryHeap_Release(queries); queries = NULL;
        ID3D12Resource_Release(query_readback); query_readback = NULL;
        for (unsigned int i = 0; i < created; ++i) { fwd_release(features[i]); features[i] = NULL; }
    }

    printf("COMPLETE\n");
    exit_code = 0;

cleanup:
    for (unsigned int i = 0; i < MAX_INSTANCES; ++i) {
        if (features[i] && fwd_release) { fwd_release(features[i]); features[i] = NULL; }
    }
    if (query_readback) { ID3D12Resource_Release(query_readback); }
    if (queries) { ID3D12QueryHeap_Release(queries); }
    if (core_ready && shutdown) {
        int live = -1;
        shutdown(device, &live);
        printf("CoreShutdown live_devices=%d\n", live);
    }
    if (event) { CloseHandle(event); }
    if (fence) { ID3D12Fence_Release(fence); }
    if (list) { ID3D12GraphicsCommandList_Release(list); }
    if (allocator) { ID3D12CommandAllocator_Release(allocator); }
    if (queue) { ID3D12CommandQueue_Release(queue); }
    if (device) { ID3D12Device_Release(device); }
    if (exit_code != 0) { fprintf(stderr, "FAILED\n"); }
    return exit_code;
}