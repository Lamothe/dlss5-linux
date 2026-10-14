// Standalone feature/evaluation probe, not a renderer or a stereo/history quality test.
#define COBJMACROS
#define WIDL_C_INLINE_WRAPPERS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <stdio.h>
#include <stdint.h>

typedef int (__cdecl *CoreInit)(unsigned long long, const wchar_t*, ID3D12Device*, int, const void*);
typedef int (__cdecl *GetCaps)(void**);
// Driver-core Shutdown1 additionally writes a live-device count to argument two.
typedef int (__cdecl *CoreShutdown)(ID3D12Device*, int*);
typedef int (__cdecl *FwdInit)(const wchar_t*, const wchar_t*, ID3D12Device*, void*);
typedef void* (__cdecl *FwdCreate)(ID3D12GraphicsCommandList*, void*, int*);
typedef void (__cdecl *FwdRelease)(void*);
typedef void (__cdecl *SetUint)(void*, const char*, unsigned int);

static void set_uint(void* params, const char* key, unsigned int value) {
    void** vtable = *(void***)params;
    ((SetUint)vtable[3])(params, key, value);
}

#include "evaluation-probe.h"
#include "boundary-probe.h"
#include "scene-probe.h"
#include "gpu-scene.h"

int main(int argc, char** argv) {
    int capture_inject_mode = (argc == 3) && strcmp(argv[1], "--capture-history-inject") == 0;
    int capture_sham_mode = (argc == 3) && strcmp(argv[1], "--capture-history-sham") == 0;
    int capture_replay_mode = (argc == 3) && strcmp(argv[1], "--capture-history-replay") == 0;
    int capture_replay2_mode = (argc == 3) && strcmp(argv[1], "--capture-history-replay2") == 0;
    int capture_injectreset_mode = (argc == 3) && strcmp(argv[1], "--capture-history-injectreset") == 0;
    int capture_history_mode = capture_inject_mode || capture_sham_mode ||
        capture_replay_mode || capture_replay2_mode || capture_injectreset_mode ||
        ((argc == 3 || argc == 4) && strcmp(argv[1], "--capture-history") == 0);
    int capture_reset_mode = (argc == 3 || argc == 4) && strcmp(argv[1], "--capture-reset-mid") == 0;
    int capture_mode = capture_reset_mode || capture_history_mode ||
                       (argc == 3 && strcmp(argv[1], "--capture") == 0);
    const char* capture_dir = capture_mode ? argv[2] : NULL;
    unsigned int capture_reset_frame = CAPTURE_RESET_FRAME;
    if ((capture_reset_mode || capture_history_mode) && argc == 4) {
        char* end = NULL;
        unsigned long parsed = strtoul(argv[3], &end, 10);
        if (!argv[3][0] || *end || parsed < 1 || parsed >= CAPTURE_FRAMES) {
            fprintf(stderr, "Reset frame must be an integer in [1, %u]\n", CAPTURE_FRAMES - 1);
            return 1;
        }
        capture_reset_frame = (unsigned int)parsed;
    }
    int boundary_mode = argc == 3 && (strcmp(argv[1], "--capture-boundary") == 0 ||
                                      strcmp(argv[1], "--capture-ladder") == 0);
    int ladder_mode = argc == 3 && strcmp(argv[1], "--capture-ladder") == 0;
    const char* boundary_dir = boundary_mode ? argv[2] : NULL;
    int tracked_mode = argc==2 && strcmp(argv[1],"--tracked-control")==0;
    int gpu_mode = tracked_mode || (argc == 2 && strcmp(argv[1], "--gpu-scene") == 0);
    GpuScene gpu_scene = {0};
    int scene_mode = gpu_mode || (argc == 2 && strcmp(argv[1], "--scene") == 0);
    int evaluate_mode = scene_mode || (argc == 2 && strcmp(argv[1], "--evaluate") == 0);
    if (argc > 1 && !evaluate_mode && !capture_mode && !boundary_mode) {
        fprintf(stderr, "Usage: feature-probe.exe [--evaluate|--scene|--gpu-scene|--capture DIR|--capture-reset-mid DIR [FRAME]|--capture-history DIR [FRAME]|--capture-history-inject DIR|--capture-history-sham DIR|--capture-boundary DIR|--capture-ladder DIR]\n");
        return 1;
    }    unsigned int frame_count = scene_mode ? 12u : PROBE_FRAMES;
    EyeInputs eyes[2] = {0};
    HistoryProbe history = {0};
    ID3D12Resource* inject_upload = NULL;
    ID3D12Resource* inject_upload2 = NULL;
    FILE* capture_manifest = NULL;
    ID3D12Device* device = NULL;
    ID3D12CommandQueue* queue = NULL;
    ID3D12CommandAllocator* allocator = NULL;
    ID3D12GraphicsCommandList* list = NULL;
    ID3D12Fence* fence = NULL;
    HANDLE event = NULL;
    void* features[2] = {NULL, NULL};
    void* caps = NULL;
    int exit_code = 1;
    int core_ready = 0;
    int submitted = 0;
    int completed = 0;
    HMODULE core = LoadLibraryW(L"_nvngx.dll");
    HMODULE forwarder = LoadLibraryW(L"nvngx.dll_nrfwd.dll");
    if (!core || !forwarder) {
        fprintf(stderr, "Missing runtime DLL, Win32 error=%lu\n", GetLastError());
        return 1;
    }
    CoreInit core_init = (CoreInit)(void*)GetProcAddress(core, "NVSDK_NGX_D3D12_Init");
    GetCaps get_caps = (GetCaps)(void*)GetProcAddress(core, "NVSDK_NGX_D3D12_GetCapabilityParameters");
    CoreShutdown shutdown = (CoreShutdown)(void*)GetProcAddress(core, "NVSDK_NGX_D3D12_Shutdown1");
    FwdInit init = (FwdInit)(void*)GetProcAddress(forwarder, "nrfwd_init");
    FwdCreate create = (FwdCreate)(void*)GetProcAddress(forwarder, "nrfwd_create");
    FwdRelease release = (FwdRelease)(void*)GetProcAddress(forwarder, "nrfwd_release");
    FwdEvaluate evaluate = (FwdEvaluate)(void*)GetProcAddress(forwarder, "nrfwd_evaluate");
    AllocateParams allocate = (AllocateParams)(void*)GetProcAddress(core, "NVSDK_NGX_D3D12_AllocateParameters");
    DestroyParams destroy = (DestroyParams)(void*)GetProcAddress(core, "NVSDK_NGX_D3D12_DestroyParameters");
    if (!evaluate || !allocate || !destroy) { fprintf(stderr, "Missing evaluation exports\n"); return 1; }
    if (!core_init || !get_caps || !shutdown || !init || !create || !release) {
        fprintf(stderr, "Missing runtime export\n");
        return 1;
    }
#define CHECK_HR(call) do { HRESULT hr = (call); if (FAILED(hr)) { fprintf(stderr, "%s => HRESULT 0x%08lx\n", #call, (unsigned long)hr); goto cleanup; } } while (0)
    CHECK_HR(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, (void**)&device));
    if (boundary_mode || capture_history_mode) { boundary_install(device); }
    // Model allocations (NVSDK_NGX_D3D12_Init through feature creation) are
    // recorded as phase 1; harness resources afterwards as phase 2.
    boundary_phase = 1;
    D3D12_COMMAND_QUEUE_DESC queue_desc = {0};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    CHECK_HR(ID3D12Device_CreateCommandQueue(device, &queue_desc, &IID_ID3D12CommandQueue, (void**)&queue));
    CHECK_HR(ID3D12Device_CreateCommandAllocator(device, D3D12_COMMAND_LIST_TYPE_DIRECT, &IID_ID3D12CommandAllocator, (void**)&allocator));
    CHECK_HR(ID3D12Device_CreateCommandList(device, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, NULL, &IID_ID3D12GraphicsCommandList, (void**)&list));
    CHECK_HR(ID3D12Device_CreateFence(device, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, (void**)&fence));
    event = CreateEventW(NULL, FALSE, FALSE, NULL);
    if (!event) { fprintf(stderr, "CreateEvent failed\n"); goto cleanup; }
    // Core export ABI: app ID, path, device, SDK version, common feature info.
    int result = core_init(0x24480451ull, L".", device, 0x15, NULL);
    printf("CoreInit => 0x%x\n", result); fflush(stdout);
    if (result != 1) { goto cleanup; }
    core_ready = 1;
    result = get_caps(&caps);
    printf("GetCapabilityParameters => 0x%x nonnull=%d\n", result, caps != NULL); fflush(stdout);
    if (result != 1 || !caps) { goto cleanup; }
    result = init(L"nvngx_dlssnr.dll", L".", device, caps);
    printf("NeuralInit => 0x%x\n", result); fflush(stdout);
    if (result != 1) { goto cleanup; }
    for (unsigned int eye = 0; eye < 2; ++eye) {
        set_uint(caps, "DLSSNR.Enabled", 1);
        set_uint(caps, "DLSSNR.Width", 720);
        set_uint(caps, "DLSSNR.Height", 720);
        set_uint(caps, "CreationNodeMask", 1);
        set_uint(caps, "VisibilityNodeMask", 1);
        set_uint(caps, "DLSSNR.Hint.Render.Preset", 0);
        set_uint(caps, "DLSSNR.UICorrection", 1);
        features[eye] = create(list, caps, &result);
        printf("Feature[%u] => 0x%x handle=%p\n", eye, result, features[eye]); fflush(stdout);
        if (result != 1 || !features[eye]) { goto cleanup; }
    }
    if (features[0] == features[1]) { fprintf(stderr, "Feature handles alias; not independent\n"); goto cleanup; }
    CHECK_HR(ID3D12GraphicsCommandList_Close(list));
    ID3D12CommandList* command_lists[] = {(ID3D12CommandList*)list};
    ID3D12CommandQueue_ExecuteCommandLists(queue, 1, command_lists);
    submitted = 1;
    CHECK_HR(ID3D12CommandQueue_Signal(queue, fence, 1));
    CHECK_HR(ID3D12Fence_SetEventOnCompletion(fence, 1, event));
    if (WaitForSingleObject(event, 30000) != WAIT_OBJECT_0 || ID3D12Fence_GetCompletedValue(fence) == UINT64_MAX) {
        fprintf(stderr, "GPU completion failed/timed out\n"); goto cleanup;
    }
    completed = 1;
    // Everything created from here on is harness-side (phase 2).
    boundary_phase = 2;
    if (capture_mode) {
        char manifest_path[512];
        // Fence values must strictly increase: the create phase used 1 and the
        // frame loop uses frame+fence_base. History mode inserts one init
        // submit at value 2 and shifts the frame loop to 3..10.
        unsigned int fence_base = capture_history_mode ? 3u : 2u;
        CreateDirectoryA(capture_dir, NULL);
        if (!prepare_eye(device, &eyes[0], 0, allocate)) { goto cleanup; }
        if (snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.txt", capture_dir) >= (int)sizeof(manifest_path)) {
            fprintf(stderr, "Capture manifest path too long\n");
            goto cleanup;
        }
        capture_manifest = fopen(manifest_path, "w");
        if (!capture_manifest) { fprintf(stderr, "Capture manifest open failed\n"); goto cleanup; }
        // Temporal-state probe (see vr/TEMPORAL-STATE.md): capture-reset-mid
        // repeats the controlled capture with a second DLSSNR.Reset at the
        // chosen frame (default 4). If reset clears all output-visible temporal
        // state and the runtime is time-homogeneous, frames N..7 replay the
        // run's own frames 0..7-N bit-identically.
        fprintf(capture_manifest, "protocol %s frames=%u size=%u context=eye0 inputs=gradient-stationary reset=%s%u\n",
                capture_history_mode ? "capture-history-v1" :
                    (capture_reset_mode ? "capture-reset-mid-v1" : "capture-v1"),
                CAPTURE_FRAMES, PROBE_SIZE,
                (capture_reset_mode || (capture_injectreset_mode) ||
                 (capture_history_mode && argc == 4)) ? "frame0+frame" : "frame",
                (capture_reset_mode || (capture_injectreset_mode) ||
                 (capture_history_mode && argc == 4)) ? capture_reset_frame : 0u);
        if (capture_history_mode) {
            // Model-owned phase-1 textures (its history allocations) are read
            // back after creation ("init", the reset-value candidate) and after
            // every evaluation (see TEMPORAL-HISTORY.md).
            if (!history_setup(device, &history)) { goto cleanup; }
            if ((capture_inject_mode || capture_sham_mode || capture_replay_mode ||
                 capture_replay2_mode || capture_injectreset_mode) &&
                (history.count == 0 || history.size[0] == 0)) {
                fprintf(stderr, "Intervention mode needs at least one history texture\n");
                goto cleanup;
            }
            if (capture_inject_mode || capture_sham_mode || capture_replay_mode ||
                capture_replay2_mode || capture_injectreset_mode) {
                // Read-side intervention: overwrite the evaluated feature's
                // history before frames 2 (uniform 1.0) and 5 (checkerboard);
                // sham mode stages the history's own bytes instead.
                if (!create_buffer(device, D3D12_HEAP_TYPE_UPLOAD, history.size[0],
                                   &inject_upload)) {
                    goto cleanup;
                }
                if (capture_replay2_mode &&
                    !create_buffer(device, D3D12_HEAP_TYPE_UPLOAD, history.size[0],
                                   &inject_upload2)) {
                    goto cleanup;
                }
                printf("HISTORY-INJECT plan=frame2:%s,frame5:%s\n",
                       capture_sham_mode ? "sham" : "uniform",
                       capture_sham_mode ? "sham" : "checkerboard");
                fflush(stdout);
            }
            if (history.count > 0) {
                CHECK_HR(ID3D12CommandAllocator_Reset(allocator));
                CHECK_HR(ID3D12GraphicsCommandList_Reset(list, allocator, NULL));
                history_record(list, &history);
                CHECK_HR(ID3D12GraphicsCommandList_Close(list));
                ID3D12CommandQueue_ExecuteCommandLists(queue, 1, command_lists);
                completed = 0;
                CHECK_HR(ID3D12CommandQueue_Signal(queue, fence, 2));
                CHECK_HR(ID3D12Fence_SetEventOnCompletion(fence, 2, event));
                if (WaitForSingleObject(event, 30000) != WAIT_OBJECT_0 ||
                    ID3D12Fence_GetCompletedValue(fence) == UINT64_MAX) {
                    fprintf(stderr, "History init GPU completion failed\n");
                    goto cleanup;
                }
                completed = 1;
                if (!history_dump(capture_dir, &history, capture_manifest, "init")) { goto cleanup; }
            }
        }
        for (unsigned int frame = 0; frame < CAPTURE_FRAMES; ++frame) {
            CHECK_HR(ID3D12CommandAllocator_Reset(allocator));
            CHECK_HR(ID3D12GraphicsCommandList_Reset(list, allocator, NULL));
            if (frame == 0) {
                for (unsigned int texture = 0; texture < 3; ++texture) { upload_texture(list, &eyes[0], texture); }
            }
            upload_texture(list, &eyes[0], 3); // fresh NaNs each evaluation
            if (capture_inject_mode && history.count > 0 && (frame == 2 || frame == 5)) {
                unsigned int pattern = (frame == 2) ? 0u : 1u;
                if (!history_inject_upload(inject_upload, &history.layout[0], pattern)) {
                    fprintf(stderr, "Inject upload failed frame=%u\n", frame);
                    goto cleanup;
                }
                history_inject_record(list, &history, inject_upload, &history.layout[0]);
                fprintf(capture_manifest,
                        "history-inject frame=%u idx=0 pattern=%s bytes=%llu\n",
                        frame, pattern == 0u ? "uniform-1.0" : "checkerboard-1.0/0.0",
                        (unsigned long long)history.size[0]);
                printf("HISTORY-INJECT frame=%u pattern=%s\n", frame,
                       pattern == 0u ? "uniform-1.0" : "checkerboard-1.0/0.0");
                fflush(stdout);
            }
            if (capture_sham_mode && history.count > 0 && (frame == 2 || frame == 5)) {
                // Content-preserving control: same upload->texture copy, but
                // staging the history's own bytes (captured after the previous
                // evaluation) back into it.
                if (!history_inject_sham(&history, inject_upload)) {
                    fprintf(stderr, "Sham staging failed frame=%u\n", frame);
                    goto cleanup;
                }
                history_inject_record(list, &history, inject_upload, &history.layout[0]);
                fprintf(capture_manifest, "history-inject frame=%u idx=0 pattern=sham\n", frame);
                printf("HISTORY-INJECT frame=%u pattern=sham\n", frame);
                fflush(stdout);
            }
            if (capture_replay_mode && history.count > 0 && (frame == 2 || frame == 5)) {
                // One-step recurrence test: the history buffer holds frame 0's
                // history (= output-00, saved at frame 0). Re-injecting it with
                // stationary inputs predicts output-NN == output-01 exactly if
                // the history is the sole temporal state.
                history_inject_record(list, &history, inject_upload, &history.layout[0]);
                fprintf(capture_manifest,
                        "history-inject frame=%u idx=0 pattern=replay-output00\n", frame);
                printf("HISTORY-INJECT frame=%u pattern=replay-output00\n", frame);
                fflush(stdout);
            }
            if (capture_replay2_mode && history.count > 0 && frame == 1) {
                // Perturb frame 1 (uniform 1.0) with a dedicated buffer, so the
                // saved output-00 bytes in inject_upload stay intact.
                if (!history_inject_upload(inject_upload2, &history.layout[0], 0u)) {
                    fprintf(stderr, "Replay2 pattern upload failed\n");
                    goto cleanup;
                }
                history_inject_record(list, &history, inject_upload2, &history.layout[0]);
                fprintf(capture_manifest,
                        "history-inject frame=1 idx=0 pattern=uniform-1.0\n");
                printf("HISTORY-INJECT frame=1 pattern=uniform-1.0\n");
                fflush(stdout);
            }
            if (capture_replay2_mode && history.count > 0 && (frame == 2 || frame == 5)) {
                // Same history as hist-replay at these frames; if the f2 output
                // matches hist-replay's f2 output the earlier f1 perturbation
                // did not matter (state = history + counter only).
                history_inject_record(list, &history, inject_upload, &history.layout[0]);
                fprintf(capture_manifest,
                        "history-inject frame=%u idx=0 pattern=replay-output00\n", frame);
                printf("HISTORY-INJECT frame=%u pattern=replay-output00\n", frame);
                fflush(stdout);
            }
            if (capture_injectreset_mode && history.count > 0 && frame == 4) {
                // Reset-frames-ignored-history test: a garbage history is
                // injected immediately before a DLSSNR.Reset frame. If reset
                // zeroes the history before the model reads it, the output
                // must equal the zero-history frame-0 result.
                if (!history_inject_upload(inject_upload, &history.layout[0], 0u)) {
                    fprintf(stderr, "Injectreset upload failed\n");
                    goto cleanup;
                }
                history_inject_record(list, &history, inject_upload, &history.layout[0]);
                fprintf(capture_manifest,
                        "history-inject frame=4 idx=0 pattern=uniform-1.0-at-reset\n");
                printf("HISTORY-INJECT frame=4 pattern=uniform-1.0-at-reset\n");
                fflush(stdout);
            }
            set_uint(eyes[0].params, "DLSSNR.Reset",
                     (frame == 0 || ((capture_reset_mode || capture_injectreset_mode ||
                                      (capture_history_mode && argc == 4)) &&
                                     frame == capture_reset_frame)) ? 1u : 0u);
            result = evaluate(list, features[0], eyes[0].params);
            printf("CaptureEvaluate frame=%u result=0x%x\n", frame, result); fflush(stdout);
            if (result != 1) { goto cleanup; }
            ++eyes[0].successful_evaluations;
            record_readback(list, &eyes[0]);
            if (capture_history_mode) { history_record(list, &history); }
            CHECK_HR(ID3D12GraphicsCommandList_Close(list));
            ID3D12CommandQueue_ExecuteCommandLists(queue, 1, command_lists);
            completed = 0;
            CHECK_HR(ID3D12CommandQueue_Signal(queue, fence, (UINT64)frame + fence_base));
            CHECK_HR(ID3D12Fence_SetEventOnCompletion(fence, (UINT64)frame + fence_base, event));
            if (WaitForSingleObject(event, 30000) != WAIT_OBJECT_0 || ID3D12Fence_GetCompletedValue(fence) == UINT64_MAX) {
                fprintf(stderr, "Capture GPU completion failed\n");
                goto cleanup;
            }
            completed = 1;
            if (!capture_dump(capture_dir, &eyes[0], frame, capture_manifest)) { goto cleanup; }
            if (capture_history_mode && history.count > 0) {
                char tag[16];
                if (snprintf(tag, sizeof(tag), "frame%02u", frame) >= (int)sizeof(tag)) {
                    fprintf(stderr, "History tag too long\n");
                    goto cleanup;
                }
                if (!history_dump(capture_dir, &history, capture_manifest, tag)) { goto cleanup; }
            }
            if ((capture_replay_mode || capture_replay2_mode) && history.count > 0 && frame == 0) {
                // Save frame 0's history (= output-00) into the injection
                // buffer; re-injected at frames 2 and 5 for the recurrence test.
                if (!history_inject_sham(&history, inject_upload)) {
                    fprintf(stderr, "Replay staging failed frame=%u\n", frame);
                    goto cleanup;
                }
                fprintf(capture_manifest, "history-stage frame=0 source=history-readback\n");
                printf("HISTORY-STAGE frame=0 saved=output00-bytes\n");
                fflush(stdout);
            }
        }
        fclose(capture_manifest);
        capture_manifest = NULL;
        printf("PASS: %s frames=%u dir=%s context=eye0\n",
               capture_history_mode ? "history capture" :
                   (capture_reset_mode ? "reset-mid capture" : "reference capture"),
               CAPTURE_FRAMES, capture_dir);
        fflush(stdout);
    }
    if (boundary_mode) {
        char manifest_path[512];
        BoundaryContext boundary = {0};
        CreateDirectoryA(boundary_dir, NULL);
        if (!prepare_eye(device, &eyes[0], 0, allocate)) { goto cleanup; }
        if (snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.txt", boundary_dir) >= (int)sizeof(manifest_path)) {
            fprintf(stderr, "Boundary manifest path too long\n");
            goto cleanup;
        }
        capture_manifest = fopen(manifest_path, "w");
        if (!capture_manifest) { fprintf(stderr, "Boundary manifest open failed\n"); goto cleanup; }
        if (ladder_mode) {
            fprintf(capture_manifest,
                    "protocol ladder-capture-v1 frames=%u size=%u context=eye0 pool=%llu\n",
                    BOUNDARY_FRAMES, PROBE_SIZE, (unsigned long long)LADDER_POOL_SIZE);
        } else {
        fprintf(capture_manifest, "protocol boundary-capture-v1 frames=%u size=%u context=eye0 window=%llu words=%u\n",
                BOUNDARY_FRAMES, PROBE_SIZE, (unsigned long long)BOUNDARY_WINDOW, BOUNDARY_WORDS);
        }
        boundary.device = device;
        boundary.queue = queue;
        boundary.allocator = allocator;
        boundary.list = list;
        boundary.fence = fence;
        boundary.event = event;
        boundary.eye = &eyes[0];
        boundary.evaluate = evaluate;
        boundary.feature = features[0];
        boundary.dir = boundary_dir;
        boundary.manifest = capture_manifest;
        boundary.sidecar = getenv("DLSS_NVAPI_CUT_FILE");
        boundary.submitted = &submitted;
        boundary.completed = &completed;
        boundary.ladder = ladder_mode;
        if (!boundary.sidecar) {
            fprintf(stderr, "DLSS_NVAPI_CUT_FILE not set: no boundary words available\n");
            goto cleanup;
        }
        if (!boundary_run(&boundary)) { goto cleanup; }
        fclose(capture_manifest);
        capture_manifest = NULL;
        if (ladder_mode) {
            printf("PASS: ladder capture frames=%u pool=%llu dir=%s context=eye0\n",
                   BOUNDARY_FRAMES, (unsigned long long)LADDER_POOL_SIZE, boundary_dir);
        } else {
        printf("PASS: boundary capture frames=%u words=%u dir=%s context=eye0\n",
               BOUNDARY_FRAMES, BOUNDARY_WORDS, boundary_dir);
        }
        fflush(stdout);
    }
    if (evaluate_mode) {
        for (unsigned int eye = 0; eye < 2; ++eye) {
            if (!prepare_eye(device, &eyes[eye], eye, allocate)) { goto cleanup; }
        }
        if (gpu_mode && !prepare_gpu_scene(device, eyes, &gpu_scene)) { goto cleanup; }
        for (unsigned int frame = 0; frame < frame_count; ++frame) {
            CHECK_HR(ID3D12CommandAllocator_Reset(allocator));
            CHECK_HR(ID3D12GraphicsCommandList_Reset(list, allocator, NULL));
            for (unsigned int position = 0; position < 2; ++position) {
                unsigned int eye = (frame + position) % 2; // alternate submission order
                if (scene_mode && !gpu_mode && !render_scene(&eyes[eye], eye, frame)) { goto cleanup; }
                if (!gpu_mode && (frame == 0 || scene_mode)) {
                    for (unsigned int texture = 0; texture < 3; ++texture) { upload_texture(list, &eyes[eye], texture); }
                }
                if (tracked_mode) {
                    gpu_scene.tracked_control=1;
                    unsigned int* c=gpu_scene.constants[eye]; memset(c,0,32*sizeof(*c));
                    c[0]=eye; c[1]=frame; c[2]=PROBE_SIZE; c[3]=PROBE_SIZE; c[4]=1;
                    // 180Y quaternion maps local -Z into scene +Z.
                    float f[24]={0,1,0,0,scene_camera_x(frame,eye),1.4f,0,0,
                        0,1,0,0,scene_camera_x(frame==0?0:frame-1,eye),1.4f,0,0,
                        -1,1,1,-1,-1,1,1,-1};
                    memcpy(c+8,f,sizeof(f));
                }
                if (gpu_mode) { record_gpu_scene(list, &eyes[eye], &gpu_scene, eye, frame); }
                upload_texture(list, &eyes[eye], 3); // fresh NaNs each evaluation: no stale output PASS
                set_uint(eyes[eye].params, "DLSSNR.Reset", frame == 0 ? 1u : 0u);
                result = evaluate(list, features[eye], eyes[eye].params);
                printf("Evaluate eye=%u frame=%u result=0x%x\n", eye, frame, result); fflush(stdout);
                if (result != 1) { goto cleanup; }
                ++eyes[eye].successful_evaluations;
                record_readback(list, &eyes[eye]);
            }
            CHECK_HR(ID3D12GraphicsCommandList_Close(list));
            ID3D12CommandQueue_ExecuteCommandLists(queue, 1, command_lists);
            completed = 0;
            CHECK_HR(ID3D12CommandQueue_Signal(queue, fence, (UINT64)frame + 2));
            CHECK_HR(ID3D12Fence_SetEventOnCompletion(fence, (UINT64)frame + 2, event));
            if (WaitForSingleObject(event, 30000) != WAIT_OBJECT_0 || ID3D12Fence_GetCompletedValue(fence) == UINT64_MAX) {
                fprintf(stderr, "Evaluation GPU completion failed\n"); goto cleanup;
            }
            completed = 1;
            for (unsigned int eye = 0; eye < 2; ++eye) {
                if (gpu_mode && !validate_gpu_inputs(&eyes[eye], &gpu_scene, eye, frame)) { goto cleanup; }
                if (!validate_readback(&eyes[eye], eye, frame, scene_mode)) { goto cleanup; }
                if (scene_mode && (frame == 0 || frame == frame_count-1)) {
                    if (!write_scene_ppm(&eyes[eye], eye, frame, 0) || !write_scene_ppm(&eyes[eye], eye, frame, 1)) { goto cleanup; }
                }
            }
        }
    }
    exit_code = 0;
cleanup:
    // Retain partial capture evidence (the launcher rejects runs without the PASS marker).
    if (capture_manifest) { fclose(capture_manifest); capture_manifest = NULL; }
    // Resource map for the offline history binding (see TEMPORAL-HISTORY.md):
    // retained on every exit path, including in-flight failures below. Only
    // dumped when the recording hook was actually installed.
    if (boundary_mode || capture_history_mode) {
        char map_path[512];
        const char* dir = capture_mode ? capture_dir : boundary_dir;
        if (dir && snprintf(map_path, sizeof(map_path), "%s/resource-map.txt", dir) < (int)sizeof(map_path)) {
            boundary_dump_map(map_path);
        }
    }
    // If submitted work did not finish, avoid destroying resources the GPU may still use.
    if (submitted && !completed) { fflush(NULL); ExitProcess(1); }
    for (unsigned int eye = 0; eye < 2; ++eye) { if (features[eye] && (eye == 0 || features[eye] != features[0])) { release(features[eye]); } }
    for (unsigned int eye = 0; eye < 2; ++eye) {
        if (!destroy_eye(&eyes[eye], destroy)) { exit_code = 1; }
    }
    for (unsigned int i = 0; i < history.count && i < HISTORY_MAX; ++i) {
        if (history.readback[i]) { ID3D12Resource_Release(history.readback[i]); }
    }
    if (inject_upload) { ID3D12Resource_Release(inject_upload); }
    if (inject_upload2) { ID3D12Resource_Release(inject_upload2); }
    destroy_gpu_scene(&gpu_scene);
    if (core_ready) {
        int live_devices = -1;
        int shutdown_result = shutdown(device, &live_devices);
        printf("CoreShutdown => 0x%x live_devices=%d\n", shutdown_result, live_devices);
        if (shutdown_result != 1 || live_devices != 0) { exit_code = 1; }
    }
    if (event) { CloseHandle(event); }
    if (fence) { ID3D12Fence_Release(fence); }
    if (list) { ID3D12GraphicsCommandList_Release(list); }
    if (allocator) { ID3D12CommandAllocator_Release(allocator); }
    if (queue) { ID3D12CommandQueue_Release(queue); }
    if (device) { ID3D12Device_Release(device); }
    // Keep model/forwarder loaded through process teardown.
    if (exit_code == 0) {
        if (capture_mode) {
            printf("PASS: capture teardown complete. NOT stereo quality or GPU timing.\n");
        } else if (boundary_mode) {
            printf("PASS: boundary teardown complete. NOT stereo quality or GPU timing.\n");
        } else if (gpu_mode) {
            printf("PASS: GPU scene evaluation frames=%u evaluations=%u,%u validated_rgb_outputs=%u,%u. NOT headset/stereo quality or GPU timing.\n", frame_count, eyes[0].successful_evaluations, eyes[1].successful_evaluations, eyes[0].validated_outputs, eyes[1].validated_outputs);
        } else if (scene_mode) {
            printf("PASS: rendered scene evaluation frames=%u evaluations=%u,%u validated_rgb_outputs=%u,%u. NOT headset/stereo quality or GPU timing.\n", frame_count, eyes[0].successful_evaluations, eyes[1].successful_evaluations, eyes[0].validated_outputs, eyes[1].validated_outputs);
        } else if (evaluate_mode) {
            printf("PASS: repeated two-context evaluation frames=%u evaluations=%u,%u validated_rgb_outputs=%u,%u. NOT stereo/history quality or GPU timing.\n", frame_count, eyes[0].successful_evaluations, eyes[1].successful_evaluations, eyes[0].validated_outputs, eyes[1].validated_outputs);
        } else {
        printf("PASS: two distinct feature handles created and GPU initialization completed. NO evaluations or stereo images tested.\n");
        }
    }
    return exit_code;
}
