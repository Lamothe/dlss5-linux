// Offscreen raster/readback oracle. No OpenXR or NGX runtime is loaded.
// Build from repository root; run with vr/probe as the working directory.
#define PROBE_MESH_RENDER
#define main feature_probe_unused_main
#include "feature-probe.c"
#undef main
#undef CHECK_HR
#include "mesh-render.h"
#include <stdlib.h>

// prepare_eye also populates NGX parameters. A local no-op ABI object lets us
// reuse its resource allocation without importing/initialising a neural runtime.
static void __cdecl probe_set_resource(void* p, const char* k, unsigned long long v) { (void)p; (void)k; (void)v; }
static void __cdecl probe_set_float(void* p, const char* k, float v) { (void)p; (void)k; (void)v; }
static void __cdecl probe_set_uint(void* p, const char* k, unsigned int v) { (void)p; (void)k; (void)v; }
static int __cdecl probe_allocate(void** result) {
    static void* table[4] = {(void*)probe_set_resource, (void*)probe_set_float, NULL, (void*)probe_set_uint};
    void** object = (void**)malloc(sizeof(*object));
    if (!object) { return 0; }
    *object = table; *result = object; return 1;
}

typedef struct ProbeImage { unsigned char* data[3]; } ProbeImage;
static void probe_free_image(ProbeImage* image) {
    for (unsigned int i = 0; i < 3; ++i) { free(image->data[i]); image->data[i] = NULL; }
}
static int probe_capture(EyeInputs* eye, GpuScene* gpu, ProbeImage* image) {
    for (unsigned int i = 0; i < 3; ++i) {
        if (!image->data[i]) { image->data[i] = (unsigned char*)malloc((size_t)eye->sizes[i]); }
        if (!image->data[i]) { return 0; }
        unsigned char* mapped = NULL;
        D3D12_RANGE range = {0, (SIZE_T)eye->sizes[i]}, no_write = {0, 0};
        if (FAILED(ID3D12Resource_Map(gpu->inputs_readback[0][i], 0, &range, (void**)&mapped))) { return 0; }
        memcpy(image->data[i], mapped, (size_t)eye->sizes[i]);
        ID3D12Resource_Unmap(gpu->inputs_readback[0][i], 0, &no_write);
    }
    return 1;
}
static const unsigned char* probe_pixel(const ProbeImage* image, const EyeInputs* eye,
                                       unsigned int texture, unsigned int x, unsigned int y) {
    const unsigned int stride[3] = {8, 4, 4};
    return image->data[texture] + eye->layouts[texture].Offset +
        (size_t)y * eye->layouts[texture].Footprint.RowPitch + x * stride[texture];
}
static int probe_ppm(const char* path, const ProbeImage* image, const EyeInputs* eye) {
    FILE* file = fopen(path, "wb");
    if (!file) { return 0; }
    int success = fprintf(file, "P6\n%u %u\n255\n", PROBE_SIZE, PROBE_SIZE) > 0;
    for (unsigned int y = 0; y < PROBE_SIZE && success; ++y) {
        for (unsigned int x = 0; x < PROBE_SIZE; ++x) {
            const uint16_t* pixel = (const uint16_t*)probe_pixel(image, eye, 0, x, y);
            unsigned char rgb[3];
            // Inputs already contain display-referred sRGB. Do not encode twice.
            for (unsigned int c = 0; c < 3; ++c) {
                float value = half_float(pixel[c]);
                if (!isfinite(value)) { success = 0; break; }
                rgb[c] = (unsigned char)(255 * fmaxf(0, fminf(1, value)) + .5f);
            }
            if (!success || fwrite(rgb, 1, 3, file) != 3) { success = 0; break; }
        }
    }
    if (fclose(file) != 0) { success = 0; }
    return success;
}
static void probe_constants(GpuScene* gpu, float camera_x) {
    unsigned int* c = gpu->constants[0];
    memset(c, 0, 32 * sizeof(*c));
    c[2] = PROBE_SIZE; c[3] = PROBE_SIZE; c[4] = 1; c[6] = 1;
    // 180 degrees about Y: local -Z looks along world +Z, local +X is -X.
    const float f[24] = {0, 1, 0, 0, camera_x, 1.4f, 1, 0,
                        0, 1, 0, 0, 0, 1.4f, 1, 0,
                        -1, 1, 1, -1, -1, 1, 1, -1};
    memcpy(c + 8, f, sizeof(f));
}
static void probe_record_readback(ID3D12GraphicsCommandList* list, EyeInputs* eye, GpuScene* gpu) {
    for (unsigned int i = 0; i < 3; ++i) {
        transition(list, eye->textures[i], eye->states[i], D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
        dst.pResource = gpu->inputs_readback[0][i]; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = eye->layouts[i];
        src.pResource = eye->textures[i]; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        ID3D12GraphicsCommandList_CopyTextureRegion(list, &dst, 0, 0, 0, &src, NULL);
        transition(list, eye->textures[i], D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        eye->states[i] = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
}
static int probe_validate(const char* label, const ProbeImage* baseline, const ProbeImage* model,
                          const EyeInputs* eye, float camera_x) {
    unsigned int covered = 0, invalid = 0, motion_bad = 0, positive = 0;
    float maximum_error = 0, minimum_depth = 1;
    for (unsigned int y = 0; y < PROBE_SIZE; ++y) {
        for (unsigned int x = 0; x < PROBE_SIZE; ++x) {
            float depth = *(const float*)probe_pixel(model, eye, 1, x, y);
            float base_depth = *(const float*)probe_pixel(baseline, eye, 1, x, y);
            const uint16_t* colour = (const uint16_t*)probe_pixel(model, eye, 0, x, y);
            const uint16_t* base_colour = (const uint16_t*)probe_pixel(baseline, eye, 0, x, y);
            const uint16_t* motion = (const uint16_t*)probe_pixel(model, eye, 2, x, y);
            float mx = half_float(motion[0]), my = half_float(motion[1]), difference = 0;
            if (!isfinite(depth) || depth < 0 || depth > 1 || !isfinite(base_depth) ||
                !isfinite(mx) || !isfinite(my)) { ++invalid; continue; }
            for (unsigned int c = 0; c < 4; ++c) {
                float value = half_float(colour[c]), base = half_float(base_colour[c]);
                if (!isfinite(value) || !isfinite(base)) { ++invalid; }
                if (c < 3) { difference += fabsf(value - base); }
            }
            if (!(depth < base_depth - .00001f && difference > .005f)) { continue; }
            ++covered; minimum_depth = fminf(minimum_depth, depth);
            // Independent inverse projection: D3D near=.1/far=20 depth -> view Z.
            // Reconstruct current pixel's world point, then project into previous
            // camera (X=0). Rotation 180Y means +.1 world camera X gives NEGATIVE
            // previous-minus-current pixel X, unlike the untracked +Z camera.
            float z = (2.0f / 19.9f) / (20.0f / 19.9f - depth);
            float local_x = (2 * ((float)x + .5f) / PROBE_SIZE - 1) * z;
            float local_y = (1 - 2 * ((float)y + .5f) / PROBE_SIZE) * z;
            float world_x = camera_x - local_x, world_y = 1.4f + local_y;
            float previous_x = ((-world_x / z) * .5f + .5f) * PROBE_SIZE;
            float previous_y = (-(world_y - 1.4f) / z * .5f + .5f) * PROBE_SIZE;
            float expected_x = previous_x - ((float)x + .5f);
            float expected_y = previous_y - ((float)y + .5f);
            float error = fmaxf(fabsf(mx - expected_x), fabsf(my - expected_y));
            maximum_error = fmaxf(maximum_error, error);
            // FP16 pixel motion plus FP32 depth inversion; static should be tiny.
            if (error > (camera_x == 0 ? .02f : .08f)) { ++motion_bad; }
            if (camera_x != 0 && (mx >= -.01f || expected_x >= -.01f)) { ++positive; }
        }
    }
    printf("Mesh probe %s: covered=%u invalid=%u motion_bad=%u wrong_sign=%u max_motion_error=%.6f min_depth=%.6f\n",
           label, covered, invalid, motion_bad, positive, maximum_error, minimum_depth);
    if (covered < 100 || invalid || motion_bad || positive) {
        fprintf(stderr, "REJECT: %s coverage/finite/depth/motion oracle failed\n", label); return 0;
    }
    return 1;
}
static int probe_unchanged(const char* label, const ProbeImage* baseline, const ProbeImage* image,
                           const EyeInputs* eye) {
    unsigned int changed = 0;
    const unsigned int strides[3] = {8, 4, 4};
    for (unsigned int i = 0; i < 3; ++i) {
        for (unsigned int y = 0; y < PROBE_SIZE; ++y) {
            for (unsigned int x = 0; x < PROBE_SIZE; ++x) {
                if (memcmp(probe_pixel(baseline, eye, i, x, y), probe_pixel(image, eye, i, x, y), strides[i])) { ++changed; }
            }
        }
    }
    printf("Mesh probe %s: changed_components=%u\n", label, changed);
    if (changed) { fprintf(stderr, "REJECT: %s wrote pixels which should be discarded\n", label); return 0; }
    return 1;
}
static int probe_triangle(ID3D12Resource* vertices, float z, float alpha, int overlap) {
    float data[6][16] = {{0}};
    const float xy[3][2] = {{-.4f, 1.1f}, {.4f, 1.1f}, {0, 1.7f}};
    for (unsigned int i = 0; i < 6; ++i) {
        float* v = data[i];
        v[0] = xy[i % 3][0]; v[1] = xy[i % 3][1];
        // Far triangle submitted second: must not overwrite the nearer triangle.
        v[2] = z + (overlap && i >= 3 ? .4f : 0);
        v[5] = -1; v[6] = .5f; v[7] = .5f; v[8] = .8f; v[10] = .5f;
        v[12] = v[13] = v[14] = 1; v[15] = alpha;
    }
    unsigned char* mapped = NULL; D3D12_RANGE no_read = {0, 0};
    if (FAILED(ID3D12Resource_Map(vertices, 0, &no_read, (void**)&mapped))) { return 0; }
    memcpy(mapped, data, sizeof(data)); ID3D12Resource_Unmap(vertices, 0, NULL); return 1;
}

int main(int argc, char** argv) {
    int skip_mesh = argc == 2 && strcmp(argv[1], "--skip-mesh") == 0;
    if (argc > 1 && !skip_mesh) { fprintf(stderr, "Usage: mesh-probe.exe [--skip-mesh]\n"); return 1; }
    EyeInputs eyes[2] = {0}; GpuScene gpu = {0}; MeshRender mesh = {0};
    ProbeImage baseline = {{0}}, moving_baseline = {{0}}, image = {{0}};
    ID3D12Device* device = NULL; ID3D12CommandQueue* queue = NULL;
    ID3D12CommandAllocator* allocator = NULL; ID3D12GraphicsCommandList* list = NULL;
    ID3D12Fence* fence = NULL; ID3D12Resource* test_vertices = NULL;
    ID3D12Resource* test_atlas = NULL; ID3D12Resource* test_upload = NULL;
    HANDLE event = NULL; int pending = 0, mesh_pending = 0, exit_code = 1;
    gpu.interactive = 1; // Preserve tracked constants; explicit post-mesh readbacks.
#define CHECK_HR(call) do { HRESULT hr = (call); if (FAILED(hr)) { fprintf(stderr, "%s => HRESULT 0x%08lx\n", #call, (unsigned long)hr); goto cleanup; } } while (0)
    CHECK_HR(D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, (void**)&device));
    D3D12_COMMAND_QUEUE_DESC queue_desc = {0}; queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    CHECK_HR(ID3D12Device_CreateCommandQueue(device, &queue_desc, &IID_ID3D12CommandQueue, (void**)&queue));
    CHECK_HR(ID3D12Device_CreateCommandAllocator(device, D3D12_COMMAND_LIST_TYPE_DIRECT, &IID_ID3D12CommandAllocator, (void**)&allocator));
    CHECK_HR(ID3D12Device_CreateCommandList(device, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, NULL, &IID_ID3D12GraphicsCommandList, (void**)&list));
    CHECK_HR(ID3D12GraphicsCommandList_Close(list));
    CHECK_HR(ID3D12Device_CreateFence(device, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, (void**)&fence));
    event = CreateEventW(NULL, FALSE, FALSE, NULL);
    if (!event) { fprintf(stderr, "CreateEvent failed\n"); goto cleanup; }
    for (unsigned int e = 0; e < 2; ++e) { if (!prepare_eye(device, &eyes[e], e, probe_allocate)) { goto cleanup; } }
    if (!prepare_gpu_scene(device, eyes, &gpu)) { goto cleanup; }
    // prepare_mesh reports possibly in-flight setup work through mesh_pending.
    if (!prepare_mesh(device, queue, eyes, &mesh, &mesh_pending)) { goto cleanup; }
    if (!create_buffer(device, D3D12_HEAP_TYPE_UPLOAD, 6 * 64, &test_vertices)) { goto cleanup; }
    // Independent opaque white atlas avoids depending on the imported model's
    // centre texel (which can legitimately be an alpha-cutout texel).
    D3D12_HEAP_PROPERTIES properties = {0}; properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    properties.CreationNodeMask = properties.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC texture = {0}; texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = texture.Height = 1; texture.DepthOrArraySize = texture.MipLevels = 1;
    texture.SampleDesc.Count = 1; texture.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
    CHECK_HR(ID3D12Device_CreateCommittedResource(device, &properties, D3D12_HEAP_FLAG_NONE, &texture,
        D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void**)&test_atlas));
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT test_layout; UINT64 test_size;
    ID3D12Device_GetCopyableFootprints(device, &texture, 0, 1, 0, &test_layout, NULL, NULL, &test_size);
    if (!create_buffer(device, D3D12_HEAP_TYPE_UPLOAD, test_size, &test_upload)) { goto cleanup; }
    unsigned char* white = NULL; D3D12_RANGE no_read = {0, 0};
    CHECK_HR(ID3D12Resource_Map(test_upload, 0, &no_read, (void**)&white));
    memset(white, 0, (size_t)test_size); memset(white + test_layout.Offset, 255, 4);
    ID3D12Resource_Unmap(test_upload, 0, NULL);
    D3D12_VERTEX_BUFFER_VIEW model_view = mesh.vertex_view;
    unsigned int model_count = mesh.vertex_count;
    // 0/1 stationary baseline/model; 2/3 moving baseline/model; 4 near triangle;
    // 5 alpha discard; 6 behind-room occlusion; 7 near/far mesh depth ordering.
    for (unsigned int frame = 0; frame < 8; ++frame) {
        float camera_x = frame == 2 || frame == 3 ? .1f : 0;
        probe_constants(&gpu, camera_x);
        mesh.vertex_view = model_view; mesh.vertex_count = model_count;
        if (frame >= 4) {
            if (!probe_triangle(test_vertices, frame == 6 ? 9 : 2, frame == 5 ? 0 : 1, frame == 7)) { goto cleanup; }
            mesh.vertex_view.BufferLocation = ID3D12Resource_GetGPUVirtualAddress(test_vertices);
            mesh.vertex_view.SizeInBytes = 6 * 64; mesh.vertex_view.StrideInBytes = 64;
            mesh.vertex_count = frame == 7 ? 6 : 3;
        }
        CHECK_HR(ID3D12CommandAllocator_Reset(allocator));
        CHECK_HR(ID3D12GraphicsCommandList_Reset(list, allocator, NULL));
        if (frame == 4) {
            D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
            dst.pResource = test_atlas; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            src.pResource = test_upload; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            src.PlacedFootprint = test_layout;
            ID3D12GraphicsCommandList_CopyTextureRegion(list, &dst, 0, 0, 0, &src, NULL);
            transition(list, test_atlas, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            D3D12_SHADER_RESOURCE_VIEW_DESC srv = {0}; srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; srv.Texture2D.MipLevels = 1;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            D3D12_CPU_DESCRIPTOR_HANDLE handle = ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(mesh.heap);
            ID3D12Device_CreateShaderResourceView(device, test_atlas, &srv, handle);
        }
        record_gpu_scene(list, &eyes[0], &gpu, 0, frame);
        if (frame != 0 && frame != 2 && !skip_mesh) { record_mesh(list, &eyes[0], &gpu, 0, &mesh); }
        probe_record_readback(list, &eyes[0], &gpu);
        CHECK_HR(ID3D12GraphicsCommandList_Close(list));
        ID3D12CommandList* lists[] = {(ID3D12CommandList*)list};
        pending = 1; ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
        UINT64 value = (UINT64)frame + 1;
        CHECK_HR(ID3D12CommandQueue_Signal(queue, fence, value));
        CHECK_HR(ID3D12Fence_SetEventOnCompletion(fence, value, event));
        if (WaitForSingleObject(event, 30000) != WAIT_OBJECT_0 ||
            ID3D12Fence_GetCompletedValue(fence) == UINT64_MAX || ID3D12Fence_GetCompletedValue(fence) < value) {
            fprintf(stderr, "Mesh probe GPU completion failed/timed out\n"); goto cleanup;
        }
        pending = 0;
        ProbeImage* target = frame == 0 ? &baseline : frame == 2 ? &moving_baseline : &image;
        if (!probe_capture(&eyes[0], &gpu, target)) { goto cleanup; }
        if (frame == 0 && !probe_ppm("mesh-baseline.ppm", target, &eyes[0])) { goto cleanup; }
        if (frame == 1) {
            if (!probe_ppm("mesh-model.ppm", target, &eyes[0]) || !probe_validate("static", &baseline, target, &eyes[0], 0)) { goto cleanup; }
        }
        if (frame == 3) {
            if (!probe_ppm("mesh-moving.ppm", target, &eyes[0]) || !probe_validate("moving", &moving_baseline, target, &eyes[0], .1f)) { goto cleanup; }
        }
        if (frame == 4 || frame == 7) {
            if (!probe_validate(frame == 4 ? "synthetic-opaque" : "mesh-depth-order", &baseline, target, &eyes[0], 0)) { goto cleanup; }
            float centre = *(const float*)probe_pixel(target, &eyes[0], 1, PROBE_SIZE / 2, PROBE_SIZE / 2);
            float expected = 20.0f / 19.9f - 2.0f / 19.9f; // world Z=2, camera Z=1.
            if (fabsf(centre - expected) > .00001f) { fprintf(stderr, "REJECT: synthetic near depth %.7f expected %.7f\n", centre, expected); goto cleanup; }
        }
        if (frame == 5 && !probe_unchanged("alpha-discard", &baseline, target, &eyes[0])) { goto cleanup; }
        if (frame == 6 && !probe_unchanged("procedural-occlusion", &baseline, target, &eyes[0])) { goto cleanup; }
    }
    exit_code = 0;
cleanup:
    // A failed signal/wait is not proof of GPU idleness. Leave live objects to
    // process teardown rather than releasing memory still referenced by the GPU.
    if (pending || mesh_pending) { fflush(NULL); ExitProcess(1); }
    probe_free_image(&baseline); probe_free_image(&moving_baseline); probe_free_image(&image);
    if (test_vertices) { ID3D12Resource_Release(test_vertices); }
    if (test_atlas) { ID3D12Resource_Release(test_atlas); }
    if (test_upload) { ID3D12Resource_Release(test_upload); }
    destroy_mesh(&mesh); destroy_gpu_scene(&gpu);
    for (unsigned int e = 0; e < 2; ++e) {
        free(eyes[e].params); eyes[e].params = NULL; (void)destroy_eye(&eyes[e], NULL);
    }
    if (event) { CloseHandle(event); }
    if (fence) { ID3D12Fence_Release(fence); }
    if (list) { ID3D12GraphicsCommandList_Release(list); }
    if (allocator) { ID3D12CommandAllocator_Release(allocator); }
    if (queue) { ID3D12CommandQueue_Release(queue); }
    if (device) { ID3D12Device_Release(device); }
    if (!exit_code) {
        printf("PASS: offscreen mesh colour/depth/motion, static/moving camera, alpha discard, procedural occlusion and mesh depth ordering. No NGX/OpenXR; NOT headset quality.\n");
    }
    return exit_code;
}
