// Offscreen composition/readback oracle. No OpenXR or NGX runtime is loaded.
// Build from repository root; run with vr/probe as the working directory.
// Default 720 square per eye; compile with -DPROBE_SIZE=64u for a cheaper test.
#define main feature_probe_unused_main
#include "feature-probe.c"
#undef main
#undef CHECK_HR
#include <stdlib.h>

// prepare_eye populates NGX parameters; a no-op ABI object avoids loading NGX.
static void __cdecl aa_set_resource(void* p, const char* k, unsigned long long v) { (void)p; (void)k; (void)v; }
static void __cdecl aa_set_float(void* p, const char* k, float v) { (void)p; (void)k; (void)v; }
static void __cdecl aa_set_uint(void* p, const char* k, unsigned int v) { (void)p; (void)k; (void)v; }
static int __cdecl aa_allocate(void** result) {
    static void* table[4] = {(void*)aa_set_resource, (void*)aa_set_float, NULL, (void*)aa_set_uint};
    void** object = (void**)malloc(sizeof(*object));
    if (!object) { return 0; }
    *object = table; *result = object; return 1;
}

static float aa_source(unsigned int eye, unsigned int texture, int diagonal,
                       unsigned int x, unsigned int y, unsigned int channel) {
    // Both source choices have real edges, but different directions/colours.
    // Opposite eye/channel values also expose descriptor-table aliasing.
    const float low[3] = {.03125f, .125f, .25f};
    const float high[3] = {.875f, .75f, .5f};
    unsigned int c = (channel + eye + (texture == 3 ? 1u : 0u)) % 3u;
    int bright = texture == 3 ? x / 3u + y / 3u >= PROBE_SIZE / 3u : x / 3u > y / 3u;
    return diagonal ? (bright ? high[c] : low[c]) : (texture == 3 ? high[c] : low[c]);
}
static int aa_fill(EyeInputs* eye, unsigned int index, int diagonal, uint16_t* sources[2]) {
    for (unsigned int t = 0; t < PROBE_TEXTURES; ++t) {
        unsigned char* mapped = NULL; D3D12_RANGE no_read = {0, 0};
        if (FAILED(ID3D12Resource_Map(eye->uploads[t], 0, &no_read, (void**)&mapped))) { return 0; }
        memset(mapped, 0, (size_t)eye->sizes[t]);
        for (unsigned int y = 0; y < PROBE_SIZE; ++y) {
            unsigned char* row = mapped + eye->layouts[t].Offset + (size_t)y * eye->layouts[t].Footprint.RowPitch;
            for (unsigned int x = 0; x < PROBE_SIZE; ++x) {
                if (t == 0 || t == 3) {
                    uint16_t* p = (uint16_t*)row + x * 4u;
                    for (unsigned int c = 0; c < 3; ++c) { p[c] = float_half(aa_source(index, t, diagonal, x, y, c)); }
                    p[3] = float_half(.375f); // compose must replace source alpha with 1.
                    memcpy(sources[t == 3 ? 1 : 0] + ((size_t)y * PROBE_SIZE + x) * 4u, p, 8);
                } else if (t == 1) { ((float*)row)[x] = .5f; }
                // Motion is zero, and is deliberately irrelevant to spatial AA.
            }
        }
        ID3D12Resource_Unmap(eye->uploads[t], 0, NULL);
    }
    return 1;
}
static float aa_decode(float value) {
    return value <= .04045f ? value / 12.92f : powf((value + .055f) / 1.055f, 2.4f);
}
// Independent CPU implementation of composeLoad/Bilinear/EdgeAA in gpu-scene.hlsl.
// The input is a copy of the actual uploaded RGBA half bits, never Presented.
typedef struct AaRgb { float c[3]; } AaRgb;
static AaRgb aa_load(const uint16_t* source, int x, int y) {
    x = x < 0 ? 0 : (x >= (int)PROBE_SIZE ? (int)PROBE_SIZE - 1 : x);
    y = y < 0 ? 0 : (y >= (int)PROBE_SIZE ? (int)PROBE_SIZE - 1 : y);
    AaRgb result;
    for (unsigned int c = 0; c < 3; ++c) {
        result.c[c] = fmaxf(0, half_float(source[((size_t)y * PROBE_SIZE + (unsigned int)x) * 4u + c]));
    }
    return result;
}
static float aa_luma(AaRgb colour) {
    return colour.c[0] * .299f + colour.c[1] * .587f + colour.c[2] * .114f;
}
static AaRgb aa_bilinear(const uint16_t* source, float x, float y) {
    x = fminf(fmaxf(x, 0), (float)PROBE_SIZE - 1);
    y = fminf(fmaxf(y, 0), (float)PROBE_SIZE - 1);
    int ix = (int)floorf(x), iy = (int)floorf(y);
    float wx = x - (float)ix, wy = y - (float)iy;
    AaRgb nw = aa_load(source, ix, iy), ne = aa_load(source, ix + 1, iy);
    AaRgb sw = aa_load(source, ix, iy + 1), se = aa_load(source, ix + 1, iy + 1), result;
    for (unsigned int c = 0; c < 3; ++c) {
        float top = nw.c[c] + wx * (ne.c[c] - nw.c[c]);
        float bottom = sw.c[c] + wx * (se.c[c] - sw.c[c]);
        result.c[c] = top + wy * (bottom - top);
    }
    return result;
}
static AaRgb aa_reference(const uint16_t* source, int x, int y, int enabled, int* edge) {
    AaRgb centre = aa_load(source, x, y);
    *edge = 0;
    if (!enabled) { return centre; }
    float nw = aa_luma(aa_load(source, x - 1, y - 1));
    float ne = aa_luma(aa_load(source, x + 1, y - 1));
    float sw = aa_luma(aa_load(source, x - 1, y + 1));
    float se = aa_luma(aa_load(source, x + 1, y + 1));
    float middle = aa_luma(centre);
    float low = fminf(middle, fminf(fminf(nw, ne), fminf(sw, se)));
    float high = fmaxf(middle, fmaxf(fmaxf(nw, ne), fmaxf(sw, se)));
    if (high - low < fmaxf(.0312f, high * .125f)) { return centre; }
    *edge = 1;
    float dx = -((nw + ne) - (sw + se)), dy = (nw + sw) - (ne + se);
    float reduce = fmaxf((nw + ne + sw + se) * .25f * (1.0f / 8.0f), 1.0f / 128.0f);
    float divisor = fminf(fabsf(dx), fabsf(dy)) + reduce;
    dx = fminf(fmaxf(dx / divisor, -8), 8);
    dy = fminf(fmaxf(dy / divisor, -8), 8);
    AaRgb near_a = aa_bilinear(source, (float)x - dx * (1.0f / 6.0f), (float)y - dy * (1.0f / 6.0f));
    AaRgb near_b = aa_bilinear(source, (float)x + dx * (1.0f / 6.0f), (float)y + dy * (1.0f / 6.0f));
    AaRgb far_a = aa_bilinear(source, (float)x - dx * .5f, (float)y - dy * .5f);
    AaRgb far_b = aa_bilinear(source, (float)x + dx * .5f, (float)y + dy * .5f);
    AaRgb candidate_a, candidate_b;
    for (unsigned int c = 0; c < 3; ++c) {
        candidate_a.c[c] = .5f * (near_a.c[c] + near_b.c[c]);
        candidate_b.c[c] = .5f * candidate_a.c[c] + .25f * (far_a.c[c] + far_b.c[c]);
    }
    float luma_b = aa_luma(candidate_b);
    return luma_b < low || luma_b > high ? candidate_a : candidate_b;
}
static int aa_validate(const unsigned char* data, const unsigned char* baseline,
                       const D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout, const uint16_t* source,
                       unsigned int eye, int diagonal, int neural, int enabled) {
    unsigned int changed = 0, nonlocal = 0, invalid = 0, mismatch = 0;
    float max_error = 0;
    for (unsigned int y = 0; y < PROBE_SIZE; ++y) {
        const uint16_t* row = (const uint16_t*)(data + layout->Offset + (size_t)y * layout->Footprint.RowPitch);
        const uint16_t* base = baseline ? (const uint16_t*)(baseline + layout->Offset + (size_t)y * layout->Footprint.RowPitch) : NULL;
        for (unsigned int x = 0; x < PROBE_SIZE; ++x) {
            const uint16_t* p = row + x * 4u;
            int edge = 0;
            AaRgb encoded = aa_reference(source, (int)x, (int)y, enabled, &edge);
            if (base && memcmp(p, base + x * 4u, 8)) {
                ++changed;
                if (!edge) { ++nonlocal; }
            }
            for (unsigned int c = 0; c < 4; ++c) {
                float v = half_float(p[c]);
                if (!isfinite(v) || v < 0 || v > 1 || (c == 3 && v != 1)) { ++invalid; }
                float expected = c == 3 ? 1 : aa_decode(encoded.c[c]);
                uint16_t h = float_half(expected);
                // CPU converter truncates; GPU stores round-to-nearest. Allow
                // one half ULP plus FP32 filter/pow error, not an sRGB-space result.
                float tolerance = c == 3 ? 0 : half_float((uint16_t)(h + 1u)) - half_float(h) + .000002f;
                float error = fabsf(v - expected);
                max_error = fmaxf(max_error, error);
                if (error > tolerance) { ++mismatch; }
            }
        }
    }
    printf("AA probe eye=%u pattern=%s neural=%d flag=%u: changed=%u nonlocal=%u invalid=%u reference_mismatch=%u max_reference_error=%.7f\n",
           eye, diagonal ? "diagonal" : "flat", neural, enabled ? 2u : 0u, changed, nonlocal, invalid, mismatch, max_error);
    if (invalid || mismatch || nonlocal) { fprintf(stderr, "REJECT: composition finite/range/alpha/local-edge/CPU FXAA linear sRGB reference oracle failed\n"); return 0; }
    if (enabled && !diagonal && changed) { fprintf(stderr, "REJECT: flat colour changed by AA\n"); return 0; }
    if (enabled && diagonal && changed < 10u) { fprintf(stderr, "REJECT: diagonal unchanged or insufficient AA edge changes\n"); return 0; }
    return 1;
}

int main(int argc, char** argv) {
    int skip_aa = argc == 2 && strcmp(argv[1], "--skip-aa") == 0;
    int bad_output = argc == 2 && strcmp(argv[1], "--bad-aa-output") == 0;
    if (argc > 1 && !skip_aa && !bad_output) { fprintf(stderr, "Usage: aa-probe.exe [--skip-aa|--bad-aa-output]\n"); return 1; }
    if (PROBE_SIZE < 32u) { fprintf(stderr, "PROBE_SIZE must be at least 32\n"); return 1; }
    EyeInputs eyes[2] = {0}; GpuScene gpu = {0};
    ID3D12Device* device = NULL; ID3D12CommandQueue* queue = NULL;
    ID3D12CommandAllocator* allocator = NULL; ID3D12GraphicsCommandList* list = NULL;
    ID3D12Fence* fence = NULL; HANDLE event = NULL;
    ID3D12Resource* readback[2] = {NULL, NULL};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layouts[2] = {0}; UINT64 sizes[2] = {0};
    unsigned char* baseline[2] = {NULL, NULL}, *output[2] = {NULL, NULL};
    uint16_t* sources[2][2] = {{NULL, NULL}, {NULL, NULL}};
    D3D12_RESOURCE_STATES presented_states[2] = {D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
    int pending = 0, exit_code = 1; UINT64 fence_value = 0;
    gpu.interactive = 1;
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
    for (unsigned int e = 0; e < 2; ++e) {
        if (!prepare_eye(device, &eyes[e], e, aa_allocate)) { goto cleanup; }
    }
    if (!prepare_gpu_scene(device, eyes, &gpu)) { goto cleanup; }
    for (unsigned int e = 0; e < 2; ++e) {
        D3D12_RESOURCE_DESC desc = ID3D12Resource_GetDesc(gpu.presented[e]);
        ID3D12Device_GetCopyableFootprints(device, &desc, 0, 1, 0, &layouts[e], NULL, NULL, &sizes[e]);
        if (layouts[e].Footprint.Format != DXGI_FORMAT_R16G16B16A16_FLOAT ||
            !create_buffer(device, D3D12_HEAP_TYPE_READBACK, sizes[e], &readback[e])) { goto cleanup; }
        baseline[e] = (unsigned char*)malloc((size_t)sizes[e]);
        output[e] = (unsigned char*)malloc((size_t)sizes[e]);
        for (unsigned int t = 0; t < 2; ++t) {
            sources[e][t] = (uint16_t*)malloc((size_t)PROBE_SIZE * PROBE_SIZE * 4u * sizeof(uint16_t));
        }
        if (!baseline[e] || !output[e] || !sources[e][0] || !sources[e][1]) { goto cleanup; }
    }
    for (int diagonal = 0; diagonal < 2; ++diagonal) {
        for (int neural = 0; neural < 2; ++neural) {
            for (int enabled = 0; enabled < 2; ++enabled) {
                // Upload rewrites and allocator reset are safe: previous submission
                // has completed before advancing to the next case.
                for (unsigned int e = 0; e < 2; ++e) { if (!aa_fill(&eyes[e], e, diagonal, sources[e])) { goto cleanup; } }
                CHECK_HR(ID3D12CommandAllocator_Reset(allocator));
                CHECK_HR(ID3D12GraphicsCommandList_Reset(list, allocator, gpu.composition));
                ID3D12GraphicsCommandList_SetComputeRootSignature(list, gpu.root);
                ID3D12GraphicsCommandList_SetDescriptorHeaps(list, 1, &gpu.heap);
                for (unsigned int e = 0; e < 2; ++e) {
                    for (unsigned int t = 0; t < PROBE_TEXTURES; ++t) {
                        upload_texture(list, &eyes[e], t);
                        transition(list, eyes[e].textures[t], eyes[e].states[t], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                        eyes[e].states[t] = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                    }
                    transition(list, gpu.presented[e], presented_states[e], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    unsigned int constants[32] = {0};
                    constants[0] = e; constants[2] = PROBE_SIZE; constants[3] = PROBE_SIZE;
                    constants[5] = (unsigned int)neural;
                    // Negative control still EXPECTS AA, but intentionally binds 0.
                    constants[6] = enabled && !skip_aa ? 2u : 0u;
                    D3D12_GPU_DESCRIPTOR_HANDLE handle = ID3D12DescriptorHeap_GetGPUDescriptorHandleForHeapStart(gpu.heap);
                    handle.ptr += (UINT64)e * 5u * gpu.descriptor_stride;
                    ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(list, 0, handle);
                    ID3D12GraphicsCommandList_SetComputeRoot32BitConstants(list, 1, 32, constants, 0);
                    ID3D12GraphicsCommandList_Dispatch(list, (PROBE_SIZE + 7u) / 8u, (PROBE_SIZE + 7u) / 8u, 1);
                    transition(list, gpu.presented[e], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    presented_states[e] = D3D12_RESOURCE_STATE_COPY_SOURCE;
                    D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
                    dst.pResource = readback[e]; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = layouts[e];
                    src.pResource = gpu.presented[e]; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                    ID3D12GraphicsCommandList_CopyTextureRegion(list, &dst, 0, 0, 0, &src, NULL);
                }
                CHECK_HR(ID3D12GraphicsCommandList_Close(list));
                ID3D12CommandList* lists[] = {(ID3D12CommandList*)list};
                pending = 1; ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
                ++fence_value;
                CHECK_HR(ID3D12CommandQueue_Signal(queue, fence, fence_value));
                CHECK_HR(ID3D12Fence_SetEventOnCompletion(fence, fence_value, event));
                if (WaitForSingleObject(event, 30000) != WAIT_OBJECT_0 ||
                    ID3D12Fence_GetCompletedValue(fence) == UINT64_MAX ||
                    ID3D12Fence_GetCompletedValue(fence) < fence_value) {
                    fprintf(stderr, "AA probe GPU completion failed/timed out\n"); goto cleanup;
                }
                pending = 0;
                for (unsigned int e = 0; e < 2; ++e) {
                    unsigned char* mapped = NULL;
                    D3D12_RANGE range = {0, (SIZE_T)sizes[e]}, no_write = {0, 0};
                    CHECK_HR(ID3D12Resource_Map(readback[e], 0, &range, (void**)&mapped));
                    memcpy(output[e], mapped, (size_t)sizes[e]);
                    // Corrupt CPU readback only: reference remains independent and
                    // the GPU/shader/bindings are unchanged by this negative control.
                    if (bad_output && enabled && diagonal && !neural && e == 0) {
                        int corrupted = 0;
                        for (unsigned int y = 0; y < PROBE_SIZE && !corrupted; ++y) {
                            for (unsigned int x = 0; x < PROBE_SIZE && !corrupted; ++x) {
                                int edge = 0;
                                (void)aa_reference(sources[e][neural], (int)x, (int)y, 1, &edge);
                                if (edge) {
                                    uint16_t* p = (uint16_t*)(output[e] + layouts[e].Offset + (size_t)y * layouts[e].Footprint.RowPitch) + x * 4u;
                                    p[0] = float_half(half_float(p[0]) + .05f);
                                    corrupted = 1;
                                    printf("Negative control: corrupted AA CPU readback eye=%u pixel=%u,%u channel=0 by +0.05\n", e, x, y);
                                }
                            }
                        }
                        if (!corrupted) { fprintf(stderr, "REJECT: negative control found no edge\n"); }
                    }
                    int valid = aa_validate(output[e], enabled ? baseline[e] : NULL, &layouts[e], sources[e][neural], e, diagonal, neural, enabled);
                    if (!enabled) { memcpy(baseline[e], mapped, (size_t)sizes[e]); }
                    ID3D12Resource_Unmap(readback[e], 0, &no_write);
                    if (!valid) { goto cleanup; }
                }
            }
        }
    }
    exit_code = 0;
cleanup:
    // Failed signal/wait does not establish GPU idleness. Do not release live
    // GPU objects or backing uploads; leave them to process teardown instead.
    if (pending) { fflush(NULL); ExitProcess(1); }
    for (unsigned int e = 0; e < 2; ++e) {
        free(baseline[e]); free(output[e]);
        for (unsigned int t = 0; t < 2; ++t) { free(sources[e][t]); }
        if (readback[e]) { ID3D12Resource_Release(readback[e]); }
    }
    destroy_gpu_scene(&gpu);
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
        printf("PASS: offscreen composition AA flat unchanged, diagonal edges changed, both eyes and colour/neural sources, all-pixel CPU FXAA reference on uploaded half inputs, local edges, linear RGB/alpha and sRGB decode. No NGX/OpenXR; NOT headset quality.\n");
    }
    return exit_code;
}
