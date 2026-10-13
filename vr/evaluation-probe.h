// Synthetic, stationary test inputs: NOT a stereo renderer or temporal-quality test.
#include <math.h>
#include <string.h>
#ifndef PROBE_SIZE
#define PROBE_SIZE 720u
#endif
#define PROBE_FRAMES 60u
#define PROBE_TEXTURES 4u

typedef int (__cdecl *FwdEvaluate)(ID3D12GraphicsCommandList*, void*, void*);
typedef int (__cdecl *AllocateParams)(void**);
typedef int (__cdecl *DestroyParams)(void*);
typedef void (__cdecl *SetFloat)(void*, const char*, float);
typedef void (__cdecl *SetResource)(void*, const char*, unsigned long long);

typedef struct EyeInputs {
    void* params;
    ID3D12Resource* textures[PROBE_TEXTURES]; // colour, depth, motion, output
    ID3D12Resource* uploads[PROBE_TEXTURES];
    ID3D12Resource* readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layouts[PROBE_TEXTURES];
    UINT64 sizes[PROBE_TEXTURES];
    D3D12_RESOURCE_STATES states[PROBE_TEXTURES];
    unsigned int successful_evaluations;
    unsigned int validated_outputs;
} EyeInputs;

static uint16_t float_half(float value) {
    // Finite test values: clamp overflow, round by truncation; support subnormals/sign.
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    unsigned int sign = (bits >> 16) & 0x8000u;
    int exponent = (int)((bits >> 23) & 255u) - 112;
    unsigned int mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) { return (uint16_t)sign; }
        return (uint16_t)(sign | ((mantissa | 0x800000u) >> (14 - exponent)));
    }
    if (exponent >= 31) { return (uint16_t)(sign | 0x7bffu); }
    return (uint16_t)(sign | ((unsigned int)exponent << 10) | (mantissa >> 13));
}
static float half_float(uint16_t value) {
    unsigned int exponent = (value >> 10) & 31u;
    unsigned int fraction = value & 1023u;
    float result;
    if (exponent == 31u) { return fraction ? NAN : ((value & 0x8000u) ? -INFINITY : INFINITY); }
    if (exponent == 0) { result = ldexpf((float)fraction, -24); }
    else { result = ldexpf(1.0f + (float)fraction / 1024.0f, (int)exponent - 15); }
    return (value & 0x8000u) ? -result : result;
}
static void set_float(void* params, const char* key, float value) {
    void** vtable = *(void***)params;
    ((SetFloat)vtable[1])(params, key, value);
}
static void set_resource(void* params, const char* key, ID3D12Resource* value) {
    void** vtable = *(void***)params;
    ((SetResource)vtable[0])(params, key, (unsigned long long)(uintptr_t)value);
}
static void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                       D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    if (before == after) { return; }
    D3D12_RESOURCE_BARRIER barrier = {0};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &barrier);
}
static int create_buffer(ID3D12Device* device, D3D12_HEAP_TYPE heap_type,
                         UINT64 size, ID3D12Resource** resource) {
    D3D12_HEAP_PROPERTIES heap = {0};
    heap.Type = heap_type; heap.CreationNodeMask = 1; heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc = {0};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size; desc.Height = 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
    desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    HRESULT result = ID3D12Device_CreateCommittedResource(device, &heap, D3D12_HEAP_FLAG_NONE,
        &desc, heap_type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST,
        NULL, &IID_ID3D12Resource, (void**)resource);
    if (FAILED(result)) { fprintf(stderr, "Buffer allocation failed 0x%08lx\n", (unsigned long)result); return 0; }
    return 1;
}
static int prepare_eye(ID3D12Device* device, EyeInputs* eye, unsigned int eye_index,
                       AllocateParams allocate) {
    if (allocate(&eye->params) != 1 || !eye->params) { fprintf(stderr, "AllocateParameters failed\n"); return 0; }
    const DXGI_FORMAT formats[] = {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32_FLOAT,
                                   DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT};
    for (unsigned int texture = 0; texture < PROBE_TEXTURES; ++texture) {
        D3D12_HEAP_PROPERTIES heap = {0};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT; heap.CreationNodeMask = 1; heap.VisibleNodeMask = 1;
        D3D12_RESOURCE_DESC desc = {0};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = PROBE_SIZE; desc.Height = PROBE_SIZE;
        desc.DepthOrArraySize = 1; desc.MipLevels = 1; desc.Format = formats[texture];
        desc.SampleDesc.Count = 1; desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
#ifdef PROBE_MESH_RENDER
        if (texture < 3) { desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET; }
#endif
        HRESULT hr = ID3D12Device_CreateCommittedResource(device, &heap, D3D12_HEAP_FLAG_NONE,
            &desc, D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void**)&eye->textures[texture]);
        if (FAILED(hr)) { fprintf(stderr, "Texture allocation failed 0x%08lx\n", (unsigned long)hr); return 0; }
        eye->states[texture] = D3D12_RESOURCE_STATE_COPY_DEST;
        ID3D12Device_GetCopyableFootprints(device, &desc, 0, 1, 0, &eye->layouts[texture], NULL, NULL, &eye->sizes[texture]);
        if (!create_buffer(device, D3D12_HEAP_TYPE_UPLOAD, eye->sizes[texture], &eye->uploads[texture])) { return 0; }
        unsigned char* mapped = NULL;
        D3D12_RANGE no_read = {0, 0};
        hr = ID3D12Resource_Map(eye->uploads[texture], 0, &no_read, (void**)&mapped);
        if (FAILED(hr)) { return 0; }
        memset(mapped, 0, (size_t)eye->sizes[texture]);
        for (unsigned int y = 0; y < PROBE_SIZE; ++y) {
            unsigned char* row = mapped + eye->layouts[texture].Offset + y * eye->layouts[texture].Footprint.RowPitch;
            for (unsigned int x = 0; x < PROBE_SIZE; ++x) {
                float u = (float)x / (PROBE_SIZE - 1); float v = (float)y / (PROBE_SIZE - 1);
                if (texture == 0) {
                    uint16_t* pixel = (uint16_t*)row + x * 4;
                    // Opposing channel gradients make swapped/aliased outputs observable.
                    pixel[0] = float_half(eye_index == 0 ? 0.15f + 0.7f * u : 0.08f + 0.1f * u);
                    pixel[1] = float_half(0.1f + 0.2f * v);
                    pixel[2] = float_half(eye_index == 1 ? 0.15f + 0.7f * u : 0.08f + 0.1f * u);
                    pixel[3] = float_half(1.0f);
                } else if (texture == 1) {
                    ((float*)row)[x] = 0.2f + 0.6f * v;
                } else if (texture == 3) {
                    uint16_t* pixel = (uint16_t*)row + x * 4;
                    for (unsigned int channel = 0; channel < 4; ++channel) { pixel[channel] = 0x7e00u; } // NaN sentinel
                }
                // Motion stays zero: these inputs/camera are stationary.
            }
        }
        ID3D12Resource_Unmap(eye->uploads[texture], 0, NULL);
    }
    if (!create_buffer(device, D3D12_HEAP_TYPE_READBACK, eye->sizes[3], &eye->readback)) { return 0; }
    static const char* keys[] = {"DLSSNR.Color", "DLSSNR.Depth", "DLSSNR.MVec", "DLSSNR.Output"};
    for (unsigned int i = 0; i < PROBE_TEXTURES; ++i) { set_resource(eye->params, keys[i], eye->textures[i]); }
    set_uint(eye->params, "DLSSNR.Enabled", 1);
    set_uint(eye->params, "DLSSNR.Width", PROBE_SIZE); set_uint(eye->params, "DLSSNR.Height", PROBE_SIZE);
    set_uint(eye->params, "DLSSNR.DepthInverted", 0);
    static const char* prefixes[] = {"Color", "Depth", "MVec", "Output"};
    for (unsigned int i = 0; i < 4; ++i) {
        char key[80];
        snprintf(key, sizeof(key), "DLSSNR.%sSubrectBaseX", prefixes[i]); set_uint(eye->params, key, 0);
        snprintf(key, sizeof(key), "DLSSNR.%sSubrectBaseY", prefixes[i]); set_uint(eye->params, key, 0);
        snprintf(key, sizeof(key), "DLSSNR.%sSubrectWidth", prefixes[i]); set_uint(eye->params, key, PROBE_SIZE);
        snprintf(key, sizeof(key), "DLSSNR.%sSubrectHeight", prefixes[i]); set_uint(eye->params, key, PROBE_SIZE);
    }
    set_float(eye->params, "DLSSNR.MVecScaleX", 1.0f); set_float(eye->params, "DLSSNR.MVecScaleY", 1.0f);
    set_float(eye->params, "DLSSNR.Intensity", 1.0f); set_uint(eye->params, "DLSSNR.Style", 0);
    set_float(eye->params, "DLSSNR.LocalStructureStrength", -1.0f);
    set_float(eye->params, "DLSSNR.LocalToneStrength", -1.0f);
    set_float(eye->params, "DLSSNR.SkinStructureStrength", -1.0f);
    set_uint(eye->params, "DLSSNR.UseAutoMask", 1);
    return 1;
}
static void upload_texture(ID3D12GraphicsCommandList* list, EyeInputs* eye, unsigned int index) {
    transition(list, eye->textures[index], eye->states[index], D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
    dst.pResource = eye->textures[index]; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource = eye->uploads[index]; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = eye->layouts[index];
    ID3D12GraphicsCommandList_CopyTextureRegion(list, &dst, 0, 0, 0, &src, NULL);
    D3D12_RESOURCE_STATES state = index == 3 ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    transition(list, eye->textures[index], D3D12_RESOURCE_STATE_COPY_DEST, state);
    eye->states[index] = state;
}
static void record_readback(ID3D12GraphicsCommandList* list, EyeInputs* eye) {
    transition(list, eye->textures[3], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
    dst.pResource = eye->readback; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = eye->layouts[3];
    src.pResource = eye->textures[3]; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    ID3D12GraphicsCommandList_CopyTextureRegion(list, &dst, 0, 0, 0, &src, NULL);
    eye->states[3] = D3D12_RESOURCE_STATE_COPY_SOURCE;
}
static int validate_readback(EyeInputs* eye, unsigned int index, unsigned int frame, int scene_mode) {
    unsigned char* mapped = NULL;
    unsigned char* input = NULL;
    D3D12_RANGE input_range = {0, (SIZE_T)eye->sizes[0]};
    if (FAILED(ID3D12Resource_Map(eye->uploads[0], 0, &input_range, (void**)&input))) { return 0; }
    D3D12_RANGE range = {0, (SIZE_T)eye->sizes[3]};
    HRESULT hr = ID3D12Resource_Map(eye->readback, 0, &range, (void**)&mapped);
    if (FAILED(hr)) { ID3D12Resource_Unmap(eye->uploads[0], 0, NULL); return 0; }
    double sums[3] = {0, 0, 0};
    double minimum[3] = {INFINITY, INFINITY, INFINITY};
    double maximum[3] = {-INFINITY, -INFINITY, -INFINITY};
    double difference = 0;
    unsigned int invalid = 0;
    for (unsigned int y = 0; y < PROBE_SIZE; ++y) {
        uint16_t* row = (uint16_t*)(mapped + eye->layouts[3].Offset + y * eye->layouts[3].Footprint.RowPitch);
        for (unsigned int x = 0; x < PROBE_SIZE; ++x) {
            for (unsigned int c = 0; c < 3; ++c) {
                float value = half_float(row[x * 4 + c]);
                if (!isfinite(value)) { ++invalid; continue; }
                sums[c] += value;
                if (value < minimum[c]) { minimum[c] = value; }
                if (value > maximum[c]) { maximum[c] = value; }
                float u = (float)x / (PROBE_SIZE - 1); float v = (float)y / (PROBE_SIZE - 1);
                uint16_t* input_row = (uint16_t*)(input + eye->layouts[0].Offset + y * eye->layouts[0].Footprint.RowPitch);
                float original = c == 1 ? 0.1f + 0.2f * v :
                    (c == (index == 0 ? 0u : 2u) ? 0.15f + 0.7f * u : 0.08f + 0.1f * u);
                difference += fabs((double)value - (scene_mode ? half_float(input_row[x*4+c]) : half_float(float_half(original))));
            }
        }
    }
    D3D12_RANGE no_write = {0, 0};
    ID3D12Resource_Unmap(eye->readback, 0, &no_write);
    ID3D12Resource_Unmap(eye->uploads[0], 0, &no_write);
    const double count = (double)PROBE_SIZE * PROBE_SIZE;
    // Checks RGB only: model alpha is not part of this milestone.
    unsigned int dominant = index == 0 ? 0u : 2u;
    int valid = invalid == 0 && maximum[dominant] - minimum[dominant] > 0.01 &&
        difference / (count * 3) > 0.00001 &&
        (scene_mode || (index == 0 ? sums[0] > sums[2] : sums[2] > sums[0]));
    printf("Output eye=%u frame=%u invalid_rgb=%u dominant_spatial_range=%.6f mean_abs_input_change=%.6f mean_rgb=%.6f,%.6f,%.6f valid=%d\n",
           index, frame, invalid, maximum[dominant]-minimum[dominant], difference/(count*3), sums[0]/count, sums[1]/count, sums[2]/count, valid);
    fflush(stdout);
    if (valid) { ++eye->validated_outputs; }
    return valid;
}
// Reference capture: dump tight-packed input/output tensors for vendor comparison.
// Files: input-colour.bin (FP16x4), input-depth.bin (F32), input-motion.bin (FP16x2),
// output-frame%02u.bin (FP16x4), manifest.txt (sizes + FNV-1a-64 digests).
#define CAPTURE_FRAMES 8u
static uint64_t capture_fnv(const unsigned char* data, size_t size) {
    uint64_t hash = 14695981039346656037ull;
    for (size_t i = 0; i < size; ++i) { hash ^= data[i]; hash *= 1099511628211ull; }
    return hash;
}
static int capture_write_file(const char* dir, const char* name, ID3D12Resource* staging,
                              const D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout,
                              SIZE_T staging_size, unsigned int channels_bytes, FILE* manifest) {
    char path[512];
    unsigned char* mapped = NULL;
    D3D12_RANGE range = {0, (SIZE_T)staging_size};
    unsigned char* tight = NULL;
    FILE* out;
    uint64_t digest;
    size_t row_bytes = (size_t)PROBE_SIZE * channels_bytes;
    size_t total = row_bytes * PROBE_SIZE;
    if (snprintf(path, sizeof(path), "%s/%s", dir, name) >= (int)sizeof(path)) {
        fprintf(stderr, "Capture path too long for %s\n", name);
        return 0;
    }
    if (FAILED(ID3D12Resource_Map(staging, 0, &range, (void**)&mapped))) {
        fprintf(stderr, "Capture map failed for %s\n", name);
        return 0;
    }
    tight = (unsigned char*)malloc(total);
    if (!tight) {
        D3D12_RANGE no_write = {0, 0};
        ID3D12Resource_Unmap(staging, 0, &no_write);
        return 0;
    }
    for (unsigned int y = 0; y < PROBE_SIZE; ++y) {
        memcpy(tight + (size_t)y * row_bytes,
               mapped + layout->Offset + (size_t)y * layout->Footprint.RowPitch, row_bytes);
    }
    {
        D3D12_RANGE no_write = {0, 0};
        ID3D12Resource_Unmap(staging, 0, &no_write);
    }
    out = fopen(path, "wb");
    if (!out) {
        fprintf(stderr, "Capture open failed for %s\n", name);
        free(tight);
        return 0;
    }
    if (fwrite(tight, 1, total, out) != total) {
        fprintf(stderr, "Capture short write for %s\n", name);
        fclose(out);
        free(tight);
        return 0;
    }
    fclose(out);
    digest = capture_fnv(tight, total);
    free(tight);
    fprintf(manifest, "%s bytes=%zu fnv=0x%016llx\n", name, total, (unsigned long long)digest);
    printf("CAPTURE file=%s bytes=%zu fnv=0x%016llx\n", name, total, (unsigned long long)digest);
    fflush(stdout);
    return 1;
}
static int capture_dump(const char* dir, EyeInputs* eye, unsigned int frame, FILE* manifest) {
    static const unsigned int channel_bytes[PROBE_TEXTURES] = {8u, 4u, 4u, 8u};
    static const char* input_names[3] = {"input-colour.bin", "input-depth.bin", "input-motion.bin"};
    char output_name[64];
    if (frame == 0) {
        for (unsigned int texture = 0; texture < 3; ++texture) {
            if (!capture_write_file(dir, input_names[texture], eye->uploads[texture],
                                    &eye->layouts[texture], eye->sizes[texture],
                                    channel_bytes[texture], manifest)) {
                return 0;
            }
        }
    }
    snprintf(output_name, sizeof(output_name), "output-frame%02u.bin", frame);
    return capture_write_file(dir, output_name, eye->readback, &eye->layouts[3],
                              eye->sizes[3], channel_bytes[3], manifest);
}
static int destroy_eye(EyeInputs* eye, DestroyParams destroy) {
    int success = 1;
    if (eye->params) {
        int result = destroy(eye->params);
        printf("DestroyParameters => 0x%x\n", result);
        if (result != 1) { success = 0; }
    }
    for (unsigned int i = 0; i < PROBE_TEXTURES; ++i) {
        if (eye->textures[i]) { ID3D12Resource_Release(eye->textures[i]); }
        if (eye->uploads[i]) { ID3D12Resource_Release(eye->uploads[i]); }
    }
    if (eye->readback) { ID3D12Resource_Release(eye->readback); }
    return success;
}
