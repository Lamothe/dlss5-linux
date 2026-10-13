// Decoder-boundary intermediate capture. Truncates the model's evaluation at
// its decoder launch (DLSS_NVAPI_CUT in the nvapi64 forwarder, see
// NVAPI-DISPATCH.md) and reads the boundary buffers back through this
// harness's own device. NOT a renderer, quality or timing test.
//
// How it works:
//   1. boundary_install() wraps the device vtable so every resource the model
//      creates through this device is recorded with its GPU virtual address
//      and byte size (buffers only; textures are recorded with size 0 and are
//      never matched).
//   2. The forwarder forwards evaluation launches only up to and including the
//      decoder kernel, and writes one sidecar line per evaluation with the
//      decoder launch's 80-byte parameter block (the boundary words).
//   3. Per frame the harness resolves words 0/1/2/4/6/7 (main input, skip
//      input, out0, counter, partial sums, weights) through the resource map
//      and copies a bounded window of each into readback buffers, after a UAV
//      barrier ordered behind the recorded kernel writes.
//   4. The output texture must still hold the uploaded NaN sentinel: with the
//      later stages never dispatched, nothing can have written it. That check
//      is the in-process canary for "the cut really happened".
#ifndef BOUNDARY_PROBE_H
#define BOUNDARY_PROBE_H

#include <stdint.h>

#define BOUNDARY_FRAMES 3u
#define BOUNDARY_WINDOW (2ull * 1024ull * 1024ull)
#define BOUNDARY_WORDS 6u
#define BOUNDARY_ARG_WORDS 10u
#define BOUNDARY_HEX_MAX (80 * 2 + 4)

static const unsigned int boundary_word_index[BOUNDARY_WORDS] = {0, 1, 2, 4, 6, 7};
static const char* boundary_word_name[BOUNDARY_WORDS] = {
    "main-input", "skip-input", "out0", "counter", "partial-sums", "weights"};

#define BOUNDARY_RESOURCE_MAX 1024

typedef struct ResourceRecord {
    D3D12_GPU_VIRTUAL_ADDRESS va;
    UINT64 size;
    ID3D12Resource* resource;
} ResourceRecord;

static ResourceRecord boundary_resources[BOUNDARY_RESOURCE_MAX];
static LONG boundary_resource_count;

static void boundary_record_resource(ID3D12Resource* resource, const D3D12_RESOURCE_DESC* desc) {
    LONG slot = InterlockedIncrement(&boundary_resource_count) - 1;
    if (slot < 0 || slot >= BOUNDARY_RESOURCE_MAX) { return; }
    boundary_resources[slot].va = ID3D12Resource_GetGPUVirtualAddress(resource);
    boundary_resources[slot].size =
        desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ? desc->Width : 0;
    boundary_resources[slot].resource = resource;
}

static const ResourceRecord* boundary_find(uint64_t va) {
    LONG count = boundary_resource_count;
    if (count > BOUNDARY_RESOURCE_MAX) { count = BOUNDARY_RESOURCE_MAX; }
    // Newest first: if a released range is reused by a later resource, the
    // live record must win over the stale one.
    for (LONG i = count - 1; i >= 0; --i) {
        const ResourceRecord* record = &boundary_resources[i];
        if (record->size > 0 && va >= record->va && va < record->va + record->size) {
            return record;
        }
    }
    return NULL;
}

// Device vtable hooks: the model allocates its tensors through the device the
// harness passes to NVSDK_NGX_D3D12_Init, so wrapping the create calls gives
// the VA -> resource map the window copies need. The original vtable is kept
// and every call is forwarded unchanged; only successful creations are added
// to the map.
static const struct ID3D12DeviceVtbl* boundary_vtbl_orig;
static struct ID3D12DeviceVtbl boundary_vtbl_hook;

static HRESULT STDMETHODCALLTYPE boundary_hook_committed(
        ID3D12Device* This, const D3D12_HEAP_PROPERTIES* heap_properties,
        D3D12_HEAP_FLAGS heap_flags, const D3D12_RESOURCE_DESC* desc,
        D3D12_RESOURCE_STATES initial_state,
        const D3D12_CLEAR_VALUE* optimized_clear_value, REFIID riid,
        void** resource) {
    HRESULT hr = boundary_vtbl_orig->CreateCommittedResource(
        This, heap_properties, heap_flags, desc, initial_state,
        optimized_clear_value, riid, resource);
    if (SUCCEEDED(hr) && resource && *resource && desc) {
        boundary_record_resource((ID3D12Resource*)*resource, desc);
    }
    return hr;
}

static HRESULT STDMETHODCALLTYPE boundary_hook_placed(
        ID3D12Device* This, ID3D12Heap* heap, UINT64 heap_offset,
        const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES initial_state,
        const D3D12_CLEAR_VALUE* optimized_clear_value, REFIID riid,
        void** resource) {
    HRESULT hr = boundary_vtbl_orig->CreatePlacedResource(
        This, heap, heap_offset, desc, initial_state, optimized_clear_value,
        riid, resource);
    if (SUCCEEDED(hr) && resource && *resource && desc) {
        boundary_record_resource((ID3D12Resource*)*resource, desc);
    }
    return hr;
}

static void boundary_install(ID3D12Device* device) {
    boundary_vtbl_orig = device->lpVtbl;
    boundary_vtbl_hook = *device->lpVtbl;
    boundary_vtbl_hook.CreateCommittedResource = boundary_hook_committed;
    boundary_vtbl_hook.CreatePlacedResource = boundary_hook_placed;
    device->lpVtbl = &boundary_vtbl_hook;
    printf("BoundaryDeviceHook installed resources_cap=%d\n", BOUNDARY_RESOURCE_MAX);
    fflush(stdout);
}

// Sidecar parsing: one `boundary eval=<n> psize=<n> pmem=<hex>` line per
// evaluation, written by the forwarder at decoder launch time. The harness
// reads the file after each evaluate returns and requires exactly one new
// line per call; anything else is a misaligned capture and fails the run.
typedef struct BoundarySidecar {
    unsigned int lines_consumed;
} BoundarySidecar;

typedef struct BoundaryWords {
    unsigned long long eval;
    unsigned int psize;
    uint64_t word[BOUNDARY_ARG_WORDS];
    char hex[BOUNDARY_HEX_MAX];
} BoundaryWords;

static int boundary_parse_line(const char* line, BoundaryWords* out) {
    const char* pmem = strstr(line, "pmem=");
    unsigned long long eval = 0;
    unsigned int psize = 0;
    size_t hexlen = 0;
    memset(out, 0, sizeof(*out));
    if (strncmp(line, "boundary eval=", 14) != 0) { return 0; }
    if (sscanf(line + 14, "%llu psize=%u", &eval, &psize) != 2) { return 0; }
    if (!pmem) { return 0; }
    pmem += 5;
    while (pmem[hexlen] && pmem[hexlen] != '\n' && pmem[hexlen] != '\r' &&
           pmem[hexlen] != ' ') {
        ++hexlen;
    }
    if (hexlen == 0 || hexlen >= sizeof(out->hex) || (hexlen & 1) != 0) { return 0; }
    // Strict: the decoder parameter block is exactly 80 bytes; anything else
    // would leave argument words uninitialised and fail closed here instead.
    if (psize != BOUNDARY_ARG_WORDS * 8 || hexlen != (size_t)psize * 2) { return 0; }
    memcpy(out->hex, pmem, hexlen);
    out->hex[hexlen] = 0;
    out->eval = eval;
    out->psize = psize;
    // Hex decode into little-endian 64-bit argument words.
    for (size_t i = 0; i < hexlen / 2 && i < sizeof(uint64_t) * BOUNDARY_ARG_WORDS; ++i) {
        unsigned int nibble[2];
        for (unsigned int k = 0; k < 2; ++k) {
            char c = pmem[i * 2 + k];
            if (c >= '0' && c <= '9') { nibble[k] = (unsigned int)(c - '0'); }
            else if (c >= 'a' && c <= 'f') { nibble[k] = (unsigned int)(c - 'a' + 10); }
            else if (c >= 'A' && c <= 'F') { nibble[k] = (unsigned int)(c - 'A' + 10); }
            else { return 0; }
        }
        if (i < sizeof(out->word)) {
            ((unsigned char*)out->word)[i] = (unsigned char)((nibble[0] << 4) | nibble[1]);
        }
    }
    return 1;
}

// Returns the number of new complete lines since the last call (must be 1),
// with the last one parsed into *out. Incomplete trailing lines are ignored.
static int boundary_next_words(const char* path, BoundarySidecar* state, BoundaryWords* out) {
    char buffer[65536];
    char line[BOUNDARY_HEX_MAX + 128];
    HANDLE file;
    DWORD got = 0;
    DWORD offset = 0;
    int seen = 0;
    file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                       NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) { return -1; }
    if (!ReadFile(file, buffer, sizeof(buffer) - 1, &got, NULL)) {
        CloseHandle(file);
        return -1;
    }
    CloseHandle(file);
    buffer[got] = 0;
    unsigned int index = 0;
    while (offset < got) {
        DWORD length = 0;
        while (offset + length < got && buffer[offset + length] != '\n') { ++length; }
        if (offset + length >= got) { break; } // incomplete trailing line
        if (length >= sizeof(line)) { return -1; }
        memcpy(line, buffer + offset, length);
        line[length] = 0;
        offset += length + 1;
        if (index >= state->lines_consumed) {
            if (!boundary_parse_line(line, out)) { return -1; }
            ++seen;
        }
        ++index;
    }
    state->lines_consumed = index;
    return seen;
}

// The output texture still holds the uploaded NaN sentinel only when no later
// stage ran: the cut's in-process canary.
static int boundary_validate_untouched(EyeInputs* eye, unsigned int frame) {
    unsigned char* mapped = NULL;
    unsigned int nan_pixels = 0;
    unsigned int other = 0;
    D3D12_RANGE range = {0, (SIZE_T)eye->sizes[3]};
    if (FAILED(ID3D12Resource_Map(eye->readback, 0, &range, (void**)&mapped))) {
        fprintf(stderr, "Boundary output map failed frame=%u\n", frame);
        return 0;
    }
    for (unsigned int y = 0; y < PROBE_SIZE; ++y) {
        uint16_t* row = (uint16_t*)(mapped + eye->layouts[3].Offset +
                                    y * eye->layouts[3].Footprint.RowPitch);
        for (unsigned int x = 0; x < PROBE_SIZE; ++x) {
            int nan = 1;
            for (unsigned int c = 0; c < 4; ++c) {
                if (row[x * 4 + c] != 0x7e00u) { nan = 0; }
            }
            if (nan) { ++nan_pixels; } else { ++other; }
        }
    }
    {
        D3D12_RANGE no_write = {0, 0};
        ID3D12Resource_Unmap(eye->readback, 0, &no_write);
    }
    printf("BoundaryOutputUntouched frame=%u nan_pixels=%u other_pixels=%u untouched=%d\n",
           frame, nan_pixels, other, other == 0);
    fflush(stdout);
    return other == 0;
}

typedef struct BoundaryContext {
    ID3D12Device* device;
    ID3D12CommandQueue* queue;
    ID3D12CommandAllocator* allocator;
    ID3D12GraphicsCommandList* list;
    ID3D12Fence* fence;
    HANDLE event;
    EyeInputs* eye;
    FwdEvaluate evaluate;
    void* feature;
    const char* dir;
    FILE* manifest;
    const char* sidecar;
    int* submitted;
    int* completed;
} BoundaryContext;

static int boundary_dump_window(const BoundaryContext* ctx, unsigned int frame,
                                unsigned int slot, const BoundaryWords* words,
                                const ResourceRecord* record, UINT64 offset,
                                UINT64 bytes, ID3D12Resource* readback) {
    unsigned char* mapped = NULL;
    FILE* out;
    char path[512];
    char name[64];
    uint64_t digest;
    D3D12_RANGE range = {0, (SIZE_T)bytes};
    if (FAILED(ID3D12Resource_Map(readback, 0, &range, (void**)&mapped))) {
        fprintf(stderr, "Boundary window map failed frame=%u word=%u\n", frame, slot);
        return 0;
    }
    digest = capture_fnv(mapped, (size_t)bytes);
    snprintf(name, sizeof(name), "frame%02u-w%u.bin", frame, boundary_word_index[slot]);
    if (snprintf(path, sizeof(path), "%s/%s", ctx->dir, name) >= (int)sizeof(path)) {
        fprintf(stderr, "Boundary window path too long\n");
        D3D12_RANGE no_write = {0, 0};
        ID3D12Resource_Unmap(readback, 0, &no_write);
        return 0;
    }
    out = fopen(path, "wb");
    if (!out || fwrite(mapped, 1, (size_t)bytes, out) != (size_t)bytes) {
        fprintf(stderr, "Boundary window write failed %s\n", name);
        if (out) { fclose(out); }
        D3D12_RANGE no_write = {0, 0};
        ID3D12Resource_Unmap(readback, 0, &no_write);
        return 0;
    }
    fclose(out);
    {
        D3D12_RANGE no_write = {0, 0};
        ID3D12Resource_Unmap(readback, 0, &no_write);
    }
    fprintf(ctx->manifest,
            "word=%u name=%s frame=%u eval=%llu va=0x%llx res_va=0x%llx res_size=%llu "
            "off=%llu bytes=%llu file=%s fnv=0x%016llx\n",
            boundary_word_index[slot], boundary_word_name[slot], frame, words->eval,
            (unsigned long long)words->word[boundary_word_index[slot]],
            (unsigned long long)record->va, (unsigned long long)record->size,
            (unsigned long long)offset, (unsigned long long)bytes, name,
            (unsigned long long)digest);
    printf("BOUNDARY frame=%u word=%u name=%s bytes=%llu fnv=0x%016llx\n", frame,
           boundary_word_index[slot], boundary_word_name[slot],
           (unsigned long long)bytes, (unsigned long long)digest);
    fflush(stdout);
    return 1;
}

static int boundary_run(BoundaryContext* ctx) {
    ID3D12CommandList* command_lists[] = {(ID3D12CommandList*)ctx->list};
    ID3D12Resource* windows[BOUNDARY_WORDS] = {0};
    BoundarySidecar sidecar = {0};
    int success = 0;
    for (unsigned int slot = 0; slot < BOUNDARY_WORDS; ++slot) {
        if (!create_buffer(ctx->device, D3D12_HEAP_TYPE_READBACK, BOUNDARY_WINDOW,
                           &windows[slot])) {
            goto done;
        }
    }
    fprintf(ctx->manifest, "sidecar=%s\n", ctx->sidecar);
    for (unsigned int frame = 0; frame < BOUNDARY_FRAMES; ++frame) {
        BoundaryWords words;
        struct { const ResourceRecord* record; UINT64 offset; UINT64 bytes; } copies[BOUNDARY_WORDS];
        const ResourceRecord* barrier_done[BOUNDARY_WORDS];
        unsigned int barriers = 0;
        int result;
        int lines;
        if (FAILED(ID3D12CommandAllocator_Reset(ctx->allocator))) {
            fprintf(stderr, "Boundary allocator reset failed\n");
            goto done;
        }
        if (FAILED(ID3D12GraphicsCommandList_Reset(ctx->list, ctx->allocator, NULL))) {
            fprintf(stderr, "Boundary list reset failed\n");
            goto done;
        }
        if (frame == 0) {
            for (unsigned int texture = 0; texture < 3; ++texture) {
                upload_texture(ctx->list, ctx->eye, texture);
            }
        }
        upload_texture(ctx->list, ctx->eye, 3); // fresh NaN sentinel each frame
        set_uint(ctx->eye->params, "DLSSNR.Reset", frame == 0 ? 1u : 0u);
        result = ctx->evaluate(ctx->list, ctx->feature, ctx->eye->params);
        printf("BoundaryEvaluate frame=%u result=0x%x\n", frame, result);
        fflush(stdout);
        if (result != 1) { goto done; }
        lines = boundary_next_words(ctx->sidecar, &sidecar, &words);
        if (lines != 1) {
            fprintf(stderr, "Boundary sidecar lines=%d for frame=%u (need exactly 1)\n",
                    lines, frame);
            goto done;
        }
        fprintf(ctx->manifest,
                "boundary-frame=%u eval=%llu psize=%u pmem=%s\n", frame,
                words.eval, words.psize, words.hex);
        for (unsigned int slot = 0; slot < BOUNDARY_WORDS; ++slot) {
            uint64_t va = words.word[boundary_word_index[slot]];
            const ResourceRecord* record = boundary_find(va);
            UINT64 offset, bytes;
            int seen = 0;
            if (!record) {
                fprintf(stderr, "Boundary word=%u va=0x%llx not mapped to a resource\n",
                        boundary_word_index[slot], (unsigned long long)va);
                goto done;
            }
            offset = va - record->va;
            bytes = record->size - offset;
            if (bytes > BOUNDARY_WINDOW) { bytes = BOUNDARY_WINDOW; }
            copies[slot].record = record;
            copies[slot].offset = offset;
            copies[slot].bytes = bytes;
            for (unsigned int t = 0; t < barriers; ++t) {
                if (barrier_done[t] == record) { seen = 1; break; }
            }
            if (!seen && barriers < BOUNDARY_WORDS) {
                D3D12_RESOURCE_BARRIER barrier = {0};
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                barrier.UAV.pResource = record->resource;
                ID3D12GraphicsCommandList_ResourceBarrier(ctx->list, 1, &barrier);
                barrier_done[barriers++] = record;
            }
            ID3D12GraphicsCommandList_CopyBufferRegion(ctx->list, windows[slot], 0,
                                                       record->resource, offset, bytes);
        }
        record_readback(ctx->list, ctx->eye);
        if (FAILED(ID3D12GraphicsCommandList_Close(ctx->list))) {
            fprintf(stderr, "Boundary list close failed\n");
            goto done;
        }
        ID3D12CommandQueue_ExecuteCommandLists(ctx->queue, 1, command_lists);
        *ctx->submitted = 1;
        *ctx->completed = 0;
        if (FAILED(ID3D12CommandQueue_Signal(ctx->queue, ctx->fence, (UINT64)frame + 2)) ||
            FAILED(ID3D12Fence_SetEventOnCompletion(ctx->fence, (UINT64)frame + 2, ctx->event)) ||
            WaitForSingleObject(ctx->event, 30000) != WAIT_OBJECT_0 ||
            ID3D12Fence_GetCompletedValue(ctx->fence) == UINT64_MAX) {
            fprintf(stderr, "Boundary GPU completion failed frame=%u\n", frame);
            goto done;
        }
        *ctx->completed = 1;
        for (unsigned int slot = 0; slot < BOUNDARY_WORDS; ++slot) {
            if (!boundary_dump_window(ctx, frame, slot, &words, copies[slot].record,
                                      copies[slot].offset, copies[slot].bytes,
                                      windows[slot])) {
                goto done;
            }
        }
        if (!capture_dump(ctx->dir, ctx->eye, frame, ctx->manifest)) { goto done; }
        if (!boundary_validate_untouched(ctx->eye, frame)) {
            fprintf(stderr, "REJECT: output texture changed despite the cut\n");
            goto done;
        }
    }
    {
        BoundaryWords trailing;
        int extra = boundary_next_words(ctx->sidecar, &sidecar, &trailing);
        if (extra != 0) {
            fprintf(stderr, "Boundary sidecar lines=%d beyond the captured frames\n", extra);
            goto done;
        }
    }
    success = 1;
done:
    for (unsigned int slot = 0; slot < BOUNDARY_WORDS; ++slot) {
        if (windows[slot]) { ID3D12Resource_Release(windows[slot]); }
    }
    return success;
}

#endif // BOUNDARY_PROBE_H
