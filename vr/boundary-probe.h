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
// The decoder's parameter block is 10 words, but an intermediate-ladder cut can
// land on any slot of the recovered 156-launch sequence: the live parameter
// sizes run 16..264 bytes (NVAPI-ARGS.md). The parse bound is that maximum.
#define BOUNDARY_ARG_WORDS_MAX 33u
#define BOUNDARY_HEX_MAX (264 * 2 + 4)

// Intermediate-ladder capture (LADDER.md): at an arbitrary recovered slot the
// model's entire working pool is read back, so consecutive cut points are
// directly comparable and the slots in between resolve to write-sets. The
// pool size is the model's own allocation, matched from the boundary words'
// resources and pinned here (see DECODER-BOUNDARY.md's allocator table).
#define LADDER_POOL_SIZE 62684160ull

static const unsigned int boundary_word_index[BOUNDARY_WORDS] = {0, 1, 2, 4, 6, 7};
static const char* boundary_word_name[BOUNDARY_WORDS] = {
    "main-input", "skip-input", "out0", "counter", "partial-sums", "weights"};

#define BOUNDARY_RESOURCE_MAX 1024

// Creation phase of a recorded resource (history-carrier binding, see
// TEMPORAL-HISTORY.md): 0 = harness setup, 1 = model (NVSDK_NGX_D3D12_Init
// through feature creation, where the model allocates its own tensors), 2 =
// harness/test after that. The phase tag is what lets the analysis tell the
// model's own allocations (e.g. its history textures) from the harness's.
static LONG boundary_phase;

typedef struct ResourceRecord {
    D3D12_GPU_VIRTUAL_ADDRESS va;
    UINT64 size;
    ID3D12Resource* resource;
    UINT32 dimension;     // D3D12_RESOURCE_DIMENSION
    UINT32 format;        // DXGI_FORMAT
    UINT64 width;
    UINT32 height;
    UINT16 depth_or_array;
    UINT16 mip_levels;
    UINT32 initial_state; // D3D12_RESOURCE_STATES at creation
    UINT32 phase;
} ResourceRecord;

static ResourceRecord boundary_resources[BOUNDARY_RESOURCE_MAX];
static LONG boundary_resource_count;

static UINT64 boundary_format_bytes(UINT32 format) {
    switch (format) {
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return 16;
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return 8;
        case DXGI_FORMAT_R32_FLOAT:
        case DXGI_FORMAT_R32_UINT:
        case DXGI_FORMAT_R32_SINT:
        case DXGI_FORMAT_R16G16_FLOAT: return 4;
        case DXGI_FORMAT_R16_FLOAT:
        case DXGI_FORMAT_R16_UINT: return 2;
        default: return 0;
    }
}

static void boundary_record_resource(ID3D12Device* device, ID3D12Resource* resource,
                                     const D3D12_RESOURCE_DESC* desc,
                                     D3D12_RESOURCE_STATES initial_state) {
    ResourceRecord* record;
    LONG slot = InterlockedIncrement(&boundary_resource_count) - 1;
    if (slot < 0 || slot >= BOUNDARY_RESOURCE_MAX) { return; }
    record = &boundary_resources[slot];
    record->va = ID3D12Resource_GetGPUVirtualAddress(resource);
    record->resource = resource;
    record->dimension = (UINT32)desc->Dimension;
    record->format = (UINT32)desc->Format;
    record->width = desc->Width;
    record->height = desc->Height;
    record->depth_or_array = desc->DepthOrArraySize;
    record->mip_levels = desc->MipLevels;
    record->initial_state = (UINT32)initial_state;
    record->phase = (UINT32)boundary_phase;
    if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        record->size = desc->Width;
    } else if (desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
        UINT64 bpp = boundary_format_bytes((UINT32)desc->Format);
        record->size = bpp ? desc->Width * desc->Height * bpp : 0;
    } else {
        record->size = 0;
    }
    (void)device;
}

// Window resolution stays buffer-only, exactly as before: textures are never
// matched by boundary_find even though they are recorded (and dumped) now.
static const ResourceRecord* boundary_find(uint64_t va) {
    LONG count = boundary_resource_count;
    if (count > BOUNDARY_RESOURCE_MAX) { count = BOUNDARY_RESOURCE_MAX; }
    // Newest first: if a released range is reused by a later resource, the
    // live record must win over the stale one.
    for (LONG i = count - 1; i >= 0; --i) {
        const ResourceRecord* record = &boundary_resources[i];
        if (record->dimension == D3D12_RESOURCE_DIMENSION_BUFFER && record->size > 0 &&
            va >= record->va && va < record->va + record->size) {
            return record;
        }
    }
    return NULL;
}

// Full resource map (history-carrier binding): one line per recorded resource
// with the fields the offline analysis needs to resolve launch-parameter
// addresses to named/typed resources.
static int boundary_dump_map(const char* path) {
    FILE* out;
    LONG count = boundary_resource_count;
    if (count > BOUNDARY_RESOURCE_MAX) { count = BOUNDARY_RESOURCE_MAX; }
    out = fopen(path, "w");
    if (!out) {
        fprintf(stderr, "Resource map open failed: %s\n", path);
        return 0;
    }
    fprintf(out, "resource-map v1 records=%ld\n", count);
    for (LONG i = 0; i < count; ++i) {
        const ResourceRecord* record = &boundary_resources[i];
        fprintf(out,
                "res idx=%ld va=0x%llx size=%llu dim=%u fmt=%u w=%llu h=%u "
                "depth=%u mips=%u init=%u phase=%u\n",
                i, (unsigned long long)record->va, (unsigned long long)record->size,
                record->dimension, record->format, (unsigned long long)record->width,
                record->height, record->depth_or_array, record->mip_levels,
                record->initial_state, record->phase);
    }
    fclose(out);
    printf("RESOURCE-MAP records=%ld file=%s\n", count, path);
    fflush(stdout);
    return 1;
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
        boundary_record_resource(This, (ID3D12Resource*)*resource, desc, initial_state);
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
        boundary_record_resource(This, (ID3D12Resource*)*resource, desc, initial_state);
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
    unsigned int count; // decoded argument words (psize / 8)
    uint64_t word[BOUNDARY_ARG_WORDS_MAX];
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
    // Strict: the parameter block must be a whole number of 64-bit argument
    // words within the live range and the hex must match it exactly. Anything
    // else would leave argument words uninitialised and fails closed here.
    // (The decoder's exact 80-byte block is asserted by the decoder-boundary
    // path itself; a ladder cut can land on any slot's parameter size.)
    if (psize == 0 || (psize & 7u) != 0 || psize > BOUNDARY_ARG_WORDS_MAX * 8u ||
        hexlen != (size_t)psize * 2) {
        return 0;
    }
    memcpy(out->hex, pmem, hexlen);
    out->hex[hexlen] = 0;
    out->eval = eval;
    out->psize = psize;
    out->count = psize / 8u;
    // Hex decode into little-endian 64-bit argument words.
    for (size_t i = 0; i < hexlen / 2 && i < BOUNDARY_ARG_WORDS_MAX * 8u; ++i) {
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
    int ladder; // 1 = capture the whole working pool at the cut point
} BoundaryContext;

// The working pool this evaluation writes, identified by the cut point's own
// bindings: the phase-1 (model-allocated) resource of pinned size that the
// boundary launch's words resolve into. Two per-eye pools of identical size
// exist (see TEMPORAL-HISTORY.md's ABBA note), so size alone can never decide;
// the words of one evaluation touch exactly one of them. The phase filter keeps
// the harness's own readback buffer (phase 2, same size) out of contention, and
// a scalar word that merely lands inside some buffer's VA range is not treated
// as an allocation the model made. Returns NULL when the words resolve into no
// such pool or into more than one - the run must fail closed rather than
// capture a plausible but wrong buffer.
static const ResourceRecord* boundary_pool_from_words(const BoundaryWords* words) {
    const ResourceRecord* found = NULL;
    unsigned int count = words->count;
    if (count > BOUNDARY_ARG_WORDS_MAX) { count = BOUNDARY_ARG_WORDS_MAX; }
    for (unsigned int i = 0; i < count; ++i) {
        uint64_t va = words->word[i];
        const ResourceRecord* record;
        if (va == 0) { continue; }
        record = boundary_find(va);
        if (record == NULL || record->size != LADDER_POOL_SIZE ||
            record->phase != 1u) {
            continue;
        }
        if (found != NULL && found != record) { return NULL; } // ambiguous
        found = record;
    }
    return found;
}

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

static int boundary_dump_pool(const BoundaryContext* ctx, unsigned int frame,
                              const ResourceRecord* pool,
                              ID3D12Resource* readback) {
    unsigned char* mapped = NULL;
    UINT64 digest;
    char name[64];
    FILE* out;
    D3D12_RANGE range = {0, (SIZE_T)LADDER_POOL_SIZE};
    D3D12_RANGE no_write = {0, 0};
    if (FAILED(ID3D12Resource_Map(readback, 0, &range, (void**)&mapped))) {
        fprintf(stderr, "Ladder pool map failed frame=%u\n", frame);
        return 0;
    }
    snprintf(name, sizeof(name), "pool-frame%02u.bin", frame);
    {
        char path[MAX_PATH];
        if (snprintf(path, sizeof(path), "%s/%s", ctx->dir, name) >= (int)sizeof(path)) {
            ID3D12Resource_Unmap(readback, 0, &no_write);
            fprintf(stderr, "Ladder pool path too long\n");
            return 0;
        }
        out = fopen(path, "wb");
        if (out == NULL) {
            ID3D12Resource_Unmap(readback, 0, &no_write);
            fprintf(stderr, "Ladder pool write failed %s\n", path);
            return 0;
        }
        if (fwrite(mapped, 1, (size_t)LADDER_POOL_SIZE, out) != (size_t)LADDER_POOL_SIZE) {
            fclose(out);
            ID3D12Resource_Unmap(readback, 0, &no_write);
            fprintf(stderr, "Ladder pool short write %s\n", path);
            return 0;
        }
        fclose(out);
    }
    digest = capture_fnv(mapped, (size_t)LADDER_POOL_SIZE);
    ID3D12Resource_Unmap(readback, 0, &no_write);
    fprintf(ctx->manifest,
            "pool-frame=%u bytes=%llu file=%s fnv=0x%016llx va=0x%llx res_va=0x%llx "
            "res_size=%llu\n",
            frame, (unsigned long long)LADDER_POOL_SIZE, name,
            (unsigned long long)digest, (unsigned long long)pool->va,
            (unsigned long long)pool->va, (unsigned long long)pool->size);
    printf("LADDER frame=%u bytes=%llu fnv=0x%016llx file=%s\n", frame,
           (unsigned long long)LADDER_POOL_SIZE, (unsigned long long)digest, name);
    fflush(stdout);
    return 1;
}

static int boundary_run(BoundaryContext* ctx) {
    ID3D12CommandList* command_lists[] = {(ID3D12CommandList*)ctx->list};
    ID3D12Resource* windows[BOUNDARY_WORDS] = {0};
    ID3D12Resource* pool_readback = NULL;
    BoundarySidecar sidecar = {0};
    int success = 0;
    for (unsigned int slot = 0; slot < BOUNDARY_WORDS; ++slot) {
        if (!create_buffer(ctx->device, D3D12_HEAP_TYPE_READBACK, BOUNDARY_WINDOW,
                           &windows[slot])) {
            goto done;
        }
    }
    if (ctx->ladder &&
        !create_buffer(ctx->device, D3D12_HEAP_TYPE_READBACK, LADDER_POOL_SIZE,
                       &pool_readback)) {
        fprintf(stderr, "Ladder readback allocation failed\n");
        goto done;
    }
    fprintf(ctx->manifest, "sidecar=%s\n", ctx->sidecar);
    for (unsigned int frame = 0; frame < BOUNDARY_FRAMES; ++frame) {
        BoundaryWords words;
        struct { const ResourceRecord* record; UINT64 offset; UINT64 bytes; } copies[BOUNDARY_WORDS];
        const ResourceRecord* barrier_done[BOUNDARY_WORDS];
        unsigned int barriers = 0;
        int result;
        int lines;
        const ResourceRecord* ladder_pool = NULL;
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
        if (!ctx->ladder && words.count != BOUNDARY_ARG_WORDS) {
            // The decoder-boundary path is defined on the decoder's exact
            // 10-word parameter block; any other size is a misaligned cut.
            fprintf(stderr, "Boundary words=%u for frame=%u (need %u)\n",
                    words.count, frame, (unsigned int)BOUNDARY_ARG_WORDS);
            goto done;
        }
        fprintf(ctx->manifest,
                "boundary-frame=%u eval=%llu psize=%u pmem=%s\n", frame,
                words.eval, words.psize, words.hex);
        if (!ctx->ladder) {
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
        } else {
            /* Ladder mode: the cut point's whole working pool, so consecutive
             * cut points can be diffed into the intervening slots' write-sets.
             * Read-only copy of a model-owned resource; a UAV barrier only, no
             * state transition (as for the history textures). */
            D3D12_RESOURCE_BARRIER barrier = {0};
            ladder_pool = boundary_pool_from_words(&words);
            if (ladder_pool == NULL) {
                fprintf(stderr,
                        "Ladder cut point words resolve into no single working pool\n");
                goto done;
            }
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.UAV.pResource = ladder_pool->resource;
            ID3D12GraphicsCommandList_ResourceBarrier(ctx->list, 1, &barrier);
            ID3D12GraphicsCommandList_CopyBufferRegion(
                ctx->list, pool_readback, 0, ladder_pool->resource, 0,
                LADDER_POOL_SIZE);
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
        if (ctx->ladder) {
            if (!boundary_dump_pool(ctx, frame, ladder_pool, pool_readback)) { goto done; }
        } else {
        for (unsigned int slot = 0; slot < BOUNDARY_WORDS; ++slot) {
            if (!boundary_dump_window(ctx, frame, slot, &words, copies[slot].record,
                                      copies[slot].offset, copies[slot].bytes,
                                      windows[slot])) {
                goto done;
            }
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
    if (pool_readback) { ID3D12Resource_Release(pool_readback); }
    return success;
}

// History-carrier capture (see TEMPORAL-HISTORY.md): read the model's own
// phase-1 textures (its `dlssnr_prev_output` history allocations) back after
// creation ("init", the reset-value candidate) and after every evaluation, so
// the offline analysis can compare history content against captured outputs.
// Read-only copies; model-owned resources are never written, and only a UAV
// barrier (no state transition) is issued on them.
#define HISTORY_MAX 4u
#define HISTORY_BYTES_MAX (16ull * 1024ull * 1024ull)

typedef struct HistoryProbe {
    ID3D12Resource* readback[HISTORY_MAX];
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout[HISTORY_MAX];
    UINT64 size[HISTORY_MAX];        // readback buffer size (footprint, padded)
    UINT64 va[HISTORY_MAX];
    unsigned int record_index[HISTORY_MAX];
    unsigned int count;
} HistoryProbe;

static int history_setup(ID3D12Device* device, HistoryProbe* hp) {
    LONG total = boundary_resource_count;
    if (total > BOUNDARY_RESOURCE_MAX) { total = BOUNDARY_RESOURCE_MAX; }
    for (LONG i = 0; i < total && hp->count < HISTORY_MAX; ++i) {
        const ResourceRecord* record = &boundary_resources[i];
        D3D12_RESOURCE_DESC desc = {0};
        UINT64 bpp;
        if (record->phase != 1) { continue; }
        if (record->dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) { continue; }
        if (record->size == 0 || record->size > HISTORY_BYTES_MAX) { continue; }
        bpp = boundary_format_bytes(record->format);
        if (!bpp) { continue; }
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = record->width;
        desc.Height = record->height;
        desc.DepthOrArraySize = record->depth_or_array;
        desc.MipLevels = record->mip_levels;
        desc.Format = (DXGI_FORMAT)record->format;
        desc.SampleDesc.Count = 1;
        ID3D12Device_GetCopyableFootprints(device, &desc, 0, 1, 0,
                                           &hp->layout[hp->count], NULL, NULL,
                                           &hp->size[hp->count]);
        if (!create_buffer(device, D3D12_HEAP_TYPE_READBACK, hp->size[hp->count],
                           &hp->readback[hp->count])) {
            return 0;
        }
        hp->record_index[hp->count] = (unsigned int)i;
        hp->va[hp->count] = record->va;
        ++hp->count;
    }
    printf("HISTORY-CANDIDATES count=%u\n", hp->count);
    fflush(stdout);
    return 1;
}

// Record one read-only copy of each history texture into its readback buffer.
// A UAV barrier orders the copy behind the model's writes; no state transition
// is issued on model-owned resources (their tracked state is unknown).
static void history_record(ID3D12GraphicsCommandList* list, HistoryProbe* hp) {
    for (unsigned int i = 0; i < hp->count; ++i) {
        const ResourceRecord* record = &boundary_resources[hp->record_index[i]];
        D3D12_RESOURCE_BARRIER barrier = {0};
        D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = record->resource;
        ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &barrier);
        dst.pResource = hp->readback[i];
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = hp->layout[i];
        src.pResource = record->resource;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = 0;
        ID3D12GraphicsCommandList_CopyTextureRegion(list, &dst, 0, 0, 0, &src, NULL);
    }
}

static int history_dump(const char* dir, HistoryProbe* hp, FILE* manifest,
                        const char* tag) {
    for (unsigned int i = 0; i < hp->count; ++i) {
        const D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout = &hp->layout[i];
        UINT64 row_bytes = (UINT64)layout->Footprint.Width *
                           boundary_format_bytes((UINT32)layout->Footprint.Format);
        size_t total = (size_t)(row_bytes * layout->Footprint.Height);
        unsigned char* mapped = NULL;
        unsigned char* tight = NULL;
        D3D12_RANGE range = {0, (SIZE_T)hp->size[i]};
        char name[64];
        char path[512];
        FILE* out;
        uint64_t digest;
        if (snprintf(name, sizeof(name), "history%u-%s.bin", i, tag) >= (int)sizeof(name)) {
            fprintf(stderr, "History name too long\n");
            return 0;
        }
        if (snprintf(path, sizeof(path), "%s/%s", dir, name) >= (int)sizeof(path)) {
            fprintf(stderr, "History path too long\n");
            return 0;
        }
        if (FAILED(ID3D12Resource_Map(hp->readback[i], 0, &range, (void**)&mapped))) {
            fprintf(stderr, "History map failed %s\n", name);
            return 0;
        }
        tight = (unsigned char*)malloc(total);
        if (!tight) {
            D3D12_RANGE no_write = {0, 0};
            ID3D12Resource_Unmap(hp->readback[i], 0, &no_write);
            return 0;
        }
        for (UINT64 y = 0; y < layout->Footprint.Height; ++y) {
            memcpy(tight + (size_t)y * row_bytes,
                   mapped + layout->Offset + (size_t)y * layout->Footprint.RowPitch,
                   (size_t)row_bytes);
        }
        {
            D3D12_RANGE no_write = {0, 0};
            ID3D12Resource_Unmap(hp->readback[i], 0, &no_write);
        }
        out = fopen(path, "wb");
        if (!out || fwrite(tight, 1, total, out) != total) {
            fprintf(stderr, "History write failed %s\n", name);
            if (out) { fclose(out); }
            free(tight);
            return 0;
        }
        fclose(out);
        digest = capture_fnv(tight, total);
        free(tight);
        fprintf(manifest,
                "history idx=%u rec=%u stage=%s va=0x%llx w=%llu h=%u fmt=%u "
                "bytes=%zu file=%s fnv=0x%016llx\n",
                i, hp->record_index[i], tag, (unsigned long long)hp->va[i],
                (unsigned long long)layout->Footprint.Width, layout->Footprint.Height,
                (unsigned int)layout->Footprint.Format, total, name,
                (unsigned long long)digest);
        printf("HISTORY idx=%u stage=%s bytes=%zu fnv=0x%016llx\n", i, tag, total,
               (unsigned long long)digest);
        fflush(stdout);
    }
    return 1;
}

// Intervention experiment (see TEMPORAL-HISTORY.md): overwrite the evaluated
// feature's history texture with a synthetic pattern before evaluation, to
// test whether its content is read at all. Pattern 0 = uniform FP16 1.0;
// pattern 1 = 16-pixel checkerboard of FP16 1.0/0.0 (in-range values, so no
// sanitisation excuse if the output fails to change).
static int history_inject_upload(ID3D12Resource* upload,
                                 const D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout,
                                 unsigned int pattern) {
    unsigned char* mapped = NULL;
    D3D12_RANGE no_read = {0, 0};
    if (FAILED(ID3D12Resource_Map(upload, 0, &no_read, (void**)&mapped))) { return 0; }
    for (UINT32 y = 0; y < layout->Footprint.Height; ++y) {
        unsigned char* row = mapped + layout->Offset + (UINT64)y * layout->Footprint.RowPitch;
        for (UINT32 x = 0; x < layout->Footprint.Width; ++x) {
            uint16_t* pixel = (uint16_t*)(row + (UINT64)x * 8u); // RGBA16F
            uint16_t value = 0x3c00u; // FP16 1.0
            if (pattern == 1u) {
                value = ((((x >> 4) + (y >> 4)) & 1u) != 0u) ? 0x3c00u : 0x0000u;
            }
            for (unsigned int c = 0; c < 4; ++c) { pixel[c] = value; }
        }
    }
    ID3D12Resource_Unmap(upload, 0, NULL);
    return 1;
}

static void history_inject_record(ID3D12GraphicsCommandList* list, HistoryProbe* hp,
                                  ID3D12Resource* upload,
                                  const D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout) {
    // Only history index 0 (the evaluated feature's history) is overwritten;
    // index 1 stays untouched as the static control.
    const ResourceRecord* record = &boundary_resources[hp->record_index[0]];
    D3D12_RESOURCE_BARRIER barrier = {0};
    D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = record->resource;
    ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &barrier);
    dst.pResource = record->resource;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;
    src.pResource = upload;
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = *layout;
    ID3D12GraphicsCommandList_CopyTextureRegion(list, &dst, 0, 0, 0, &src, NULL);
}

// Sham intervention (content-preserving): stage the history's own read-back
// bytes into the injection upload buffer, so the same upload->texture copy runs
// before the evaluation without changing any content. If the sham run
// reproduces the reference bit-for-bit while the pattern runs diverge, the
// causality is the injected *content*, not the copy machinery.
static int history_inject_sham(HistoryProbe* hp, ID3D12Resource* upload) {
    unsigned char* src = NULL;
    unsigned char* dst = NULL;
    D3D12_RANGE read = {0, (SIZE_T)hp->size[0]};
    D3D12_RANGE no_read = {0, 0};
    if (FAILED(ID3D12Resource_Map(hp->readback[0], 0, &read, (void**)&src))) { return 0; }
    if (FAILED(ID3D12Resource_Map(upload, 0, &no_read, (void**)&dst))) {
        D3D12_RANGE no_write = {0, 0};
        ID3D12Resource_Unmap(hp->readback[0], 0, &no_write);
        return 0;
    }
    memcpy(dst, src, (size_t)hp->size[0]);
    ID3D12Resource_Unmap(upload, 0, NULL);
    {
        D3D12_RANGE no_write = {0, 0};
        ID3D12Resource_Unmap(hp->readback[0], 0, &no_write);
    }
    return 1;
}

#endif // BOUNDARY_PROBE_H
