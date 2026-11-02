// Windowed desktop demonstration of NVIDIA DLSS Neural Rendering (NGX feature 18) on
// Linux through Proton: a D3D12 application that draws its own scene, evaluates the
// model on that frame, and presents the raw render beside the neural output.
//
// This is a learning/demonstration harness, not a product and not a quality claim. The
// model is NVIDIA's and is not shipped with this repository; it is loaded from the
// working directory beside the executable, as is the caller-gate forwarder.
//
// Usage: nr-desktop.exe [--frames N] [--dump PREFIX]
//   --frames N   exit after N presented frames (default: run until Escape)
//   --dump P     write P<n>.ppm (raw) and P<n>-neural.ppm at frame 90, plus a line of
//                numeric comparison. Pixels are display-referred sRGB in both.
// Keys: F10 cycles side-by-side / neural only / raw only. Escape closes.
#define COBJMACROS
#define WIDL_C_INLINE_WRAPPERS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef void (__cdecl *SetUint)(void*, const char*, unsigned int);

// The uint setter is vtable slot 3 of the driver core's parameter block; the header's
// typed overloads are not available to a standalone caller (see vr/evaluation-probe.h).
static void set_uint(void* params, const char* key, unsigned int value) {
    void** vtable = *(void***)params;
    ((SetUint)vtable[3])(params, key, value);
}

#include "evaluation-probe.h"

#define PANEL PROBE_SIZE
#define WINDOW_WIDTH (2 * PANEL)
#define WINDOW_HEIGHT PANEL
#define DUMP_FRAME 90

typedef int (__cdecl *CoreInit)(unsigned long long, const wchar_t*, ID3D12Device*, int, const void*);
typedef int (__cdecl *GetCaps)(void**);
typedef int (__cdecl *CoreShutdown)(ID3D12Device*, int*);
typedef int (__cdecl *FwdInit)(const wchar_t*, const wchar_t*, ID3D12Device*, void*);
typedef void* (__cdecl *FwdCreate)(ID3D12GraphicsCommandList*, void*, int*);
typedef void (__cdecl *FwdRelease)(void*);

static volatile LONG g_quit;
static volatile LONG g_mode;  // 0 side by side, 1 neural only, 2 raw only

static const wchar_t* mode_title(LONG mode) {
    switch (mode) {
        case 1: return L"DLSS Neural Rendering (DX12 under Proton) \x2014 neural only \x2014 F10 switches, Esc closes";
        case 2: return L"DLSS Neural Rendering (DX12 under Proton) \x2014 raw only \x2014 F10 switches, Esc closes";
        default: return L"DLSS Neural Rendering (DX12 under Proton) \x2014 raw | neural \x2014 F10 switches, Esc closes";
    }
}

static LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
    if (message == WM_CLOSE || message == WM_DESTROY) { InterlockedExchange(&g_quit, 1); return 0; }
    if (message == WM_KEYDOWN) {
        if (wparam == VK_ESCAPE) { InterlockedExchange(&g_quit, 1); return 0; }
        if (wparam == VK_F10) {
            LONG mode = (g_mode + 1) % 3;
            InterlockedExchange(&g_mode, mode);
            SetWindowTextW(window, mode_title(mode));
            return 0;
        }
    }
    return DefWindowProcW(window, message, wparam, lparam);
}

static HWND create_window(HINSTANCE instance) {
    WNDCLASSEXW cls = {0};
    cls.cbSize = sizeof(cls);
    cls.style = CS_HREDRAW | CS_VREDRAW;
    cls.lpfnWndProc = window_proc;
    cls.hInstance = instance;
    cls.hCursor = LoadCursorW(NULL, (LPCWSTR)IDC_ARROW);
    cls.hbrBackground = NULL;  // the swapchain covers the client area before the first present
    cls.lpszClassName = L"NRDesktopWindow";
    if (!RegisterClassExW(&cls)) { return NULL; }
    RECT rect = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
    HWND window = CreateWindowExW(0, cls.lpszClassName, mode_title(0),
        WS_OVERLAPPEDWINDOW & ~WS_MAXIMIZEBOX & ~WS_THICKFRAME,
        CW_USEDEFAULT, CW_USEDEFAULT, rect.right - rect.left, rect.bottom - rect.top,
        NULL, NULL, instance, NULL);
    return window;
}

static int compile_shader(const wchar_t* path, const char* entry, const char* target, ID3DBlob** blob) {
    ID3DBlob* errors = NULL;
    HRESULT hr = D3DCompileFromFile(path, NULL, NULL, entry, target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, blob, &errors);
    if (FAILED(hr)) {
        if (errors) { fprintf(stderr, "Shader %ls %s: %.*s\n", path, entry, (int)ID3D10Blob_GetBufferSize(errors), (char*)ID3D10Blob_GetBufferPointer(errors)); }
        else { fprintf(stderr, "Shader %ls %s failed 0x%08lx\n", path, entry, (unsigned long)hr); }
        if (errors) { ID3D10Blob_Release(errors); }
        return 0;
    }
    if (errors) { ID3D10Blob_Release(errors); }
    return 1;
}

// Scene root signature: one descriptor table of five UAVs plus the 32 scene constants.
static int create_scene_pipeline(ID3D12Device* device, ID3D12RootSignature** root, ID3D12PipelineState** pipeline) {
    ID3DBlob* shader = NULL;
    ID3DBlob* signature = NULL;
    ID3DBlob* errors = NULL;
    if (!compile_shader(L"../gpu-scene.hlsl", "main", "cs_5_1", &shader)) { return 0; }
    D3D12_DESCRIPTOR_RANGE range = {0};
    range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    range.NumDescriptors = 5;
    D3D12_ROOT_PARAMETER parameters[2] = {0};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[0].DescriptorTable.NumDescriptorRanges = 1;
    parameters[0].DescriptorTable.pDescriptorRanges = &range;
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[1].Constants.Num32BitValues = 32;
    D3D12_ROOT_SIGNATURE_DESC desc = {0};
    desc.NumParameters = 2;
    desc.pParameters = parameters;
    HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &errors);
    if (FAILED(hr)) {
        if (errors) { fprintf(stderr, "Scene root signature: %.*s\n", (int)ID3D10Blob_GetBufferSize(errors), (char*)ID3D10Blob_GetBufferPointer(errors)); }
        goto failed;
    }
    hr = ID3D12Device_CreateRootSignature(device, 0, ID3D10Blob_GetBufferPointer(signature), ID3D10Blob_GetBufferSize(signature), &IID_ID3D12RootSignature, (void**)root);
    if (FAILED(hr)) { goto failed; }
    D3D12_COMPUTE_PIPELINE_STATE_DESC state = {0};
    state.pRootSignature = *root;
    state.CS.pShaderBytecode = ID3D10Blob_GetBufferPointer(shader);
    state.CS.BytecodeLength = ID3D10Blob_GetBufferSize(shader);
    hr = ID3D12Device_CreateComputePipelineState(device, &state, &IID_ID3D12PipelineState, (void**)pipeline);
    if (FAILED(hr)) { goto failed; }
    ID3D10Blob_Release(shader);
    ID3D10Blob_Release(signature);
    return 1;
failed:
    fprintf(stderr, "Scene pipeline failed 0x%08lx\n", (unsigned long)hr);
    if (signature) { ID3D10Blob_Release(signature); }
    if (shader) { ID3D10Blob_Release(shader); }
    return 0;
}

// Present root signature: two source textures plus the mode/width constants.
static int create_present_pipeline(ID3D12Device* device, ID3D12RootSignature** root, ID3D12PipelineState** pipeline) {
    ID3DBlob* vertex = NULL;
    ID3DBlob* pixel = NULL;
    ID3DBlob* signature = NULL;
    ID3DBlob* errors = NULL;
    if (!compile_shader(L"../desktop-present.hlsl", "VSMain", "vs_5_1", &vertex) ||
        !compile_shader(L"../desktop-present.hlsl", "PSMain", "ps_5_1", &pixel)) { return 0; }
    D3D12_DESCRIPTOR_RANGE range = {0};
    range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    range.NumDescriptors = 2;
    D3D12_ROOT_PARAMETER parameters[2] = {0};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[0].DescriptorTable.NumDescriptorRanges = 1;
    parameters[0].DescriptorTable.pDescriptorRanges = &range;
    parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[1].Constants.Num32BitValues = 4;
    parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_SIGNATURE_DESC desc = {0};
    desc.NumParameters = 2;
    desc.pParameters = parameters;
    HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &errors);
    if (FAILED(hr)) {
        if (errors) { fprintf(stderr, "Present root signature: %.*s\n", (int)ID3D10Blob_GetBufferSize(errors), (char*)ID3D10Blob_GetBufferPointer(errors)); }
        goto failed;
    }
    hr = ID3D12Device_CreateRootSignature(device, 0, ID3D10Blob_GetBufferPointer(signature), ID3D10Blob_GetBufferSize(signature), &IID_ID3D12RootSignature, (void**)root);
    if (FAILED(hr)) { goto failed; }
    D3D12_GRAPHICS_PIPELINE_STATE_DESC state = {0};
    state.pRootSignature = *root;
    state.VS.pShaderBytecode = ID3D10Blob_GetBufferPointer(vertex);
    state.VS.BytecodeLength = ID3D10Blob_GetBufferSize(vertex);
    state.PS.pShaderBytecode = ID3D10Blob_GetBufferPointer(pixel);
    state.PS.BytecodeLength = ID3D10Blob_GetBufferSize(pixel);
    state.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    state.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    state.RasterizerState.DepthClipEnable = TRUE;
    state.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    state.SampleMask = UINT_MAX;
    state.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    state.NumRenderTargets = 1;
    state.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    state.SampleDesc.Count = 1;
    hr = ID3D12Device_CreateGraphicsPipelineState(device, &state, &IID_ID3D12PipelineState, (void**)pipeline);
    if (FAILED(hr)) { goto failed; }
    ID3D10Blob_Release(vertex);
    ID3D10Blob_Release(pixel);
    ID3D10Blob_Release(signature);
    return 1;
failed:
    fprintf(stderr, "Present pipeline failed 0x%08lx\n", (unsigned long)hr);
    if (vertex) { ID3D10Blob_Release(vertex); }
    if (pixel) { ID3D10Blob_Release(pixel); }
    if (signature) { ID3D10Blob_Release(signature); }
    return 0;
}

static int create_swapchain(HWND window, ID3D12CommandQueue* queue, IDXGISwapChain3** swapchain, ID3D12Resource* buffers[2]) {
    IDXGIFactory4* factory = NULL;
    HRESULT hr = CreateDXGIFactory2(0, &IID_IDXGIFactory4, (void**)&factory);
    if (FAILED(hr)) { fprintf(stderr, "DXGI factory failed 0x%08lx\n", (unsigned long)hr); return 0; }
    DXGI_SWAP_CHAIN_DESC1 desc = {0};
    desc.Width = WINDOW_WIDTH;
    desc.Height = WINDOW_HEIGHT;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = 2;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    desc.Scaling = DXGI_SCALING_STRETCH;
    IDXGISwapChain1* base = NULL;
    hr = IDXGIFactory4_CreateSwapChainForHwnd(factory, (IUnknown*)queue, window, &desc, NULL, NULL, &base);
    IDXGIFactory4_Release(factory);
    if (FAILED(hr)) { fprintf(stderr, "Swapchain failed 0x%08lx\n", (unsigned long)hr); return 0; }
    hr = IDXGISwapChain1_QueryInterface(base, &IID_IDXGISwapChain3, (void**)swapchain);
    if (FAILED(hr)) { IDXGISwapChain1_Release(base); return 0; }
    for (unsigned int i = 0; i < 2; ++i) {
        hr = IDXGISwapChain3_GetBuffer(*swapchain, i, &IID_ID3D12Resource, (void**)&buffers[i]);
        if (FAILED(hr)) { return 0; }
    }
    return 1;
}

static int write_ppm_rgba8(const char* path, const unsigned char* mapped, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout,
                           unsigned int width, unsigned int height) {
    FILE* file = fopen(path, "wb");
    if (!file) { fprintf(stderr, "Cannot write %s\n", path); return 0; }
    fprintf(file, "P6\n%u %u\n255\n", width, height);
    for (unsigned int y = 0; y < height; ++y) {
        const unsigned char* row = mapped + layout->Offset + (size_t)y * layout->Footprint.RowPitch;
        for (unsigned int x = 0; x < width; ++x) { fwrite(row + (size_t)x * 4, 1, 3, file); }
    }
    fclose(file);
    return 1;
}

static int write_ppm(const char* path, const unsigned char* mapped, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layout,
                     unsigned int width, unsigned int height, double* mean_out) {
    FILE* file = fopen(path, "wb");
    if (!file) { fprintf(stderr, "Cannot write %s\n", path); return 0; }
    fprintf(file, "P6\n%u %u\n255\n", width, height);
    double sum = 0;
    unsigned long long count = 0;
    for (unsigned int y = 0; y < height; ++y) {
        const uint16_t* row = (const uint16_t*)(mapped + layout->Offset + (size_t)y * layout->Footprint.RowPitch);
        for (unsigned int x = 0; x < width; ++x) {
            unsigned char rgb[3];
            for (unsigned int c = 0; c < 3; ++c) {
                float value = half_float(row[x * 4 + c]);
                if (!isfinite(value)) { value = 0; }
                if (value < 0) { value = 0; }
                if (value > 1) { value = 1; }
                rgb[c] = (unsigned char)(value * 255.0f + 0.5f);
                sum += rgb[c];
                ++count;
            }
            fwrite(rgb, 1, 3, file);
        }
    }
    fclose(file);
    if (mean_out) { *mean_out = count ? sum / (double)count / 255.0 : 0; }
    return 1;
}

int main(int argc, char** argv) {
    unsigned int frame_limit = 0;
    const char* dump_prefix = NULL;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) { frame_limit = (unsigned int)strtoul(argv[++i], NULL, 10); }
        else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) { dump_prefix = argv[++i]; }
        else { fprintf(stderr, "Usage: nr-desktop.exe [--frames N] [--dump PREFIX]\n"); return 2; }
    }
    HINSTANCE instance = GetModuleHandleW(NULL);
    HWND window = create_window(instance);
    if (!window) { fprintf(stderr, "Window creation failed %lu\n", GetLastError()); return 1; }
    ShowWindow(window, SW_SHOW);
    UpdateWindow(window);
    // Keep the demonstration above other windows: it is meant to be looked at while it runs.
    SetWindowPos(window, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW);

    int exit_code = 1;
    ID3D12Device* device = NULL;
    ID3D12CommandQueue* queue = NULL;
    ID3D12CommandAllocator* allocator = NULL;
    ID3D12GraphicsCommandList* list = NULL;
    ID3D12Fence* fence = NULL;
    HANDLE event = NULL;
    IDXGISwapChain3* swapchain = NULL;
    ID3D12Resource* backbuffers[2] = {NULL, NULL};
    ID3D12DescriptorHeap* rtv_heaps[2] = {NULL, NULL};
    ID3D12DescriptorHeap* heap = NULL;
    ID3D12RootSignature* scene_root = NULL;
    ID3D12PipelineState* scene_pipeline = NULL;
    ID3D12RootSignature* present_root = NULL;
    ID3D12PipelineState* present_pipeline = NULL;
    ID3D12Resource* dump_raw = NULL;
    ID3D12Resource* dump_neural = NULL;
    ID3D12Resource* dump_window = NULL;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT window_layout = {0};
    UINT64 window_size = 0;
    void* feature = NULL;
    void* caps = NULL;
    EyeInputs eye = {0};
    int core_ready = 0;
    UINT64 fence_value = 0;
    unsigned int frames = 0;

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
    set_uint(caps, "DLSSNR.Enabled", 1);
    set_uint(caps, "DLSSNR.Width", PANEL);
    set_uint(caps, "DLSSNR.Height", PANEL);
    set_uint(caps, "CreationNodeMask", 1);
    set_uint(caps, "VisibilityNodeMask", 1);
    set_uint(caps, "DLSSNR.Hint.Render.Preset", 0);
    set_uint(caps, "DLSSNR.UICorrection", 1);
    feature = fwd_create(list, caps, &result);
    printf("Feature => 0x%x handle=%p\n", result, feature); fflush(stdout);
    if (result != 1 || !feature) { goto cleanup; }
    CHECK_HR(ID3D12GraphicsCommandList_Close(list));
    {
        ID3D12CommandList* lists[] = {(ID3D12CommandList*)list};
        ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
        CHECK_HR(ID3D12CommandQueue_Signal(queue, fence, ++fence_value));
        CHECK_HR(ID3D12Fence_SetEventOnCompletion(fence, fence_value, event));
        if (WaitForSingleObject(event, 60000) != WAIT_OBJECT_0) { fprintf(stderr, "Feature creation did not complete\n"); goto cleanup; }
    }

    if (!prepare_eye(device, &eye, 0, allocate)) { goto cleanup; }
    if (!create_scene_pipeline(device, &scene_root, &scene_pipeline)) { goto cleanup; }
    if (!create_present_pipeline(device, &present_root, &present_pipeline)) { goto cleanup; }
    if (!create_swapchain(window, queue, &swapchain, backbuffers)) { goto cleanup; }

    // One shader-visible heap: UAVs 0..4 for the scene, SRVs 5..6 for presentation.
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {0};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = 7;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    CHECK_HR(ID3D12Device_CreateDescriptorHeap(device, &heap_desc, &IID_ID3D12DescriptorHeap, (void**)&heap));
    UINT stride = ID3D12Device_GetDescriptorHandleIncrementSize(device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(heap);
    for (unsigned int i = 0; i < 5; ++i) {
        ID3D12Resource* target = eye.textures[i < 4 ? i : 3];
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {0};
        uav.Format = i == 1 ? DXGI_FORMAT_R32_FLOAT : i == 2 ? DXGI_FORMAT_R16G16_FLOAT : DXGI_FORMAT_R16G16B16A16_FLOAT;
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        D3D12_CPU_DESCRIPTOR_HANDLE handle = cpu;
        handle.ptr += (SIZE_T)i * stride;
        ID3D12Device_CreateUnorderedAccessView(device, target, NULL, &uav, handle);
    }
    for (unsigned int i = 0; i < 2; ++i) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv = {0};
        srv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        D3D12_CPU_DESCRIPTOR_HANDLE handle = cpu;
        handle.ptr += (SIZE_T)(5 + i) * stride;
        ID3D12Device_CreateShaderResourceView(device, i == 0 ? eye.textures[0] : eye.textures[3], &srv, handle);
    }
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = ID3D12DescriptorHeap_GetGPUDescriptorHandleForHeapStart(heap);

    if (dump_prefix) {
        if (!create_buffer(device, D3D12_HEAP_TYPE_READBACK, eye.sizes[0], &dump_raw)) { goto cleanup; }
        if (!create_buffer(device, D3D12_HEAP_TYPE_READBACK, eye.sizes[3], &dump_neural)) { goto cleanup; }
        D3D12_RESOURCE_DESC window_desc = ID3D12Resource_GetDesc(backbuffers[0]);
        UINT window_rows = 0;
        ID3D12Device_GetCopyableFootprints(device, &window_desc, 0, 1, 0, &window_layout, &window_rows, &window_size, NULL);
        if (!create_buffer(device, D3D12_HEAP_TYPE_READBACK, window_size, &dump_window)) { goto cleanup; }
    }

    printf("READY panel=%u window=%ux%u (F10 switches, Esc closes)\n", PANEL, WINDOW_WIDTH, WINDOW_HEIGHT); fflush(stdout);

    unsigned int dumped = 0;
    while (!InterlockedCompareExchange(&g_quit, 0, 0)) {
        MSG message;
        while (PeekMessageW(&message, NULL, 0, 0, PM_REMOVE)) {
            if (message.message == WM_QUIT) { InterlockedExchange(&g_quit, 1); }
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        if (InterlockedCompareExchange(&g_quit, 0, 0)) { break; }

        CHECK_HR(ID3D12CommandAllocator_Reset(allocator));
        CHECK_HR(ID3D12GraphicsCommandList_Reset(list, allocator, NULL));
        for (unsigned int i = 0; i < 3; ++i) {
            transition(list, eye.textures[i], eye.states[i], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            eye.states[i] = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        transition(list, eye.textures[3], eye.states[3], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        eye.states[3] = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

        ID3D12GraphicsCommandList_SetComputeRootSignature(list, scene_root);
        ID3D12GraphicsCommandList_SetPipelineState(list, scene_pipeline);
        ID3D12GraphicsCommandList_SetDescriptorHeaps(list, 1, &heap);
        ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(list, 0, gpu);
        unsigned int constants[32] = {0};
        constants[0] = 0;            // eye
        constants[1] = frames;       // frame: the camera pans with it
        constants[2] = PANEL;
        constants[3] = PANEL;
        constants[4] = 0;            // tracked pose: use the shader's own camera path
        constants[5] = 0;            // neural input ignored by the scene entry
        constants[6] = 1;            // Pad0: procedural room
        ID3D12GraphicsCommandList_SetComputeRoot32BitConstants(list, 1, 32, constants, 0);
        ID3D12GraphicsCommandList_Dispatch(list, (PANEL + 7) / 8, (PANEL + 7) / 8, 1);
        for (unsigned int i = 0; i < 3; ++i) {
            transition(list, eye.textures[i], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            eye.states[i] = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
        set_uint(eye.params, "DLSSNR.Reset", frames == 0 ? 1u : 0u);
        result = fwd_evaluate(list, feature, eye.params);
        if (result != 1) { fprintf(stderr, "Evaluate frame %u => 0x%x\n", frames, result); goto cleanup; }

        int dump_now = dump_prefix && frames == DUMP_FRAME && !dumped;
        if (dump_now) {
            transition(list, eye.textures[0], eye.states[0], D3D12_RESOURCE_STATE_COPY_SOURCE);
            eye.states[0] = D3D12_RESOURCE_STATE_COPY_SOURCE;
            transition(list, eye.textures[3], eye.states[3], D3D12_RESOURCE_STATE_COPY_SOURCE);
            eye.states[3] = D3D12_RESOURCE_STATE_COPY_SOURCE;
            D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
            dst.pResource = dump_raw; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = eye.layouts[0];
            src.pResource = eye.textures[0]; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            ID3D12GraphicsCommandList_CopyTextureRegion(list, &dst, 0, 0, 0, &src, NULL);
            dst.pResource = dump_neural; dst.PlacedFootprint = eye.layouts[3];
            src.pResource = eye.textures[3];
            ID3D12GraphicsCommandList_CopyTextureRegion(list, &dst, 0, 0, 0, &src, NULL);
        }
        transition(list, eye.textures[0], eye.states[0], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        eye.states[0] = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        transition(list, eye.textures[3], eye.states[3], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        eye.states[3] = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

        UINT backbuffer_index = IDXGISwapChain3_GetCurrentBackBufferIndex(swapchain);
        ID3D12Resource* backbuffer = backbuffers[backbuffer_index];
        transition(list, backbuffer, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
        // One RTV heap per swapchain buffer, created on first use so both buffers are
        // described without a second heap-binding rule in the present pass.
        if (!rtv_heaps[backbuffer_index]) {
            D3D12_DESCRIPTOR_HEAP_DESC rtv_desc = {0};
            rtv_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
            rtv_desc.NumDescriptors = 1;
            rtv_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
            CHECK_HR(ID3D12Device_CreateDescriptorHeap(device, &rtv_desc, &IID_ID3D12DescriptorHeap, (void**)&rtv_heaps[backbuffer_index]));
            D3D12_CPU_DESCRIPTOR_HANDLE slot = ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(rtv_heaps[backbuffer_index]);
            ID3D12Device_CreateRenderTargetView(device, backbuffer, NULL, slot);
        }
        D3D12_CPU_DESCRIPTOR_HANDLE rtv = ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(rtv_heaps[backbuffer_index]);
        ID3D12GraphicsCommandList_OMSetRenderTargets(list, 1, &rtv, FALSE, NULL);
        D3D12_VIEWPORT viewport = {0, 0, (float)WINDOW_WIDTH, (float)WINDOW_HEIGHT, 0, 1};
        D3D12_RECT scissor = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
        ID3D12GraphicsCommandList_RSSetViewports(list, 1, &viewport);
        ID3D12GraphicsCommandList_RSSetScissorRects(list, 1, &scissor);
        ID3D12GraphicsCommandList_SetGraphicsRootSignature(list, present_root);
        ID3D12GraphicsCommandList_SetPipelineState(list, present_pipeline);
        ID3D12GraphicsCommandList_SetDescriptorHeaps(list, 1, &heap);
        D3D12_GPU_DESCRIPTOR_HANDLE sources = gpu;
        sources.ptr += (UINT64)5 * stride;
        ID3D12GraphicsCommandList_SetGraphicsRootDescriptorTable(list, 0, sources);
        unsigned int present_constants[4] = {(unsigned int)g_mode, PANEL, PANEL, 0};
        ID3D12GraphicsCommandList_SetGraphicsRoot32BitConstants(list, 1, 4, present_constants, 0);
        ID3D12GraphicsCommandList_IASetPrimitiveTopology(list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ID3D12GraphicsCommandList_DrawInstanced(list, 3, 1, 0, 0);
        if (dump_now) {
            // Record exactly what is about to be presented, so the composition can be
            // checked without a host screenshot.
            transition(list, backbuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION dst = {0}, src = {0};
            dst.pResource = dump_window; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = window_layout;
            src.pResource = backbuffer; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            ID3D12GraphicsCommandList_CopyTextureRegion(list, &dst, 0, 0, 0, &src, NULL);
            transition(list, backbuffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
        } else {
            transition(list, backbuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
        }

        CHECK_HR(ID3D12GraphicsCommandList_Close(list));
        ID3D12CommandList* lists[] = {(ID3D12CommandList*)list};
        ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
        CHECK_HR(ID3D12CommandQueue_Signal(queue, fence, ++fence_value));
        CHECK_HR(ID3D12Fence_SetEventOnCompletion(fence, fence_value, event));
        if (WaitForSingleObject(event, 30000) != WAIT_OBJECT_0) { fprintf(stderr, "Frame %u did not complete\n", frames); goto cleanup; }
        CHECK_HR(IDXGISwapChain3_Present(swapchain, 1, 0));

        if (dump_now) {
            double mean_raw = 0, mean_neural = 0;
            unsigned char* mapped = NULL;
            char path[512];
            D3D12_RANGE range = {0, (SIZE_T)eye.sizes[0]};
            if (SUCCEEDED(ID3D12Resource_Map(dump_raw, 0, &range, (void**)&mapped))) {
                snprintf(path, sizeof(path), "%s-raw.ppm", dump_prefix);
                if (!write_ppm(path, mapped, &eye.layouts[0], PANEL, PANEL, &mean_raw)) { goto cleanup; }
                ID3D12Resource_Unmap(dump_raw, 0, NULL);
            }
            range.End = (SIZE_T)eye.sizes[3];
            if (SUCCEEDED(ID3D12Resource_Map(dump_neural, 0, &range, (void**)&mapped))) {
                snprintf(path, sizeof(path), "%s-neural.ppm", dump_prefix);
                if (!write_ppm(path, mapped, &eye.layouts[3], PANEL, PANEL, &mean_neural)) { goto cleanup; }
                ID3D12Resource_Unmap(dump_neural, 0, NULL);
            }
            snprintf(path, sizeof(path), "%s-window.ppm", dump_prefix);
            D3D12_RANGE window_range = {0, (SIZE_T)window_size};
            if (SUCCEEDED(ID3D12Resource_Map(dump_window, 0, &window_range, (void**)&mapped))) {
                if (!write_ppm_rgba8(path, mapped, &window_layout, WINDOW_WIDTH, WINDOW_HEIGHT)) { goto cleanup; }
                ID3D12Resource_Unmap(dump_window, 0, NULL);
            }
            printf("DUMP frame=%u raw=%s-raw.ppm neural=%s-neural.ppm window=%s-window.ppm mean_raw=%.4f mean_neural=%.4f\n", frames, dump_prefix, dump_prefix, dump_prefix, mean_raw, mean_neural);
            fflush(stdout);
            dumped = 1;
        }
        ++frames;
        if (frames % 120 == 0) { printf("frame=%u\n", frames); fflush(stdout); }
        if (frame_limit && frames >= frame_limit) { break; }
    }
    printf("PRESENTED frames=%u\n", frames);
    exit_code = 0;

cleanup:
    if (exit_code != 0) { fprintf(stderr, "FAILED after %u frames\n", frames); }
    if (feature && fwd_release) { fwd_release(feature); }
    if (core_ready && shutdown) {
        int live_devices = -1;
        int shutdown_result = shutdown(device, &live_devices);
        printf("CoreShutdown => 0x%x live_devices=%d\n", shutdown_result, live_devices);
        if (shutdown_result != 1 || live_devices != 0) { exit_code = 1; }
    }
    for (unsigned int i = 0; i < 2; ++i) {
        if (rtv_heaps[i]) { ID3D12DescriptorHeap_Release(rtv_heaps[i]); }
        if (backbuffers[i]) { ID3D12Resource_Release(backbuffers[i]); }
    }
    if (dump_raw) { ID3D12Resource_Release(dump_raw); }
    if (dump_neural) { ID3D12Resource_Release(dump_neural); }
    if (dump_window) { ID3D12Resource_Release(dump_window); }
    if (swapchain) { IDXGISwapChain3_Release(swapchain); }
    if (present_pipeline) { ID3D12PipelineState_Release(present_pipeline); }
    if (present_root) { ID3D12RootSignature_Release(present_root); }
    if (scene_pipeline) { ID3D12PipelineState_Release(scene_pipeline); }
    if (scene_root) { ID3D12RootSignature_Release(scene_root); }
    if (heap) { ID3D12DescriptorHeap_Release(heap); }
    if (eye.params && destroy) { destroy(eye.params); }
    for (unsigned int i = 0; i < PROBE_TEXTURES; ++i) {
        if (eye.uploads[i]) { ID3D12Resource_Release(eye.uploads[i]); }
        if (eye.textures[i]) { ID3D12Resource_Release(eye.textures[i]); }
    }
    if (eye.readback) { ID3D12Resource_Release(eye.readback); }
    if (event) { CloseHandle(event); }
    if (fence) { ID3D12Fence_Release(fence); }
    if (list) { ID3D12GraphicsCommandList_Release(list); }
    if (allocator) { ID3D12CommandAllocator_Release(allocator); }
    if (queue) { ID3D12CommandQueue_Release(queue); }
    if (device) { ID3D12Device_Release(device); }
    if (window) { DestroyWindow(window); }
    return exit_code;
}