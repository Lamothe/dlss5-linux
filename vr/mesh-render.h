#ifndef DLSS_MESH_RENDER_H
#define DLSS_MESH_RENDER_H
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
// Include after evaluation-probe.h and gpu-scene.h. Caller must fence before destroy.
// model.bin: magic[8]="DLMESH01", LE uint32 vertexCount,width,height,reserved=0;
// vertexCount flattened triangle vertices (16 LE IEEE floats, 64 bytes):
// position3,normal3,atlasUV2,roughness,metallic,alphaCutoff,pad=0,linearBaseFactor4.
// Then width*height*4 RGBA8 pixels: sRGB RGB, linear alpha. No trailing bytes.
// World coordinates are already transformed/placed by the offline converter.
// Alpha BLEND is approximated as MASK; no mipmaps, normal maps or mesh shadows.
typedef struct MeshRender {
    ID3D12RootSignature* root;
    ID3D12PipelineState* pipeline;
    ID3D12Resource* vertices;
    ID3D12Resource* atlas;
    ID3D12Resource* atlas_upload;
    ID3D12Resource* scene_depth[2];
    ID3D12Resource* depth;
    ID3D12DescriptorHeap* heap;
    ID3D12DescriptorHeap* rtv;
    ID3D12DescriptorHeap* dsv;
    UINT srv_stride,rtv_stride,vertex_count;
    D3D12_VERTEX_BUFFER_VIEW vertex_view;
} MeshRender;
static void destroy_mesh(MeshRender* mesh) {
    for (unsigned int i=0;i<2;++i) { if (mesh->scene_depth[i]) { ID3D12Resource_Release(mesh->scene_depth[i]); } }
    if (mesh->vertices) { ID3D12Resource_Release(mesh->vertices); }
    if (mesh->atlas) { ID3D12Resource_Release(mesh->atlas); }
    if (mesh->atlas_upload) { ID3D12Resource_Release(mesh->atlas_upload); }
    if (mesh->depth) { ID3D12Resource_Release(mesh->depth); }
    if (mesh->heap) { ID3D12DescriptorHeap_Release(mesh->heap); }
    if (mesh->rtv) { ID3D12DescriptorHeap_Release(mesh->rtv); }
    if (mesh->dsv) { ID3D12DescriptorHeap_Release(mesh->dsv); }
    if (mesh->pipeline) { ID3D12PipelineState_Release(mesh->pipeline); }
    if (mesh->root) { ID3D12RootSignature_Release(mesh->root); }
    memset(mesh,0,sizeof(*mesh));
}
static int mesh_read_model(unsigned char** bytes,uint32_t* count,uint32_t* width,uint32_t* height) {
    FILE* file=fopen("model.bin","rb");
    unsigned char* data=NULL;
    int success=0;
    if (!file) { fprintf(stderr,"Mesh: cannot open model.bin\n"); return 0; }
    if (_fseeki64(file,0,SEEK_END)!=0) { goto done; }
    int64_t length=_ftelli64(file);
    if (length<24 || _fseeki64(file,0,SEEK_SET)!=0) { goto done; }
    unsigned char header[24];
    if (fread(header,1,24,file)!=24 || memcmp(header,"DLMESH01",8)!=0) { goto done; }
    uint32_t reserved;
    memcpy(count,header+8,4); memcpy(width,header+12,4); memcpy(height,header+16,4); memcpy(&reserved,header+20,4);
    // Explicit caps bound memory, D3D12 texture dimensions and UINT vertex-view size.
    if (reserved || !*count || *count%3 || *count>16000000u || !*width || !*height || *width>8192 || *height>8192) { goto done; }
    uint64_t vertices=(uint64_t)*count*64, pixels=(uint64_t)*width**height*4;
    uint64_t expected=24+vertices+pixels;
    if (expected!=(uint64_t)length || vertices>UINT_MAX || expected>SIZE_MAX) { goto done; }
    data=(unsigned char*)malloc((size_t)(vertices+pixels));
    if (!data || fread(data,1,(size_t)(vertices+pixels),file)!=(size_t)(vertices+pixels)) { goto done; }
    for (uint32_t v=0;v<*count;++v) {
        float values[16]; memcpy(values,data+(size_t)v*64,64);
        for (unsigned int j=0;j<16;++j) { if (!isfinite(values[j])) { goto done; } }
        for (unsigned int j=0;j<3;++j) { if (fabsf(values[j])>10000 || fabsf(values[j+3])>1.01f) { goto done; } }
        float normal_length=values[3]*values[3]+values[4]*values[4]+values[5]*values[5];
        if (normal_length<0.5f || normal_length>1.5f || values[6]<0 || values[6]>1 || values[7]<0 || values[7]>1 || values[11]!=0) { goto done; }
        for (unsigned int j=8;j<11;++j) { if (values[j]<0 || values[j]>1) { goto done; } }
        for (unsigned int j=12;j<16;++j) { if (values[j]<0 || values[j]>1) { goto done; } }
    }
    *bytes=data; data=NULL; success=1;
done:
    free(data); fclose(file);
    if (!success) { fprintf(stderr,"Mesh: invalid/truncated model.bin or unsupported bounds\n"); }
    return success;
}
static int prepare_mesh(ID3D12Device* device,ID3D12CommandQueue* queue,EyeInputs* eyes,MeshRender* mesh,int* uncertain) {
    *uncertain=0;
    unsigned char* data=NULL;
    uint32_t count=0,width=0,height=0;
    ID3DBlob* vs=NULL; ID3DBlob* ps=NULL; ID3DBlob* blob=NULL; ID3DBlob* errors=NULL;
    ID3D12CommandAllocator* allocator=NULL;
    ID3D12GraphicsCommandList* list=NULL;
    ID3D12Fence* fence=NULL;
    HANDLE event=NULL;
    HRESULT hr=E_FAIL;
    int success=0;
    if (!mesh_read_model(&data,&count,&width,&height)) { goto done; }
    for (unsigned int e=0;e<2;++e) {
        for (unsigned int i=0;i<3;++i) {
            D3D12_RESOURCE_DESC existing=ID3D12Resource_GetDesc(eyes[e].textures[i]);
            if (!(existing.Flags&D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)) {
                fprintf(stderr,"Mesh: eye inputs require ALLOW_RENDER_TARGET flag\n"); goto done;
            }
        }
    }
    hr=D3DCompileFromFile(L"../mesh-render.hlsl",NULL,NULL,"vertexMain","vs_5_1",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&vs,&errors);
    if (FAILED(hr)) { goto done; }
    if (errors) { ID3D10Blob_Release(errors); errors=NULL; }
    hr=D3DCompileFromFile(L"../mesh-render.hlsl",NULL,NULL,"pixelMain","ps_5_1",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&ps,&errors);
    if (FAILED(hr)) { goto done; }
    if (errors) { ID3D10Blob_Release(errors); errors=NULL; }
    D3D12_DESCRIPTOR_RANGE range={0}; range.RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV; range.NumDescriptors=2;
    D3D12_ROOT_PARAMETER params[2]={0};
    params[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges=1; params[0].DescriptorTable.pDescriptorRanges=&range;
    params[0].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
    params[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; params[1].Constants.Num32BitValues=32;
    D3D12_STATIC_SAMPLER_DESC sampler={0}; sampler.Filter=D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU=sampler.AddressV=sampler.AddressW=D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxAnisotropy=1; sampler.ComparisonFunc=D3D12_COMPARISON_FUNC_ALWAYS;
    sampler.ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_SIGNATURE_DESC root={0}; root.NumParameters=2; root.pParameters=params;
    root.NumStaticSamplers=1; root.pStaticSamplers=&sampler; root.Flags=D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    hr=D3D12SerializeRootSignature(&root,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&errors);
    if (FAILED(hr)) { goto done; }
    hr=ID3D12Device_CreateRootSignature(device,0,ID3D10Blob_GetBufferPointer(blob),ID3D10Blob_GetBufferSize(blob),&IID_ID3D12RootSignature,(void**)&mesh->root);
    if (FAILED(hr)) { goto done; }
    const D3D12_INPUT_ELEMENT_DESC elements[]={
        {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,12,D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,24,D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"TEXCOORD",1,DXGI_FORMAT_R32G32B32A32_FLOAT,0,32,D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,48,D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0}
    };
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso={0}; pso.pRootSignature=mesh->root;
    pso.VS.pShaderBytecode=ID3D10Blob_GetBufferPointer(vs); pso.VS.BytecodeLength=ID3D10Blob_GetBufferSize(vs);
    pso.PS.pShaderBytecode=ID3D10Blob_GetBufferPointer(ps); pso.PS.BytecodeLength=ID3D10Blob_GetBufferSize(ps);
    pso.InputLayout.pInputElementDescs=elements; pso.InputLayout.NumElements=5;
    pso.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID; pso.RasterizerState.CullMode=D3D12_CULL_MODE_NONE;
    pso.RasterizerState.FrontCounterClockwise=TRUE; pso.RasterizerState.DepthClipEnable=TRUE;
    pso.BlendState.IndependentBlendEnable=TRUE;
    for (unsigned int i=0;i<8;++i) {
        pso.BlendState.RenderTarget[i].SrcBlend=D3D12_BLEND_ONE; pso.BlendState.RenderTarget[i].DestBlend=D3D12_BLEND_ZERO;
        pso.BlendState.RenderTarget[i].BlendOp=D3D12_BLEND_OP_ADD;
        pso.BlendState.RenderTarget[i].SrcBlendAlpha=D3D12_BLEND_ONE; pso.BlendState.RenderTarget[i].DestBlendAlpha=D3D12_BLEND_ZERO;
        pso.BlendState.RenderTarget[i].BlendOpAlpha=D3D12_BLEND_OP_ADD;
        pso.BlendState.RenderTarget[i].LogicOp=D3D12_LOGIC_OP_NOOP;
        pso.BlendState.RenderTarget[i].RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;
    }
    pso.DepthStencilState.DepthEnable=TRUE; pso.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ALL;
    pso.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_LESS;
    pso.DepthStencilState.StencilReadMask=255; pso.DepthStencilState.StencilWriteMask=255;
    pso.DepthStencilState.FrontFace.StencilFailOp=pso.DepthStencilState.FrontFace.StencilDepthFailOp=pso.DepthStencilState.FrontFace.StencilPassOp=D3D12_STENCIL_OP_KEEP;
    pso.DepthStencilState.FrontFace.StencilFunc=D3D12_COMPARISON_FUNC_ALWAYS; pso.DepthStencilState.BackFace=pso.DepthStencilState.FrontFace;
    pso.SampleMask=UINT_MAX; pso.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso.NumRenderTargets=3; pso.RTVFormats[0]=DXGI_FORMAT_R16G16B16A16_FLOAT; pso.RTVFormats[1]=DXGI_FORMAT_R32_FLOAT; pso.RTVFormats[2]=DXGI_FORMAT_R16G16_FLOAT;
    pso.DSVFormat=DXGI_FORMAT_D32_FLOAT; pso.SampleDesc.Count=1;
    hr=ID3D12Device_CreateGraphicsPipelineState(device,&pso,&IID_ID3D12PipelineState,(void**)&mesh->pipeline);
    if (FAILED(hr)) { goto done; }
    // Static upload-heap vertex buffer avoids a second large temporary allocation.
    if (!create_buffer(device,D3D12_HEAP_TYPE_UPLOAD,(UINT64)count*64,&mesh->vertices)) { goto done; }
    unsigned char* mapped=NULL; D3D12_RANGE no_read={0,0};
    hr=ID3D12Resource_Map(mesh->vertices,0,&no_read,(void**)&mapped); if (FAILED(hr)) { goto done; }
    memcpy(mapped,data,(size_t)count*64); ID3D12Resource_Unmap(mesh->vertices,0,NULL);
    mesh->vertex_count=count; mesh->vertex_view.BufferLocation=ID3D12Resource_GetGPUVirtualAddress(mesh->vertices);
    mesh->vertex_view.SizeInBytes=count*64; mesh->vertex_view.StrideInBytes=64;
    D3D12_HEAP_PROPERTIES properties={0}; properties.Type=D3D12_HEAP_TYPE_DEFAULT; properties.CreationNodeMask=1; properties.VisibleNodeMask=1;
    D3D12_RESOURCE_DESC texture={0}; texture.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width=width; texture.Height=height; texture.DepthOrArraySize=1; texture.MipLevels=1; texture.SampleDesc.Count=1;
    texture.Format=DXGI_FORMAT_R8G8B8A8_TYPELESS;
    hr=ID3D12Device_CreateCommittedResource(device,&properties,D3D12_HEAP_FLAG_NONE,&texture,D3D12_RESOURCE_STATE_COPY_DEST,NULL,&IID_ID3D12Resource,(void**)&mesh->atlas);
    if (FAILED(hr)) { goto done; }
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint; UINT64 upload_size;
    ID3D12Device_GetCopyableFootprints(device,&texture,0,1,0,&footprint,NULL,NULL,&upload_size);
    if (!create_buffer(device,D3D12_HEAP_TYPE_UPLOAD,upload_size,&mesh->atlas_upload)) { goto done; }
    hr=ID3D12Resource_Map(mesh->atlas_upload,0,&no_read,(void**)&mapped); if (FAILED(hr)) { goto done; }
    memset(mapped,0,(size_t)upload_size);
    for (uint32_t y=0;y<height;++y) { memcpy(mapped+footprint.Offset+(size_t)y*footprint.Footprint.RowPitch,data+(size_t)count*64+(size_t)y*width*4,(size_t)width*4); }
    ID3D12Resource_Unmap(mesh->atlas_upload,0,NULL);
    texture.Width=PROBE_SIZE; texture.Height=PROBE_SIZE; texture.Format=DXGI_FORMAT_R32_FLOAT;
    for (unsigned int e=0;e<2;++e) {
        hr=ID3D12Device_CreateCommittedResource(device,&properties,D3D12_HEAP_FLAG_NONE,&texture,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,NULL,&IID_ID3D12Resource,(void**)&mesh->scene_depth[e]);
        if (FAILED(hr)) { goto done; }
    }
    texture.Format=DXGI_FORMAT_D32_FLOAT; texture.Flags=D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE clear={0}; clear.Format=DXGI_FORMAT_D32_FLOAT; clear.DepthStencil.Depth=1;
    hr=ID3D12Device_CreateCommittedResource(device,&properties,D3D12_HEAP_FLAG_NONE,&texture,D3D12_RESOURCE_STATE_DEPTH_WRITE,&clear,&IID_ID3D12Resource,(void**)&mesh->depth);
    if (FAILED(hr)) { goto done; }
    D3D12_DESCRIPTOR_HEAP_DESC heap={0}; heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; heap.NumDescriptors=4; heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    hr=ID3D12Device_CreateDescriptorHeap(device,&heap,&IID_ID3D12DescriptorHeap,(void**)&mesh->heap); if (FAILED(hr)) { goto done; }
    mesh->srv_stride=ID3D12Device_GetDescriptorHandleIncrementSize(device,heap.Type);
    heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_RTV; heap.NumDescriptors=6; heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    hr=ID3D12Device_CreateDescriptorHeap(device,&heap,&IID_ID3D12DescriptorHeap,(void**)&mesh->rtv); if (FAILED(hr)) { goto done; }
    mesh->rtv_stride=ID3D12Device_GetDescriptorHandleIncrementSize(device,heap.Type);
    heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_DSV; heap.NumDescriptors=1;
    hr=ID3D12Device_CreateDescriptorHeap(device,&heap,&IID_ID3D12DescriptorHeap,(void**)&mesh->dsv); if (FAILED(hr)) { goto done; }
    ID3D12Device_CreateDepthStencilView(device,mesh->depth,NULL,ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(mesh->dsv));
    for (unsigned int e=0;e<2;++e) {
        D3D12_CPU_DESCRIPTOR_HANDLE handle=ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(mesh->heap); handle.ptr+=(SIZE_T)e*2*mesh->srv_stride;
        D3D12_SHADER_RESOURCE_VIEW_DESC srv={0}; srv.Format=DXGI_FORMAT_R8G8B8A8_UNORM_SRGB; srv.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; srv.Texture2D.MipLevels=1;
        ID3D12Device_CreateShaderResourceView(device,mesh->atlas,&srv,handle); handle.ptr+=mesh->srv_stride;
        srv.Format=DXGI_FORMAT_R32_FLOAT; ID3D12Device_CreateShaderResourceView(device,mesh->scene_depth[e],&srv,handle);
        handle=ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(mesh->rtv); handle.ptr+=(SIZE_T)e*3*mesh->rtv_stride;
        for (unsigned int i=0;i<3;++i) { ID3D12Device_CreateRenderTargetView(device,eyes[e].textures[i],NULL,handle); handle.ptr+=mesh->rtv_stride; }
    }
    hr=ID3D12Device_CreateCommandAllocator(device,D3D12_COMMAND_LIST_TYPE_DIRECT,&IID_ID3D12CommandAllocator,(void**)&allocator); if (FAILED(hr)) { goto done; }
    hr=ID3D12Device_CreateCommandList(device,0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator,NULL,&IID_ID3D12GraphicsCommandList,(void**)&list); if (FAILED(hr)) { goto done; }
    D3D12_TEXTURE_COPY_LOCATION dst={0},src={0}; dst.pResource=mesh->atlas; dst.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource=mesh->atlas_upload; src.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint=footprint;
    ID3D12GraphicsCommandList_CopyTextureRegion(list,&dst,0,0,0,&src,NULL);
    transition(list,mesh->atlas,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    hr=ID3D12GraphicsCommandList_Close(list); if (FAILED(hr)) { goto done; }
    hr=ID3D12Device_CreateFence(device,0,D3D12_FENCE_FLAG_NONE,&IID_ID3D12Fence,(void**)&fence); if (FAILED(hr)) { goto done; }
    event=CreateEventA(NULL,FALSE,FALSE,NULL); if (!event) { goto done; }
    hr=ID3D12Fence_SetEventOnCompletion(fence,1,event); if (FAILED(hr)) { goto done; }
    ID3D12CommandList* submitted=(ID3D12CommandList*)list;
    *uncertain=1; ID3D12CommandQueue_ExecuteCommandLists(queue,1,&submitted);
    hr=ID3D12CommandQueue_Signal(queue,fence,1); if (FAILED(hr)) { goto done; }
    if (WaitForSingleObject(event,30000)!=WAIT_OBJECT_0 || ID3D12Fence_GetCompletedValue(fence)==UINT64_MAX) { hr=E_FAIL; goto done; }
    ID3D12Resource_Release(mesh->atlas_upload); mesh->atlas_upload=NULL;
    fprintf(stderr,"Mesh: loaded %u triangles, %ux%u atlas; alpha-test, no mipmaps/normal maps/shadows\n",count/3,width,height);
    *uncertain=0; success=1;
done:
    if (!success) { fprintf(stderr,"Mesh setup failed HRESULT=0x%08lx\n",(unsigned long)hr); }
    if (errors) { fprintf(stderr,"Mesh shader: %.*s\n",(int)ID3D10Blob_GetBufferSize(errors),(char*)ID3D10Blob_GetBufferPointer(errors)); ID3D10Blob_Release(errors); }
    if (blob) { ID3D10Blob_Release(blob); } if (vs) { ID3D10Blob_Release(vs); } if (ps) { ID3D10Blob_Release(ps); }
    if (*uncertain) { free(data); fprintf(stderr,"Mesh upload GPU completion uncertain; process exit required\n"); return 0; }
    if (event) { CloseHandle(event); } if (fence) { ID3D12Fence_Release(fence); }
    if (list) { ID3D12GraphicsCommandList_Release(list); } if (allocator) { ID3D12CommandAllocator_Release(allocator); }
    free(data); return success;
}
static void record_mesh(ID3D12GraphicsCommandList* list,EyeInputs* eye,GpuScene* gpu,unsigned int index,MeshRender* mesh) {
    transition(list,eye->textures[1],eye->states[1],D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(list,mesh->scene_depth[index],D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12GraphicsCommandList_CopyResource(list,mesh->scene_depth[index],eye->textures[1]);
    transition(list,mesh->scene_depth[index],D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    for (unsigned int i=0;i<3;++i) {
        transition(list,eye->textures[i],i==1?D3D12_RESOURCE_STATE_COPY_SOURCE:eye->states[i],D3D12_RESOURCE_STATE_RENDER_TARGET);
    }
    D3D12_CPU_DESCRIPTOR_HANDLE target=ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(mesh->rtv); target.ptr+=(SIZE_T)index*3*mesh->rtv_stride;
    D3D12_CPU_DESCRIPTOR_HANDLE depth=ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(mesh->dsv);
    ID3D12GraphicsCommandList_ClearDepthStencilView(list,depth,D3D12_CLEAR_FLAG_DEPTH,1,0,0,NULL);
    ID3D12GraphicsCommandList_OMSetRenderTargets(list,3,&target,TRUE,&depth);
    D3D12_VIEWPORT viewport={0,0,(float)PROBE_SIZE,(float)PROBE_SIZE,0,1}; D3D12_RECT scissor={0,0,PROBE_SIZE,PROBE_SIZE};
    ID3D12GraphicsCommandList_RSSetViewports(list,1,&viewport); ID3D12GraphicsCommandList_RSSetScissorRects(list,1,&scissor);
    ID3D12GraphicsCommandList_SetGraphicsRootSignature(list,mesh->root); ID3D12GraphicsCommandList_SetPipelineState(list,mesh->pipeline);
    ID3D12GraphicsCommandList_SetDescriptorHeaps(list,1,&mesh->heap);
    D3D12_GPU_DESCRIPTOR_HANDLE srv=ID3D12DescriptorHeap_GetGPUDescriptorHandleForHeapStart(mesh->heap); srv.ptr+=(UINT64)index*2*mesh->srv_stride;
    ID3D12GraphicsCommandList_SetGraphicsRootDescriptorTable(list,0,srv);
    ID3D12GraphicsCommandList_SetGraphicsRoot32BitConstants(list,1,32,gpu->constants[index],0);
    ID3D12GraphicsCommandList_IASetPrimitiveTopology(list,D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D12GraphicsCommandList_IASetVertexBuffers(list,0,1,&mesh->vertex_view);
    ID3D12GraphicsCommandList_DrawInstanced(list,mesh->vertex_count,1,0,0);
    ID3D12GraphicsCommandList_OMSetRenderTargets(list,0,NULL,FALSE,NULL);
    for (unsigned int i=0;i<3;++i) {
        transition(list,eye->textures[i],D3D12_RESOURCE_STATE_RENDER_TARGET,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        eye->states[i]=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
}
#endif
