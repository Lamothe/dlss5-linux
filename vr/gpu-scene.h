#include <d3dcompiler.h>
typedef struct GpuScene {
    ID3D12RootSignature* root;
    ID3D12PipelineState* pipeline;
    ID3D12PipelineState* composition;
    ID3D12Resource* presented[2];
    unsigned int constants[2][32];
    int interactive;
    int tracked_control;
    ID3D12DescriptorHeap* heap;
    ID3D12Resource* inputs_readback[2][3];
    ID3D12Resource* poison_uploads[2][3];
    UINT descriptor_stride;
} GpuScene;
static void destroy_gpu_scene(GpuScene* scene) {
    for (unsigned int eye=0;eye<2;++eye) {
        if (scene->presented[eye]) { ID3D12Resource_Release(scene->presented[eye]); }
        for (unsigned int i=0;i<3;++i) {
            if (scene->poison_uploads[eye][i]) { ID3D12Resource_Release(scene->poison_uploads[eye][i]); }
            if (scene->inputs_readback[eye][i]) { ID3D12Resource_Release(scene->inputs_readback[eye][i]); }
        }
    }
    if (scene->heap) { ID3D12DescriptorHeap_Release(scene->heap); }
    if (scene->composition) { ID3D12PipelineState_Release(scene->composition); }
    if (scene->pipeline) { ID3D12PipelineState_Release(scene->pipeline); }
    if (scene->root) { ID3D12RootSignature_Release(scene->root); }
}
static int prepare_gpu_scene(ID3D12Device* device, EyeInputs* eyes, GpuScene* scene) {
    ID3DBlob* shader=NULL; ID3DBlob* blob=NULL; ID3DBlob* errors=NULL;
    int success=0;
    HRESULT hr=D3DCompileFromFile(L"../gpu-scene.hlsl",NULL,NULL,"main","cs_5_1",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&shader,&errors);
    if (FAILED(hr)) {
        if (errors) { fprintf(stderr,"Shader: %.*s\n",(int)ID3D10Blob_GetBufferSize(errors),(char*)ID3D10Blob_GetBufferPointer(errors)); }
        goto done;
    }
    if (errors) { ID3D10Blob_Release(errors); errors=NULL; }
    D3D12_DESCRIPTOR_RANGE range={0};
    range.RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_UAV; range.NumDescriptors=5;
    D3D12_ROOT_PARAMETER params[2]={0};
    params[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges=1; params[0].DescriptorTable.pDescriptorRanges=&range;
    params[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[1].Constants.Num32BitValues=32;
    D3D12_ROOT_SIGNATURE_DESC desc={0}; desc.NumParameters=2; desc.pParameters=params;
    hr=D3D12SerializeRootSignature(&desc,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&errors);
    if (FAILED(hr)) { goto done; }
    hr=ID3D12Device_CreateRootSignature(device,0,ID3D10Blob_GetBufferPointer(blob),ID3D10Blob_GetBufferSize(blob),&IID_ID3D12RootSignature,(void**)&scene->root);
    if (FAILED(hr)) { goto done; }
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline={0};
    pipeline.pRootSignature=scene->root;
    pipeline.CS.pShaderBytecode=ID3D10Blob_GetBufferPointer(shader);
    pipeline.CS.BytecodeLength=ID3D10Blob_GetBufferSize(shader);
    hr=ID3D12Device_CreateComputePipelineState(device,&pipeline,&IID_ID3D12PipelineState,(void**)&scene->pipeline);
    if (FAILED(hr)) { goto done; }
    ID3D10Blob_Release(shader); shader=NULL;
    hr=D3DCompileFromFile(L"../gpu-scene.hlsl",NULL,NULL,"compose","cs_5_1",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&shader,&errors);
    if (FAILED(hr)) { goto done; }
    pipeline.CS.pShaderBytecode=ID3D10Blob_GetBufferPointer(shader);
    pipeline.CS.BytecodeLength=ID3D10Blob_GetBufferSize(shader);
    hr=ID3D12Device_CreateComputePipelineState(device,&pipeline,&IID_ID3D12PipelineState,(void**)&scene->composition);
    if (FAILED(hr)) { goto done; }
    D3D12_DESCRIPTOR_HEAP_DESC heap={0}; heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.NumDescriptors=10; heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    hr=ID3D12Device_CreateDescriptorHeap(device,&heap,&IID_ID3D12DescriptorHeap,(void**)&scene->heap);
    if (FAILED(hr)) { goto done; }
    scene->descriptor_stride=ID3D12Device_GetDescriptorHandleIncrementSize(device,heap.Type);
    D3D12_CPU_DESCRIPTOR_HANDLE handle=ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(scene->heap);
    for (unsigned int eye=0;eye<2;++eye) {
        handle=ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(scene->heap);
        handle.ptr+=(SIZE_T)eye*5*scene->descriptor_stride;
        for (unsigned int i=0;i<3;++i) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav={0};
            uav.Format=eyes[eye].layouts[i].Footprint.Format; uav.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;
            ID3D12Device_CreateUnorderedAccessView(device,eyes[eye].textures[i],NULL,&uav,handle);
            handle.ptr+=scene->descriptor_stride;
            if (!create_buffer(device,D3D12_HEAP_TYPE_READBACK,eyes[eye].sizes[i],&scene->inputs_readback[eye][i])) { goto done; }
            if (!create_buffer(device,D3D12_HEAP_TYPE_UPLOAD,eyes[eye].sizes[i],&scene->poison_uploads[eye][i])) { goto done; }
            unsigned char* poison=NULL;
            D3D12_RANGE no_read={0,0};
            hr=ID3D12Resource_Map(scene->poison_uploads[eye][i],0,&no_read,(void**)&poison);
            if (FAILED(hr)) { goto done; }
            for (unsigned int y=0;y<PROBE_SIZE;++y) {
                unsigned char* row=poison+eyes[eye].layouts[i].Offset+y*eyes[eye].layouts[i].Footprint.RowPitch;
                if (i==1) { for (unsigned int x=0;x<PROBE_SIZE;++x) { ((float*)row)[x]=NAN; } }
                else {
                    unsigned int components=i==0?4u:2u;
                    for (unsigned int x=0;x<PROBE_SIZE*components;++x) { ((uint16_t*)row)[x]=0x7e00u; }
                }
            }
            ID3D12Resource_Unmap(scene->poison_uploads[eye][i],0,NULL);
        }
    }
        D3D12_HEAP_PROPERTIES properties={0}; properties.Type=D3D12_HEAP_TYPE_DEFAULT;
        properties.CreationNodeMask=1; properties.VisibleNodeMask=1;
        D3D12_RESOURCE_DESC target={0}; target.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        target.Width=PROBE_SIZE; target.Height=PROBE_SIZE; target.DepthOrArraySize=1;
        target.MipLevels=1; target.SampleDesc.Count=1; target.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;
        target.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        // Descriptor layout: colour, depth, motion, neural output, linear presentation.
        for (unsigned int eye=0;eye<2;++eye) {
            hr=ID3D12Device_CreateCommittedResource(device,&properties,D3D12_HEAP_FLAG_NONE,&target,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,NULL,&IID_ID3D12Resource,(void**)&scene->presented[eye]);
            if (FAILED(hr)) { goto done; }
            handle=ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(scene->heap);
            handle.ptr+=(SIZE_T)(eye*5+3)*scene->descriptor_stride;
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav={0}; uav.Format=target.Format; uav.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;
            ID3D12Device_CreateUnorderedAccessView(device,eyes[eye].textures[3],NULL,&uav,handle);
            handle.ptr+=scene->descriptor_stride;
            ID3D12Device_CreateUnorderedAccessView(device,scene->presented[eye],NULL,&uav,handle);
        }
    success=1;
done:
    if (!success) { fprintf(stderr,"GPU scene setup failed HRESULT=0x%08lx\n",(unsigned long)hr); }
    if (errors) { ID3D10Blob_Release(errors); }
    if (blob) { ID3D10Blob_Release(blob); }
    if (shader) { ID3D10Blob_Release(shader); }
    return success;
}
static void record_gpu_scene(ID3D12GraphicsCommandList* list, EyeInputs* eye, GpuScene* scene, unsigned int index, unsigned int frame) {
    for (unsigned int i=0;i<3;++i) {
        if (scene->interactive) {
            transition(list,eye->textures[i],eye->states[i],D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            continue;
        }
        ID3D12Resource* mirror=eye->uploads[i];
        eye->uploads[i]=scene->poison_uploads[index][i];
        upload_texture(list,eye,i);
        eye->uploads[i]=mirror;
        transition(list,eye->textures[i],eye->states[i],D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    ID3D12GraphicsCommandList_SetComputeRootSignature(list,scene->root);
    ID3D12GraphicsCommandList_SetPipelineState(list,scene->pipeline);
    ID3D12GraphicsCommandList_SetDescriptorHeaps(list,1,&scene->heap);
    D3D12_GPU_DESCRIPTOR_HANDLE handle=ID3D12DescriptorHeap_GetGPUDescriptorHandleForHeapStart(scene->heap);
    handle.ptr+=(UINT64)index*5*scene->descriptor_stride;
    ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(list,0,handle);
    unsigned int constants[32]={index,frame,PROBE_SIZE,PROBE_SIZE};
    ID3D12GraphicsCommandList_SetComputeRoot32BitConstants(list,1,32,(scene->interactive||scene->tracked_control)?scene->constants[index]:constants,0);
    ID3D12GraphicsCommandList_Dispatch(list,(PROBE_SIZE+7)/8,(PROBE_SIZE+7)/8,1);
    for (unsigned int i=0;i<3;++i) {
        if (scene->interactive) {
            transition(list,eye->textures[i],D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            eye->states[i]=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; continue;
        }
        transition(list,eye->textures[i],D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION dst={0},src={0};
        dst.pResource=scene->inputs_readback[index][i]; dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint=eye->layouts[i];
        src.pResource=eye->textures[i]; src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        ID3D12GraphicsCommandList_CopyTextureRegion(list,&dst,0,0,0,&src,NULL);
        transition(list,eye->textures[i],D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        eye->states[i]=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
}
// Independent CPU intersection oracle at sparse pixel centres.
static float oracle_z(unsigned int x,unsigned int y,unsigned int eye,unsigned int frame) {
    SceneVector origin={scene_camera_x(frame,eye),1.4f,0};
    SceneVector ray=scene_normalise((SceneVector){2*((float)x+.5f)/PROBE_SIZE-1,1-2*((float)y+.5f)/PROBE_SIZE,1});
    float t=6/ray.z;
    if (ray.y < -.0001f) { t=fminf(t,-origin.y/ray.y); }
    const SceneVector centres[2]={{-.45f,.65f,2.6f},{.65f,.45f,3.4f}};
    const float radii[2]={.65f,.45f};
    for (unsigned int i=0;i<2;++i) {
        SceneVector offset=scene_add(origin,scene_scale(centres[i],-1));
        float b=scene_dot(offset,ray),d=b*b-scene_dot(offset,offset)+radii[i]*radii[i];
        if (d>0) { float hit=-b-sqrtf(d); if (hit>.1f) { t=fminf(t,hit); } }
    }
    return t*ray.z;
}
// Independent CPU integral for numerical colour checks at visible floor samples.
static float floor_wave_integral(float x) {
    float phase=x-2*floorf(x/2);
    return phase<=1 ? phase : 2-phase;
}
static float floor_wave_average(float centre,float width) {
    if (width<.0001f) { return ((int)floorf(centre)&1) ? -1.f : 1.f; }
    float result=(floor_wave_integral(centre+width*.5f)-floor_wave_integral(centre-width*.5f))/width;
    return fmaxf(-1,fminf(1,result));
}
// Called only after the GPU fence. Retain GPU-generated inputs in existing host
// uploads for output comparison/snapshots; validates depth/motion numerically.
static int validate_gpu_inputs(EyeInputs* eye, GpuScene* scene, unsigned int index, unsigned int frame) {
    for (unsigned int i=0;i<3;++i) {
        unsigned char* source=NULL; unsigned char* destination=NULL;
        D3D12_RANGE read={0,(SIZE_T)eye->sizes[i]},no_read={0,0};
        if (FAILED(ID3D12Resource_Map(scene->inputs_readback[index][i],0,&read,(void**)&source))) { return 0; }
        if (FAILED(ID3D12Resource_Map(eye->uploads[i],0,&no_read,(void**)&destination))) {
            ID3D12Resource_Unmap(scene->inputs_readback[index][i],0,NULL); return 0;
        }
        memcpy(destination,source,(size_t)eye->sizes[i]);
        ID3D12Resource_Unmap(eye->uploads[i],0,NULL);
        unsigned int invalid=0,moving=0;
        double sum=0;
        for (unsigned int y=0;y<PROBE_SIZE;++y) {
            unsigned char* row=source+eye->layouts[i].Offset+y*eye->layouts[i].Footprint.RowPitch;
            for (unsigned int x=0;x<PROBE_SIZE;++x) {
                if (i==0) {
                    for (unsigned int c=0;c<4;++c) { float v=half_float(((uint16_t*)row)[x*4+c]); if (!isfinite(v)||v<0||v>1) { ++invalid; } }
                } else if (i==1) {
                    float v=((float*)row)[x]; if (!isfinite(v)||v<0||v>1) { ++invalid; } sum+=v;
                } else {
                    float vx=half_float(((uint16_t*)row)[x*2]),vy=half_float(((uint16_t*)row)[x*2+1]);
                    if (!isfinite(vx)||!isfinite(vy)||fabsf(vy)>(scene->tracked_control?.001f:0) || (frame==0 && fabsf(vx)>(scene->tracked_control?.001f:0)) || (frame>0 && (scene->tracked_control ? vx>=0 : vx<=0))) { ++invalid; }
                    if (vx!=0) { ++moving; } sum+=fabsf(vx);
                }
            }
        }
        if (i==0) {
            unsigned int checked=0;
            for (unsigned int y=601;y<PROBE_SIZE;y+=29) {
                for (unsigned int x=17;x<PROBE_SIZE;x+=53) {
                    unsigned int scene_x=scene->tracked_control?PROBE_SIZE-1-x:x;
                    float rx=2*((float)scene_x+.5f)/PROBE_SIZE-1;
                    float ry=1-2*((float)y+.5f)/PROBE_SIZE;
                    float z=-1.4f/ry;
                    if (fabsf(oracle_z(scene_x,y,index,frame)-z)>.0001f) { continue; }
                    float step=2.f/PROBE_SIZE;
                    float wx=2*(fabsf(1.4f*step/ry)+fabsf(1.4f*rx*step/(ry*ry)));
                    float wz=2*fabsf(1.4f*step/(ry*ry));
                    float px=scene_camera_x(frame,index)+rx*z;
                    float parity=floor_wave_average(px*2,wx)*floor_wave_average(z*2,wz);
                    float linear=(.35f-.15f*parity)*(.12f+.65f*.8f/sqrtf(.25f+.64f+.16f));
                    float expected=linear<=.0031308f?linear*12.92f:1.055f*powf(linear,1/2.4f)-.055f;
                    unsigned char* row=source+eye->layouts[i].Offset+y*eye->layouts[i].Footprint.RowPitch;
                    for (unsigned int c=0;c<3;++c) {
                        float actual=half_float(((uint16_t*)row)[x*4+c]);
                        if (!isfinite(actual)||fabsf(actual-expected)>.0015f) { ++invalid; }
                    }
                    ++checked;
                }
            }
            printf("FilteredFloor eye=%u frame=%u checked=%u invalid=%u\n",index,frame,checked,invalid);
            if (!checked) { ++invalid; }
        }
        if (i>0) {
            for (unsigned int y=35;y<PROBE_SIZE;y+=79) {
                for (unsigned int x=27;x<PROBE_SIZE;x+=83) {
                    unsigned char* row=source+eye->layouts[i].Offset+y*eye->layouts[i].Footprint.RowPitch;
                    float z=oracle_z(scene->tracked_control?PROBE_SIZE-1-x:x,y,index,frame);
                    float expected=i==1 ? 20.0f/19.9f-2/(19.9f*z) :
                        (scene_camera_x(frame,index)-scene_camera_x(frame==0?0:frame-1,index))/z*PROBE_SIZE*.5f;
                    if (scene->tracked_control && i==2) { expected=-expected; }
                    float actual=i==1 ? ((float*)row)[x] : half_float(((uint16_t*)row)[x*2]);
                    float tolerance=i==1 ? .00001f : .002f;
                    if (!isfinite(actual)||fabsf(actual-expected)>tolerance) { ++invalid; }
                }
            }
        }
        ID3D12Resource_Unmap(scene->inputs_readback[index][i],0,&no_read);
        printf("GPUInput eye=%u frame=%u texture=%u invalid=%u moving=%u mean=%.6f\n",index,frame,i,invalid,moving,sum/(PROBE_SIZE*PROBE_SIZE));
        if (invalid || (i==2 && frame>0 && moving!=PROBE_SIZE*PROBE_SIZE)) { return 0; }
    }
    return 1;
}
