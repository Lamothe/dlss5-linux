// Minimal tracked OpenXR test; existing probe retained as regression entrypoint.
#define PROBE_MESH_RENDER 1
#define main retained_probe_main
#include "feature-probe.c"
#undef main
#include "mesh-render.h"
#include "city-collision.h"
#include <dxgi1_4.h>
#define XR_USE_PLATFORM_WIN32
#define XR_NO_PROTOTYPES
#define XR_USE_GRAPHICS_API_D3D12
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#define XR_FUNCTIONS(X) X(xrGetInstanceProcAddr) X(xrCreateInstance) X(xrDestroyInstance) X(xrGetSystem) X(xrCreateSession) X(xrDestroySession) X(xrCreateReferenceSpace) X(xrDestroySpace) X(xrEnumerateSwapchainFormats) X(xrCreateSwapchain) X(xrDestroySwapchain) X(xrEnumerateSwapchainImages) X(xrPollEvent) X(xrBeginSession) X(xrEndSession) X(xrRequestExitSession) X(xrWaitFrame) X(xrBeginFrame) X(xrEndFrame) X(xrLocateViews) X(xrAcquireSwapchainImage) X(xrWaitSwapchainImage) X(xrReleaseSwapchainImage) X(xrStringToPath) X(xrCreateActionSet) X(xrDestroyActionSet) X(xrCreateAction) X(xrSuggestInteractionProfileBindings) X(xrAttachSessionActionSets) X(xrSyncActions) X(xrGetActionStateFloat) X(xrGetActionStateBoolean) X(xrGetActionStateVector2f)
#define DECLARE(name) static PFN_##name name;
XR_FUNCTIONS(DECLARE)
#undef DECLARE
#define XR_CHECK(call) do { XrResult r=(call); if (XR_FAILED(r)) { fprintf(stderr,"%s failed %d\n",#call,r); goto cleanup; } } while(0)
#define DX_CHECK(call) do { HRESULT h=(call); if (FAILED(h)) { fprintf(stderr,"%s failed 0x%08lx\n",#call,(unsigned long)h); goto cleanup; } } while(0)
typedef struct Locomotion {
    float x,y,z,yaw;
    int move_ready,turn_ready,snap_armed,city;
    const CityCollision* collision;
    XrTime last_time;
    unsigned int moves,snaps;
} Locomotion;
static float clamp_float(float value,float low,float high) {
    return fminf(high,fmaxf(low,value));
}
// Combined render poses are in scene coordinates. Compositor poses remain physical.
static XrView virtual_view(const XrView* physical,int room,const Locomotion* locomotion) {
    XrView result=*physical;
    XrQuaternionf q={physical->pose.orientation.z,physical->pose.orientation.w,-physical->pose.orientation.x,-physical->pose.orientation.y};
    float yaw=room?locomotion->yaw:0.f,c=cosf(yaw),s=sinf(yaw),h=sinf(yaw*.5f),w=cosf(yaw*.5f);
    result.pose.orientation=(XrQuaternionf){w*q.x+h*q.z,w*q.y+h*q.w,w*q.z-h*q.x,w*q.w-h*q.y};
    float x=-physical->pose.position.x,z=-physical->pose.position.z+(room&&!locomotion->city?1.f:0.f);
    result.pose.position=(XrVector3f){c*x+s*z+(room?locomotion->x:0.f),physical->pose.position.y+1.4f+(room?locomotion->y:0.f),-s*x+c*z+(room?locomotion->z:0.f)};
    return result;
}
static void update_locomotion(Locomotion* locomotion,const XrView physical[2],XrVector2f move,XrVector2f turn,int move_active,int turn_active,XrTime time,int smoke,unsigned int frame,int* history) {
    float dt=locomotion->last_time&&time>locomotion->last_time?clamp_float((float)((double)(time-locomotion->last_time)/1e9),0.f,.05f):0.f;
    locomotion->last_time=time;
    XrView head=virtual_view(&physical[0],1,locomotion);
    XrView other=virtual_view(&physical[1],1,locomotion);
    float head_x=(head.pose.position.x+other.pose.position.x)*.5f,head_z=(head.pose.position.z+other.pose.position.z)*.5f;
    int snap=0;
    if (smoke) {
        // Smoke turns are scheduled only; live controller input cannot alter them.
        snap=frame==350?1:0; locomotion->snap_armed=0;
    }
    float angle=0.f;
    if (!smoke&&turn_active&&locomotion->turn_ready&&isfinite(turn.x)&&isfinite(turn.y)&&fabsf(turn.x)>.2f) {
        // Continuous yaw, 60 degrees/s at full deflection, around the head centre.
        angle=-copysignf(clamp_float((fabsf(turn.x)-.2f)/.8f,0.f,1.f),turn.x)*dt*(3.14159265358979323846f/3.f);
    }
    if (snap) { angle=-(float)snap*(3.14159265358979323846f/6.f); }
    if (angle!=0.f) {
        // Turn around the head centre, not the LOCAL origin or either eye.
        float c=cosf(angle),s=sinf(angle);
        float x=head_x-locomotion->x,z=head_z-locomotion->z;
        locomotion->x=head_x-(c*x+s*z); locomotion->z=head_z-(-s*x+c*z);
        locomotion->yaw=remainderf(locomotion->yaw+angle,2.f*3.14159265358979323846f);
        if (snap) { ++locomotion->snaps; *history=0; }
        head=virtual_view(&physical[0],1,locomotion);
    }
    if (smoke) { move=(XrVector2f){0.f,frame>=100&&frame<130?1.f:0.f}; move_active=1; dt=.01f; }
    float before_x=locomotion->x,before_y=locomotion->y,before_z=locomotion->z;
    // Right-stick vertical inspection movement: 0.5 m/s; diagonal turns take priority.
    if (!smoke&&turn_active&&locomotion->turn_ready&&isfinite(turn.x)&&isfinite(turn.y)
        &&fabsf(turn.x)<=.2f&&fabsf(turn.y)>.2f) {
        locomotion->y+=copysignf(clamp_float((fabsf(turn.y)-.2f)/.8f,0.f,1.f)*.5f*dt,turn.y);
    }
    if (move_active&&locomotion->move_ready&&isfinite(move.x)&&isfinite(move.y)) {
        float magnitude=sqrtf(move.x*move.x+move.y*move.y);
        if (magnitude>.2f) {
            // Project head forward onto the floor; ignore pitch/roll for movement.
            XrQuaternionf q=head.pose.orientation;
            float forward_x=-2.f*(q.x*q.z+q.w*q.y),forward_z=2.f*(q.x*q.x+q.y*q.y)-1.f;
            float length=sqrtf(forward_x*forward_x+forward_z*forward_z);
            if (length>1e-4f) {
                forward_x/=length; forward_z/=length;
                float speed=clamp_float((magnitude-.2f)/.8f,0.f,1.f)*dt*(locomotion->city?2.f:1.f)/magnitude; // City: 2 m/s; room: 1 m/s maximum.
                // In Y-up coordinates, right is (-forward_z, forward_x).
                locomotion->x+=(move.y*forward_x-move.x*forward_z)*speed;
                locomotion->z+=(move.y*forward_z+move.x*forward_x)*speed;
            }
        }
    }
    locomotion->move_ready=move_active; locomotion->turn_ready=turn_active;
    if (locomotion->city) {
        // Resolve only the requested virtual translation, around the tracked head
        // centre. Never correct physical walking, leaning, turning or recentering.
        XrView start={0}; Locomotion origin=*locomotion;
        origin.x=before_x; origin.y=before_y; origin.z=before_z;
        start=virtual_view(&physical[0],1,&origin);
        XrView start_other=virtual_view(&physical[1],1,&origin);
        float position[3]={(start.pose.position.x+start_other.pose.position.x)*.5f,
            (start.pose.position.y+start_other.pose.position.y)*.5f,
            (start.pose.position.z+start_other.pose.position.z)*.5f};
        float requested[3]={locomotion->x-before_x,locomotion->y-before_y,locomotion->z-before_z},resolved[3];
        city_collision_move(locomotion->collision,position,requested,resolved);
        locomotion->x=before_x+resolved[0]; locomotion->y=before_y+resolved[1]; locomotion->z=before_z+resolved[2];
    } else {
        // Bounds only: no furniture collision or physical-play-space protection.
        // Constrain both rendered eye positions, including physical head translation.
        XrView bounded[2]={virtual_view(&physical[0],1,locomotion),virtual_view(&physical[1],1,locomotion)};
        float min_x=fminf(bounded[0].pose.position.x,bounded[1].pose.position.x),max_x=fmaxf(bounded[0].pose.position.x,bounded[1].pose.position.x);
        float min_y=fminf(bounded[0].pose.position.y,bounded[1].pose.position.y),max_y=fmaxf(bounded[0].pose.position.y,bounded[1].pose.position.y);
        locomotion->y+=clamp_float(0.f,.25f-min_y,(locomotion->city?15.f:2.95f)-max_y);
        float min_z=fminf(bounded[0].pose.position.z,bounded[1].pose.position.z),max_z=fmaxf(bounded[0].pose.position.z,bounded[1].pose.position.z);
        locomotion->x+=clamp_float(0.f,(locomotion->city?-10.f:-2.6f)-min_x,(locomotion->city?10.f:2.6f)-max_x);
        locomotion->z+=clamp_float(0.f,(locomotion->city?-20.f:.4f)-min_z,(locomotion->city?40.f:6.5f)-max_z);
    }
    int moved=fabsf(locomotion->x-before_x)>1e-6f||fabsf(locomotion->y-before_y)>1e-6f||fabsf(locomotion->z-before_z)>1e-6f;
    if (moved) { ++locomotion->moves; }
    if (snap|| (angle!=0.f&&frame%60==0)|| (moved&&(locomotion->moves==1||frame%60==0||smoke))) {
        XrView actual=virtual_view(&physical[0],1,locomotion);
        XrView actual_other=virtual_view(&physical[1],1,locomotion);
        printf("DEMO LOCOMOTION frame=%u source=%s snap=%d camera_x=%.4f camera_y=%.4f camera_z=%.4f yaw_degrees=%.1f dt=%.4f history_reset=%d bounds_only=%d\n",frame,smoke?"smoke":"controller",snap,(actual.pose.position.x+actual_other.pose.position.x)*.5f,(actual.pose.position.y+actual_other.pose.position.y)*.5f,(actual.pose.position.z+actual_other.pose.position.z)*.5f,locomotion->yaw*180.f/3.14159265358979323846f,dt,snap!=0,!locomotion->city); fflush(stdout);
    }
}
static void view_constants(GpuScene* gpu,unsigned int eye,unsigned int frame,const XrView* current,const XrView* previous,int neural,int room,int city,unsigned int lighting) {
    unsigned int* c=gpu->constants[eye]; memset(c,0,32*sizeof(*c));
    c[0]=eye; c[1]=frame; c[2]=PROBE_SIZE; c[3]=PROBE_SIZE; c[4]=1; c[5]=neural?1u:0u; c[6]=(city?4u:(room?1u:0u))|2u; c[7]=lighting;
    float f[24]={current->pose.orientation.x,current->pose.orientation.y,current->pose.orientation.z,current->pose.orientation.w,
        current->pose.position.x,current->pose.position.y,current->pose.position.z,0,
        previous->pose.orientation.x,previous->pose.orientation.y,previous->pose.orientation.z,previous->pose.orientation.w,
        previous->pose.position.x,previous->pose.position.y,previous->pose.position.z,0,
        tanf(current->fov.angleLeft),tanf(current->fov.angleRight),tanf(current->fov.angleUp),tanf(current->fov.angleDown),
        tanf(previous->fov.angleLeft),tanf(previous->fov.angleRight),tanf(previous->fov.angleUp),tanf(previous->fov.angleDown)};
    memcpy(c+8,f,sizeof(f));
}
static void compose_eye(ID3D12GraphicsCommandList* list,EyeInputs* eye,GpuScene* gpu,unsigned int index,ID3D12Resource* destination) {
    // Inputs are written/read through UAV descriptors for the composition shader.
    transition(list,eye->textures[0],eye->states[0],D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    eye->states[0]=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    D3D12_RESOURCE_BARRIER barrier={0}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource=eye->textures[3]; ID3D12GraphicsCommandList_ResourceBarrier(list,1,&barrier);
    ID3D12GraphicsCommandList_SetComputeRootSignature(list,gpu->root);
    ID3D12GraphicsCommandList_SetPipelineState(list,gpu->composition);
    ID3D12GraphicsCommandList_SetDescriptorHeaps(list,1,&gpu->heap);
    D3D12_GPU_DESCRIPTOR_HANDLE handle=ID3D12DescriptorHeap_GetGPUDescriptorHandleForHeapStart(gpu->heap);
    handle.ptr+=(UINT64)index*5*gpu->descriptor_stride;
    ID3D12GraphicsCommandList_SetComputeRootDescriptorTable(list,0,handle);
    ID3D12GraphicsCommandList_SetComputeRoot32BitConstants(list,1,32,gpu->constants[index],0);
    ID3D12GraphicsCommandList_Dispatch(list,(PROBE_SIZE+7)/8,(PROBE_SIZE+7)/8,1);
    transition(list,gpu->presented[index],D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(list,destination,D3D12_RESOURCE_STATE_RENDER_TARGET,D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12GraphicsCommandList_CopyResource(list,destination,gpu->presented[index]);
    transition(list,destination,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_RENDER_TARGET);
    transition(list,gpu->presented[index],D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}
typedef struct FrameTiming {
    int neural;
    double scene_ms, neural_ms, compose_ms, gpu_ms, frame_wall_ms, wait_ms, cadence_ms, period_ms;
} FrameTiming;
static double elapsed_ms(LARGE_INTEGER begin,LARGE_INTEGER end,LARGE_INTEGER frequency) {
    return 1000.0*(double)(end.QuadPart-begin.QuadPart)/(double)frequency.QuadPart;
}
int main(int argc,char** argv) {
    int neural=0,room=0,smoke=0,model=0,city=0,supplied=0,automotive=0;
    for (int i=1;i<argc;++i) {
        if (!strcmp(argv[i],"--neural")) { neural=1; }
        else if (!strcmp(argv[i],"--automotive")) { automotive=1; supplied=1; city=1; room=1; }
        else if (!strcmp(argv[i],"--automotive-smoke")) { automotive=1; supplied=1; city=1; room=1; smoke=1; }
        else if (!strcmp(argv[i],"--supplied-city")) { supplied=1; city=1; room=1; }
        else if (!strcmp(argv[i],"--supplied-city-smoke")) { supplied=1; city=1; room=1; smoke=1; }
        else if (!strcmp(argv[i],"--city")) { city=1; room=1; }
        else if (!strcmp(argv[i],"--city-smoke")) { city=1; room=1; smoke=1; }
        else if (!strcmp(argv[i],"--model")) { model=1; room=1; }
        else if (!strcmp(argv[i],"--room")) { room=1; }
        else if (!strcmp(argv[i],"--room-smoke")) { room=1; smoke=1; }
        else { fprintf(stderr,"Usage: xr-scene.exe [--neural] [--room|--room-smoke] [--model|--city|--city-smoke]\n"); return 1; }
    }
    if (city&&model) { fprintf(stderr,"Select either --model or --city\n"); return 1; }
    int interactive=room&&!smoke,exit_requested=0;
    unsigned int frame_limit=600u,lighting=0,toggles=0;
    FrameTiming* timings=calloc(interactive?1u:frame_limit,sizeof(*timings));
    if (!timings) { return 1; }
    XrActionSet action_set=XR_NULL_HANDLE; XrAction trigger=XR_NULL_HANDLE,light_action=XR_NULL_HANDLE,move_action=XR_NULL_HANDLE,turn_action=XR_NULL_HANDLE,exit_action=XR_NULL_HANDLE;
    XrPath hands[2]={XR_NULL_PATH,XR_NULL_PATH}; int trigger_down=0,light_down=0,input_ready=0,smoke_stage=0,smoke_light=0;
    Locomotion locomotion={0}; int focused=0;
    locomotion.city=city; if (city) { locomotion.x=automotive?0.f:supplied?10.f:-5.5f; }
    CityCollision collision={0};
    if (city) {
        if (!city_collision_load(&collision,automotive?"automotive-collision.bin":supplied?"supplied-city-collision.bin":"city-collision.bin")) {
            fprintf(stderr,"Invalid or missing city-collision.bin\n"); free(timings); return 1;
        }
        if (supplied) {
            // Final four exporter records are perimeter slabs: derive navigation
            // limits from them rather than the old assembled street dimensions.
            if (collision.count<4) { city_collision_free(&collision); free(timings); return 1; }
            CityCollisionBox* edge=collision.boxes+collision.count-4;
            collision.custom_limits=1;
            collision.low[0]=edge[0].max[0]+CITY_COLLISION_RADIUS;
            collision.high[0]=edge[1].min[0]-CITY_COLLISION_RADIUS;
            collision.low[1]=.25f; collision.high[1]=29.f;
            collision.low[2]=edge[2].max[2]+CITY_COLLISION_RADIUS;
            collision.high[2]=edge[3].min[2]-CITY_COLLISION_RADIUS;
        }
        locomotion.collision=&collision;
        printf("DEMO CITY COLLISION boxes=%u head_radius=%.2f\n",collision.count,CITY_COLLISION_RADIUS);
    }
    LARGE_INTEGER cpu_frequency={0},last_completion={0};
    ID3D12QueryHeap* queries=NULL; ID3D12Resource* timestamp_readback=NULL;
    UINT64 timestamp_frequency=0;
    if (!QueryPerformanceFrequency(&cpu_frequency)||cpu_frequency.QuadPart<=0) { city_collision_free(&collision); free(timings); return 1; }
    int code=1,running=0,pending=0,core_ready=0; unsigned int frames=0;
    XrInstance instance=XR_NULL_HANDLE; XrSession session=XR_NULL_HANDLE; XrSpace space=XR_NULL_HANDLE;
    XrSystemId system=XR_NULL_SYSTEM_ID; XrSwapchain chains[2]={XR_NULL_HANDLE,XR_NULL_HANDLE};
    XrSwapchainImageD3D12KHR* images[2]={NULL,NULL}; uint32_t image_counts[2]={0,0};
    ID3D12Device* device=NULL; ID3D12CommandQueue* queue=NULL; ID3D12CommandAllocator* allocator=NULL;
    ID3D12GraphicsCommandList* list=NULL; ID3D12Fence* fence=NULL; HANDLE event=NULL;
    IDXGIFactory4* factory=NULL; IDXGIAdapter1* adapter=NULL;
    EyeInputs eyes[2]={0}; GpuScene gpu={0}; MeshRender mesh={0}; int mesh_pending=0; void* caps=NULL; void* features[2]={NULL,NULL};
    CoreShutdown shutdown=NULL; FwdRelease release=NULL; DestroyParams destroy=NULL; FwdEvaluate evaluate=NULL;
    HMODULE loader=LoadLibraryW(L"openxr_loader.dll"),core=NULL,forwarder=NULL;
    if (!loader) { fprintf(stderr,"OpenXR loader missing\n"); goto cleanup; }
#define LOAD(name) name=(PFN_##name)(void*)GetProcAddress(loader,#name); if (!name) { fprintf(stderr,"Missing " #name "\n"); goto cleanup; }
    XR_FUNCTIONS(LOAD)
#undef LOAD
    const char* extension="XR_KHR_D3D12_enable";
    XrInstanceCreateInfo ici={XR_TYPE_INSTANCE_CREATE_INFO};
    strcpy(ici.applicationInfo.applicationName,"DLSS stereo testbed"); ici.applicationInfo.apiVersion=XR_CURRENT_API_VERSION;
    ici.enabledExtensionCount=1; ici.enabledExtensionNames=&extension;
    XR_CHECK(xrCreateInstance(&ici,&instance));
    XrSystemGetInfo sgi={XR_TYPE_SYSTEM_GET_INFO}; sgi.formFactor=XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    XR_CHECK(xrGetSystem(instance,&sgi,&system));
    PFN_xrGetD3D12GraphicsRequirementsKHR requirements=NULL;
    XR_CHECK(xrGetInstanceProcAddr(instance,"xrGetD3D12GraphicsRequirementsKHR",(PFN_xrVoidFunction*)&requirements));
    if (!requirements) { goto cleanup; }
    XrGraphicsRequirementsD3D12KHR req={XR_TYPE_GRAPHICS_REQUIREMENTS_D3D12_KHR};
    XR_CHECK(requirements(instance,system,&req));
    DX_CHECK(CreateDXGIFactory1(&IID_IDXGIFactory4,(void**)&factory));
    for (UINT i=0;;++i) {
        HRESULT h=IDXGIFactory4_EnumAdapters1(factory,i,&adapter);
        if (h==DXGI_ERROR_NOT_FOUND) { fprintf(stderr,"Required XR adapter not found\n"); goto cleanup; }
        if (FAILED(h)) { goto cleanup; }
        DXGI_ADAPTER_DESC1 desc; DX_CHECK(IDXGIAdapter1_GetDesc1(adapter,&desc));
        if (memcmp(&desc.AdapterLuid,&req.adapterLuid,sizeof(LUID))==0) { break; }
        IDXGIAdapter1_Release(adapter); adapter=NULL;
    }
    DX_CHECK(D3D12CreateDevice((IUnknown*)adapter,req.minFeatureLevel,&IID_ID3D12Device,(void**)&device));
    D3D12_COMMAND_QUEUE_DESC qd={0}; qd.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
    DX_CHECK(ID3D12Device_CreateCommandQueue(device,&qd,&IID_ID3D12CommandQueue,(void**)&queue));
    DX_CHECK(ID3D12Device_CreateCommandAllocator(device,qd.Type,&IID_ID3D12CommandAllocator,(void**)&allocator));
    DX_CHECK(ID3D12Device_CreateCommandList(device,0,qd.Type,allocator,NULL,&IID_ID3D12GraphicsCommandList,(void**)&list));
    DX_CHECK(ID3D12Device_CreateFence(device,0,D3D12_FENCE_FLAG_NONE,&IID_ID3D12Fence,(void**)&fence));
    event=CreateEventW(NULL,FALSE,FALSE,NULL); if (!event) { goto cleanup; }
    DX_CHECK(ID3D12CommandQueue_GetTimestampFrequency(queue,&timestamp_frequency));
    if (!timestamp_frequency) { fprintf(stderr,"Invalid GPU timestamp frequency\n"); goto cleanup; }
    D3D12_QUERY_HEAP_DESC query_desc={0}; query_desc.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP; query_desc.Count=8;
    DX_CHECK(ID3D12Device_CreateQueryHeap(device,&query_desc,&IID_ID3D12QueryHeap,(void**)&queries));
    D3D12_HEAP_PROPERTIES timing_heap={0}; timing_heap.Type=D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC timing_desc={0}; timing_desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
    timing_desc.Width=8*sizeof(UINT64); timing_desc.Height=1; timing_desc.DepthOrArraySize=1;
    timing_desc.MipLevels=1; timing_desc.SampleDesc.Count=1; timing_desc.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    DX_CHECK(ID3D12Device_CreateCommittedResource(device,&timing_heap,D3D12_HEAP_FLAG_NONE,&timing_desc,D3D12_RESOURCE_STATE_COPY_DEST,NULL,&IID_ID3D12Resource,(void**)&timestamp_readback));
    XrGraphicsBindingD3D12KHR binding={XR_TYPE_GRAPHICS_BINDING_D3D12_KHR}; binding.device=device; binding.queue=queue;
    XrSessionCreateInfo sci={XR_TYPE_SESSION_CREATE_INFO}; sci.next=&binding; sci.systemId=system;
    XR_CHECK(xrCreateSession(instance,&sci,&session));
    if (room) {
        XR_CHECK(xrStringToPath(instance,"/user/hand/left",&hands[0]));
        XR_CHECK(xrStringToPath(instance,"/user/hand/right",&hands[1]));
        XrActionSetCreateInfo asi={XR_TYPE_ACTION_SET_CREATE_INFO};
        strcpy(asi.actionSetName,"demo"); strcpy(asi.localizedActionSetName,"Demo controls");
        XR_CHECK(xrCreateActionSet(instance,&asi,&action_set));
        XrActionCreateInfo aci={XR_TYPE_ACTION_CREATE_INFO};
        aci.countSubactionPaths=2; aci.subactionPaths=hands;
        aci.actionType=XR_ACTION_TYPE_FLOAT_INPUT;
        strcpy(aci.actionName,"neural_toggle"); strcpy(aci.localizedActionName,"Toggle neural rendering");
        XR_CHECK(xrCreateAction(action_set,&aci,&trigger));
        aci.actionType=XR_ACTION_TYPE_BOOLEAN_INPUT;
        strcpy(aci.actionName,"lighting_toggle"); strcpy(aci.localizedActionName,"Toggle lighting");
        XR_CHECK(xrCreateAction(action_set,&aci,&light_action));
        aci.actionType=XR_ACTION_TYPE_VECTOR2F_INPUT;
        aci.countSubactionPaths=1; aci.subactionPaths=&hands[0];
        strcpy(aci.actionName,"move"); strcpy(aci.localizedActionName,"Move (bounds only)");
        XR_CHECK(xrCreateAction(action_set,&aci,&move_action));
        aci.subactionPaths=&hands[1];
        strcpy(aci.actionName,"smooth_turn"); strcpy(aci.localizedActionName,"Smooth turn");
        XR_CHECK(xrCreateAction(action_set,&aci,&turn_action));
        aci.actionType=XR_ACTION_TYPE_BOOLEAN_INPUT;
        strcpy(aci.actionName,"exit_demo"); strcpy(aci.localizedActionName,"Exit demo (B)");
        XR_CHECK(xrCreateAction(action_set,&aci,&exit_action));
        XrPath profile=XR_NULL_PATH; XrActionSuggestedBinding bindings[7]={0};
        XR_CHECK(xrStringToPath(instance,"/interaction_profiles/oculus/touch_controller",&profile));
        const char* paths[]={"/user/hand/left/input/trigger/value","/user/hand/right/input/trigger/value","/user/hand/left/input/x/click","/user/hand/right/input/a/click","/user/hand/left/input/thumbstick","/user/hand/right/input/thumbstick","/user/hand/right/input/b/click"};
        for (unsigned int i=0;i<7;++i) { bindings[i].action=i<2?trigger:(i<4?light_action:(i==4?move_action:(i==5?turn_action:exit_action))); XR_CHECK(xrStringToPath(instance,paths[i],&bindings[i].binding)); }
        XrInteractionProfileSuggestedBinding suggest={XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
        suggest.interactionProfile=profile; suggest.countSuggestedBindings=(uint32_t)(sizeof(bindings)/sizeof(bindings[0])); suggest.suggestedBindings=bindings;
        XR_CHECK(xrSuggestInteractionProfileBindings(instance,&suggest));
        printf("DEMO BINDINGS count=%u left_thumbstick=move right_thumbstick=smooth_turn\n",suggest.countSuggestedBindings);
        XrSessionActionSetsAttachInfo attach={XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO}; attach.countActionSets=1; attach.actionSets=&action_set;
        XR_CHECK(xrAttachSessionActionSets(session,&attach));
    }
    XrReferenceSpaceCreateInfo rsi={XR_TYPE_REFERENCE_SPACE_CREATE_INFO}; rsi.referenceSpaceType=XR_REFERENCE_SPACE_TYPE_LOCAL; rsi.poseInReferenceSpace.orientation.w=1;
    XR_CHECK(xrCreateReferenceSpace(session,&rsi,&space));
    uint32_t format_count=0; XR_CHECK(xrEnumerateSwapchainFormats(session,0,&format_count,NULL));
    int64_t* formats=calloc(format_count,sizeof(*formats)); if (!formats) { goto cleanup; }
    XrResult fr=xrEnumerateSwapchainFormats(session,format_count,&format_count,formats);
    int found=0; if (XR_SUCCEEDED(fr)) { for (uint32_t i=0;i<format_count;++i) { if (formats[i]==DXGI_FORMAT_R16G16B16A16_FLOAT) { found=1; } } }
    free(formats); if (!found) { fprintf(stderr,"Runtime lacks linear FP16 swapchain format\n"); goto cleanup; }
    for (unsigned int eye=0;eye<2;++eye) {
        XrSwapchainCreateInfo ci={XR_TYPE_SWAPCHAIN_CREATE_INFO}; ci.usageFlags=XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT|XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
        ci.format=DXGI_FORMAT_R16G16B16A16_FLOAT; ci.sampleCount=1; ci.width=PROBE_SIZE; ci.height=PROBE_SIZE; ci.faceCount=1; ci.arraySize=1; ci.mipCount=1;
        XR_CHECK(xrCreateSwapchain(session,&ci,&chains[eye]));
        XR_CHECK(xrEnumerateSwapchainImages(chains[eye],0,&image_counts[eye],NULL));
        images[eye]=calloc(image_counts[eye],sizeof(*images[eye])); if (!images[eye]||!image_counts[eye]) { goto cleanup; }
        for (uint32_t i=0;i<image_counts[eye];++i) { images[eye][i].type=XR_TYPE_SWAPCHAIN_IMAGE_D3D12_KHR; }
        XR_CHECK(xrEnumerateSwapchainImages(chains[eye],image_counts[eye],&image_counts[eye],(XrSwapchainImageBaseHeader*)images[eye]));
    }
    // Existing verified runtime ABI; keep neural model uninitialised in off mode.
    if (neural||room) {
        core=LoadLibraryW(L"nvngx.dll"); forwarder=LoadLibraryW(L"nvngx.dll_nrfwd.dll");
        if (!core||!forwarder) { fprintf(stderr,"Neural runtime library load failed error=%lu\n",(unsigned long)GetLastError()); goto cleanup; }
        CoreInit init=(CoreInit)(void*)GetProcAddress(core,"NVSDK_NGX_D3D12_Init");
        GetCaps get_caps=(GetCaps)(void*)GetProcAddress(core,"NVSDK_NGX_D3D12_GetCapabilityParameters");
        AllocateParams allocate=(AllocateParams)(void*)GetProcAddress(core,"NVSDK_NGX_D3D12_AllocateParameters");
        shutdown=(CoreShutdown)(void*)GetProcAddress(core,"NVSDK_NGX_D3D12_Shutdown1"); destroy=(DestroyParams)(void*)GetProcAddress(core,"NVSDK_NGX_D3D12_DestroyParameters");
        FwdInit nr_init=(FwdInit)(void*)GetProcAddress(forwarder,"nrfwd_init"); FwdCreate create=(FwdCreate)(void*)GetProcAddress(forwarder,"nrfwd_create");
        evaluate=(FwdEvaluate)(void*)GetProcAddress(forwarder,"nrfwd_evaluate"); release=(FwdRelease)(void*)GetProcAddress(forwarder,"nrfwd_release");
        if (!init||!get_caps||!allocate||!shutdown||!destroy||!nr_init||!create||!evaluate||!release) { fprintf(stderr,"Missing neural runtime export\n"); goto cleanup; }
        int init_result=init(0x24480451ull,L".",device,0x15,NULL);
        printf("CoreInit => 0x%x\n",init_result); fflush(stdout);
        if (init_result!=1) { goto cleanup; } core_ready=1;
        int caps_result=get_caps(&caps); printf("GetCapabilityParameters => 0x%x nonnull=%d\n",caps_result,caps!=NULL); fflush(stdout);
        if (caps_result!=1||!caps) { goto cleanup; }
        int nr_result=nr_init(L"nvngx_dlssnr.dll",L".",device,caps); printf("NeuralInit => 0x%x\n",nr_result); fflush(stdout);
        if (nr_result!=1) { goto cleanup; }
        for (unsigned int eye=0;eye<2;++eye) {
            set_uint(caps,"DLSSNR.Width",PROBE_SIZE); set_uint(caps,"DLSSNR.Height",PROBE_SIZE);
            set_uint(caps,"DLSSNR.Hint.Render.Preset",0); set_uint(caps,"DLSSNR.DepthInverted",0);
            set_uint(caps,"CreationNodeMask",1); set_uint(caps,"VisibilityNodeMask",1);
            set_uint(caps,"DLSSNR.Enabled",1); set_uint(caps,"DLSSNR.UICorrection",1);
            int result=0; features[eye]=create(list,caps,&result);
            printf("Feature[%u] => 0x%x handle=%p\n",eye,result,features[eye]); fflush(stdout);
            if (result!=1||!features[eye]||!prepare_eye(device,&eyes[eye],eye,allocate)) { goto cleanup; }
        }
        if (features[0]==features[1]) { goto cleanup; }
    } else {
        // Resource setup needs parameter allocator; use ordinary driver core only,
        // never load the neural model or forwarder in this baseline.
        core=LoadLibraryW(L"nvngx.dll"); if (!core) { goto cleanup; }
        AllocateParams allocate=(AllocateParams)(void*)GetProcAddress(core,"NVSDK_NGX_D3D12_AllocateParameters");
        destroy=(DestroyParams)(void*)GetProcAddress(core,"NVSDK_NGX_D3D12_DestroyParameters");
        if (!allocate||!destroy) { goto cleanup; }
        for (unsigned int eye=0;eye<2;++eye) { if (!prepare_eye(device,&eyes[eye],eye,allocate)) { goto cleanup; } }
    }
    gpu.interactive=1; if (!prepare_gpu_scene(device,eyes,&gpu)) { goto cleanup; }
    if ((model||city)&&!prepare_mesh_file(device,queue,eyes,&mesh,&mesh_pending,automotive?"automotive.bin":supplied?"supplied-city.bin":city?"city.bin":"model.bin")) { goto cleanup; }
    DX_CHECK(ID3D12GraphicsCommandList_Close(list));
    ID3D12CommandList* commands[]={(ID3D12CommandList*)list};
    pending=1; ID3D12CommandQueue_ExecuteCommandLists(queue,1,commands);
    DX_CHECK(ID3D12CommandQueue_Signal(queue,fence,1)); DX_CHECK(ID3D12Fence_SetEventOnCompletion(fence,1,event));
    if (WaitForSingleObject(event,30000)!=WAIT_OBJECT_0||ID3D12Fence_GetCompletedValue(fence)==UINT64_MAX) { goto cleanup; } pending=0;
    ULONGLONG start=GetTickCount64(); XrView previous[2]={{XR_TYPE_VIEW},{XR_TYPE_VIEW}}; int history=0; unsigned int frame_attempts=0; XrTime recenter_time=0;
    printf("XR READY neural=%d fixed_resolution=%u interactive=%d exit=right_B\n",neural,PROBE_SIZE,interactive); fflush(stdout);
    while (interactive||(GetTickCount64()-start<120000 && frames<frame_limit)) {
        XrEventDataBuffer ev={XR_TYPE_EVENT_DATA_BUFFER}; XrResult er;
        while ((er=xrPollEvent(instance,&ev))==XR_SUCCESS) {
            if (ev.type==XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
                XrSessionState state=((XrEventDataSessionStateChanged*)&ev)->state;
                printf("XR STATE %d\n",state); fflush(stdout);
                int next_focused=state==XR_SESSION_STATE_FOCUSED;
                if (focused!=next_focused) {
                    focused=next_focused; history=0; input_ready=0;
                    locomotion.last_time=0; locomotion.move_ready=0; locomotion.turn_ready=0; locomotion.snap_armed=0;
                }
                if (state==XR_SESSION_STATE_READY) { XrSessionBeginInfo bi={XR_TYPE_SESSION_BEGIN_INFO}; bi.primaryViewConfigurationType=XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO; XR_CHECK(xrBeginSession(session,&bi)); running=1; history=0; }
                if (state==XR_SESSION_STATE_STOPPING) { XR_CHECK(xrEndSession(session)); running=0; history=0; last_completion.QuadPart=0; if (exit_requested) { goto finished; } }
                if (state==XR_SESSION_STATE_EXITING||state==XR_SESSION_STATE_LOSS_PENDING) { goto finished; }
            } else if (ev.type==XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING) {
                XrEventDataReferenceSpaceChangePending* change=(XrEventDataReferenceSpaceChangePending*)&ev;
                if (change->referenceSpaceType==XR_REFERENCE_SPACE_TYPE_LOCAL) { recenter_time=change->changeTime; }
            } else if (ev.type==XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) { goto finished; }
            ev=(XrEventDataBuffer){XR_TYPE_EVENT_DATA_BUFFER};
        }
        if (er!=XR_EVENT_UNAVAILABLE) { fprintf(stderr,"PollEvent failed %d\n",er); goto cleanup; }
        if (!running) { last_completion.QuadPart=0; Sleep(10); continue; }
        LARGE_INTEGER wait_begin,work_begin,completion; QueryPerformanceCounter(&wait_begin);
        XrFrameWaitInfo wi={XR_TYPE_FRAME_WAIT_INFO}; XrFrameState fs={XR_TYPE_FRAME_STATE}; XR_CHECK(xrWaitFrame(session,&wi,&fs));
        QueryPerformanceCounter(&work_begin);
        ++frame_attempts;
        XrVector2f move={0},turn={0}; int move_active=0,turn_active=0;
        if (room) {
            int pressed=0,light_pressed=0,trigger_active=0,light_active=0,input_available=0;
            if (focused) {
                XrActiveActionSet active={action_set,XR_NULL_PATH};
                XrActionsSyncInfo sync={XR_TYPE_ACTIONS_SYNC_INFO}; sync.countActiveActionSets=1; sync.activeActionSets=&active;
                XrResult sync_result=xrSyncActions(session,&sync);
                if (XR_FAILED(sync_result)) { fprintf(stderr,"Action sync failed %d\n",sync_result); goto cleanup; }
                if (sync_result==XR_SUCCESS) {
                    if (interactive&&!exit_requested) {
                        XrActionStateGetInfo exit_get={XR_TYPE_ACTION_STATE_GET_INFO}; exit_get.action=exit_action; exit_get.subactionPath=hands[1];
                        XrActionStateBoolean exit_state={XR_TYPE_ACTION_STATE_BOOLEAN}; XR_CHECK(xrGetActionStateBoolean(session,&exit_get,&exit_state));
                        if (exit_state.isActive&&exit_state.currentState&&exit_state.changedSinceLastSync) {
                            XR_CHECK(xrRequestExitSession(session)); exit_requested=1;
                            printf("DEMO EXIT requested_by=right_B\n"); fflush(stdout);
                        }
                    }
                    input_available=1;
                    XrActionStateGetInfo vector_get={XR_TYPE_ACTION_STATE_GET_INFO}; vector_get.action=move_action; vector_get.subactionPath=hands[0];
                    XrActionStateVector2f vector={XR_TYPE_ACTION_STATE_VECTOR2F}; XR_CHECK(xrGetActionStateVector2f(session,&vector_get,&vector));
                    move=vector.currentState; move_active=vector.isActive;
                    vector_get.action=turn_action; vector_get.subactionPath=hands[1]; vector=(XrActionStateVector2f){XR_TYPE_ACTION_STATE_VECTOR2F};
                    XR_CHECK(xrGetActionStateVector2f(session,&vector_get,&vector)); turn=vector.currentState; turn_active=vector.isActive;
                    if (frame_attempts%60==0) { printf("DEMO STICKS move_active=%d move=%.3f,%.3f turn_active=%d turn=%.3f,%.3f\n",move_active,move.x,move.y,turn_active,turn.x,turn.y); }
                    for (unsigned int hand=0;hand<2;++hand) {
                        XrActionStateGetInfo get={XR_TYPE_ACTION_STATE_GET_INFO}; get.subactionPath=hands[hand]; get.action=trigger;
                        XrActionStateFloat state={XR_TYPE_ACTION_STATE_FLOAT}; XR_CHECK(xrGetActionStateFloat(session,&get,&state));
                        if (state.isActive) { trigger_active=1; }
                        if (state.isActive && state.currentState>(trigger_down?.35f:.7f)) { pressed=1; }
                        get.action=light_action;
                        XrActionStateBoolean light={XR_TYPE_ACTION_STATE_BOOLEAN}; XR_CHECK(xrGetActionStateBoolean(session,&get,&light));
                        if (light.isActive) { light_active=1; }
                        if (light.isActive&&light.currentState) { light_pressed=1; }
                    }
                }
            }
            int auto_toggle=smoke&&((frames>=200&&smoke_stage==0)||(frames>=400&&smoke_stage==1));
            input_available=input_available&&trigger_active&&light_active;
            if (!input_available) { input_ready=0; }
            int controller_toggle=!smoke&&input_available&&input_ready&&pressed&&!trigger_down;
            int controller_light=!smoke&&input_available&&input_ready&&light_pressed&&!light_down;
            if (controller_toggle||auto_toggle) {
                neural=!neural; history=0; ++toggles;
                if (auto_toggle) { ++smoke_stage; }
                printf("DEMO TOGGLE frame=%u neural=%d source=%s\n",frames,neural,auto_toggle?"smoke":"controller"); fflush(stdout);
            }
            if (controller_light||(smoke&&frames>=300&&!smoke_light)) {
                lighting=1-lighting; history=0;
                if (smoke) { smoke_light=1; }
                printf("DEMO LIGHT frame=%u lighting=%u\n",frames,lighting); fflush(stdout);
            }
            if (input_available) { trigger_down=pressed; light_down=light_pressed; input_ready=1; }
        }
        if (recenter_time && fs.predictedDisplayTime>=recenter_time) { history=0; recenter_time=0; locomotion.last_time=0; locomotion.move_ready=0; locomotion.turn_ready=0; locomotion.snap_armed=0; }
        XrFrameBeginInfo bi={XR_TYPE_FRAME_BEGIN_INFO}; XR_CHECK(xrBeginFrame(session,&bi));
        XrCompositionLayerProjectionView projection[2]={{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW},{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}};
        XrCompositionLayerProjection layer={XR_TYPE_COMPOSITION_LAYER_PROJECTION}; layer.space=space; layer.viewCount=2; layer.views=projection;
        XrView views[2]={{XR_TYPE_VIEW},{XR_TYPE_VIEW}}; XrViewState vs={XR_TYPE_VIEW_STATE}; uint32_t count=0;
        int submit=0;
        if (fs.shouldRender) {
            XrViewLocateInfo li={XR_TYPE_VIEW_LOCATE_INFO}; li.viewConfigurationType=XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO; li.displayTime=fs.predictedDisplayTime; li.space=space;
            XR_CHECK(xrLocateViews(session,&li,&vs,2,&count,views));
            if (count==2 && (vs.viewStateFlags&(XR_VIEW_STATE_POSITION_VALID_BIT|XR_VIEW_STATE_ORIENTATION_VALID_BIT))==(XR_VIEW_STATE_POSITION_VALID_BIT|XR_VIEW_STATE_ORIENTATION_VALID_BIT)) {
                if (room) { update_locomotion(&locomotion,views,move,turn,move_active,turn_active,fs.predictedDisplayTime,smoke,frames,&history); }
                XrView virtual_views[2]={virtual_view(&views[0],room,&locomotion),virtual_view(&views[1],room,&locomotion)};
                DX_CHECK(ID3D12CommandAllocator_Reset(allocator)); DX_CHECK(ID3D12GraphicsCommandList_Reset(list,allocator,NULL));
                uint32_t acquired[2]={0,0};
                for (unsigned int eye=0;eye<2;++eye) {
                    XrSwapchainImageAcquireInfo ai={XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO}; XR_CHECK(xrAcquireSwapchainImage(chains[eye],&ai,&acquired[eye]));
                    XrSwapchainImageWaitInfo sw={XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO}; sw.timeout=1000000000;
                    XrResult wait=xrWaitSwapchainImage(chains[eye],&sw); if (wait!=XR_SUCCESS) { fprintf(stderr,"Swapchain wait %d\n",wait); goto cleanup; }
                    view_constants(&gpu,eye,frames,&virtual_views[eye],history?&previous[eye]:&virtual_views[eye],neural,room,city,lighting);
                    ID3D12GraphicsCommandList_EndQuery(list,queries,D3D12_QUERY_TYPE_TIMESTAMP,eye*4);
                    record_gpu_scene(list,&eyes[eye],&gpu,eye,frames);
                    if (model||city) { record_mesh(list,&eyes[eye],&gpu,eye,&mesh); }
                    ID3D12GraphicsCommandList_EndQuery(list,queries,D3D12_QUERY_TYPE_TIMESTAMP,eye*4+1);
                    transition(list,eyes[eye].textures[3],eyes[eye].states[3],D3D12_RESOURCE_STATE_UNORDERED_ACCESS); eyes[eye].states[3]=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                    if (neural) { set_uint(eyes[eye].params,"DLSSNR.Reset",history?0:1); if (evaluate(list,features[eye],eyes[eye].params)!=1) { fprintf(stderr,"Neural eye %u failed\n",eye); goto cleanup; } }
                    ID3D12GraphicsCommandList_EndQuery(list,queries,D3D12_QUERY_TYPE_TIMESTAMP,eye*4+2);
                    compose_eye(list,&eyes[eye],&gpu,eye,images[eye][acquired[eye]].texture);
                    ID3D12GraphicsCommandList_EndQuery(list,queries,D3D12_QUERY_TYPE_TIMESTAMP,eye*4+3);
                    projection[eye].pose=views[eye].pose; projection[eye].fov=views[eye].fov;
                    projection[eye].subImage.swapchain=chains[eye]; projection[eye].subImage.imageRect.extent=(XrExtent2Di){PROBE_SIZE,PROBE_SIZE};
                }
                ID3D12GraphicsCommandList_ResolveQueryData(list,queries,D3D12_QUERY_TYPE_TIMESTAMP,0,8,timestamp_readback,0);
                DX_CHECK(ID3D12GraphicsCommandList_Close(list)); pending=1;
                ID3D12CommandQueue_ExecuteCommandLists(queue,1,commands);
                DX_CHECK(ID3D12CommandQueue_Signal(queue,fence,(UINT64)frames+2)); DX_CHECK(ID3D12Fence_SetEventOnCompletion(fence,(UINT64)frames+2,event));
                if (WaitForSingleObject(event,30000)!=WAIT_OBJECT_0||ID3D12Fence_GetCompletedValue(fence)==UINT64_MAX) { goto cleanup; } pending=0;
                UINT64* ticks=NULL; D3D12_RANGE timing_range={0,8*sizeof(UINT64)};
                DX_CHECK(ID3D12Resource_Map(timestamp_readback,0,&timing_range,(void**)&ticks));
                FrameTiming* timing=&timings[interactive?0u:frames]; timing->neural=neural;
                int valid_ticks=1;
                for (unsigned int i=1;i<8;++i) { if (ticks[i]<ticks[i-1]) { valid_ticks=0; } }
                for (unsigned int eye=0;eye<2;++eye) {
                    timing->scene_ms+=1000.0*(double)(ticks[eye*4+1]-ticks[eye*4])/timestamp_frequency;
                    timing->neural_ms+=1000.0*(double)(ticks[eye*4+2]-ticks[eye*4+1])/timestamp_frequency;
                    timing->compose_ms+=1000.0*(double)(ticks[eye*4+3]-ticks[eye*4+2])/timestamp_frequency;
                }
                timing->gpu_ms=1000.0*(double)(ticks[7]-ticks[0])/timestamp_frequency;
                D3D12_RANGE written={0,0}; ID3D12Resource_Unmap(timestamp_readback,0,&written);
                if (!valid_ticks||!isfinite(timing->gpu_ms)||timing->gpu_ms<=0) { fprintf(stderr,"Invalid GPU timing\n"); goto cleanup; }
                timing->wait_ms=elapsed_ms(wait_begin,work_begin,cpu_frequency);
                timing->period_ms=(double)fs.predictedDisplayPeriod/1000000.0;
                for (unsigned int eye=0;eye<2;++eye) { XrSwapchainImageReleaseInfo ri={XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO}; XR_CHECK(xrReleaseSwapchainImage(chains[eye],&ri)); previous[eye]=virtual_views[eye]; }
                history=1; submit=1; ++frames;
            } else { history=0; locomotion.last_time=0; locomotion.move_ready=0; locomotion.turn_ready=0; locomotion.snap_armed=0; }
        } else { history=0; locomotion.last_time=0; locomotion.move_ready=0; locomotion.turn_ready=0; locomotion.snap_armed=0; }
        const XrCompositionLayerBaseHeader* layers[]={(const XrCompositionLayerBaseHeader*)&layer};
        XrFrameEndInfo end={XR_TYPE_FRAME_END_INFO}; end.displayTime=fs.predictedDisplayTime; end.environmentBlendMode=XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        end.layerCount=submit?1:0; end.layers=submit?layers:NULL; XR_CHECK(xrEndFrame(session,&end));
        if (submit) {
            QueryPerformanceCounter(&completion);
            timings[interactive?0u:frames-1].frame_wall_ms=elapsed_ms(work_begin,completion,cpu_frequency);
            timings[interactive?0u:frames-1].cadence_ms=last_completion.QuadPart?elapsed_ms(last_completion,completion,cpu_frequency):0;
            last_completion=completion;
            if (interactive) {
                FrameTiming* t=&timings[0];
                printf("TIMING %u,%u,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",frames,PROBE_SIZE,t->neural,t->scene_ms,t->neural_ms,t->compose_ms,t->gpu_ms,t->frame_wall_ms,t->wait_ms,t->cadence_ms,t->period_ms);
                memset(t,0,sizeof(*t));
            }
        } else { last_completion.QuadPart=0; }
        if (frame_attempts<=3) { printf("XR FRAME attempt=%u shouldRender=%u views=%u flags=0x%llx\n",frame_attempts,fs.shouldRender,count,(unsigned long long)vs.viewStateFlags); fflush(stdout); }
        if (submit && frames%60==0) { printf("XR SUBMITTED frames=%u neural=%d\n",frames,neural); fflush(stdout); }
    }
finished:
    if (smoke&&(frames!=600||toggles!=2||smoke_stage!=2||!smoke_light||lighting!=1||locomotion.moves<30||locomotion.snaps!=1)) { fprintf(stderr,"Incomplete room smoke test\n"); goto cleanup; }
    if (room) { printf("DEMO COMPLETE frames=%u toggles=%u lighting=%u locomotion_moves=%u snap_turns=%u bounds_only=1\n",frames,toggles,lighting,locomotion.moves,locomotion.snaps); }
    if (frames>0||exit_requested) { code=0; }
cleanup:
    city_collision_free(&collision);
    if (mesh_pending) { ExitProcess((UINT)code); }
    if (!code&&!interactive) {
        printf("TIMING_HEADER frame,resolution,neural,scene_ms,neural_ms,compose_ms,gpu_ms,frame_wall_ms,wait_ms,cadence_ms,period_ms\n");
        for (unsigned int i=0;i<frames;++i) {
            FrameTiming* t=&timings[i];
            printf("TIMING %u,%u,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",i+1,PROBE_SIZE,t->neural,t->scene_ms,t->neural_ms,t->compose_ms,t->gpu_ms,t->frame_wall_ms,t->wait_ms,t->cadence_ms,t->period_ms);
        }
    }
    if (pending) { fprintf(stderr,"GPU completion uncertain; process exit owns cleanup\n"); ExitProcess(1); }
    if (release) { for (unsigned int eye=0;eye<2;++eye) { if (features[eye] && (eye==0 || features[eye]!=features[0])) { release(features[eye]); } } }
    for (unsigned int eye=0;eye<2;++eye) { if (destroy && !destroy_eye(&eyes[eye],destroy)) { code=1; } }
    destroy_mesh(&mesh);
    destroy_gpu_scene(&gpu);
    if (core_ready) { int live=-1; if (shutdown(device,&live)!=1||live!=0) { code=1; } }
    // Runtime images are borrowed: never release their D3D resource interfaces.
    for (unsigned int eye=0;eye<2;++eye) { free(images[eye]); if (chains[eye]) { xrDestroySwapchain(chains[eye]); } }
    if (space) { xrDestroySpace(space); } if (session) { xrDestroySession(session); } if (action_set) { xrDestroyActionSet(action_set); } if (instance) { xrDestroyInstance(instance); }
    if (event) { CloseHandle(event); } if (list) { ID3D12GraphicsCommandList_Release(list); }
    if (allocator) { ID3D12CommandAllocator_Release(allocator); } if (fence) { ID3D12Fence_Release(fence); }
    if (timestamp_readback) { ID3D12Resource_Release(timestamp_readback); }
    if (queries) { ID3D12QueryHeap_Release(queries); }
    if (queue) { ID3D12CommandQueue_Release(queue); } if (device) { ID3D12Device_Release(device); }
    if (adapter) { IDXGIAdapter1_Release(adapter); } if (factory) { IDXGIFactory4_Release(factory); }
    if (!code) { printf("PASS: XR submitted frames=%u neural=%d. Visual/stereo quality and latency unverified.\n",frames,neural); }
    free(timings);
    return code;
}
