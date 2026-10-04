// CPU ray-rendered, fixed-world scene. Metres, symmetric 90-degree FOV.
// This offscreen test does not use headset poses/FOV or submit OpenXR images.
typedef struct SceneVector { float x, y, z; } SceneVector;
static SceneVector scene_add(SceneVector a, SceneVector b) { return (SceneVector){a.x+b.x,a.y+b.y,a.z+b.z}; }
static SceneVector scene_scale(SceneVector a, float s) { return (SceneVector){a.x*s,a.y*s,a.z*s}; }
static float scene_dot(SceneVector a, SceneVector b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
static SceneVector scene_normalise(SceneVector a) { return scene_scale(a, 1.0f/sqrtf(scene_dot(a,a))); }
static float scene_camera_x(unsigned int frame, unsigned int eye) {
    return 0.12f*sinf((float)frame*0.05f) + (eye == 0 ? -0.032f : 0.032f);
}
static int render_scene(EyeInputs* eye, unsigned int index, unsigned int frame) {
    unsigned char* mapped[3] = {NULL,NULL,NULL};
    D3D12_RANGE no_read = {0,0};
    for (unsigned int i=0;i<3;++i) {
        if (FAILED(ID3D12Resource_Map(eye->uploads[i],0,&no_read,(void**)&mapped[i]))) {
            for (unsigned int j=0;j<i;++j) { ID3D12Resource_Unmap(eye->uploads[j],0,NULL); }
            return 0;
        }
    }
    float camera_x = scene_camera_x(frame,index);
    float previous_x = scene_camera_x(frame == 0 ? 0 : frame-1,index);
    SceneVector origin = {camera_x,1.4f,0};
    const SceneVector centres[2] = {{-0.45f,0.65f,2.6f},{0.65f,0.45f,3.4f}};
    const float radii[2] = {0.65f,0.45f};
    const SceneVector colours[2] = {{0.75f,0.18f,0.08f},{0.08f,0.28f,0.8f}};
    unsigned int foreground=0, moving=0;
    double depth_sum=0, motion_sum=0;
    for (unsigned int y=0;y<PROBE_SIZE;++y) {
        uint16_t* colour=(uint16_t*)(mapped[0]+eye->layouts[0].Offset+y*eye->layouts[0].Footprint.RowPitch);
        float* depth=(float*)(mapped[1]+eye->layouts[1].Offset+y*eye->layouts[1].Footprint.RowPitch);
        uint16_t* motion=(uint16_t*)(mapped[2]+eye->layouts[2].Offset+y*eye->layouts[2].Footprint.RowPitch);
        for (unsigned int x=0;x<PROBE_SIZE;++x) {
            float nx=2.0f*((float)x+0.5f)/PROBE_SIZE-1.0f;
            float ny=1.0f-2.0f*((float)y+0.5f)/PROBE_SIZE;
            SceneVector direction=scene_normalise((SceneVector){nx,ny,1});
            float distance=6.0f/direction.z;
            SceneVector normal={0,0,-1};
            SceneVector albedo={0.38f,0.4f,0.42f};
            if (direction.y < -0.0001f) {
                float floor_distance=-origin.y/direction.y;
                if (floor_distance<distance) {
                    distance=floor_distance; normal=(SceneVector){0,1,0};
                    SceneVector p=scene_add(origin,scene_scale(direction,distance));
                    int checker=((int)floorf(p.x*2)+(int)floorf(p.z*2))&1;
                    float shade=checker?0.5f:0.2f; albedo=(SceneVector){shade,shade,shade};
                }
            }
            for (unsigned int sphere=0;sphere<2;++sphere) {
                SceneVector offset=scene_add(origin,scene_scale(centres[sphere],-1));
                float b=scene_dot(offset,direction);
                float discriminant=b*b-scene_dot(offset,offset)+radii[sphere]*radii[sphere];
                if (discriminant>0) {
                    float t=-b-sqrtf(discriminant);
                    if (t>0.1f && t<distance) {
                        distance=t;
                        normal=scene_normalise(scene_add(scene_add(origin,scene_scale(direction,t)),scene_scale(centres[sphere],-1)));
                        albedo=colours[sphere]; ++foreground;
                    }
                }
            }
            SceneVector point=scene_add(origin,scene_scale(direction,distance));
            SceneVector light=scene_normalise((SceneVector){-0.5f,0.8f,-0.4f});
            float illumination=0.12f+0.65f*fmaxf(0,scene_dot(normal,light));
            // Encode display-referred proxy using sRGB transfer, not linear HDR values.
            float components[3]={albedo.x*illumination,albedo.y*illumination,albedo.z*illumination};
            for (unsigned int c=0;c<3;++c) {
                float linear=components[c];
                float encoded=linear<=0.0031308f?linear*12.92f:1.055f*powf(linear,1.0f/2.4f)-0.055f;
                colour[x*4+c]=float_half(encoded);
            }
            colour[x*4+3]=float_half(1);
            // D3D projection depth: +Z forward, normal depth, near0.1m far20m.
            depth[x]=20.0f/19.9f-2.0f/(19.9f*point.z);
            // Previous minus current projected position, in pixels. Stationary world,
            // parallel cameras translating in X: Y vector is exactly zero.
            float mx=(camera_x-previous_x)/point.z*0.5f*PROBE_SIZE;
            motion[x*2]=float_half(mx); motion[x*2+1]=0;
            depth_sum+=depth[x]; motion_sum+=fabsf(mx);
            if (fabsf(mx)>0.0001f) { ++moving; }
        }
    }
    for (unsigned int i=0;i<3;++i) { ID3D12Resource_Unmap(eye->uploads[i],0,NULL); }
    printf("Scene eye=%u frame=%u camera_x_m=%.6f foreground_hits=%u moving_pixels=%u mean_depth=%.6f mean_abs_motion_px=%.6f\n",
        index,frame,camera_x,foreground,moving,depth_sum/(PROBE_SIZE*PROBE_SIZE),motion_sum/(PROBE_SIZE*PROBE_SIZE));
    return foreground>0 && (frame==0 || moving>0);
}
static int write_scene_ppm(EyeInputs* eye, unsigned int index, unsigned int frame, int output) {
    unsigned char* mapped=NULL;
    ID3D12Resource* resource=output?eye->readback:eye->uploads[0];
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout=eye->layouts[output?3:0];
    D3D12_RANGE range={0,(SIZE_T)eye->sizes[output?3:0]};
    if (FAILED(ID3D12Resource_Map(resource,0,&range,(void**)&mapped))) { return 0; }
    char filename[100];
    snprintf(filename,sizeof(filename),"scene-eye%u-frame%u-%s.ppm",index,frame,output?"output":"input");
    FILE* file=fopen(filename,"wb");
    int success=file!=NULL;
    if (file) {
        if (fprintf(file,"P6\n%u %u\n255\n",PROBE_SIZE,PROBE_SIZE)<0) { success=0; }
        for (unsigned int y=0;y<PROBE_SIZE;++y) {
            uint16_t* row=(uint16_t*)(mapped+layout.Offset+y*layout.Footprint.RowPitch);
            for (unsigned int x=0;x<PROBE_SIZE;++x) {
                unsigned char rgb[3];
                for (unsigned int c=0;c<3;++c) {
                    float value=half_float(row[x*4+c]);
                    rgb[c]=(unsigned char)(255.0f*fminf(1,fmaxf(0,value))+0.5f);
                }
                if (fwrite(rgb,1,3,file)!=3) { success=0; }
            }
        }
        if (fclose(file)!=0) { success=0; }
    }
    D3D12_RANGE no_write={0,0}; ID3D12Resource_Unmap(resource,0,&no_write);
    return success;
}
