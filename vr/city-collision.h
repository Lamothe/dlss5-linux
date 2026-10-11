#ifndef CITY_COLLISION_H
#define CITY_COLLISION_H
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(float)==4,"Collision payload requires float32");
#define CITY_COLLISION_RADIUS .2f
#define CITY_COLLISION_MAX_BOXES 4096u
typedef struct CityCollisionBox { float min[3],max[3]; } CityCollisionBox;
typedef struct CityCollision { CityCollisionBox* boxes; uint32_t count; int custom_limits; float low[3],high[3]; } CityCollision;

static inline void city_collision_free(CityCollision* city) {
    free(city->boxes); city->boxes=NULL; city->count=0;
}
static inline uint32_t city_collision_u32(const unsigned char* bytes) {
    return (uint32_t)bytes[0]|((uint32_t)bytes[1]<<8)|((uint32_t)bytes[2]<<16)|((uint32_t)bytes[3]<<24);
}
// DLCOLL01, little-endian uint32 count, then min XYZ/max XYZ float32 records.
// XYZ are renderer Y-up coordinates, i.e. Blender (-X,Z,Y), proper rotation (same handedness).
static inline int city_collision_load(CityCollision* city,const char* path) {
    FILE* file=fopen(path,"rb"); unsigned char header[12]; CityCollision loaded={0}; int valid=0;
    if (!file) { return 0; }
    if (fread(header,1,sizeof(header),file)!=sizeof(header)||memcmp(header,"DLCOLL01",8)) { goto done; }
    loaded.count=city_collision_u32(header+8);
    if (!loaded.count||loaded.count>CITY_COLLISION_MAX_BOXES) { goto done; }
    loaded.boxes=calloc(loaded.count,sizeof(*loaded.boxes));
    if (!loaded.boxes) { goto done; }
    for (uint32_t i=0;i<loaded.count;++i) {
        unsigned char record[24]; float values[6];
        if (fread(record,1,sizeof(record),file)!=sizeof(record)) { goto done; }
        for (int j=0;j<6;++j) {
            uint32_t bits=city_collision_u32(record+4*j); memcpy(&values[j],&bits,sizeof(bits));
            if (!isfinite(values[j])||fabsf(values[j])>10000.f) { goto done; }
        }
        for (int axis=0;axis<3;++axis) {
            if (values[axis]>=values[axis+3]) { goto done; }
            loaded.boxes[i].min[axis]=values[axis]; loaded.boxes[i].max[axis]=values[axis+3];
        }
    }
    if (fgetc(file)!=EOF||ferror(file)) { goto done; }
    valid=1;
done:
    if (fclose(file)) { valid=0; }
    if (valid) { city_collision_free(city); *city=loaded; }
    else { city_collision_free(&loaded); }
    return valid;
}
// Conservative sphere envelope: expand each AABB by the 0.2 m head radius.
// Sweep rather than endpoint-test so thin walls cannot be skipped. If physical
// tracking starts inside a box, allow escape/tangential motion, never depenetrate.
static inline int city_collision_hit(const CityCollisionBox* box,const float p[3],const float d[3],float* fraction,int* axis_out,float* normal_out) {
    float lo[3],hi[3]; int inside=1;
    for (int a=0;a<3;++a) {
        lo[a]=box->min[a]-CITY_COLLISION_RADIUS; hi[a]=box->max[a]+CITY_COLLISION_RADIUS;
        if (p[a]<lo[a]||p[a]>hi[a]) { inside=0; }
    }
    if (inside) {
        float nearest=INFINITY,normal=0.f; int axis=0;
        for (int a=0;a<3;++a) {
            if (p[a]-lo[a]<nearest) { nearest=p[a]-lo[a]; axis=a; normal=-1.f; }
            if (hi[a]-p[a]<nearest) { nearest=hi[a]-p[a]; axis=a; normal=1.f; }
        }
        if (d[axis]*normal<0.f) { *fraction=0.f; *axis_out=axis; *normal_out=normal; return 1; }
        return 0;
    }
    float enter=0.f,exit=1.f,normal=0.f; int axis=-1;
    for (int a=0;a<3;++a) {
        if (fabsf(d[a])<1e-9f) { if (p[a]<lo[a]||p[a]>hi[a]) { return 0; } continue; }
        float first=(lo[a]-p[a])/d[a],last=(hi[a]-p[a])/d[a],n=-1.f;
        if (first>last) { float swap=first; first=last; last=swap; n=1.f; }
        if (first>=enter) { enter=first; axis=a; normal=n; }
        exit=fminf(exit,last);
        if (enter>exit) { return 0; }
    }
    if (axis<0||enter<0.f||enter>1.f||exit<0.f) { return 0; }
    *fraction=enter; *axis_out=axis; *normal_out=normal; return 1;
}
static inline void city_collision_move(const CityCollision* city,const float start[3],const float requested[3],float result[3]) {
    float p[3]={start[0],start[1],start[2]};
    float length=sqrtf(requested[0]*requested[0]+requested[1]*requested[1]+requested[2]*requested[2]);
    if (!isfinite(length)||length>100.f) { memset(result,0,3*sizeof(*result)); return; }
    int steps=(int)ceilf(length/.05f); if (steps<1) { steps=1; }
    for (int step=0;step<steps;++step) {
        float d[3]={requested[0]/steps,requested[1]/steps,requested[2]/steps};
        // Virtual-motion limits only. Tracking beyond a limit is not clamped.
        const float defaults_low[3]={-7.7f+CITY_COLLISION_RADIUS,.25f,-20.7f+CITY_COLLISION_RADIUS};
        const float defaults_high[3]={7.7f-CITY_COLLISION_RADIUS,15.f,41.7f-CITY_COLLISION_RADIUS};
        const float* low=city&&city->custom_limits?city->low:defaults_low;
        const float* high=city&&city->custom_limits?city->high:defaults_high;
        for (int a=0;a<3;++a) {
            if (d[a]<0.f) { d[a]=fmaxf(d[a],fminf(0.f,low[a]-p[a])); }
            else { d[a]=fminf(d[a],fmaxf(0.f,high[a]-p[a])); }
        }
        for (int slide=0;slide<3;++slide) {
            float fraction=1.f,normal=0.f; int axis=-1;
            if (city) { for (uint32_t i=0;i<city->count;++i) {
                float hit,n; int a;
                if (city_collision_hit(&city->boxes[i],p,d,&hit,&a,&n)&&hit<=fraction) { fraction=hit; axis=a; normal=n; }
            } }
            for (int a=0;a<3;++a) { p[a]+=d[a]*fraction; d[a]*=1.f-fraction; }
            if (axis<0) { break; }
            if (d[axis]*normal<0.f) { d[axis]=0.f; }
        }
    }
    for (int a=0;a<3;++a) { result[a]=p[a]-start[a]; }
}
#endif
