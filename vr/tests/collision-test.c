#include "../city-collision.h"
#include <assert.h>

static void write_u32(FILE* file,uint32_t value) {
    unsigned char bytes[4]={(unsigned char)value,(unsigned char)(value>>8),(unsigned char)(value>>16),(unsigned char)(value>>24)};
    assert(fwrite(bytes,1,4,file)==4);
}
static void payload(const char* path,int variant) {
    FILE* file=fopen(path,"wb"); assert(file);
    assert(fwrite(variant==1?"BADMAGIC":"DLCOLL01",1,8,file)==8);
    write_u32(file,variant==2?0u:variant==3?CITY_COLLISION_MAX_BOXES+1u:1u);
    const float values[6]={-.5f,0.f,1.f,.5f,2.f,1.001f};
    for (int i=0;i<(variant==4?5:6);++i) {
        float value=variant==5&&i==0?NAN:variant==6&&i==3?-.6f:variant==8&&i==3?INFINITY:values[i];
        uint32_t bits; memcpy(&bits,&value,4); write_u32(file,bits);
    }
    if (variant==7) { assert(fputc(0,file)!=EOF); }
    assert(!fclose(file));
}
static void move_check(const CityCollision* city,const float start[3],const float delta[3],float x,float y,float z) {
    float result[3]; city_collision_move(city,start,delta,result);
    assert(fabsf(start[0]+result[0]-x)<2e-4f);
    assert(fabsf(start[1]+result[1]-y)<2e-4f);
    assert(fabsf(start[2]+result[2]-z)<2e-4f);
}
int main(int argc,char** argv) {
    assert(argc==2); const char* path=argv[1]; CityCollision city={0};
    assert(!city_collision_load(&city,path));
    payload(path,0); assert(city_collision_load(&city,path)); assert(city.count==1);
    CityCollisionBox* original=city.boxes;
    for (int variant=1;variant<=8;++variant) {
        payload(path,variant); assert(!city_collision_load(&city,path)); assert(city.boxes==original&&city.count==1);
    }
    // Thin wall cannot be tunnelled through, even by a large movement.
    move_check(&city,(float[3]){0,1,0},(float[3]){0,0,10},0,1,.8f);
    move_check(&city,(float[3]){0,1,2},(float[3]){0,0,-10},0,1,1.201f);
    // Slide along the wall, then proceed once clear of the car-sized AABB.
    float result[3]; city_collision_move(&city,(float[3]){0,1,0},(float[3]){1,0,2},result);
    assert(result[0]>.99f && result[2]>.8f);
    // Passing above the obstacle, and vertical entry from above/below.
    move_check(&city,(float[3]){0,2.3f,0},(float[3]){0,0,3},0,2.3f,3);
    move_check(&city,(float[3]){0,3,1},(float[3]){0,-2,0},0,2.2f,1);
    // Tracking already overlaps geometry: idle is unchanged; escape is allowed.
    move_check(&city,(float[3]){0,1,1},(float[3]){0,0,0},0,1,1);
    move_check(&city,(float[3]){0,1,1},(float[3]){0,0,-1},0,1,0);
    // Bounds affect virtual displacement only, including recovery from tracking
    // outside the city, rather than teleporting the physical head back inside.
    move_check(NULL,(float[3]){20,1,0},(float[3]){0,0,0},20,1,0);
    move_check(NULL,(float[3]){20,1,0},(float[3]){-1,0,0},19,1,0);
    move_check(NULL,(float[3]){20,1,0},(float[3]){1,0,0},20,1,0);
    move_check(NULL,(float[3]){0,1,0},(float[3]){10,0,50},7.5f,1,41.5f);
    // Two simultaneous perpendicular walls stop corner entry.
    CityCollisionBox walls[2]={{{1,-1,-10},{1.001f,4,10}},{{-10,-1,1},{10,4,1.001f}}};
    CityCollision corner={.boxes=walls,.count=2}; move_check(&corner,(float[3]){0,1,0},(float[3]){5,0,5},.8f,1,.8f);
    city_collision_free(&city); assert(!city.boxes&&!city.count); assert(!remove(path));
    puts("PASS: collision metadata validation, head radius, swept thin-wall/corner blocking, sliding, vertical clearance and tracking escape");
    return 0;
}
