#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
// Set C_INCLUDE_PATH=vr for the extracted host test runner.
#include "city-collision.h"
typedef int64_t XrTime;
typedef struct {float x,y;} XrVector2f;
typedef struct {float x,y,z;} XrVector3f;
typedef struct {float x,y,z,w;} XrQuaternionf;
typedef struct {XrQuaternionf orientation;XrVector3f position;} Pose;
typedef struct {Pose pose;} XrView;
// Locomotion functions are inserted from xr-scene.c by the host test runner.
int main(void) {
 CityCollision supplied={.custom_limits=1,.low={-38,.25f,-26},.high={15,29,26}};
 float moved[3];
 city_collision_move(&supplied,(float[3]){10,1.6f,0},(float[3]){.1f,0,0},moved);
 assert(fabsf(moved[0]-.1f)<.00001f);
 city_collision_move(&supplied,(float[3]){-25,1.6f,-23},(float[3]){-1,0,-1},moved);
 assert(fabsf(moved[0]+1)<.0001f&&fabsf(moved[2]+1)<.0001f);
 city_collision_move(&supplied,(float[3]){14.9f,1.6f,0},(float[3]){1,0,0},moved);
 assert(fabsf(moved[0]-.1f)<.0001f);

 XrView eyes[2]={0}; eyes[0].pose.orientation.w=eyes[1].pose.orientation.w=1;
 Locomotion loc={0}; int history=1; XrTime time=1000000000;
 update_locomotion(&loc,eyes,(XrVector2f){0},(XrVector2f){0,1},0,1,time,0,0,&history);
 assert(loc.y==0); time+=50000000;
 update_locomotion(&loc,eyes,(XrVector2f){0},(XrVector2f){0,1},0,1,time,0,1,&history);
 assert(fabsf(loc.y-.025f)<1e-6f); assert(history==1);
 float before=loc.y;
 update_locomotion(&loc,eyes,(XrVector2f){0},(XrVector2f){0,-1},0,1,time+=50000000,0,2,&history);
 assert(loc.y<before); assert(fabsf(loc.y)<1e-6f);
 for(int i=0;i<200;i++) update_locomotion(&loc,eyes,(XrVector2f){0},(XrVector2f){0,1},0,1,time+=50000000,0,3,&history);
 assert(fabsf(virtual_view(&eyes[0],1,&loc).pose.position.y-2.95f)<1e-5f);
 for(int i=0;i<300;i++) update_locomotion(&loc,eyes,(XrVector2f){0},(XrVector2f){0,-1},0,1,time+=50000000,0,3,&history);
 assert(fabsf(virtual_view(&eyes[0],1,&loc).pose.position.y-.25f)<1e-5f);
 before=loc.y;
 update_locomotion(&loc,eyes,(XrVector2f){0},(XrVector2f){1,1},0,1,time+=50000000,0,3,&history);
 assert(loc.y==before);
 update_locomotion(&loc,eyes,(XrVector2f){0},(XrVector2f){.3f,1},0,1,time+=50000000,0,3,&history); assert(loc.y==before);
 update_locomotion(&loc,eyes,(XrVector2f){0},(XrVector2f){0,1},0,0,time+=50000000,0,3,&history); assert(loc.y==before);
 update_locomotion(&loc,eyes,(XrVector2f){0},(XrVector2f){0,1},0,1,time+=50000000,0,3,&history); assert(loc.y==before);
 loc=(Locomotion){.city=1}; time=1000000000;
 update_locomotion(&loc,eyes,(XrVector2f){0,1},(XrVector2f){0,1},1,1,time,0,4,&history);
 update_locomotion(&loc,eyes,(XrVector2f){0,1},(XrVector2f){0,1},1,1,time+=50000000,0,4,&history);
 assert(fabsf(loc.z-.1f)<1e-5f); assert(fabsf(loc.y-.025f)<1e-5f);
 for(int i=0;i<900;i++) update_locomotion(&loc,eyes,(XrVector2f){0,1},(XrVector2f){0,1},1,1,time+=50000000,0,4,&history);
 XrView city=virtual_view(&eyes[0],1,&loc); assert(fabsf(city.pose.position.z-41.5f)<1e-4f); assert(fabsf(city.pose.position.y-15)<1e-4f);
 for(int i=0;i<900;i++) update_locomotion(&loc,eyes,(XrVector2f){0,-1},(XrVector2f){0,-1},1,1,time+=50000000,0,4,&history);
 city=virtual_view(&eyes[0],1,&loc); assert(fabsf(city.pose.position.z+20.5f)<1e-4f); assert(fabsf(city.pose.position.y-.25f)<1e-4f);
 for(int i=0;i<500;i++) update_locomotion(&loc,eyes,(XrVector2f){1,0},(XrVector2f){0,0},1,1,time+=50000000,0,4,&history);
 city=virtual_view(&eyes[0],1,&loc); assert(fabsf(city.pose.position.x+7.5f)<1e-4f);
 for(int i=0;i<500;i++) update_locomotion(&loc,eyes,(XrVector2f){-1,0},(XrVector2f){0,0},1,1,time+=50000000,0,4,&history);
 city=virtual_view(&eyes[0],1,&loc); assert(fabsf(city.pose.position.x-7.5f)<1e-4f);
 // Smooth yaw accumulates over held input, preserving history and head pivot.
 loc=(Locomotion){.city=1,.turn_ready=1}; eyes[0].pose.position.x=eyes[1].pose.position.x=.4f;
 XrView before_turn=virtual_view(&eyes[0],1,&loc);
 history=1; loc.last_time=time;
 for(int i=0;i<10;i++) update_locomotion(&loc,eyes,(XrVector2f){0},(XrVector2f){1,0},0,1,time+=50000000,0,4,&history);
 assert(loc.snaps==0 && history==1); assert(fabsf(loc.yaw+3.14159265358979323846f/6)<1e-5f);
 XrView after_turn=virtual_view(&eyes[0],1,&loc);
 assert(fabsf(after_turn.pose.position.x-before_turn.pose.position.x)<1e-5f);
 assert(fabsf(after_turn.pose.position.z-before_turn.pose.position.z)<1e-5f);
 float yaw=loc.yaw;
 update_locomotion(&loc,eyes,(XrVector2f){0},(XrVector2f){.1f,0},0,1,time+=50000000,0,4,&history); assert(loc.yaw==yaw);
 update_locomotion(&loc,eyes,(XrVector2f){0},(XrVector2f){1,0},0,0,time+=50000000,0,4,&history); assert(loc.yaw==yaw);
 update_locomotion(&loc,eyes,(XrVector2f){0},(XrVector2f){1,0},0,1,time+=50000000,0,4,&history); assert(loc.yaw==yaw);
 puts("PASS: city collision/physical tracking and room movement/height/bounds; smooth 60-degree/s yaw, deadzone, activation, head pivot and history");
}
