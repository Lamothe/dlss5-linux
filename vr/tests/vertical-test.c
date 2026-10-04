#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
typedef int64_t XrTime;
typedef struct {float x,y;} XrVector2f;
typedef struct {float x,y,z;} XrVector3f;
typedef struct {float x,y,z,w;} XrQuaternionf;
typedef struct {XrQuaternionf orientation;XrVector3f position;} Pose;
typedef struct {Pose pose;} XrView;
// Locomotion functions are inserted from xr-scene.c by the host test runner.
int main(void) {
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
 update_locomotion(&loc,eyes,(XrVector2f){0},(XrVector2f){0,1},0,0,time+=50000000,0,3,&history); assert(loc.y==before);
 update_locomotion(&loc,eyes,(XrVector2f){0},(XrVector2f){0,1},0,1,time+=50000000,0,3,&history); assert(loc.y==before);
 puts("PASS vertical up/down, speed, limits, inactive/reactivation, diagonal suppression, continuous history");
}
