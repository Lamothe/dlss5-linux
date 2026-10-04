// Shared-world GPU ray scene. Metres, +Z forward, symmetric square 90-degree FOV.
RWTexture2D<float4> Colour : register(u0);
RWTexture2D<float> Depth : register(u1);
RWTexture2D<float2> Motion : register(u2);
RWTexture2D<float4> NeuralOutput : register(u3);
RWTexture2D<float4> Presented : register(u4);
cbuffer Scene : register(b0) {
 uint Eye; uint Frame; uint Width; uint Height;
 uint Tracked; uint Neural; uint Pad0; uint Pad1; // Pad0: room=1, spatial AA=2.
 float4 Rotation; float4 Position;
 float4 PreviousRotation; float4 PreviousPosition;
 float4 Tangents; float4 PreviousTangents;
};
float3 rotateVector(float4 q,float3 v) { return v+2*cross(q.xyz,cross(q.xyz,v)+q.w*v); }
float3 inverseRotate(float4 q,float3 v) { return rotateVector(float4(-q.xyz,q.w),v); }
float3 pixelRay(float2 pixel);
float cameraX(uint f);
float3 roomStatus(float3 colour,float3 origin,float3 direction,float2 pixel);
// Compose reads only its selected source; never filter Presented in place.
float3 composeLoad(int2 pixel) {
    pixel=clamp(pixel,int2(0,0),int2(Width,Height)-1);
    return max(0,Neural!=0 ? NeuralOutput[pixel].rgb : Colour[pixel].rgb);
}
// RW textures have no sampler: integer coordinates denote texel centres.
float3 composeBilinear(float2 pixel) {
    pixel=clamp(pixel,float2(0,0),float2(Width,Height)-1);
    int2 base=int2(floor(pixel)); float2 weight=frac(pixel);
    return lerp(lerp(composeLoad(base),composeLoad(base+int2(1,0)),weight.x),
                lerp(composeLoad(base+int2(0,1)),composeLoad(base+int2(1,1)),weight.x),weight.y);
}
float composeLuma(float3 encoded) { return dot(encoded,float3(0.299,0.587,0.114)); }
// FXAA-like spatial edge filter in sRGB space, with no temporal history.
float3 composeEdgeAA(int2 pixel,float3 centre) {
    float nw=composeLuma(composeLoad(pixel+int2(-1,-1)));
    float ne=composeLuma(composeLoad(pixel+int2(1,-1)));
    float sw=composeLuma(composeLoad(pixel+int2(-1,1)));
    float se=composeLuma(composeLoad(pixel+int2(1,1)));
    float middle=composeLuma(centre);
    float low=min(middle,min(min(nw,ne),min(sw,se)));
    float high=max(middle,max(max(nw,ne),max(sw,se)));
    if (high-low<max(0.0312,high*0.125)) { return centre; }
    float2 direction=float2(-((nw+ne)-(sw+se)),(nw+sw)-(ne+se));
    float reduce=max((nw+ne+sw+se)*0.25*(1.0/8.0),1.0/128.0);
    direction=clamp(direction/(min(abs(direction.x),abs(direction.y))+reduce),-8,8);
    float2 position=float2(pixel);
    float3 candidateA=0.5*(composeBilinear(position-direction*(1.0/6.0))+
                           composeBilinear(position+direction*(1.0/6.0)));
    float3 candidateB=0.5*candidateA+0.25*(composeBilinear(position-direction*0.5)+
                                         composeBilinear(position+direction*0.5));
    float lumaB=composeLuma(candidateB);
    return lumaB<low || lumaB>high ? candidateA : candidateB;
}
[numthreads(8,8,1)]
void compose(uint3 id : SV_DispatchThreadID) {
 if (id.x>=Width || id.y>=Height) { return; }
 float3 encoded=max(0, Neural != 0 ? NeuralOutput[id.xy].rgb : Colour[id.xy].rgb);
 if ((Pad0&2)!=0) { encoded=composeEdgeAA(int2(id.xy),encoded); }
 float3 decoded=float3(encoded.x<=0.04045?encoded.x/12.92:pow((encoded.x+0.055)/1.055,2.4),
                       encoded.y<=0.04045?encoded.y/12.92:pow((encoded.y+0.055)/1.055,2.4),
                       encoded.z<=0.04045?encoded.z/12.92:pow((encoded.z+0.055)/1.055,2.4));
 if ((Pad0&1)!=0) {
     float3 origin=Tracked!=0?Position.xyz:float3(cameraX(Frame),1.4,0);
     decoded=roomStatus(decoded,origin,normalize(pixelRay(float2(id.xy))),float2(id.xy));
 }
 Presented[id.xy]=float4(decoded,1);
}
float cameraX(uint f) { return 0.12 * sin(float(f) * 0.05) + (Eye == 0 ? -0.032 : 0.032); }
// Integral of a period-two square wave (+1 on even cells, -1 on odd).
float checkerIntegral(float x) { return 1-abs(1-(x-2*floor(x/2))); }
float checkerAverage(float centre,float width) {
    if (width<0.0001) { return (int(floor(centre)) & 1) != 0 ? -1 : 1; }
    return clamp((checkerIntegral(centre+width*0.5)-checkerIntegral(centre-width*0.5))/width,-1,1);
}
float3 pixelRay(float2 pixel) {
    float2 uv=(pixel+0.5)/float2(Width,Height);
    if (Tracked != 0) { return rotateVector(Rotation,float3(lerp(Tangents.x,Tangents.y,uv.x),lerp(Tangents.z,Tangents.w,uv.y),-1)); }
    return float3(2*uv.x-1,1-2*uv.y,1);
}
// Pad0 bit 0 selects this static furnished room; Pad1 selects cool daylight / warm light.
// All traces are analytic and bounded. Parallel slabs never divide by zero.
struct RoomHit { float distance; float3 normal; uint material; float3 position; float3 colour; };
bool roomSlab(float origin,float direction,float low,float high,float3 axis,
              inout float nearT,inout float farT,inout float3 nearN,inout float3 farN) {
    if (abs(direction)<0.000001) { return origin>=low && origin<=high; }
    float a=(low-origin)/direction, b=(high-origin)/direction;
    float3 n=-axis, f=axis;
    if (a>b) { float swap=a; a=b; b=swap; n=axis; f=-axis; }
    if (a>nearT) { nearT=a; nearN=n; }
    if (b<farT) { farT=b; farN=f; }
    return nearT<=farT;
}
void roomBox(float3 origin,float3 direction,float3 low,float3 high,uint material,inout RoomHit hit) {
    float nearT=-1e20, farT=1e20;
    float3 nearN=0, farN=0;
    if (!roomSlab(origin.x,direction.x,low.x,high.x,float3(1,0,0),nearT,farT,nearN,farN)) { return; }
    if (!roomSlab(origin.y,direction.y,low.y,high.y,float3(0,1,0),nearT,farT,nearN,farN)) { return; }
    if (!roomSlab(origin.z,direction.z,low.z,high.z,float3(0,0,1),nearT,farT,nearN,farN)) { return; }
    float t=nearT>0.001 ? nearT : farT;
    if (t>0.001 && t<hit.distance) { hit.distance=t; hit.normal=nearT>0.001?nearN:farN; hit.material=material; }
}
void roomSphere(float3 origin,float3 direction,float3 centre,float radius,uint material,inout RoomHit hit) {
    float3 offset=origin-centre;
    float b=dot(offset,direction), discriminant=b*b-dot(offset,offset)+radius*radius;
    if (discriminant<0) { return; }
    float root=sqrt(discriminant), t=-b-root;
    if (t<=0.001) { t=-b+root; }
    if (t>0.001 && t<hit.distance) { hit.distance=t; hit.normal=normalize(origin+direction*t-centre); hit.material=material; }
}
RoomHit roomEmpty(float distance) {
    RoomHit hit; hit.distance=distance; hit.normal=float3(0,1,0); hit.material=9; hit.position=0; hit.colour=0;
    return hit;
}
// A coarse furniture-only shadow trace avoids a second full room trace per pixel.
float roomShadow(float3 p,float3 lightDirection,float lightDistance) {
    RoomHit blocker=roomEmpty(lightDistance);
    roomBox(p,lightDirection,float3(-1.45,0.78,3.0),float3(0.65,0.92,4.45),2,blocker);
    roomBox(p,lightDirection,float3(1.25,0.08,3.15),float3(2.7,0.6,5.4),3,blocker);
    roomBox(p,lightDirection,float3(2.5,0.55,3.15),float3(2.8,1.35,5.4),3,blocker);
    roomBox(p,lightDirection,float3(-2.7,0.1,6.35),float3(-1.35,2.35,6.8),2,blocker);
    roomSphere(p,lightDirection,float3(-0.8,1.14,3.65),0.22,4,blocker);
    roomSphere(p,lightDirection,float3(0.15,1.08,3.7),0.16,5,blocker);
    return blocker.distance<lightDistance ? 0.28 : 1;
}
RoomHit renderRoom(float3 origin,float3 direction,float2 pixel) {
    RoomHit hit=roomEmpty(20);
    // Interior dimensions: X [-3,3], Y [0,3.2], Z [0,7]. Left window is a real opening.
    roomBox(origin,direction,float3(-3.15,-0.15,0),float3(3.15,0,7),1,hit);
    roomBox(origin,direction,float3(-3.15,3.2,0),float3(3.15,3.35,7),0,hit);
    roomBox(origin,direction,float3(3,0,0),float3(3.15,3.2,7),0,hit);
    roomBox(origin,direction,float3(-3,0,7),float3(3,3.2,7.15),0,hit);
    roomBox(origin,direction,float3(-3,0,-0.15),float3(3,3.2,0),0,hit);
    roomBox(origin,direction,float3(-3.15,0,0),float3(-3,3.2,2.1),0,hit);
    roomBox(origin,direction,float3(-3.15,0,4.8),float3(-3,3.2,7),0,hit);
    roomBox(origin,direction,float3(-3.15,0,2.1),float3(-3,1.15,4.8),0,hit);
    roomBox(origin,direction,float3(-3.15,2.75,2.1),float3(-3,3.2,4.8),0,hit);
    roomBox(origin,direction,float3(-3.02,1.08,2.03),float3(-2.85,1.2,4.87),2,hit);
    roomBox(origin,direction,float3(-3.02,2.7,2.03),float3(-2.9,2.82,4.87),2,hit);
    roomBox(origin,direction,float3(-3.02,1.2,3.41),float3(-2.93,2.7,3.49),2,hit);
    // Static exterior surface through the opening gives the bright view valid world motion.
    roomBox(origin,direction,float3(-5.6,-15,-15),float3(-5.5,15,22),9,hit);
    // Wood table, four individual legs and glossy metal / lacquer display objects.
    roomBox(origin,direction,float3(-1.45,0.78,3),float3(0.65,0.92,4.45),2,hit);
    [unroll] for (uint leg=0;leg<4;++leg) {
        float x=(leg&1)!=0 ? 0.43 : -1.33, z=(leg&2)!=0 ? 4.2 : 3.12;
        roomBox(origin,direction,float3(x,0,z),float3(x+0.12,0.78,z+0.12),2,hit);
    }
    roomSphere(origin,direction,float3(-0.8,1.14,3.65),0.22,4,hit);
    roomSphere(origin,direction,float3(0.15,1.08,3.7),0.16,5,hit);
    roomBox(origin,direction,float3(-0.38,0.92,4.0),float3(-0.12,1.27,4.26),4,hit);
    // Sofa: base, back, arms and separate rectangular fabric seat/back cushions.
    roomBox(origin,direction,float3(1.25,0.08,3.15),float3(2.7,0.45,5.4),3,hit);
    roomBox(origin,direction,float3(2.5,0.45,3.15),float3(2.8,1.35,5.4),3,hit);
    roomBox(origin,direction,float3(1.18,0.4,3.08),float3(2.8,0.9,3.28),3,hit);
    roomBox(origin,direction,float3(1.18,0.4,5.27),float3(2.8,0.9,5.47),3,hit);
    [unroll] for (uint cushion=0;cushion<3;++cushion) {
        float z=3.32+float(cushion)*0.64;
        roomBox(origin,direction,float3(1.28,0.45,z),float3(2.46,0.62,z+0.6),3,hit);
        roomBox(origin,direction,float3(2.33,0.63,z),float3(2.51,1.24,z+0.6),6,hit);
    }
    // Open shelf with contrasting books and a ceramic object.
    roomBox(origin,direction,float3(-2.7,0.1,6.35),float3(-2.6,2.35,6.8),2,hit);
    roomBox(origin,direction,float3(-1.45,0.1,6.35),float3(-1.35,2.35,6.8),2,hit);
    [unroll] for (uint shelf=0;shelf<4;++shelf) {
        float y=0.15+float(shelf)*0.7;
        roomBox(origin,direction,float3(-2.7,y,6.35),float3(-1.35,y+0.08,6.8),2,hit);
    }
    [unroll] for (uint book=0;book<5;++book) {
        float x=-2.5+float(book)*0.17;
        roomBox(origin,direction,float3(x,0.93,6.42),float3(x+0.13,1.35+0.06*float(book&1),6.68),(book&1)!=0?5:6,hit);
    }
    roomSphere(origin,direction,float3(-1.95,1.76,6.56),0.13,5,hit);
    // Large contrasting matte panels and a physical status-board surface.
    roomBox(origin,direction,float3(0.85,1.3,6.92),float3(1.7,2.55,7),5,hit);
    roomBox(origin,direction,float3(1.85,1.3,6.92),float3(2.7,2.55,7),6,hit);
    roomBox(origin,direction,float3(-1.05,1.5,6.83),float3(1.05,2.12,6.99),2,hit);
    roomBox(origin,direction,float3(-0.98,1.56,6.81),float3(0.98,2.06,6.84),7,hit);
    hit.position=origin+direction*hit.distance;
    if (hit.material==9) {
        hit.colour=lerp(float3(0.55,0.7,0.85),float3(0.95,0.9,0.72),saturate(0.7-0.09*hit.position.y));
        return hit;
    }
    float3 p=hit.position, n=hit.normal;
    // Explicit neighbouring-ray footprint, low-pass filtered before any high-frequency detail.
    float footprint=hit.distance*max(length(normalize(pixelRay(pixel+float2(1,0)))-direction),
                                    length(normalize(pixelRay(pixel+float2(0,1)))-direction));
    float3 albedo=float3(0.68,0.7,0.68); float roughness=0.8, metallic=0;
    if (hit.material==1 || hit.material==2) {
        float grainAxis=p.z+0.13*sin(p.x*4)+0.025*sin(p.z*3);
        float grain=sin(grainAxis*65)*exp(-pow(footprint*65,2));
        float broad=sin(grainAxis*7)*0.06;
        albedo=float3(0.38,0.19,0.075)*(1+broad+0.13*grain);
        if (hit.material==1) {
            float seam=1-smoothstep(0.006+footprint,0.015+footprint,abs(frac(p.x*1.5)-0.5)/1.5);
            albedo*=1-0.16*seam; albedo*=1.25;
        }
        roughness=0.46;
    } else if (hit.material==3 || hit.material==6) {
        float weave=sin((p.x+p.z)*420)*sin(p.y*420+p.z*210)*exp(-pow(footprint*420,2));
        albedo=(hit.material==3?float3(0.09,0.29,0.34):float3(0.68,0.51,0.21))*(1+0.07*weave);
        roughness=0.95;
    } else if (hit.material==4) { albedo=float3(0.65,0.7,0.75); metallic=0.92; roughness=0.15;
    } else if (hit.material==5) { albedo=float3(0.55,0.09,0.045); roughness=0.26;
    } else if (hit.material==7) { albedo=float3(0.015,0.022,0.028); }
    float3 lightPosition=Pad1==0?float3(-2.4,2.65,3.4):float3(0.4,2.85,2.5);
    float3 lightVector=lightPosition-p; float lightDistance=max(length(lightVector),0.001);
    float3 l=lightVector/lightDistance, v=-direction;
    float3 lightColour=Pad1==0?float3(0.92,1.02,1.15):float3(1.35,0.91,0.56);
    float attenuation=4/(1+0.22*lightDistance*lightDistance);
    float visibility=roomShadow(p+n*0.006,l,lightDistance);
    float diffuse=max(0,dot(n,l));
    float3 halfVector=l+v; halfVector/=max(length(halfVector),0.00001);
    float specular=pow(max(0,dot(n,halfVector)),lerp(100,5,roughness));
    float3 fresnel=lerp(float3(0.04,0.04,0.04),albedo,metallic);
    float3 ambient=Pad1==0?float3(0.17,0.2,0.24):float3(0.13,0.105,0.085);
    hit.colour=albedo*ambient*(0.65+0.35*max(0,n.y))+
        lightColour*attenuation*visibility*(albedo*(1-metallic)*diffuse+fresnel*specular*(1-0.65*roughness));
    // Broad static environment highlight makes polished objects readable without recursive reflection.
    if (metallic>0) {
        float3 reflection=reflect(direction,n);
        hit.colour+=albedo*(0.12+0.45*pow(saturate(dot(reflection,normalize(float3(-1,0.45,0.1)))),12));
    }
    hit.colour=min(hit.colour,1);
    return hit;
}
// 3x5 bitmap glyphs, top row in low bits. Status is composed AFTER neural processing.
uint roomLetter(uint letter) {
    if (letter==0) { return 24573; } // N
    if (letter==1) { return 29391; } // E
    if (letter==2) { return 31597; } // U
    if (letter==3) { return 23275; } // R
    if (letter==4) { return 23530; } // A
    if (letter==5) { return 29257; } // L
    if (letter==6) { return 31599; } // O
    if (letter==7) { return 4815; } // F
    return 0;
}
float3 roomStatus(float3 colour,float3 origin,float3 direction,float2 pixel) {
    // Same plane as the physical board in main; no head-locked or per-eye screen coordinates.
    if (direction.z<=0.000001) { return colour; }
    float t=(6.81-origin.z)/direction.z;
    if (t<=0.001) { return colour; }
    float3 p=origin+t*direction;
    if (abs(p.x)>0.98 || p.y<1.56 || p.y>2.06) { return colour; }
    // Depth is an SRV-state resource after main, so do not read its UAV in compose.
    // Only the small board rectangle retraces geometry; unused shading is optimised out.
    RoomHit visible=renderRoom(origin,direction,pixel);
    if (t>visible.distance+0.002) { return colour; }
    // Ten character slots, 4 columns each; one blank slot separates NEURAL and ON/OFF.
    float2 cell=float2((p.x+0.88)/0.044,(1.94-p.y)/0.052);
    if (cell.x<0 || cell.x>=40 || cell.y<0 || cell.y>=5) { return colour; }
    uint column=uint(floor(cell.x)), row=uint(floor(cell.y)), slot=column/4;
    uint letter=8;
    if (slot<6) { letter=slot; }
    else if (slot==7) { letter=6; }
    else if (slot==8) { letter=Neural!=0?0:7; }
    else if (slot==9 && Neural==0) { letter=7; }
    bool ink=(column%4)<3 && ((roomLetter(letter)>>(row*3+column%4))&1)!=0;
    if (ink) { return Neural!=0?float3(0.22,0.95,0.48):float3(1,0.65,0.22); }
    return colour;
}
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= Width || id.y >= Height) { return; }
    float cx = cameraX(Frame), px = cameraX(Frame == 0 ? 0 : Frame-1);
    float3 origin = float3(cx,1.4,0);
    float2 ndc = float2(2*(float(id.x)+0.5)/Width-1,1-2*(float(id.y)+0.5)/Height);
    float3 direction = normalize(float3(ndc,1));
    if (Tracked != 0) {
        float u=(float(id.x)+0.5)/Width, v=(float(id.y)+0.5)/Height;
        float3 localRay=normalize(float3(lerp(Tangents.x,Tangents.y,u),lerp(Tangents.z,Tangents.w,v),-1));
        origin=Position.xyz;
        direction=rotateVector(Rotation,localRay);
    }
    float distance; float3 normal, albedo, hitPosition, linearColour;
    if ((Pad0&1)!=0) {
        RoomHit room=renderRoom(origin,direction,float2(id.xy));
        distance=room.distance; normal=room.normal; hitPosition=room.position; linearColour=room.colour;
    } else {
    distance = direction.z>0.0001 ? (6-origin.z)/direction.z : 20;
    if (distance<0.1) { distance=20; }
    normal = float3(0,0,-1); albedo = float3(0.38,0.4,0.42);
    if (direction.y < -0.0001) {
        float t = -origin.y/direction.y;
        if (t>0.1 && t < distance) {
            distance=t; normal=float3(0,1,0);
            float3 p=origin+direction*t;
            // Compute derivatives explicitly: compute shaders have no ddx/ddy.
            float3 ray=pixelRay(float2(id.xy));
            float3 dx=pixelRay(float2(id.xy)+float2(1,0))-ray;
            float3 dy=pixelRay(float2(id.xy)+float2(0,1))-ray;
            float2 footprintX=origin.y*(ray.xz*dx.y-ray.y*dx.xz)/(ray.y*ray.y);
            float2 footprintY=origin.y*(ray.xz*dy.y-ray.y*dy.xz)/(ray.y*ray.y);
            // Axis-aligned box approximation of the projected footprint (metres).
            float2 width=2*(abs(footprintX)+abs(footprintY));
            float parity=checkerAverage(p.x*2,width.x)*checkerAverage(p.z*2,width.y);
            albedo=0.35-0.15*parity;
        }
    }
    float3 centres[2]={float3(-0.45,0.65,2.6),float3(0.65,0.45,3.4)};
    float radii[2]={0.65,0.45};
    float3 colours[2]={float3(0.75,0.18,0.08),float3(0.08,0.28,0.8)};
    [unroll] for (uint sphere=0;sphere<2;++sphere) {
        float3 offset=origin-centres[sphere];
        float b=dot(offset,direction);
        float discriminant=b*b-dot(offset,offset)+radii[sphere]*radii[sphere];
        if (discriminant>0) {
            float t=-b-sqrt(discriminant);
            if (t<=0.1) { t=-b+sqrt(discriminant); }
            if (t>0.1 && t<distance) {
                distance=t; normal=normalize(origin+direction*t-centres[sphere]); albedo=colours[sphere];
            }
        }
    }
    hitPosition=origin+direction*distance;
    float lighting=0.12+0.65*max(0,dot(normal,normalize(float3(-0.5,0.8,-0.4))));
    linearColour=albedo*lighting;
    }
    float3 encoded=float3(linearColour.x<=0.0031308?linearColour.x*12.92:1.055*pow(linearColour.x,1/2.4)-0.055,
                          linearColour.y<=0.0031308?linearColour.y*12.92:1.055*pow(linearColour.y,1/2.4)-0.055,
                          linearColour.z<=0.0031308?linearColour.z*12.92:1.055*pow(linearColour.z,1/2.4)-0.055);
    Colour[id.xy]=float4(encoded,1);
    float viewZ=hitPosition.z;
    float2 motion=float2((cx-px)/hitPosition.z*Width*0.5,0);
    if (Tracked != 0) {
        float3 local=inverseRotate(Rotation,hitPosition-Position.xyz);
        float3 previous=inverseRotate(PreviousRotation,hitPosition-PreviousPosition.xyz);
        viewZ=max(0.1,-local.z);
        float2 previousPixel=float2((previous.x/max(0.1,-previous.z)-PreviousTangents.x)/(PreviousTangents.y-PreviousTangents.x)*Width,
          (previous.y/max(0.1,-previous.z)-PreviousTangents.z)/(PreviousTangents.w-PreviousTangents.z)*Height)-0.5;
        motion=previousPixel-float2(id.xy);
    }
    Depth[id.xy]=saturate(20/19.9-2/(19.9*viewZ));
    Motion[id.xy]=motion;
}
