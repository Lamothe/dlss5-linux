// Static world-space mesh. Atlas RGB is sRGB; factors and lighting are linear.
// First scope: no normal/metallic-roughness maps, skinning, blending or shadows.
Texture2D<float4> Atlas : register(t0);
Texture2D<float> SceneDepth : register(t1);
SamplerState AtlasSampler : register(s0);
cbuffer Scene : register(b0) {
 uint Eye; uint Frame; uint Width; uint Height;
 uint Tracked; uint Neural; uint Pad0; uint Pad1;
 float4 Rotation; float4 Position;
 float4 PreviousRotation; float4 PreviousPosition;
 float4 Tangents; float4 PreviousTangents;
};
float3 rotateVector(float4 q,float3 v) { return v+2*cross(q.xyz,cross(q.xyz,v)+q.w*v); }
float3 inverseRotate(float4 q,float3 v) { return rotateVector(float4(-q.xyz,q.w),v); }
float cameraX(uint f) { return 0.12*sin(float(f)*0.05)+(Eye==0?-0.032:0.032); }
float3 cameraPosition() { return Tracked!=0?Position.xyz:float3(cameraX(Frame),1.4,0); }
float4 projectWorld(float3 world,bool previous) {
 float3 local;
 float4 t;
 float z;
 if (Tracked!=0) {
  local=inverseRotate(previous?PreviousRotation:Rotation,world-(previous?PreviousPosition.xyz:Position.xyz));
  t=previous?PreviousTangents:Tangents; z=-local.z;
 } else {
  local=world-float3(cameraX(previous?(Frame>0?Frame-1:0):Frame),1.4,0);
  t=float4(-1,1,1,-1); z=local.z;
 }
 // NDC Y is up, whereas tangent interpolation follows screen rows downwards.
 float x=(2*local.x-(t.x+t.y)*z)/(t.y-t.x);
 float y=((t.z+t.w)*z-2*local.y)/(t.w-t.z);
 return float4(x,y,(20.0/19.9)*z-2.0/19.9,z);
}
struct Vertex {
 float3 position:POSITION; float3 normal:NORMAL; float2 uv:TEXCOORD0;
 float4 properties:TEXCOORD1; float4 factor:COLOR0;
};
struct Varying {
 float4 position:SV_POSITION;
 float3 world:TEXCOORD0; float3 normal:TEXCOORD1; float2 uv:TEXCOORD2;
 float4 properties:TEXCOORD3; float4 factor:COLOR0;
};
Varying vertexMain(Vertex input) {
 Varying output;
 output.position=projectWorld(input.position,false); output.world=input.position;
 output.normal=input.normal; output.uv=input.uv; output.properties=input.properties;
 output.factor=input.factor; return output;
}
float3 encodeSRGB(float3 v) {
 v=max(v,0);
 return float3(v.x<=0.0031308?v.x*12.92:1.055*pow(v.x,1.0/2.4)-0.055,
               v.y<=0.0031308?v.y*12.92:1.055*pow(v.y,1.0/2.4)-0.055,
               v.z<=0.0031308?v.z*12.92:1.055*pow(v.z,1.0/2.4)-0.055);
}
struct Targets { float4 colour:SV_Target0; float depth:SV_Target1; float2 motion:SV_Target2; };
Targets pixelMain(Varying input) {
 // Snapshot avoids simultaneously reading and rendering to the procedural depth.
 clip(SceneDepth.Load(int3(int2(input.position.xy),0))-input.position.z);
 // Single atlas mip: bilinear filtering only. Converter must provide tile gutters.
 float4 texel=Atlas.SampleLevel(AtlasSampler,input.uv,0);
 clip(texel.a*input.factor.a-input.properties.z);
 float3 albedo=texel.rgb*input.factor.rgb;
 float3 n=input.normal/max(length(input.normal),0.00001);
 float3 v=cameraPosition()-input.world; v/=max(length(v),0.00001);
 // Orient two-sided shading towards the viewer, independent of mirrored nodes.
 if (dot(n,v)<0) { n=-n; }
 float3 lightPosition=Pad1==0?float3(-2.4,2.65,3.4):float3(0.4,2.85,2.5);
 float3 l=lightPosition-input.world; float distance=max(length(l),0.001); l/=distance;
 float3 h=l+v; h/=max(length(h),0.00001);
 float roughness=saturate(input.properties.x),metallic=saturate(input.properties.y);
 float3 lightColour=Pad1==0?float3(0.92,1.02,1.15):float3(1.35,0.91,0.56);
 float3 ambient=Pad1==0?float3(0.17,0.2,0.24):float3(0.13,0.105,0.085);
 float3 fresnel=lerp(float3(0.04,0.04,0.04),albedo,metallic);
 float specular=pow(max(0,dot(n,h)),lerp(100,5,roughness));
 float3 colour=albedo*ambient*(0.65+0.35*max(0,n.y))+lightColour*(4/(1+0.22*distance*distance))*
  (albedo*(1-metallic)*max(0,dot(n,l))+fresnel*specular*(1-0.65*roughness));
 float4 previous=projectWorld(input.world,true);
 float2 previousPixel=float2(previous.x,-previous.y)/max(previous.w,0.1);
 previousPixel=(previousPixel*0.5+0.5)*float2(Width,Height);
 Targets output; output.colour=float4(encodeSRGB(colour),1); output.depth=input.position.z;
 output.motion=previousPixel-input.position.xy; return output;
}
