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
 float depth=(Pad0&4)!=0?(100.0/99.9)*z-10.0/99.9:(20.0/19.9)*z-2.0/19.9;
  return float4(x,y,depth,z);
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
// Ground-only tile-local mip pyramid, kept inside each original atlas cell.
// Explicit level sampling avoids atlas bleed; derivatives select the footprint.
float3 surfaceLevel(float2 tile, float2 local, uint level, bool normal) {
 uint size=368;float2 start=0;
 if(level>0) {
   start.x=376;
   for(uint i=1;i<=8;i++) {
     size=size/2;
     if(i==level) break;
     start.y+=size+8;
   }
 }
 float2 uv=(tile*744+start+4+saturate(local)*size)/8192;
 if(!normal) return Atlas.SampleLevel(AtlasSampler,uv,0).rgb;
 float2 pixel=uv*8192-.5;int2 origin=int2(floor(pixel));float2 f=frac(pixel);
 float3 a=encodeSRGB(Atlas.Load(int3(origin,0)).rgb);
 float3 b=encodeSRGB(Atlas.Load(int3(origin+int2(1,0),0)).rgb);
 float3 c=encodeSRGB(Atlas.Load(int3(origin+int2(0,1),0)).rgb);
 float3 d=encodeSRGB(Atlas.Load(int3(origin+int2(1,1),0)).rgb);
 return lerp(lerp(a,b,f.x),lerp(c,d,f.x),f.y)*2-1;
}
float3 surfaceSample(float2 tile,float2 local,float lod,bool normal) {
 uint level=(uint)floor(lod);
 return lerp(surfaceLevel(tile,local,level,normal),surfaceLevel(tile,local,min(level+1,8),normal),frac(lod));
}
struct Targets { float4 colour:SV_Target0; float depth:SV_Target1; float2 motion:SV_Target2; };
Targets pixelMain(Varying input) {
 // Snapshot avoids simultaneously reading and rendering to the procedural depth.
 clip(SceneDepth.Load(int3(int2(input.position.xy),0))-input.position.z);
 bool ground=input.properties.w>=129;
 float2 colourTile=floor(input.uv/(744.0/8192));
 float2 local=(input.uv*8192-colourTile*744-4.5)/735;
 // Conservative major-axis footprint: stable at grazing angles, no undersampling.
 float2 dx=ddx(input.uv)*(8192.0/735*368),dy=ddy(input.uv)*(8192.0/735*368);
 float footprint=sqrt(dot(dx,dx)+dot(dy,dy));
 float lod=clamp(log2(max(footprint,1)),0,8);
 float4 texel=Atlas.SampleLevel(AtlasSampler,input.uv,0);
 if(ground) texel=float4(surfaceSample(colourTile,local,lod,false),1);
 if(input.properties.z>=0) { clip(texel.a*input.factor.a-input.properties.z); }
 float3 albedo=texel.rgb*input.factor.rgb;
 float3 n=input.normal/max(length(input.normal),0.00001);
 // City normal tiles share the atlas; RGB is decoded back to linear data.
 if (input.properties.w>0) {
   const float grid=11.0;
   float index=input.properties.w-1-(ground?128:0);
   float2 target=float2(fmod(index,grid),floor(index/grid));
   float tileSize=744.0/8192.0;
   float2 normalUV=input.uv+(target-floor(input.uv/tileSize))*tileSize;
   // Decode each loaded texel before interpolation: normals are data, not sRGB colour.
   uint aw,ah; Atlas.GetDimensions(aw,ah);
   float2 pixel=normalUV*float2(aw,ah)-.5;
   int2 origin=int2(floor(pixel)); float2 blend=frac(pixel);
   int2 limit=int2(aw,ah)-1;
   float3 a=encodeSRGB(Atlas.Load(int3(clamp(origin,int2(0,0),limit),0)).rgb);
   float3 b=encodeSRGB(Atlas.Load(int3(clamp(origin+int2(1,0),int2(0,0),limit),0)).rgb);
   float3 c=encodeSRGB(Atlas.Load(int3(clamp(origin+int2(0,1),int2(0,0),limit),0)).rgb);
   float3 d=encodeSRGB(Atlas.Load(int3(clamp(origin+int2(1,1),int2(0,0),limit),0)).rgb);
   float3 mapped=lerp(lerp(a,b,blend.x),lerp(c,d,blend.x),blend.y)*2-1;
   if(ground) mapped=surfaceSample(target,local,lod,true);
   float3 dp1=ddx(input.world),dp2=ddy(input.world);
   float2 duv1=ddx(input.uv),duv2=ddy(input.uv);
   float determinant=duv1.x*duv2.y-duv1.y*duv2.x;
   if (abs(determinant)>1e-10) {
     float3 tangent=(dp1*duv2.y-dp2*duv1.y)/determinant;
     tangent-=n*dot(n,tangent);
     if (length(tangent)>1e-6) {
       tangent=normalize(tangent);
       float3 bitangent=(dp2*duv1.x-dp1*duv2.x)/determinant;
       bitangent-=n*dot(n,bitangent)+tangent*dot(tangent,bitangent);
       if(length(bitangent)>1e-6) { n=normalize(tangent*mapped.x+normalize(bitangent)*(-mapped.y)+n*mapped.z); }
     }
   }
 }
 float3 v=cameraPosition()-input.world; v/=max(length(v),0.00001);
 // Orient two-sided shading towards the viewer, independent of mirrored nodes.
 if (dot(n,v)<0) { n=-n; }
 float3 lightPosition=Pad1==0?float3(-2.4,2.65,3.4):float3(0.4,2.85,2.5);
 float3 l=lightPosition-input.world; float distance=max(length(l),0.001); l/=distance;
  bool city=(Pad0&4)!=0;
  if (city) { l=normalize(Pad1==0?float3(-0.45,0.8,-0.35):float3(0.5,0.65,0.3)); }
 float3 h=l+v; h/=max(length(h),0.00001);
 float roughness=saturate(input.properties.x),metallic=saturate(input.properties.y);
 float3 lightColour=Pad1==0?float3(0.92,1.02,1.15):float3(1.35,0.91,0.56);
 float3 ambient=Pad1==0?float3(0.17,0.2,0.24):float3(0.13,0.105,0.085);
 if (city) { ambient=Pad1==0?float3(0.3,0.35,0.42):float3(0.32,0.28,0.24); }
  float3 fresnel=lerp(float3(0.04,0.04,0.04),albedo,metallic);
 float specular=pow(max(0,dot(n,h)),lerp(100,5,roughness));
 float3 colour=albedo*ambient*(0.65+0.35*max(0,n.y))+lightColour*(city?1.1:(4/(1+0.22*distance*distance)))*
  (albedo*(1-metallic)*max(0,dot(n,l))+fresnel*specular*(1-0.65*roughness));
 float4 previous=projectWorld(input.world,true);
 float2 previousPixel=float2(previous.x,-previous.y)/max(previous.w,0.1);
 previousPixel=(previousPixel*0.5+0.5)*float2(Width,Height);
 Targets output; output.colour=float4(encodeSRGB(colour),input.properties.z<0?saturate(texel.a*input.factor.a):1); output.depth=input.position.z;
 output.motion=previousPixel-input.position.xy; return output;
}
Targets pixelLinear(Varying input) {
 Targets result=pixelMain(input);float3 v=result.colour.rgb;
 result.colour.rgb=float3(v.x<=.04045?v.x/12.92:pow((v.x+.055)/1.055,2.4),v.y<=.04045?v.y/12.92:pow((v.y+.055)/1.055,2.4),v.z<=.04045?v.z/12.92:pow((v.z+.055)/1.055,2.4));
 return result;
}
