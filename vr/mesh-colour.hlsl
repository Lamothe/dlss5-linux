RWTexture2D<float4> Colour:register(u0);
cbuffer Mode:register(b0) { uint Encode; uint Size; }
[numthreads(8,8,1)]
void main(uint3 id:SV_DispatchThreadID) {
 if(id.x>=Size||id.y>=Size) return;
 float4 value=Colour[id.xy];float3 v=max(value.rgb,0);
 if(Encode!=0) v=float3(v.x<=.0031308?v.x*12.92:1.055*pow(v.x,1/2.4)-.055,v.y<=.0031308?v.y*12.92:1.055*pow(v.y,1/2.4)-.055,v.z<=.0031308?v.z*12.92:1.055*pow(v.z,1/2.4)-.055);
 else v=float3(v.x<=.04045?v.x/12.92:pow((v.x+.055)/1.055,2.4),v.y<=.04045?v.y/12.92:pow((v.y+.055)/1.055,2.4),v.z<=.04045?v.z/12.92:pow((v.z+.055)/1.055,2.4));
 Colour[id.xy]=float4(v,value.a);
}
