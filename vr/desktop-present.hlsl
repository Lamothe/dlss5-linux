// Windowed presentation for the desktop DLSSNR demo: fullscreen triangle, no sampler.
// The scene and the model both work in display-referred sRGB values, so the swapchain is
// R8G8B8A8_UNORM and the values are written through unchanged (saturate only).
cbuffer Present : register(b0) {
    uint Mode;    // 0 = raw | neural side by side, 1 = neural only, 2 = raw only
    uint Width;   // scene width in pixels (one panel)
    uint Height;  // scene height in pixels
    uint Pad0;
};
Texture2D<float4> Raw : register(t0);
Texture2D<float4> Neural : register(t1);

struct VSOut {
    float4 position : SV_POSITION;
};

VSOut VSMain(uint id : SV_VertexID) {
    // Oversized triangle covering the viewport; no vertex buffer needed.
    float2 uv = float2((id << 1) & 2, id & 2);
    VSOut output;
    output.position = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return output;
}

float4 PSMain(VSOut input) : SV_TARGET {
    int2 pixel = int2(input.position.xy);
    int width = (int)Width;
    int height = (int)Height;
    if (Mode == 0) {
        if (pixel.y >= height) { return float4(0, 0, 0, 1); }
        // A one-pixel white seam makes the split unmistakable at any display scale.
        if (pixel.x == width - 1 || pixel.x == width) { return float4(1, 1, 1, 1); }
        if (pixel.x < width) { return float4(saturate(Raw.Load(int3(pixel, 0)).rgb), 1); }
        return float4(saturate(Neural.Load(int3(int2(pixel.x - width, pixel.y), 0)).rgb), 1);
    }
    // Single-source modes: the panel is centred at 1:1, letterboxed rather than stretched.
    int x = pixel.x - (2 * width - width) / 2;
    if (x < 0 || x >= width || pixel.y >= height) { return float4(0, 0, 0, 1); }
    float3 colour = Mode == 1 ? Neural.Load(int3(int2(x, pixel.y), 0)).rgb
                              : Raw.Load(int3(int2(x, pixel.y), 0)).rgb;
    return float4(saturate(colour), 1);
}