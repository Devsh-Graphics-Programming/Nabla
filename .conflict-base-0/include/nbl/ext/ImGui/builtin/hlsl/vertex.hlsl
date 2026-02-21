#include "common.hlsl"
#include "psinput.hlsl"

using namespace nbl::ext::imgui;

[[vk::push_constant]] struct PushConstants pc;

struct VSInput
{
    [[vk::location(0)]] float2 position : POSITION;
    [[vk::location(1)]] float2 uv : TEXCOORD0;
    [[vk::location(2)]] float4 color : COLOR0;
};

/*
    we use Indirect Indexed draw call to render whole GUI, note we do a cross 
    platform trick and use base instance index as replacement for gl_DrawID 
    to request per object data with BDA
*/

PSInput VSMain(VSInput input, uint drawID : SV_InstanceID)
{
    PSInput output;
    output.color = input.color;
    output.uv = input.uv;
    output.drawID = drawID;

    // BDA for requesting object data
    const PerObjectData self = vk::RawBufferLoad<PerObjectData>(pc.elementBDA + sizeof(PerObjectData)* drawID);

    // NDC [-1, 1] range
    output.position = float4(input.position * pc.scale + pc.translate, 0, 1);
    
    const float32_t2 vMin = nbl::hlsl::glsl::unpackSnorm2x16(self.aabbMin);
    const float32_t2 vMax = nbl::hlsl::glsl::unpackSnorm2x16(self.aabbMax);

    // clip planes calculations, axis aligned
    output.clip[0] = output.position.x - vMin.x;
    output.clip[1] = output.position.y - vMin.y;
    output.clip[2] = vMax.x - output.position.x;
    output.clip[3] = vMax.y - output.position.y;
	
    return output;
}