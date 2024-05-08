#include "common.hlsl"

[[vk::push_constant]] struct PushConstants pc;

PSInput VSMain(VSInput input)
{
    PSInput output;
    output.color = input.color;
    output.uv = input.uv;
    output.position = float4(input.position * pc.scale + pc.translate, 0, 1);
	
    return output;
}