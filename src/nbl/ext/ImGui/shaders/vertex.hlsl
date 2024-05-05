#include "common.hlsl"

struct VSInput
{
    [[vk::location(0)]] float2 position : POSITION;
	[[vk::location(1)]] float2 uv : TEXCOORD0;
    [[vk::location(2)]] float4 color : COLOR0;
};

struct PushConstants
{
    float2 scale;
    float2 translate;
};

[[vk::push_constant]] struct PushConstants pc;

PSInput VSMain(VSInput input)
{
    PSInput output;
    output.color = input.color;
    output.uv = input.uv;
    output.position = float4(input.position * pc.scale + pc.translate, 0, 1);
	
    return output;
}