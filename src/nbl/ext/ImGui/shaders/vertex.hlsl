#include "common.hlsl"

[[vk::push_constant]] struct PushConstants pc;

PSInput VSMain(VSInput input, uint drawID : SV_InstanceID)
{
    PSInput output;
    output.color = input.color;
    output.uv = input.uv;
    output.position = float4(input.position * pc.scale + pc.translate, 0, 1);

    // BDA for requesting object data
    const PerObjectData self = vk::RawBufferLoad<PerObjectData>(pc.elementBDA + sizeof(PerObjectData)* drawID);

	// TODO
    output.clip[0] = 69;
    output.clip[1] = 69;
	
    return output;
}