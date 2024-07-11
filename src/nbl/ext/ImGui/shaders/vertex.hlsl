#include "common.hlsl"

[[vk::push_constant]] struct PushConstants pc;

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

    // BDA for requesting object data
    const PerObjectData self = vk::RawBufferLoad<PerObjectData>(pc.elementBDA + sizeof(PerObjectData)* drawID);

    // NDC [-1, 1] range
    output.position = float4(input.position * pc.scale + pc.translate, 0, 1);
    const float2 vMin = self.aabbMin.unpack();
    const float2 vMax = self.aabbMax.unpack();

    // clip planes calculations, axis aligned
    output.clip[0] = output.position.x - vMin.x;
    output.clip[1] = output.position.y - vMin.y;
    output.clip[2] = vMax.x - output.position.x;
    output.clip[3] = vMax.y - output.position.y;
	
    return output;
}