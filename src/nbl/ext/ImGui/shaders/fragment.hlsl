#include "common.hlsl"

[[vk::push_constant]] struct PushConstants pc;

// separable image samplers, one to handle all textures
[[vk::binding(0,0)]] Texture2DArray texture;
[[vk::binding(1,0)]] SamplerState samplerState;

/*
    we use Indirect Indexed draw call to render whole GUI, note we do a cross 
    platform trick and use base instance index as replacement for gl_DrawID 
    to request per object data with BDA
*/

float4 PSMain(PSInput input, uint drawID : SV_InstanceID) : SV_Target0
{
    // BDA for requesting object data
    const PerObjectData self = vk::RawBufferLoad<PerObjectData>(pc.elementBDA + sizeof(PerObjectData)* drawID);

    return input.color * texture.Sample(samplerState, float32_t3(input.uv, self.texId));
}