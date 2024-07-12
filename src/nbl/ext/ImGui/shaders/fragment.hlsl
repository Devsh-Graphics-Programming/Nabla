#include "common.hlsl"

[[vk::push_constant]] struct PushConstants pc;

// single separable image sampler to handle all textures we descriptor-index
[[vk::binding(0,0)]] Texture2D textures[NBL_MAX_IMGUI_TEXTURES];
[[vk::binding(1,0)]] SamplerState samplerState;

/*
    we use Indirect Indexed draw call to render whole GUI, note we do a cross 
    platform trick and use base instance index as replacement for gl_DrawID 
    to request per object data with BDA
*/

float4 PSMain(PSInput input, uint drawID : SV_InstanceID) : SV_Target0
{
    // BDA for requesting object data
    // TODO: move this to vertex shader, then pass along as interpolant
    const PerObjectData self = vk::RawBufferLoad<PerObjectData>(pc.elementBDA + sizeof(PerObjectData)* drawID);

    return input.color * textures[NonUniformResourceIndex(self.texId)].Sample(samplerState, input.uv);
}