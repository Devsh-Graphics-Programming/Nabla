#include "common.hlsl"

[[vk::push_constant]] struct PushConstants pc;

// single separable image sampler to handle all textures we descriptor-index
[[vk::binding(0,0)]] Texture2D textures[NBL_MAX_IMGUI_TEXTURES];
[[vk::binding(1,0)]] SamplerState samplerStates[NBL_MAX_IMGUI_TEXTURES];

/*
    we use Indirect Indexed draw call to render whole GUI, note we do a cross 
    platform trick and use base instance index as replacement for gl_DrawID 
    to request per object data with BDA
*/

float4 PSMain(PSInput input) : SV_Target0
{
    // BDA for requesting object data
    const PerObjectData self = vk::RawBufferLoad<PerObjectData>(pc.elementBDA + sizeof(PerObjectData)* input.drawID);

    float4 texel = textures[NonUniformResourceIndex(self.texId)].Sample(samplerStates[self.texId], input.uv) * input.color;

    if(self.texId != 0) // TMP!
        texel.w = 1.f;

    return texel;
}