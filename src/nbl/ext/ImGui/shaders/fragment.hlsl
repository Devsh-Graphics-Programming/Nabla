#ifndef NBL_TEXTURES_BINDING
#error "NBL_TEXTURES_BINDING must be defined!"
#endif

#ifndef NBL_SAMPLER_STATES_BINDING
#error "NBL_SAMPLER_STATES_BINDING must be defined!"
#endif

#ifndef NBL_RESOURCES_SET
#error "NBL_RESOURCES_SET must be defined!"
#endif

#ifndef NBL_RESOURCES_COUNT
#error "NBL_RESOURCES_COUNT must be defined!"
#endif

#include "common.hlsl"

[[vk::push_constant]] struct PushConstants pc;

// separable image samplers to handle textures we do descriptor-index
[[vk::binding(NBL_TEXTURES_BINDING, NBL_RESOURCES_SET)]] Texture2D textures[NBL_RESOURCES_COUNT];
[[vk::binding(NBL_SAMPLER_STATES_BINDING, NBL_RESOURCES_SET)]] SamplerState samplerStates[NBL_RESOURCES_COUNT];

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