#include "common.hlsl"

[[vk::combinedImageSampler]][[vk::binding(0, 0)]] Texture2D sampleTexture : register(t0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]] SamplerState linearSampler : register(s0);

float4 PSMain(PSInput input) : SV_Target0
{
    return input.color * sampleTexture.Sample(linearSampler, input.uv);
}