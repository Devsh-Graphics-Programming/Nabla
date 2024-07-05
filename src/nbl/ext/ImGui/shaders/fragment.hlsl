#include "common.hlsl"

[[vk::push_constant]] struct PushConstants pc;

[[vk::binding(0, 1)]] StructuredBuffer<PerObjectData> perObject : register(t0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]] Texture2D sampleTexture : register(t0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]] SamplerState linearSampler : register(s0);

float4 PSMain(PSInput input, uint drawID : SV_InstanceID) : SV_Target0
{
    PerObjectData objectData = perObject[drawID];

    // convert NDC coordinates to window coordinates
    float2 windowPos = (input.position.xy * 0.5 + 0.5) * pc.viewport.zw + pc.viewport.xy;

    // scissor pass
    if (windowPos.x < objectData.scissor.x || windowPos.x > (objectData.scissor.x + objectData.scissor.z) || windowPos.y < objectData.scissor.y || windowPos.y > (objectData.scissor.y + objectData.scissor.w)) 
        discard;

    return input.color * sampleTexture.Sample(linearSampler, input.uv);
}