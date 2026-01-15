#include "nbl/ext/EnvmapImportanceSampling/builtin/hlsl/common.hlsl"

using namespace nbl;
using namespace nbl::hlsl;
using namespace nbl::hlsl::ext::envmap_importance_sampling;

[[vk::push_constant]] SLumaGenPushConstants pc;

[[vk::combinedImageSampler]][[vk::binding(0, 0)]] Texture2D<float32_t4> envMap;
[[vk::combinedImageSampler]][[vk::binding(0, 0)]] SamplerState envMapSampler;

[[vk::binding(1, 0)]] RWTexture2D<float32_t> outImage;

// TODO(kevinyu): Temporary to make nsc compiles
#define LUMA_MAP_GEN_WORKGROUP_DIM 16

[numthreads(LUMA_MAP_GEN_WORKGROUP_DIM, LUMA_MAP_GEN_WORKGROUP_DIM, 1)]
[shader("compute")]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{	
	if (all(threadID < pc.lumaMapResolution))
	{

		const float32_t2 uv = (float32_t2(threadID.xy) + float32_t2(0.5, 0.5)) / float32_t2(pc.lumaMapResolution);
		const float32_t3 envMapSample = envMap.Sample(envMapSampler, uv).rgb;
		const float32_t luma = hlsl::dot(float32_t4(envMapSample, 1.0f), pc.luminanceScales) * sin(numbers::pi<float32_t> * uv.y);

		outImage[threadID.xy] = luma;
	}
}
