#include "common.hlsl"

using namespace nbl;
using namespace nbl::hlsl;
using namespace nbl::hlsl::ext::envmap_importance_sampling;

[[vk::push_constant]] SLumaGenPushConstants pc;

[[vk::binding(0, 0)]] Texture2D<float32_t4> envMap;
[[vk::binding(1, 0)]] RWTexture2D<float32_t> outImage;

[numthreads(WORKGROUP_DIM, WORKGROUP_DIM, 1)]
[shader("compute")]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{	
	if (all(threadID < pc.lumaMapResolution))
	{

		const float uv_y = (float(threadID.y) + float(0.5f)) / pc.lumaMapResolution.y;
		const float32_t3 envMapSample = envMap.Load(float32_t3(threadID.xy, 0));
		const float32_t luma = hlsl::dot(envMapSample, pc.lumaRGBCoefficients) * sin(numbers::pi<float32_t> * uv_y);

		outImage[threadID.xy] = luma;
	}
}
