#include "common.hlsl"

using namespace nbl;
using namespace nbl::hlsl;
using namespace nbl::hlsl::sampling::hierarchical_image;

[[vk::push_constant]] SLumaGenPushConstants pc;

// TODO: Use layer texture, to implement envmap importance sampling for cube map
[[vk::binding(0, 0)]] Texture2D<float32_t4> envMap;
[[vk::binding(1, 0)]] RWTexture2D<float32_t> outImage;

[numthreads(GEN_LUMA_WORKGROUP_DIM, GEN_LUMA_WORKGROUP_DIM, 1)]
[shader("compute")]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{	
	if (all(threadID.xy < uint32_t2(pc.lumaMapWidth, pc.lumaMapHeight)))
	{

		const float uv_y = (float(threadID.y) + float(0.5f)) / pc.lumaMapHeight;
		const float32_t3 envMapSample = envMap.Load(float32_t3(threadID.xy, 0));
		const float32_t luma = hlsl::dot(envMapSample, pc.lumaRGBCoefficients) * sin(numbers::pi<float32_t> * uv_y);

		outImage[threadID.xy] = luma;
	}
}
