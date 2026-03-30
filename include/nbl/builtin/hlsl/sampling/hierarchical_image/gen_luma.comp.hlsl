#include "common.hlsl"
#include "nbl/builtin/hlsl/limits.hlsl"

using namespace nbl;
using namespace nbl::hlsl;
using namespace nbl::hlsl::sampling::hierarchical_image;

[[vk::push_constant]] SLumaGenPushConstants pc;

[[vk::binding(0, 0)]] Texture2DArray<float32_t4> envMap;
[[vk::binding(1, 0)]] RWTexture2DArray<float32_t> outImage;

[numthreads(GenLumaWorkgroupDim, GenLumaWorkgroupDim, 1)]
[shader("compute")]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{	
	if (all(threadID.xyz < uint32_t3(pc.lumaMapWidth, pc.lumaMapHeight, pc.lumaMapLayer)))
	{
		const float uv_y = (float(threadID.y) + float(0.5f)) / pc.lumaMapHeight;
		const float32_t3 envMapSample = envMap.Load(int4(threadID.xyz, 0));
		float32_t luma = hlsl::dot(envMapSample, pc.lumaRGBCoefficients) * sin(numbers::pi<float32_t> * uv_y);

		// We reduce the luma of the corner texel since we want to do "corner sampling" when generating warp map.
		if (threadID.x == 0 || threadID.x == (pc.lumaMapWidth - 1))
			luma *= 0.5f;
		if (threadID.y == 0 || threadID.y == (pc.lumaMapHeight - 1))
			luma *= 0.5f;

		luma = max(luma, nbl::hlsl::numeric_limits<float32_t>::min);

		outImage[threadID.xyz] = luma;
	}
}
