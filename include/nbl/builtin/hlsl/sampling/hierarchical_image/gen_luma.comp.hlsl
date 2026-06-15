#include "common.hlsl"
#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/colorspace.hlsl"
#include "nbl/builtin/hlsl/sampling/spherical_mapping.hlsl"

using namespace nbl;
using namespace nbl::hlsl;
using namespace nbl::hlsl::sampling::hierarchical_image;

[[vk::push_constant]] SLumaGenPushConstants pc;

[[vk::binding(0, 0)]] Texture2DArray<float32_t4> envMap;
[[vk::binding(1, 0)]] RWTexture2DArray<float32_t> outImage;

template<typename Colorspace = colorspace::scRGB>
float32_t calcLuma(float32_t3 col)
{
  return hlsl::dot(Colorspace::ToXYZ()[1], col);
}

[numthreads(GenLumaWorkgroupDim, GenLumaWorkgroupDim, 1)]
[shader("compute")]
void main(uint32_t3 threadID : SV_DispatchThreadID)
{	
	if (all(threadID.xyz < uint32_t3(pc.lumaMapWidth, pc.lumaMapHeight, pc.lumaMapLayer)))
	{
		const float uv = (float32_t2(threadID.xy) + promote<float32_t2>(0.5)) / float32_t2(pc.lumaMapWidth, pc.lumaMapHeight);
		const float32_t3 envMapSample = envMap.Load(int4(threadID.xyz, 0));
		// Ask(kevin): I have to call generate first to get the cache, which is a lot of wasted calculation. Is this okay?
		sampling::SphericalMapping postWarp;
		sampling::SphericalMapping::cache_type postWarpCache;
		postWarp.generate(uv, postWarpCache);
		float32_t luma = calcLuma(envMapSample) / postWarp.forwardPdf(uv, postWarpCache);

		// We reduce the luma of the corner texel since we want to do "corner sampling" when generating warp map.
		if (threadID.x == 0 || threadID.x == (pc.lumaMapWidth - 1))
			luma *= 0.5f;
		if (threadID.y == 0 || threadID.y == (pc.lumaMapHeight - 1))
			luma *= 0.5f;

		luma = max(luma, nbl::hlsl::numeric_limits<float32_t>::min);

		outImage[threadID.xyz] = luma;
	}
}
