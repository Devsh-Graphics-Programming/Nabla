layout(constant_id = 0) const uint kComputeWGSize = 1024u;
layout(local_size_x=1024) in; // bug : https://github.com/KhronosGroup/SPIRV-Cross/issues/671#issuecomment-603943800

layout(constant_id = 1) const uint EII_COLOR = 0u;
layout(constant_id = 2) const uint EII_ALBEDO = 1u;
layout(constant_id = 3) const uint EII_NORMAL = 2u;

#include "./CommonPushConstants.h"

layout(push_constant, row_major) uniform PushConstants{
	CommonPushConstants data;
} pc;

#define SHARED_CHANNELS 3u
shared float repackBuffer[kComputeWGSize * SHARED_CHANNELS];


uvec3 clampCoords(in ivec3 inCoord)
{
	return uvec3(uvec2(clamp(inCoord.xy, ivec2(0, 0), ivec2(pc.data.imageWidth, gl_NumWorkGroups.y))), inCoord.z);
}





void ltswap(inout vec4 a, inout vec4 b)
{
	bool swap = b.w < a.w;
	vec3 tmp = a.rgb;
	a.rgb = swap ? b.rgb : a.rgb;
	b.rgb = swap ? tmp : b.rgb;
}

#define DENOISER_EXPOSURE_BIAS -1.0