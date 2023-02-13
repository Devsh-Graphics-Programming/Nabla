#ifndef _NBL_GLSL_EXT_ENVMAP_SAMPLING_FUNCTIONS_INCLUDED_
#define _NBL_GLSL_EXT_ENVMAP_SAMPLING_FUNCTIONS_INCLUDED_

#include <nbl/builtin/glsl/sampling/envmap.glsl>
#include <nbl/builtin/glsl/math/functions.glsl>
#include <nbl/builtin/glsl/bxdf/common.glsl>

float nbl_glsl_ext_HierarchicalWarp_deferred_pdf(in vec3 rayDirection, in sampler2D luminance, ivec2 luminanceMapSize, int lastLuminanceMip)
{
	const vec2 envmapUV = nbl_glsl_sampling_envmap_uvCoordFromDirection(rayDirection);

	float sinTheta = length(rayDirection.xy);
	float sumLum = texelFetch(luminance, ivec2(0), lastLuminanceMip).r;
	float lum = textureLod(luminance, envmapUV, 0).r;
	float bigfactor = float(luminanceMapSize.x*luminanceMapSize.y)/sumLum;
	return bigfactor*(lum/(sinTheta*2.f*nbl_glsl_PI*nbl_glsl_PI));
}

float nbl_glsl_ext_HierarchicalWarp_deferred_pdf(in vec3 rayDirection, in sampler2D luminance)
{
	const ivec2 luminanceMapSize = textureSize(luminance, 0);
	int lastLuminanceMip = int(log2(luminanceMapSize.x)); // TODO: later turn into push constant
	return nbl_glsl_ext_HierarchicalWarp_deferred_pdf(rayDirection, luminance, luminanceMapSize, lastLuminanceMip);
}


vec3 nbl_glsl_ext_HierarchicalWarp_generate(out float pdf, in vec2 rand, in sampler2D warpMap, ivec2 warpMapSize, ivec2 lastWarpMapPixel, float pdfConstant)
{
	vec2 xi = rand;
	const vec2 unnormCoord = xi*lastWarpMapPixel;

	const vec2 interpolant = fract(unnormCoord);

	vec2 warpSampleCoord = (unnormCoord+vec2(0.5f))/vec2(warpMapSize);
	const vec4 dirsX = textureGather(warpMap, warpSampleCoord, 0); // 0_1, 1_1, 1_0, 0_0
	const vec4 dirsY = textureGather(warpMap, warpSampleCoord, 1); // 0_1, 1_1, 1_0, 0_0
	const mat4x2 uvs = transpose(mat2x4(dirsX,dirsY));

	const vec2 xDiffs[] = {
		uvs[2]-uvs[3],
		uvs[1]-uvs[0]
	};
	const vec2 yVals[] = {
		xDiffs[0]*interpolant.x+uvs[3],
		xDiffs[1]*interpolant.x+uvs[0]
	};
	const vec2 yDiff = yVals[1]-yVals[0];
	const vec2 uv = yDiff*interpolant.y+yVals[0];

	float sinTheta;
	const vec3 L = nbl_glsl_sampling_envmap_directionFromUVCoord(uv, sinTheta);
	
	const float detInterpolJacobian = determinant(mat2(
		mix(xDiffs[0],xDiffs[1],interpolant.y), // first column dFdx
		yDiff // second column dFdy
	));

	pdf = pdfConstant/abs(sinTheta*detInterpolJacobian);
	return L;
}

vec3 nbl_glsl_ext_HierarchicalWarp_generate(out float pdf, in vec2 rand, in sampler2D warpMap)
{
	const ivec2 warpMapSize = textureSize(warpMap,0);
	const ivec2 lastWarpMapPixel = warpMapSize - ivec2(1.f);
	float pdfConstant = 1.f/(2.f*nbl_glsl_PI*nbl_glsl_PI*float(lastWarpMapPixel.x*lastWarpMapPixel.y));
	return nbl_glsl_ext_HierarchicalWarp_generate(pdf, rand, warpMap, warpMapSize, lastWarpMapPixel, pdfConstant);
}

nbl_glsl_LightSample nbl_glsl_ext_HierarchicalWarp_generate(out float pdf, in vec2 rand, in sampler2D warpMap, in nbl_glsl_AnisotropicViewSurfaceInteraction interaction)
{
	const vec3 L = nbl_glsl_ext_HierarchicalWarp_generate(pdf, rand, warpMap);
	return nbl_glsl_createLightSample(L, interaction);
}

#endif