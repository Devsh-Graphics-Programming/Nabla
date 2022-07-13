#ifndef _NBL_GLSL_BLIT_INCLUDED_
#define _NBL_GLSL_BLIT_INCLUDED_

#ifndef _NBL_GLSL_BLIT_MAIN_DEFINED_

#include <nbl/builtin/glsl/blit/multi_dimensional_array_addressing.glsl>

#include <nbl/builtin/glsl/blit/parameters.glsl>
nbl_glsl_blit_parameters_t nbl_glsl_blit_getParameters();

vec4 nbl_glsl_blit_getData(in vec3 texCoord, in uint layerIdx);
void nbl_glsl_blit_setData(in vec4 data, in uvec3 coord, in uint layerIdx);
vec4 nbl_glsl_blit_getKernelWeight(in uint index);
void nbl_glsl_blit_addToHistogram(in uint bucketIndex, in uint layerIdx);

#ifndef _NBL_GLSL_BLIT_ALPHA_BIN_COUNT_
#error _NBL_GLSL_BLIT_ALPHA_BIN_COUNT_ must be defined
#endif

#define scratchShared _NBL_GLSL_SCRATCH_SHARED_DEFINED_

// p in input space
ivec3 getMinKernelWindowCoord(in vec3 p, in vec3 minSupport)
{
	return ivec3(ceil(p - vec3(0.5f) + minSupport));
}

void nbl_glsl_blit_main()
{
	const nbl_glsl_blit_parameters_t params = nbl_glsl_blit_getParameters();

	const uvec3 outputTexelsPerWG = params.outputTexelsPerWG;

	const vec3 scale = params.fScale;
	const vec3 halfScale = scale * vec3(0.5f);

	const uvec3 minOutputPixel = gl_WorkGroupID * outputTexelsPerWG;
	const vec3 minOutputPixelCenterOfWG = vec3(minOutputPixel)*scale + halfScale;
	const ivec3 regionStartCoord = getMinKernelWindowCoord(minOutputPixelCenterOfWG, params.negativeSupport); // this can be negative, in which case HW sampler takes care of wrapping for us

	const uvec3 preloadRegion = params.preloadRegion;

	for (uint virtualInvocation = gl_LocalInvocationIndex; virtualInvocation < preloadRegion.x * preloadRegion.y * preloadRegion.z; virtualInvocation += _NBL_GLSL_WORKGROUP_SIZE_)
	{
		const ivec3 inputPixelCoord = regionStartCoord + ivec3(nbl_glsl_multi_dimensional_array_addressing_snakeCurveInverse(virtualInvocation, preloadRegion));

		vec3 inputTexCoord = (inputPixelCoord + vec3(0.5f)) / params.inDim;
		
		const vec4 loadedData = nbl_glsl_blit_getData(inputTexCoord, gl_WorkGroupID.z);
		for (uint ch = 0; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
			scratchShared[ch][virtualInvocation] = loadedData[ch];
	}
	barrier();

	const uvec3 iterationRegions[3] = uvec3[]( uvec3(outputTexelsPerWG.x, preloadRegion.yz), uvec3(outputTexelsPerWG.yx, preloadRegion.z), outputTexelsPerWG.yxz );

	uint readScratchOffset = 0;
	uint writeScratchOffset = params.secondScratchOffset;
	for (uint axis = 0; axis < _NBL_GLSL_BLIT_DIM_COUNT_; ++axis)
	{
		const uvec3 iterationRegion = iterationRegions[axis];
		for (uint virtualInvocation = gl_LocalInvocationIndex; virtualInvocation < iterationRegion.x * iterationRegion.y * iterationRegion.z; virtualInvocation += _NBL_GLSL_WORKGROUP_SIZE_)
		{
			const uvec3 virtualInvocationID = nbl_glsl_multi_dimensional_array_addressing_snakeCurveInverse(virtualInvocation, iterationRegion);

			uint outputPixel = virtualInvocationID.x;
			if (axis == 2)
				outputPixel = virtualInvocationID.z;
			outputPixel += minOutputPixel[axis];

			if (outputPixel >= params.outDim[axis])
				break;

			const int minKernelWindow = int(ceil((outputPixel + 0.5f) * scale[axis] - 0.5f + params.negativeSupport[axis]));

			// Combined stride for the two non-blitting dimensions, tightly coupled and experimentally derived with/by `iterationRegion` above and the general order of iteration we use to avoid
			// read bank conflicts.
			uint combinedStride;
			{
				if (axis == 0)
					combinedStride = virtualInvocationID.z * preloadRegion.y + virtualInvocationID.y;
				else if (axis == 1)
					combinedStride = virtualInvocationID.z * outputTexelsPerWG.x + virtualInvocationID.y;
				else if (axis == 2)
					combinedStride = virtualInvocationID.y * outputTexelsPerWG.y + virtualInvocationID.x;
			}

			uint offset = readScratchOffset + (minKernelWindow - regionStartCoord[axis]) + combinedStride*preloadRegion[axis];
			const uint windowPhase = outputPixel % params.phaseCount[axis];

			uint kernelWeightIndex;
			if (axis == 0)
				kernelWeightIndex = windowPhase * params.windowDim.x;
			else if (axis == 1)
				kernelWeightIndex = params.phaseCount.x * params.windowDim.x + windowPhase * params.windowDim.y;
			else if (axis == 2)
				kernelWeightIndex = params.phaseCount.x * params.windowDim.x + params.phaseCount.y * params.windowDim.y + windowPhase * params.windowDim.z;

			vec4 kernelWeight = nbl_glsl_blit_getKernelWeight(kernelWeightIndex);

			vec4 accum = vec4(0.f);
			for (uint ch = 0; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
				accum[ch] = scratchShared[ch][offset] * kernelWeight[ch];

			for (uint i = 1; i < params.windowDim[axis]; ++i)
			{
				kernelWeightIndex++;
				offset++;

				kernelWeight = nbl_glsl_blit_getKernelWeight(kernelWeightIndex);
				for (uint ch = 0; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
					accum[ch] += scratchShared[ch][offset] * kernelWeight[ch];
			}

			const bool lastPass = (axis == (_NBL_GLSL_BLIT_DIM_COUNT_ - 1));
			if (lastPass)
			{
				// Tightly coupled with iteration order (`iterationRegions`)
				uvec3 outCoord = virtualInvocationID.yxz;
				if (axis == 0)
					outCoord = virtualInvocationID.xyz;
				outCoord += minOutputPixel;

				const uint bucketIndex = uint(round(clamp(accum.a, 0, 1) * float(_NBL_GLSL_BLIT_ALPHA_BIN_COUNT_)));
				nbl_glsl_blit_addToHistogram(bucketIndex, gl_WorkGroupID.z);

				nbl_glsl_blit_setData(accum, outCoord, gl_WorkGroupID.z);
			}
			else
			{
				uint scratchOffset = writeScratchOffset;
				if (axis == 0)
					scratchOffset += nbl_glsl_multi_dimensional_array_addressing_snakeCurve(virtualInvocationID.yxz, iterationRegions[0].yxz);
				else 
					scratchOffset += writeScratchOffset + nbl_glsl_multi_dimensional_array_addressing_snakeCurve(virtualInvocationID.zxy, iterationRegions[1].zxy);
				
				for (uint ch = 0; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
					scratchShared[ch][scratchOffset] = accum[ch];
				
			}
		}
		
		const uint tmp = readScratchOffset;
		readScratchOffset = writeScratchOffset;
		writeScratchOffset = tmp;
		barrier();
	}
}

#undef scratchShared

#define _NBL_GLSL_BLIT_MAIN_DEFINED_
#endif

#endif