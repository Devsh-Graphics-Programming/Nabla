#ifndef _NBL_GLSL_BLIT_INCLUDED_
#define _NBL_GLSL_BLIT_INCLUDED_

#ifndef _NBL_GLSL_BLIT_MAIN_DEFINED_

#include <nbl/builtin/glsl/blit/parameters.glsl>
nbl_glsl_blit_parameters_t nbl_glsl_blit_getParameters();

vec4 nbl_glsl_blit_getData(in vec3 texCoord, in uint layerIdx);
void nbl_glsl_blit_setData(in vec4 data, in uvec3 coord, in uint layerIdx);

void nbl_glsl_blit_addToHistogram(in uint bucketIndex, in uint layerIdx);

#define scratchShared _NBL_GLSL_SCRATCH_SHARED_DEFINED_

uvec3 linearIndexTo3DIndex(in uint linearIndex, in uvec3 gridDim)
{
	uvec3 index3D;

	const uint areaXY = gridDim.x * gridDim.y;
	index3D.z = linearIndex / areaXY;
	index3D.y = (linearIndex - (index3D.z * areaXY)) / gridDim.x;
	index3D.x = linearIndex - (index3D.y*gridDim.x) - (index3D.z*areaXY);

	return index3D;	
}

uint nbl_glsl_multi_dimensional_array_addressing_snakeCurve(in uvec3 coordinate, in uvec3 extents)
{
	return (coordinate.z * extents.y + coordinate.y) * extents.x + coordinate.x;
}

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
	const ivec3 regionStartCoord = getMinKernelWindowCoord(minOutputPixelCenterOfWG, params.negativeSupport); // this can be negative, in which case we wrap

	const uvec3 preloadRegion = params.preloadRegion;

	for (uint virtualInvocation = gl_LocalInvocationIndex; virtualInvocation < preloadRegion.x * preloadRegion.y * preloadRegion.z; virtualInvocation += _NBL_GLSL_WORKGROUP_SIZE_)
	{
		const ivec3 inputPixelCoord = regionStartCoord + ivec3(linearIndexTo3DIndex(virtualInvocation, preloadRegion));

		vec3 inputTexCoord = (inputPixelCoord + vec3(0.5f)) / params.inDim;
		
		const vec4 loadedData = nbl_glsl_blit_getData(inputTexCoord, gl_WorkGroupID.z);
		for (uint ch = 0; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
			scratchShared[ch][virtualInvocation] = loadedData[ch];
	}
	barrier();

	const uvec3 iterationRegions[3] = uvec3[]( uvec3(outputTexelsPerWG.x, preloadRegion.yz), uvec3(outputTexelsPerWG.yx, preloadRegion.z), outputTexelsPerWG.yxz );

	uint readOffset = 0;
	for (uint axis = 0; axis < _NBL_GLSL_BLIT_DIM_COUNT_; ++axis)
	{
		const uvec3 iterationRegion = iterationRegions[axis];
		for (uint virtualInvocation = gl_LocalInvocationIndex; virtualInvocation < iterationRegion.x * iterationRegion.y * iterationRegion.z; virtualInvocation += _NBL_GLSL_WORKGROUP_SIZE_)
		{
			const uvec3 virtualInvocationID = linearIndexTo3DIndex(virtualInvocation, iterationRegion);

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

			uint offset = (minKernelWindow - regionStartCoord[axis]) + combinedStride*preloadRegion[axis];
			const uint init_offset = offset;
			const uint windowPhase = outputPixel % params.phaseCount[axis];

			uint kernelWeightIndex;
			if (axis == 0)
				kernelWeightIndex = windowPhase * params.windowDim.x;
			else if (axis == 1)
				kernelWeightIndex = params.phaseCount.x * params.windowDim.x + windowPhase * params.windowDim.y;
			else if (axis == 2)
				kernelWeightIndex = params.phaseCount.x * params.windowDim.x + params.phaseCount.y * params.windowDim.y + windowPhase * params.windowDim.z;

			/*
			* For some reason the following doesn't work?!! Driver bug or UB somewhere?
				uvec3 kernelWeightLUTCoord = uvec3(0, 0, 0);
				for (uint i = 0; i < _NBL_GLSL_BLIT_DIM_COUNT_ - 1; ++i)
					kernelWeightLUTCoord[i] = params.phaseCount[i];
				kernelWeightLUTCoord[axis] = windowPhase;
			
				uint kernelWeightIndex = uint(dot(kernelWeightLUTCoord, params.windowDim));
			*/

			const uint init_kernelWeightIndex = kernelWeightIndex;
			// Todo(achal): getter here
			vec4 kernelWeight = texelFetch(_NBL_GLSL_BLIT_KERNEL_WEIGHTS_DESCRIPTOR_DEFINED_, int(kernelWeightIndex));
			// vec4 kernelWeight = vec4(1.f);

			vec4 accum = vec4(0.f);
			for (uint ch = 0; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
				accum[ch] = scratchShared[ch][params.offset*readOffset + offset] * kernelWeight[ch];

			for (uint i = 1; i < params.windowDim[axis]; ++i)
			{
				kernelWeightIndex++;
				offset++;

				kernelWeight = texelFetch(_NBL_GLSL_BLIT_KERNEL_WEIGHTS_DESCRIPTOR_DEFINED_, int(kernelWeightIndex));
				for (uint ch = 0; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
					accum[ch] += scratchShared[ch][params.offset*readOffset + offset] * kernelWeight[ch];
			}

			const bool lastPass = (axis == (_NBL_GLSL_BLIT_DIM_COUNT_ - 1));
			if (lastPass)
			{
				// Tightly coupled with iteration order (`iterationRegions`)
				uvec3 outCoord = virtualInvocationID.yxz;
				if (axis == 0)
					outCoord = virtualInvocationID.xyz;
				outCoord += minOutputPixel;

				const uint bucketIndex = packUnorm4x8(vec4(accum.a, 0.f, 0.f, 0.f));
				nbl_glsl_blit_addToHistogram(bucketIndex, gl_WorkGroupID.z);

				nbl_glsl_blit_setData(accum, outCoord, gl_WorkGroupID.z);
			}
			else
			{
				if (axis == 0)
				{
					for (uint ch = 0; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
						scratchShared[ch][params.offset * (1 - readOffset) + nbl_glsl_multi_dimensional_array_addressing_snakeCurve(virtualInvocationID.yxz, iterationRegions[0].yxz)] = accum[ch];
				}
				else if (axis == 1)
				{
					for (uint ch = 0; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
						scratchShared[ch][params.offset * (1 - readOffset) + nbl_glsl_multi_dimensional_array_addressing_snakeCurve(virtualInvocationID.zxy, iterationRegions[1].zxy)] = accum[ch];
				}
			}
		}
		readOffset ^= 1;
		barrier();
	}
}

#undef scratchShared

#define _NBL_GLSL_BLIT_MAIN_DEFINED_
#endif

#endif