#ifndef _NBL_GLSL_BLIT_INCLUDED_
#define _NBL_GLSL_BLIT_INCLUDED_

#ifndef _NBL_GLSL_BLIT_MAIN_DEFINED_

#include <nbl/builtin/glsl/blit/parameters.glsl>
nbl_glsl_blit_parameters_t nbl_glsl_blit_getParameters();

vec4 nbl_glsl_blit_getData(in vec3 texCoord, in uint layerIdx);
void nbl_glsl_blit_setData(in vec4 data, in uvec3 coord, in uint layerIdx);

vec4 nbl_glsl_blit_getCachedWeightsPremultiplied(in uvec3 lutCoord);
void nbl_glsl_blit_addToHistogram(in uint bucketIndex, in uint layerIdx);

#define scratchShared _NBL_GLSL_SCRATCH_SHARED_DEFINED_

uint roundUpToPoT(in uint value)
{
	return 1u << (1u + findMSB(value - 1u));
}

uvec3 linearIndexTo3DIndex(in uint linearIndex, in uvec3 gridDim)
{
	uvec3 index3d;
	const uint itemsPerSlice = gridDim.x * gridDim.y;

	index3d.z = linearIndex / itemsPerSlice;

	const uint sliceLocalIndex = linearIndex % itemsPerSlice;

	index3d.y = sliceLocalIndex / gridDim.x;
	index3d.x = sliceLocalIndex % gridDim.x;

	return index3d;	
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

ivec3 getMaxKernelWindowCoord(in vec3 p, in vec3 maxSupport)
{
	return ivec3(floor(p - vec3(0.5f) + maxSupport));
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

#if 1
	const uvec3 iterationRegions[3] = uvec3[]( uvec3(outputTexelsPerWG.x, preloadRegion.yz), uvec3(outputTexelsPerWG.yx, preloadRegion.z), outputTexelsPerWG.yxz );

	uint readOffset = 0;
	for (uint axis = 0; axis < _NBL_GLSL_BLIT_DIM_COUNT_; ++axis)
	{
		const uvec3 iterationRegion = iterationRegions[axis];
		for (uint virtualInvocation = gl_LocalInvocationIndex; virtualInvocation < iterationRegion.x * iterationRegion.y * iterationRegion.z; virtualInvocation += _NBL_GLSL_WORKGROUP_SIZE_)
		{
			// Todo(achal): Remove % from inside linearIndexTo3DIndex
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
#endif

#if 0
	// X pass
	uvec3 iterationRegion = uvec3(outputTexelsPerWG.x, preloadRegion.yz); // (3, 5, 6) 
	for (uint virtualInvocation = gl_LocalInvocationIndex; virtualInvocation < iterationRegion.x*iterationRegion.y*iterationRegion.z; virtualInvocation += _NBL_GLSL_WORKGROUP_SIZE_)
	{
		const uvec3 virtualInvocationID = linearIndexTo3DIndex(virtualInvocation, iterationRegion);

		const uint outputPixel = virtualInvocationID[0] + minOutputPixel.x;
		if (outputPixel >= params.outDim.x)
			break;

		const int minKernelWindowX = int(ceil((outputPixel + 0.5f)*scale.x - 0.5f + params.negativeSupport.x)); // this is just a 1D version of getMinKernelWindowCoord

		// Todo(achal): I haven't ran into any bugs with this but, is this correct?
		uint offset = (minKernelWindowX - regionStartCoord.x) + (virtualInvocationID[2] * preloadRegion.y + virtualInvocationID[1]) * preloadRegion.x;
		const uint init_offset = offset;

		const uint phase = outputPixel % params.phaseCount.x;

		uint kernelWeightsIndex = phase*params.windowDim.x;
		vec4 kernelWeight = texelFetch(_NBL_GLSL_BLIT_KERNEL_WEIGHTS_DESCRIPTOR_DEFINED_, int(kernelWeightsIndex));
		// vec4 kernelWeight = vec4(1.f);

		vec4 accum = vec4(0.f);
		for (uint ch = 0; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
			accum[ch] = scratchShared[ch][offset] * kernelWeight[ch];

		for (uint i = 1; i < params.windowDim.x; ++i)
		{
			kernelWeightsIndex = phase * params.windowDim.x + i;
			kernelWeight = texelFetch(_NBL_GLSL_BLIT_KERNEL_WEIGHTS_DESCRIPTOR_DEFINED_, int(kernelWeightsIndex));
			offset += 1;
			for (uint ch = 0; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
				accum[ch] += scratchShared[ch][offset] * kernelWeight[ch];
		}

#if _NBL_GLSL_BLIT_DIM_COUNT_>1
		for (uint ch = 0; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
			scratchShared[ch][params.offset + (virtualInvocationID.y + preloadRegion.y * (virtualInvocationID.x + virtualInvocationID.z * outputTexelsPerWG.x))] = accum[ch];
#else
		nbl_glsl_blit_setData(accum, ivec3(virtualInvocationID.xyz + minOutputPixel));
#endif
	}
	barrier();

#if _NBL_GLSL_BLIT_DIM_COUNT_>1
	// Y pass
	iterationRegion = uvec3(outputTexelsPerWG.yx, preloadRegion.z);
	for (uint virtualInvocation = gl_LocalInvocationIndex; virtualInvocation < iterationRegion.x * iterationRegion.y * iterationRegion.z; virtualInvocation += _NBL_GLSL_WORKGROUP_SIZE_)
	{
		const uvec3 virtualInvocationID = linearIndexTo3DIndex(virtualInvocation, iterationRegion);

		const uint outputPixel = virtualInvocationID.x + minOutputPixel.y;
		if (outputPixel >= params.outDim.y)
			break;

		const int minKernelWindowY = int(ceil((outputPixel + 0.5f) * scale.y - 0.5f + params.negativeSupport.y));
		uint offset = virtualInvocationID.z * preloadRegion.y*outputTexelsPerWG.x + virtualInvocationID.y * preloadRegion.y + (minKernelWindowY - regionStartCoord.y);

		const uint phase = outputPixel % params.phaseCount.y;

		uint kernelWeightsIndex = params.phaseCount.x*params.windowDim.x + phase*params.windowDim.y;

		
		vec4 kernelWeight = texelFetch(_NBL_GLSL_BLIT_KERNEL_WEIGHTS_DESCRIPTOR_DEFINED_, int(kernelWeightsIndex));
		// vec4 kernelWeight = vec4(1.f);

		vec4 accum = vec4(0.f);
		for (uint ch = 0; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
			accum[ch] = scratchShared[ch][params.offset + offset] * kernelWeight[ch];

		for (uint i = 1; i < params.windowDim.y; ++i)
		{
			kernelWeightsIndex = params.phaseCount.x*params.windowDim.x + phase * params.windowDim.y + i;
			kernelWeight = texelFetch(_NBL_GLSL_BLIT_KERNEL_WEIGHTS_DESCRIPTOR_DEFINED_, int(kernelWeightsIndex));
			offset += 1;
			for (uint ch = 0; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
				accum[ch] += scratchShared[ch][params.offset + offset] * kernelWeight[ch];
		}

#if _NBL_GLSL_BLIT_DIM_COUNT_>2
		for (uint ch = 0; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
			scratchShared[ch][virtualInvocationID.z + preloadRegion.z * (virtualInvocationID.x + virtualInvocationID.y * outputTexelsPerWG.y)] = accum[ch];
#else
		nbl_glsl_blit_setData(accum, ivec3(virtualInvocationID.yxz+minOutputPixel));
#endif
	}
	barrier();
#endif

#if _NBL_GLSL_BLIT_DIM_COUNT_>2
	// Z pass
	iterationRegion = outputTexelsPerWG.yxz;
	for (uint virtualInvocation = gl_LocalInvocationIndex; virtualInvocation < iterationRegion.x * iterationRegion.y * iterationRegion.z; virtualInvocation += _NBL_GLSL_WORKGROUP_SIZE_)
	{
		const uvec3 virtualInvocationID = linearIndexTo3DIndex(virtualInvocation, iterationRegion);

		const uint outputPixel = virtualInvocationID.z + minOutputPixel.z;
		if (outputPixel >= params.outDim.z)
			break;

		const int minKernelWindowZ = int(ceil((outputPixel + 0.5f) * scale.z - 0.5f + params.negativeSupport.z));

		uint offset = (minKernelWindowZ - regionStartCoord.z) + (virtualInvocationID.x * preloadRegion.z + virtualInvocationID.y * preloadRegion.z * outputTexelsPerWG.y);
		const uint init_offset = offset;

		const uint phase = outputPixel % params.phaseCount.z;

		uint kernelWeightsIndex = params.phaseCount.x * params.windowDim.x + params.phaseCount.y * params.windowDim.y + phase * params.windowDim.z;
		vec4 kernelWeight = texelFetch(_NBL_GLSL_BLIT_KERNEL_WEIGHTS_DESCRIPTOR_DEFINED_, int(kernelWeightsIndex));
		// vec4 kernelWeight = vec4(1.f);

		vec4 accum = vec4(0.f);
		for (uint ch = 0; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
			accum[ch] = scratchShared[ch][offset] * kernelWeight[ch];

		for (uint i = 1; i < params.windowDim.z; ++i)
		{
			kernelWeightsIndex = params.phaseCount.x * params.windowDim.x + params.phaseCount.y * params.windowDim.y + phase*params.windowDim.z + i;
			kernelWeight = texelFetch(_NBL_GLSL_BLIT_KERNEL_WEIGHTS_DESCRIPTOR_DEFINED_, int(kernelWeightsIndex));
			offset += 1;
			for (uint ch = 0; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
				accum[ch] += scratchShared[ch][offset] * kernelWeight[ch];
		}

		nbl_glsl_blit_setData(accum, ivec3(virtualInvocationID.yxz+minOutputPixel));
	}
#endif

#endif

#if 0
	const uint windowPixelCount = params.windowDim.x * params.windowDim.y * params.windowDim.z;

	const uint windowsPerStep = _NBL_GLSL_WORKGROUP_SIZE_ / windowPixelCount;
	const uint stepCount = (params.windowsPerWG + windowsPerStep - 1) / windowsPerStep;

	const uint totalWindowCount = params.outDim.x * params.outDim.y * params.outDim.z;

	for (uint step = 0u; step < stepCount; ++step)
	{
		const uint stepLocalWindowIndex = gl_LocalInvocationIndex / windowPixelCount;
		if (stepLocalWindowIndex >= windowsPerStep)
			break;

		const uint wgLocalWindowIndex = stepLocalWindowIndex + step * windowsPerStep;
		if (wgLocalWindowIndex >= params.windowsPerWG)
			break;

		// It could be the case that the last workgroup processes LESS THAN windowsPerWG windows
		const uint globalWindowIndex = gl_WorkGroupID.x * params.windowsPerWG + wgLocalWindowIndex;
		if (globalWindowIndex >= totalWindowCount)
			break;

		uvec3 globalWindowID = linearIndexTo3DIndex(globalWindowIndex, params.outDim);

		const vec3 outputPixelCenter = (globalWindowID + vec3(0.5f)) * scale;

		const ivec3 windowMinCoord = ivec3(ceil(outputPixelCenter - vec3(0.5f) - abs(params.negativeSupport))); // this can be negative

		const uint windowLocalPixelIndex = gl_LocalInvocationIndex % windowPixelCount;
		uvec3 windowLocalPixelID = linearIndexTo3DIndex(windowLocalPixelIndex, params.windowDim);

		const ivec3 inputPixelCoord = windowMinCoord + ivec3(windowLocalPixelID);
		const vec3 inputPixelCenter = vec3(inputPixelCoord) + vec3(0.5f);

		const uvec3 windowPhase = globalWindowID % params.phaseCount;
		uvec3 lutIndex;
		lutIndex.x = windowPhase.x * params.windowDim.x + windowLocalPixelID.x;
		lutIndex.y = params.phaseCount.x * params.windowDim.x + windowPhase.y * params.windowDim.y + windowLocalPixelID.y;
		lutIndex.z = params.phaseCount.x * params.windowDim.x + params.phaseCount.y * params.windowDim.y + windowLocalPixelID.z;
		const vec4 premultWeights = nbl_glsl_blit_getCachedWeightsPremultiplied(lutIndex);

		vec3 inputTexCoord = (inputPixelCoord + vec3(0.5)) / params.inDim;
		if (_NBL_GLSL_BLIT_DIM_COUNT_ < 3)
			inputTexCoord[_NBL_GLSL_BLIT_DIM_COUNT_] = gl_GlobalInvocationID[_NBL_GLSL_BLIT_DIM_COUNT_];

		const vec4 loadedData = nbl_glsl_blit_getData(inputTexCoord) * premultWeights;
		for (uint ch = 0u; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
			scratchShared[ch][wgLocalWindowIndex * windowPixelCount + windowLocalPixelIndex] = loadedData[ch];
	}
	barrier();
#endif

#if 0
	for (uint ch = 0u; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
	{
		const uvec3 stride = uvec3(1u, params.windowDim.x, params.windowDim.x * params.windowDim.y);

		for (uint axis = 0u; axis < _NBL_GLSL_BLIT_DIM_COUNT_; ++axis)
		{
			const uint stride = stride[axis];
			const uint elementCount = (windowPixelCount * params.windowsPerWG) / stride;

			const uint adderLength = params.windowDim[axis];
			const uint paddedAdderLength = roundUpToPoT(adderLength);
			const uint adderCount = elementCount / adderLength;
			const uint addersPerStep = _NBL_GLSL_WORKGROUP_SIZE_ / paddedAdderLength;
			const uint adderStepCount = (adderCount + addersPerStep - 1) / addersPerStep;

			for (uint adderStep = 0u; adderStep < adderStepCount; ++adderStep)
			{
				const uint stepLocalAdderIndex = gl_LocalInvocationIndex / paddedAdderLength;
				const uint wgLocalAdderIndex = adderStep * addersPerStep + stepLocalAdderIndex;
				const uint adderLocalPixelIndex = gl_LocalInvocationIndex % paddedAdderLength;

				for (uint s = paddedAdderLength / 2u; s > 0u; s >>= 1u)
				{
					if ((adderLocalPixelIndex < s) && (stepLocalAdderIndex < addersPerStep) && (wgLocalAdderIndex < adderCount))
					{
						float addend = 0.f;
						if (adderLocalPixelIndex + s < adderLength)
							addend = scratchShared[ch][(wgLocalAdderIndex * adderLength + adderLocalPixelIndex + s) * stride];

						scratchShared[ch][(wgLocalAdderIndex * adderLength + adderLocalPixelIndex) * stride] += addend;
					}
					barrier();
				}
			}
		}
		barrier();
	}

	for (uint step = 0u; step < stepCount; ++step)
	{
		const bool firstInvocationOfWindow = (gl_LocalInvocationIndex % windowPixelCount) == 0u ? true : false;
		if (!firstInvocationOfWindow)
			break;

		const uint stepLocalWindowIndex = gl_LocalInvocationIndex / windowPixelCount;
		if (stepLocalWindowIndex >= windowsPerStep) // otherwise some invocations in this step might interfere with next step's windows.
			break;

		const uint wgLocalWindowIndex = stepLocalWindowIndex + step * windowsPerStep;
		if (wgLocalWindowIndex >= params.windowsPerWG) // otherwise some invocations in this workgroup might interfere with next workgroup's windows.
			break;

		const uint globalWindowIndex = gl_WorkGroupID.x * params.windowsPerWG + wgLocalWindowIndex;
		if (globalWindowIndex >= totalWindowCount)
			break;

		vec4 dataToStore;
		for (uint ch = 0u; ch < _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_; ++ch)
			dataToStore[ch] = scratchShared[ch][wgLocalWindowIndex * windowPixelCount];

		// Todo(achal): Need to pull this out in setData
#if NBL_GLSL_EQUAL(_NBL_GLSL_BLIT_DIM_COUNT_, 1)
	#define LAYER_IDX gl_GlobalInvocationID.y
#elif NBL_GLSL_EQUAL(_NBL_GLSL_BLIT_DIM_COUNT_, 2)
	#define LAYER_IDX gl_GlobalInvocationID.z
#elif NBL_GLSL_EQUAL(_NBL_GLSL_BLIT_DIM_COUNT_, 3)
	#define LAYER_IDX 0
#else
	#error _NBL_GLSL_BLIT_DIM_COUNT_ not supported
#endif

		const uint bucketIndex = packUnorm4x8(vec4(dataToStore.a, 0.f, 0.f, 0.f));
		nbl_glsl_blit_addToHistogram(bucketIndex, LAYER_IDX);

#undef LAYER_IDX

		uvec3 globalWindowID = linearIndexTo3DIndex(globalWindowIndex, params.outDim);

		nbl_glsl_blit_setData(dataToStore, ivec3(globalWindowID));
	}
#endif

}

#undef scratchShared

#define _NBL_GLSL_BLIT_MAIN_DEFINED_
#endif

#endif