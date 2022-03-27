#ifndef _NBL_GLSL_BLIT_INCLUDED_
#define _NBL_GLSL_BLIT_INCLUDED_

#ifndef _NBL_GLSL_BLIT_MAIN_DEFINED_

#include <../blit/parameters.glsl>
nbl_glsl_blit_parameters_t nbl_glsl_blit_getParameters();

nbl_glsl_blit_pixel_t nbl_glsl_blit_getData(in ivec3 coord);
void nbl_glsl_blit_setData(in nbl_glsl_blit_pixel_t data, in ivec3 coord);

float nbl_glsl_blit_getCachedWeightsPremultiplied(in uvec3 lutCoord);
void nbl_glsl_blit_addToHistogram(in uint bucketIndex);

#define scratchShared _NBL_GLSL_SCRATCH_SHARED_DEFINED_

uint roundUpToPoT(in uint value)
{
	return 1u << (1u + findMSB(value - 1u));
}

void nbl_glsl_blit_main()
{
	const nbl_glsl_blit_parameters_t params = nbl_glsl_blit_getParameters();

	const vec3 scale = vec3(params.inDim) / vec3(params.outDim);

	const uint windowPixelCount = params.windowDim.x * params.windowDim.y * params.windowDim.z;

	// Todo(achal): assert on the CPU windowPixelCount <= _NBL_GLSL_WORKGROUP_SIZE_
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

		uvec3 globalWindowID;
		{
			const uint windowsPerSlice = params.outDim.x * params.outDim.y;

			globalWindowID.z = globalWindowIndex / windowsPerSlice;

			const uint sliceLocalIndex = globalWindowIndex % windowsPerSlice;

			globalWindowID.y = sliceLocalIndex / params.outDim.x;
			globalWindowID.x = sliceLocalIndex % params.outDim.x;
		}

		const vec3 outputPixelCenter = (globalWindowID + vec3(0.5f)) * scale;

		const ivec3 windowMinCoord = ivec3(ceil(outputPixelCenter - vec3(0.5f) - abs(params.negativeSupport))); // this can be negative

		const uint windowLocalPixelIndex = gl_LocalInvocationIndex % windowPixelCount;
		uvec3 windowLocalPixelID;
		{
			const uint pixelsPerSlice = params.windowDim.x * params.windowDim.y;

			windowLocalPixelID.z = windowLocalPixelIndex / pixelsPerSlice;

			const uint sliceLocalIndex = windowLocalPixelIndex % pixelsPerSlice;

			windowLocalPixelID.y = sliceLocalIndex / params.windowDim.x;
			windowLocalPixelID.x = sliceLocalIndex % params.windowDim.x;
		}

		const ivec3 inputPixelCoord = windowMinCoord + ivec3(windowLocalPixelID);
		const vec3 inputPixelCenter = vec3(inputPixelCoord) + vec3(0.5f);

		const uvec3 windowPhase = globalWindowID % params.phaseCount;
		uvec3 lutIndex;
		lutIndex.x = windowPhase.x * params.windowDim.x + windowLocalPixelID.x;
		lutIndex.y = params.phaseCount.x * params.windowDim.x + windowPhase.y * params.windowDim.y + windowLocalPixelID.y;
		lutIndex.z = params.phaseCount.x * params.windowDim.x + params.phaseCount.y * params.windowDim.y + windowLocalPixelID.z;

		const float premultWeights = nbl_glsl_blit_getCachedWeightsPremultiplied(lutIndex);
		const vec4 loadedData = nbl_glsl_blit_getData(inputPixelCoord).data * premultWeights;
		for (uint ch = 0u; ch < _NBL_GLSL_BLIT_CHANNEL_COUNT_; ++ch)
			scratchShared[ch][wgLocalWindowIndex * windowPixelCount + windowLocalPixelIndex] = loadedData[ch];
	}
	barrier();

	for (uint ch = 0u; ch < _NBL_GLSL_BLIT_CHANNEL_COUNT_; ++ch)
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

		nbl_glsl_blit_pixel_t dataToStore;
		for (uint ch = 0u; ch < _NBL_GLSL_BLIT_CHANNEL_COUNT_; ++ch)
			dataToStore.data[ch] = scratchShared[ch][wgLocalWindowIndex * windowPixelCount];

		// Doing vec4(result) should set the alpha component to 0 for all cases when nbl_glsl_blit_input_pixel_t != vec4
		// const uint bucketIndex = packUnorm4x8(vec4(vec4(result).a, 0.f, 0.f, 0.f));
		// nbl_glsl_blit_addToHistogram(bucketIndex);

		uvec3 globalWindowID;
		{
			const uint windowsPerSlice = params.outDim.x * params.outDim.y;

			globalWindowID.z = globalWindowIndex / windowsPerSlice;

			const uint sliceLocalIndex = globalWindowIndex % windowsPerSlice;

			globalWindowID.y = sliceLocalIndex / params.outDim.x;
			globalWindowID.x = sliceLocalIndex % params.outDim.x;
		}

		nbl_glsl_blit_setData(dataToStore, ivec3(globalWindowID));
	}
}

#define _NBL_GLSL_BLIT_MAIN_DEFINED_
#endif

#endif