#ifndef _NBL_GLSL_BLIT_INCLUDED_
#define _NBL_GLSL_BLIT_INCLUDED_

#ifndef _NBL_GLSL_BLIT_MAIN_DEFINED_

#include <../blit/parameters.glsl>
nbl_glsl_blit_parameters_t nbl_glsl_blit_getParameters();

nbl_glsl_blit_input_pixel_t nbl_glsl_blit_getData(in ivec3 coord);
void nbl_glsl_blit_setData(in uvec3 coord, in vec4 data);
float nbl_glsl_blit_getCachedWeightsPremultiplied(in uvec3 lutCoord);

#define scratchShared _NBL_GLSL_SCRATCH_SHARED_DEFINED_

void nbl_glsl_blit_main()
{
	const nbl_glsl_blit_parameters_t params = nbl_glsl_blit_getParameters();

	const vec3 scale = vec3(params.inDim) / vec3(params.outDim);

	const uint windowPixelCount = params.windowDim.x * params.windowDim.y * params.windowDim.z; // 21

	// Todo(achal): assert on the CPU windowPixelCount <= _NBL_GLSL_WORKGROUP_SIZE_
	const uint windowsPerStep = _NBL_GLSL_WORKGROUP_SIZE_ / windowPixelCount; // 256/21 = 12
	const uint stepCount = (params.windowsPerWG + windowsPerStep - 1) / windowsPerStep; // 4

	const uint totalWindowCount = params.outDim.x * params.outDim.y * params.outDim.z;

	for (uint step = 0u; step < stepCount; ++step)
	{
		const uint stepLocalWindowIndex = gl_LocalInvocationIndex / windowPixelCount;
		if (stepLocalWindowIndex >= windowsPerStep)
			break;

		const uint wgLocalWindowIndex = stepLocalWindowIndex + step * windowsPerStep;

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

			windowLocalPixelID.x = sliceLocalIndex / params.windowDim.x;
			windowLocalPixelID.y = sliceLocalIndex % params.windowDim.x;
		}

		const ivec3 inputPixelCoord = windowMinCoord + ivec3(windowLocalPixelID);
		const vec3 inputPixelCenter = vec3(inputPixelCoord) + vec3(0.5f);

		const uvec3 windowPhase = globalWindowID % params.phaseCount;
		uvec3 lutIndex;
		lutIndex.x = windowPhase.x * params.windowDim.x + windowLocalPixelID.x;
		lutIndex.y = params.phaseCount.x * params.windowDim.x + windowPhase.y * params.windowDim.y + windowLocalPixelID.y;
		lutIndex.z = params.phaseCount.x * params.windowDim.x + params.phaseCount.y * params.windowDim.y + windowLocalPixelID.z;

		const float premultWeights = nbl_glsl_blit_getCachedWeightsPremultiplied(lutIndex);
		scratchShared[wgLocalWindowIndex * windowPixelCount + windowLocalPixelIndex] = nbl_glsl_blit_input_pixel_t(nbl_glsl_blit_getData(inputPixelCoord)) * premultWeights;
	}
	barrier();

	const uvec3 stride = uvec3(1u, params.windowDim.x, params.windowDim.x * params.windowDim.y);
	const uint axisCount = 2u; // Todo(achal): Get it via push constants
	for (uint axis = 0u; axis < axisCount; ++axis)
	{
		const uint stride = stride[axis]; // { 1, 3, 21 }
		const uint elementCount = (windowPixelCount * params.windowsPerWG) / stride; // { 21*48/1 = 1008, 21*48/3 = 336, 21*48/21 = 48 }

		const uint adderLength = params.windowDim[axis]; // { 3, 7, 1 }
		const uint adderCount = elementCount / adderLength; // { 1008/3 = 336, 336/7 = 48, 48/1 = 48 }
		const uint addersPerStep = _NBL_GLSL_WORKGROUP_SIZE_ / adderLength; // { 256/3 = 85, 256/7 = 36, 256/1 = 256 }
		const uint adderStepCount = (adderCount + addersPerStep - 1) / addersPerStep; // { (336+85-1)/85 = 4, (48+36-1)/36 = 2, (48+256-1)/256 = 1 }

		for (uint adderStep = 0u; adderStep < adderStepCount; ++adderStep)
		{
			const uint wgLocalAdderIndex = adderStep * addersPerStep + gl_LocalInvocationIndex / adderLength;
			const uint adderLocalPixelIndex = gl_LocalInvocationIndex % adderLength;

			// To make this code dynamically uniform we have to ensure that the following for loop runs at least once
			// even if adderLength = 1, hence the `max`
			const uint reduceStepCount = max(uint(ceil(log2(float(adderLength)))), 1u); // { 2, 3, 1 }
			for (uint reduceStep = 0u; reduceStep < reduceStepCount; ++reduceStep)
			{
				const uint offset = (1u << reduceStep); // { {1,2}, {1,2,4}, {1} }
				const uint baseIndex = (1u << (reduceStep + 1u)) * adderLocalPixelIndex;

				if ((baseIndex < adderLength) && (wgLocalAdderIndex < adderCount))
				{
					vec4 addend = vec4(0.f);
					if (baseIndex + offset < adderLength) // Don't need any kind of finer bounds checking here since we're ensuring that all windows fit COMPLETELY in a workgroup
						addend = scratchShared[((baseIndex + wgLocalAdderIndex * adderLength) + offset) * stride];

					scratchShared[(baseIndex + wgLocalAdderIndex * adderLength) * stride] += addend;
				}
				barrier();
			}
		}
	}

	for (uint step = 0u; step < stepCount; ++step)
	{
		const bool firstInvocationOfWindow = (gl_LocalInvocationIndex % windowPixelCount) == 0u ? true : false;
		if (!firstInvocationOfWindow)
			break;

		const uint stepLocalWindowIndex = gl_LocalInvocationIndex / windowPixelCount;
		if (stepLocalWindowIndex >= windowsPerStep) // This is important, otherwise some invocations in this step might interfere with next step's windows.
			break;

		// This doesn't need additional bounds checking because windows are packed tightly in workgroups. Same cannot be said for above where there might be some
		// empty space at the end of a step-- where a full window could not fit.
		const uint wgLocalWindowIndex = stepLocalWindowIndex + step * windowsPerStep;
		const uint globalWindowIndex = gl_WorkGroupID.x * params.windowsPerWG + wgLocalWindowIndex;
		if (globalWindowIndex >= totalWindowCount)
			break;

		const nbl_glsl_blit_input_pixel_t result = scratchShared[wgLocalWindowIndex * windowPixelCount];

		const uint bucketIndex = packUnorm4x8(vec4(result.a, 0.f, 0.f, 0.f));
		atomicAdd(alphaHistogram.data[bucketIndex], 1u);

		uvec3 globalWindowID;
		{
			const uint windowsPerSlice = params.outDim.x * params.outDim.y;

			globalWindowID.z = globalWindowIndex / windowsPerSlice;

			const uint sliceLocalIndex = globalWindowIndex % windowsPerSlice;

			globalWindowID.y = sliceLocalIndex / params.outDim.x;
			globalWindowID.x = sliceLocalIndex % params.outDim.x;
		}

		nbl_glsl_blit_setData(globalWindowID, vec4(result));
	}
}

#define _NBL_GLSL_BLIT_MAIN_DEFINED_
#endif

#endif