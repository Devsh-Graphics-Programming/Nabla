#ifndef _NBL_GLSL_EXT_RADIXSORT_INCLUDED_
#define _NBL_GLSL_EXT_RADIXSORT_INCLUDED_

#define _NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ (_NBL_GLSL_WORKGROUP_SIZE_ * 2 * 4)
shared uint scratch_shared[_NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_];
#define _NBL_GLSL_SCRATCH_SHARED_DEFINED_ scratch_shared

#include <nbl/builtin/glsl/workgroup/ballot.glsl>

#include "nbl/builtin/glsl/ext/RadixSort/parameters.glsl"

#ifndef _NBL_GLSL_EXT_RADIXSORT_GET_PARAMETERS_DEFINED_
#error "You need to define `nbl_glsl_ext_RadixSort_getParameters` and mark `_NBL_GLSL_EXT_RADIXSORT_GET_PARAMETERS_DEFINED_`!"
#endif

uint nbl_glsl_ext_RadixSort_extractDigit(in uint key)
{
	return (key >> nbl_glsl_ext_RadixSort_Parameters_t_getShift()) & 0xf;
}

uint nbl_glsl_ext_RadixSort_workgroupCompact(in uint digit, in bool save_local_histogram)
{
	uint scatter_idx = 0u;
	uint local_histogram[NUM_BUCKETS];
	for (int i = 0; i < NUM_BUCKETS; ++i)
	{
		const bool predicate = (i == digit);
		nbl_glsl_workgroupBallot(predicate);
		local_histogram[i] = nbl_glsl_workgroupBallotInclusiveBitCount();

		if (predicate)
			scatter_idx = local_histogram[i] - 1;
	}

	uint last_of_wg_idx = min((gl_WorkGroupID.x + 1u) * _NBL_GLSL_WORKGROUP_SIZE_ - 1u, nbl_glsl_ext_RadixSort_Parameters_t_getElementCountTotal() - 1u);
	if (save_local_histogram && (gl_GlobalInvocationID.x == last_of_wg_idx))
	{
		for (int i = 0; i < NUM_BUCKETS; ++i)
			scratch_shared[i] = local_histogram[i];
	}
	barrier();

	return scatter_idx;
}

uvec2 nbl_glsl_ext_RadixSort_workgroupSort(in uvec2 data, in uint scatter_idx)
{
	if (gl_LocalInvocationIndex == 0u)
	{
		uint sum = 0;
		scratch_shared[NUM_BUCKETS] = sum;
		for (int i = 1; i < NUM_BUCKETS; ++i)
			scratch_shared[i + NUM_BUCKETS] = (sum += scratch_shared[i - 1]);
	}
	barrier();

	uint digit = nbl_glsl_ext_RadixSort_extractDigit(data.x);
	uint local_offset = scratch_shared[NUM_BUCKETS + digit];
	scatter_idx += local_offset;
	memoryBarrier();

	scratch_shared[scatter_idx] = data.x;
	scratch_shared[_NBL_GLSL_WORKGROUP_SIZE_ + scatter_idx] = data.y;
	barrier();

	return uvec2(scratch_shared[gl_LocalInvocationIndex], scratch_shared[gl_LocalInvocationIndex + _NBL_GLSL_WORKGROUP_SIZE_]);
}

#endif