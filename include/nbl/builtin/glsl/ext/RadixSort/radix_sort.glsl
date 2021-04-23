#ifndef _NBL_GLSL_EXT_RADIXSORT_INCLUDED_
#define _NBL_GLSL_EXT_RADIXSORT_INCLUDED_

#include <nbl/builtin/glsl/workgroup/ballot.glsl>

#include "nbl/builtin/glsl/ext/RadixSort/parameters.glsl"

#ifndef _NBL_GLSL_EXT_RADIXSORT_HISTOGRAM_SET_HISTOGRAM_DECLARED_
void nbl_glsl_ext_RadixSort_setHistogram(in uint idx, in uint val);
#define _NBL_GLSL_EXT_RADIXSORT_HISTOGRAM_SET_HISTOGRAM_DECLARED_
#endif

#ifndef _NBL_GLSL_EXT_RADIXSORT_BUCKET_COUNT_
#error "You need to define `_NBL_GLSL_EXT_RADIXSORT_BUCKET_COUNT_`!"
#endif

#ifndef _NBL_GLSL_EXT_RADIXSORT_GET_PARAMETERS_DEFINED_
#error "You need to define `nbl_glsl_ext_RadixSort_getParameters` and mark `_NBL_GLSL_EXT_RADIXSORT_GET_PARAMETERS_DEFINED_`!"
#endif

uint nbl_glsl_ext_RadixSort_extractDigit(in uint key)
{
	return (key >> nbl_glsl_ext_RadixSort_Parameters_t_getShift()) & (_NBL_GLSL_EXT_RADIXSORT_BUCKET_COUNT_ - 1u);
}

uint nbl_glsl_ext_RadixSort_workgroupCompact(in uint digit)
{
	uint scatter_idx = 0u;
	uint local_histogram[_NBL_GLSL_EXT_RADIXSORT_BUCKET_COUNT_];
	for (int i = 0; i < _NBL_GLSL_EXT_RADIXSORT_BUCKET_COUNT_; ++i)
	{
		const bool predicate = (i == digit);
		nbl_glsl_workgroupBallot(predicate);
		local_histogram[i] = nbl_glsl_workgroupBallotInclusiveBitCount();

		if (predicate)
			scatter_idx = local_histogram[i];
	}

#ifdef _NBL_GLSL_EXT_RADIXSORT_HISTOGRAM_SET_HISTOGRAM_DEFINED_

	uint last_of_wg_idx = min((gl_WorkGroupID.x + 1u) * _NBL_GLSL_WORKGROUP_SIZE_ - 1u, nbl_glsl_ext_RadixSort_Parameters_t_getElementCountTotal() - 1u);
	if (gl_GlobalInvocationID.x == last_of_wg_idx)
	{
		for (int i = 0; i < _NBL_GLSL_EXT_RADIXSORT_BUCKET_COUNT_; ++i)
			nbl_glsl_ext_RadixSort_setHistogram(i * gl_NumWorkGroups.x + gl_WorkGroupID.x, local_histogram[i]);

		// There is no reason to do the scan here except for the coincidence that only the things which define
		// _NBL_GLSL_EXT_RADIXSORT_HISTOGRAM_SET_HISTOGRAM_DEFINED_ require it, to locally sort elements (default_histogram.comp, here)
		uint sum = 0;
		_NBL_GLSL_SCRATCH_SHARED_DEFINED_[0] = sum;
		for (int i = 1; i < _NBL_GLSL_EXT_RADIXSORT_BUCKET_COUNT_; ++i)
			_NBL_GLSL_SCRATCH_SHARED_DEFINED_[i] = (sum += local_histogram[i - 1]);
	}
	barrier();

#endif

	return scatter_idx - 1u;
}

#endif