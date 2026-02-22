#ifndef _NBL_GLSL_EXT_RADIXSORT_INCLUDED_
#define _NBL_GLSL_EXT_RADIXSORT_INCLUDED_

#include <nbl/builtin/glsl/workgroup/ballot.glsl>

#include "nbl/builtin/glsl/ext/RadixSort/parameters.glsl"

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

uint nbl_glsl_ext_RadixSort_workgroupCompactAndHistogram(in uint digit, inout uint local_histogram[_NBL_GLSL_EXT_RADIXSORT_BUCKET_COUNT_])
{
	uint scatter_idx = 0u;
	for (int i = 0; i < _NBL_GLSL_EXT_RADIXSORT_BUCKET_COUNT_; ++i)
	{
		const bool predicate = (i == digit);
		nbl_glsl_workgroupBallot(predicate);
		local_histogram[i] = nbl_glsl_workgroupBallotInclusiveBitCount();

		if (predicate)
			scatter_idx = local_histogram[i];
	}
	return scatter_idx - 1u;
}

uint nbl_glsl_ext_RadixSort_workgroupCompact(in uint digit)
{
	uint local_histogram[_NBL_GLSL_EXT_RADIXSORT_BUCKET_COUNT_];
	return nbl_glsl_ext_RadixSort_workgroupCompactAndHistogram(digit, local_histogram);
}

#endif