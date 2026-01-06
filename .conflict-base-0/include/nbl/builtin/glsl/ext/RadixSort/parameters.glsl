#ifndef _NBL_GLSL_EXT_RADIXSORT_PARAMETERS_INCLUDED_
#define _NBL_GLSL_EXT_RADIXSORT_PARAMETERS_INCLUDED_

#include "nbl/builtin/glsl/ext/RadixSort/parameters_struct.glsl"

#ifndef _NBL_GLSL_EXT_RADIXSORT_GET_PARAMETERS_DECLARED_
nbl_glsl_ext_RadixSort_Parameters_t nbl_glsl_ext_RadixSort_getParameters();
#define _NBL_GLSL_EXT_RADIXSORT_GET_PARAMETERS_DECLARED_
#endif

#ifndef _NBL_GLSL_EXT_RADIXSORT_PARAMETERS_METHODS_DEFINED_

uint nbl_glsl_ext_RadixSort_Parameters_t_getShift()
{
    nbl_glsl_ext_RadixSort_Parameters_t params = nbl_glsl_ext_RadixSort_getParameters();
    return params.shift;
}

uint nbl_glsl_ext_RadixSort_Parameters_t_getElementCountTotal()
{
    nbl_glsl_ext_RadixSort_Parameters_t params = nbl_glsl_ext_RadixSort_getParameters();
    return params.element_count_total;
}

#define _NBL_GLSL_EXT_RADIXSORT_PARAMETERS_METHODS_DEFINED_
#endif

#endif