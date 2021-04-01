#ifndef _NBL_GLSL_EXT_SCAN_PARAMETERS_INCLUDED_
#define _NBL_GLSL_EXT_SCAN_PARAMETERS_INCLUDED_

#include "nbl/builtin/glsl/ext/Scan/parameters_struct.glsl"

#ifndef _NBL_GLSL_EXT_SCAN_GET_PARAMETERS_DECLARED_
nbl_glsl_ext_Scan_Parameters_t nbl_glsl_ext_Scan_getParameters();
#define _NBL_GLSL_EXT_SCAN_GET_PARAMETERS_DECLARED_
#endif

#ifndef _NBL_GLSL_EXT_SCAN_PARAMETERS_METHODS_DEFINED_

uint nbl_glsl_ext_Scan_Parameters_t_getStride()
{
    nbl_glsl_ext_Scan_Parameters_t params = nbl_glsl_ext_Scan_getParameters();
    return params.stride;
}

uint nbl_glsl_ext_Scan_Parameters_t_getElementCountPass()
{
    nbl_glsl_ext_Scan_Parameters_t params = nbl_glsl_ext_Scan_getParameters();
    return params.element_count_pass;
}

uint nbl_glsl_ext_Scan_Parameters_t_getElementCountTotal()
{
    nbl_glsl_ext_Scan_Parameters_t params = nbl_glsl_ext_Scan_getParameters();
    return params.element_count_total;
}

#define _NBL_GLSL_EXT_SCAN_PARAMETERS_METHODS_DEFINED_
#endif

#endif