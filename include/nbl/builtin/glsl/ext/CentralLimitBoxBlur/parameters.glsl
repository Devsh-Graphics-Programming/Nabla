#ifndef _NBL_GLSL_EXT_BLUR_PARAMETERS_INCLUDED_
#define _NBL_GLSL_EXT_BLUR_PARAMETERS_INCLUDED_

#include "nbl/builtin/glsl/ext/CentralLimitBoxBlur/parameters_struct.glsl"

#ifndef _NBL_GLSL_EXT_BLUR_GET_PARAMETERS_DECLARED_
nbl_glsl_ext_Blur_Parameters_t nbl_glsl_ext_Blur_getParameters();
#define _NBL_GLSL_EXT_BLUR_GET_PARAMETERS_DECLARED_
#endif

#ifndef _NBL_GLSL_EXT_BLUR_PARAMETERS_METHODS_DEFINED_

uvec3 nbl_glsl_ext_Blur_Parameters_t_getDimensions()
{
    nbl_glsl_ext_Blur_Parameters_t params = nbl_glsl_ext_Blur_getParameters();
    return params.input_dimensions.xyz;
}

float nbl_glsl_ext_Blur_Parameters_t_getRadius()
{
    nbl_glsl_ext_Blur_Parameters_t params = nbl_glsl_ext_Blur_getParameters();
    return params.radius;
}

uint nbl_glsl_ext_Blur_Parameters_t_getDirection()
{
    nbl_glsl_ext_Blur_Parameters_t params = nbl_glsl_ext_Blur_getParameters();
    return (params.input_dimensions.w >> 30) & 0x3u;
}

uint nbl_glsl_ext_Blur_Parameters_t_getChannelCount()
{
    nbl_glsl_ext_Blur_Parameters_t params = nbl_glsl_ext_Blur_getParameters();
    return (params.input_dimensions.w >> 28) & 0x3u;
}

uint nbl_glsl_ext_Blur_Parameters_t_getWrapMode()
{
    nbl_glsl_ext_Blur_Parameters_t params = nbl_glsl_ext_Blur_getParameters();
    return (params.input_dimensions.w >> 26) & 0x3u;
}

uvec4 nbl_glsl_ext_Blur_Parameters_t_getInputStrides()
{
    nbl_glsl_ext_Blur_Parameters_t params = nbl_glsl_ext_Blur_getParameters();
    return params.input_strides;
}

uvec4 nbl_glsl_ext_Blur_Parameters_t_getOutputStrides()
{
    nbl_glsl_ext_Blur_Parameters_t params = nbl_glsl_ext_Blur_getParameters();
    return params.output_strides;
}

#define _NBL_GLSL_EXT_BLUR_PARAMETERS_METHODS_DEFINED_
#endif

#endif