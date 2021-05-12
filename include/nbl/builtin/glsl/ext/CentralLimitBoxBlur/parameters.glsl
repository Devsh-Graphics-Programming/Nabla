#ifndef _NBL_GLSL_EXT_BLUR_PARAMETERS_INCLUDED_
#define _NBL_GLSL_EXT_BLUR_PARAMETERS_INCLUDED_

#include "nbl/builtin/glsl/ext/CentralLimitBoxBlur/parameters_struct.glsl"

#ifndef _NBL_GLSL_EXT_BLUR_GET_PARAMETERS_DECLARED_
nbl_glsl_ext_Blur_Parameters_t nbl_glsl_ext_Blur_getParameters();
#define _NBL_GLSL_EXT_BLUR_GET_PARAMETERS_DECLARED_
#endif

#ifndef _NBL_GLSL_EXT_BLUR_PARAMETERS_METHODS_DEFINED_

uint nbl_glsl_ext_Blur_Parameters_t_getWidth()
{
    nbl_glsl_ext_Blur_Parameters_t params = nbl_glsl_ext_Blur_getParameters();
    return params.width;
}

uint nbl_glsl_ext_Blur_Parameters_t_getHeight()
{
    nbl_glsl_ext_Blur_Parameters_t params = nbl_glsl_ext_Blur_getParameters();
    return params.height;
}

float nbl_glsl_ext_Blur_Parameters_t_getRadius()
{
    nbl_glsl_ext_Blur_Parameters_t params = nbl_glsl_ext_Blur_getParameters();
    return params.radius;
}

uint nbl_glsl_ext_Blur_Parameters_t_getDirection()
{
    nbl_glsl_ext_Blur_Parameters_t params = nbl_glsl_ext_Blur_getParameters();
    return params.direction;
}

#define _NBL_GLSL_EXT_BLUR_PARAMETERS_METHODS_DEFINED_
#endif

#endif