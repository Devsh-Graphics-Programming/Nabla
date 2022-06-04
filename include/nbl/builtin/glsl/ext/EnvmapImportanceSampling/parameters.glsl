#ifndef _NBL_GLSL_EXT_ENVMAP_SAMPLING_PARAMETERS_STRUCT_INCLUDED_
#define _NBL_GLSL_EXT_ENVMAP_SAMPLING_PARAMETERS_STRUCT_INCLUDED_
	
struct LumaMipMapGenShaderData_t
{
    vec4 luminanceScales;
    uint calcLuma;
    vec3 padding;
};

#endif
