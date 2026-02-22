#ifndef _NBL_GLSL_EXT_ENVMAP_SAMPLING_PARAMETERS_STRUCT_INCLUDED_
#define _NBL_GLSL_EXT_ENVMAP_SAMPLING_PARAMETERS_STRUCT_INCLUDED_

struct nbl_glsl_ext_EnvmapSampling_LumaGenShaderData_t
{
    vec4 luminanceScales;
	uvec2 lumaMapResolution;
	uvec2 padding;
};

struct nbl_glsl_ext_EnvmapSampling_LumaMeasurement_t
{
	float xDirSum;
	float yDirSum;
	float zDirSum;
	float weightSum;
	float maxLuma;
};

#endif
