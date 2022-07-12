#ifndef _NBL_BUILTIN_GLSL_EXT_MITSUBA_LOADER_INSTANCE_DATA_STRUCT_INCLUDED_
#define _NBL_BUILTIN_GLSL_EXT_MITSUBA_LOADER_INSTANCE_DATA_STRUCT_INCLUDED_

#include <nbl/builtin/glsl/material_compiler/material_data.glsl>

struct nbl_glsl_ext_Mitsuba_Loader_instance_data_t
{
	mat4x3 tform;
	// TODO: remove, faster to invert in shader probably
	vec3 normalMatrixRow0;
	uint padding0;
	vec3 normalMatrixRow1;
	uint padding1;
	vec3 normalMatrixRow2;
	uint determinantSignBit;
	// end of TODO
	nbl_glsl_MC_material_data_t material;
};


#endif