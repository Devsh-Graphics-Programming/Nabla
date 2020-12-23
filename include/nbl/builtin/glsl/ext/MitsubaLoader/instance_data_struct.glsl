#ifndef _NBL_BUILTIN_SHADERS_LOADERS_MITSUBA_INSTANCE_DATA_STRUCT_GLSL_INCLUDED_
#define _NBL_BUILTIN_SHADERS_LOADERS_MITSUBA_INSTANCE_DATA_STRUCT_GLSL_INCLUDED_

struct InstanceData
{
	mat4x3 tform;
	vec3 normalMatrixRow0;
	uint front_instr_offset;
	vec3 normalMatrixRow1;
	uint front_rem_pdf_count;
	vec3 normalMatrixRow2;
	uint _padding;//not needed
	uvec2 emissive;
	float determinant;
	uint front_prefetch_count;
	uint front_nprecomp_count;
	uint front_genchoice_count;
	uint front_prefetch_offset;
	uint back_instr_offset;
	uint back_rem_pdf_count;
	uint back_prefetch_count;
	uint back_nprecomp_count;
	uint back_genchoice_count;
	uint back_prefetch_offset;
};

#endif