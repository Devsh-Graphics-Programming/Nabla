#ifndef _IRR_BUILTIN_SHADERS_LOADERS_MITSUBA_INSTANCE_DATA_STRUCT_GLSL_INCLUDED_
#define _IRR_BUILTIN_SHADERS_LOADERS_MITSUBA_INSTANCE_DATA_STRUCT_GLSL_INCLUDED_

struct InstanceData
{
	mat4x3 tform;
	vec3 normalMatrixRow0;
	uint instr_offset;
	vec3 normalMatrixRow1;
	uint rem_pdf_count;
	vec3 normalMatrixRow2;
	uint _padding;//not needed
	uvec2 emissive;
	uint prefetch_count;
	uint nprecomp_count;
	uint genchoice_count;
	uint prefetch_offset;
};

#endif