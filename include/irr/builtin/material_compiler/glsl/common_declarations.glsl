#ifndef _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_COMMON_DECLARATIONS_INCLUDED_
#define _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_COMMON_DECLARATIONS_INCLUDED_

#include <irr/builtin/glsl/virtual_texturing/extensions.glsl>

#define instr_t uvec2
#define reg_t uint
#define params_t mat4x3
#define bxdf_eval_t vec3
#define eval_and_pdf_t vec4

#define INSTR_1ST_DWORD(instr) (instr).x
#define INSTR_2ND_DWORD(instr) (instr).y

struct bsdf_data_t
{
	uvec4 data[sizeof_bsdf_data];
};

struct instr_stream_t
{
	uint offset;
	uint count;
};

#endif