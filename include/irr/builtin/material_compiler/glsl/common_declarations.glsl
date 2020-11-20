#ifndef _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_COMMON_DECLARATIONS_INCLUDED_
#define _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_COMMON_DECLARATIONS_INCLUDED_

#include <irr/builtin/glsl/virtual_texturing/extensions.glsl>
#include <irr/builtin/glsl/colorspace/encodeCIEXYZ.glsl>

#define instr_t uvec2
#define prefetch_instr_t uvec4
#define reg_t uint
#define params_t mat2x3
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

// all vectors (and dot products) have untouched orientation relatively to shader inputs
// therefore MC_precomputed_t::NdotV can be used to determine if we are inside a material
// (in case of precomp.NdotV<0.0, currInteraction will be set with -precomp.N)
struct MC_precomputed_t
{
	vec3 N;
	vec3 V;
	vec3 pos;
	bool frontface;
};

#define CIE_XYZ_Luma_Y_coeffs transpose(irr_glsl_sRGBtoXYZ)[1]

//#define MATERIAL_COMPILER_USE_SWTICH
#ifdef MATERIAL_COMPILER_USE_SWTICH
#define BEGIN_CASES(X)	switch (X) {
#define CASE_BEGIN(X,C) case C:
#define CASE_END		break;
#define CASE_OTHERWISE	default:
#define END_CASES		break; }
#else
#define BEGIN_CASES(X)
#define CASE_BEGIN(X,C) if (X==C)
#define CASE_END		else
#define CASE_OTHERWISE
#define END_CASES
#endif

#endif