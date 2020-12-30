// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_MATERIAL_COMPILER_GLSL_COMMON_DECLARATIONS_INCLUDED_
#define _NBL_BUILTIN_MATERIAL_COMPILER_GLSL_COMMON_DECLARATIONS_INCLUDED_

#include <nbl/builtin/glsl/virtual_texturing/extensions.glsl>
#include <nbl/builtin/glsl/colorspace/encodeCIEXYZ.glsl>
#include <nbl/builtin/glsl/bxdf/common.glsl>

#define nbl_glsl_instr_t uvec2
#define nbl_glsl_prefetch_instr_t uvec4
#define nbl_glsl_reg_t uint
#define nbl_glsl_params_t mat2x3
#define nbl_glsl_bxdf_eval_t vec3
#define nbl_glsl_eval_and_pdf_t vec4

struct nbl_glsl_bsdf_data_t
{
	uvec4 data[sizeof_bsdf_data];
};

struct nbl_glsl_instr_stream_t
{
	uint offset;
	uint count;
};

// all vectors (and dot products) have untouched orientation relatively to shader inputs
// therefore MC_precomputed_t::NdotV can be used to determine if we are inside a material
// (in case of precomp.NdotV<0.0, currInteraction will be set with -precomp.N)
struct nbl_glsl_MC_precomputed_t
{
	vec3 N;
	vec3 V;
	vec3 pos;
	bool frontface;
};

struct nbl_glsl_MC_microfacet_t
{
	nbl_glsl_AnisotropicMicrofacetCache inner;
	float TdotH2;
	float BdotH2;
};
void nbl_glsl_finalizeMicrofacet(inout nbl_glsl_MC_microfacet_t mf)
{
	mf.TdotH2 = mf.inner.TdotH * mf.inner.TdotH;
	mf.BdotH2 = mf.inner.BdotH * mf.inner.BdotH;
}

struct nbl_glsl_MC_interaction_t
{
	nbl_glsl_AnisotropicViewSurfaceInteraction inner;
	float TdotV2;
	float BdotV2;
};
void nbl_glsl_finalizeInteraction(inout nbl_glsl_MC_interaction_t i)
{
	i.TdotV2 = i.inner.TdotV * i.inner.TdotV;
	i.BdotV2 = i.inner.BdotV * i.inner.BdotV;
}

#define NBL_GLSL_MC_ALPHA_EPSILON 1.0e-08

#define NBL_GLSL_MC_CIE_XYZ_Luma_Y_coeffs transpose(nbl_glsl_sRGBtoXYZ)[1]

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