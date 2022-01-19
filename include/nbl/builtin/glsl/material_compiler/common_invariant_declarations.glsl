// Copyright (C) 2020-2021 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_MATERIAL_COMPILER_COMMON_INVARIANT_DECLARATIONS_INCLUDED_
#define _NBL_BUILTIN_GLSL_MATERIAL_COMPILER_COMMON_INVARIANT_DECLARATIONS_INCLUDED_

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

#define nbl_glsl_MC_instr_t uvec2
#define nbl_glsl_MC_prefetch_instr_t uvec4
#define nbl_glsl_MC_reg_t uint
#define nbl_glsl_MC_params_t mat2x3
#define nbl_glsl_MC_bxdf_spectrum_t vec3

struct nbl_glsl_MC_instr_stream_t
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
	bool frontface;
};


#include <nbl/builtin/glsl/colorspace/encodeCIEXYZ.glsl>

#define NBL_GLSL_MC_CIE_XYZ_Luma_Y_coeffs transpose(nbl_glsl_sRGBtoXYZ)[1]


#include <nbl/builtin/glsl/bxdf/common.glsl>

#define NBL_GLSL_MC_ALPHA_EPSILON 1.0e-8

struct nbl_glsl_MC_interaction_t
{
	nbl_glsl_AnisotropicViewSurfaceInteraction inner;
	float TdotV2;
	float BdotV2;
};
void nbl_glsl_MC_finalizeInteraction(inout nbl_glsl_MC_interaction_t i)
{
	i.TdotV2 = i.inner.TdotV * i.inner.TdotV;
	i.BdotV2 = i.inner.BdotV * i.inner.BdotV;
}

struct nbl_glsl_MC_microfacet_t
{
	nbl_glsl_AnisotropicMicrofacetCache inner;
	float TdotH2;
	float BdotH2;
};

void nbl_glsl_MC_finalizeMicrofacet(inout nbl_glsl_MC_microfacet_t mf)
{
	mf.TdotH2 = mf.inner.TdotH * mf.inner.TdotH;
	mf.BdotH2 = mf.inner.BdotH * mf.inner.BdotH;
}

#include <nbl/builtin/glsl/format/decode.glsl>

struct nbl_glsl_MC_oriented_material_t
{
	uvec2 emissive;
	uint prefetch_offset;
	uint prefetch_count;
	uint instr_offset;
	uint rem_pdf_count;
	uint nprecomp_count;
	uint genchoice_count;
};
vec3 nbl_glsl_MC_oriented_material_t_getEmissive(in nbl_glsl_MC_oriented_material_t orientedMaterial)
{
	return nbl_glsl_decodeRGB19E7(orientedMaterial.emissive);
}
//rem'n'pdf and eval use the same instruction stream
nbl_glsl_MC_instr_stream_t nbl_glsl_MC_oriented_material_t_getEvalStream(in nbl_glsl_MC_oriented_material_t orientedMaterial)
{
	return nbl_glsl_MC_instr_stream_t( orientedMaterial.instr_offset,orientedMaterial.rem_pdf_count );
}
nbl_glsl_MC_instr_stream_t nbl_glsl_MC_oriented_material_t_getRemAndPdfStream(in nbl_glsl_MC_oriented_material_t orientedMaterial)
{
	return nbl_glsl_MC_instr_stream_t( orientedMaterial.instr_offset,orientedMaterial.rem_pdf_count );
}
nbl_glsl_MC_instr_stream_t nbl_glsl_MC_oriented_material_t_getGenChoiceStream(in nbl_glsl_MC_oriented_material_t orientedMaterial)
{
	return nbl_glsl_MC_instr_stream_t( orientedMaterial.instr_offset+orientedMaterial.rem_pdf_count,orientedMaterial.genchoice_count );
}
nbl_glsl_MC_instr_stream_t nbl_glsl_MC_oriented_material_t_getTexPrefetchStream(in nbl_glsl_MC_oriented_material_t orientedMaterial)
{
	return nbl_glsl_MC_instr_stream_t( orientedMaterial.prefetch_offset,orientedMaterial.prefetch_count );
}
nbl_glsl_MC_instr_stream_t nbl_glsl_MC_oriented_material_t_getNormalPrecompStream(in nbl_glsl_MC_oriented_material_t orientedMaterial)
{
	return nbl_glsl_MC_instr_stream_t( orientedMaterial.instr_offset+orientedMaterial.rem_pdf_count+orientedMaterial.genchoice_count,orientedMaterial.nprecomp_count );
}

struct nbl_glsl_MC_material_data_t
{
	nbl_glsl_MC_oriented_material_t front;
	nbl_glsl_MC_oriented_material_t back;
};
nbl_glsl_MC_oriented_material_t nbl_glsl_MC_material_data_t_getOriented(in nbl_glsl_MC_material_data_t material, in bool frontface)
{
	return frontface ? material.front:material.back;
}

#endif