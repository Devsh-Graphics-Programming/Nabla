// Copyright (C) 2020-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_GLSL_MATERIAL_COMPILER_ORIENTED_MATERIAL_INCLUDED_
#define _NBL_BUILTIN_GLSL_MATERIAL_COMPILER_ORIENTED_MATERIAL_INCLUDED_

struct nbl_glsl_MC_oriented_material_t
{
	uvec2 emissive;
	// TODO: derive/define upper bounds for instruction counts and bitpack them!
	uint prefetch_offset;
	uint prefetch_count;
	uint instr_offset;
	uint rem_pdf_count;
	uint nprecomp_count;
	uint genchoice_count;
};

#ifndef __cplusplus
#include <nbl/builtin/glsl/format/decode.glsl>
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
#endif

#endif