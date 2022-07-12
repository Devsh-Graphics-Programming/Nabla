// Copyright (C) 2020-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_GLSL_MATERIAL_COMPILER_MATERIAL_DATA_INCLUDED_
#define _NBL_BUILTIN_GLSL_MATERIAL_COMPILER_MATERIAL_DATA_INCLUDED_


#include <nbl/builtin/glsl/material_compiler/oriented_material.glsl>


struct nbl_glsl_MC_material_data_t
{
	nbl_glsl_MC_oriented_material_t front;
	nbl_glsl_MC_oriented_material_t back;
};

#ifndef __cplusplus
nbl_glsl_MC_oriented_material_t nbl_glsl_MC_material_data_t_getOriented(in nbl_glsl_MC_material_data_t material, in bool frontface)
{
	return frontface ? material.front:material.back;
}
#endif

#endif