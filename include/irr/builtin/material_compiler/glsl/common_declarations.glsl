// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_COMMON_DECLARATIONS_INCLUDED_
#define _IRR_BUILTIN_MATERIAL_COMPILER_GLSL_COMMON_DECLARATIONS_INCLUDED_

#include <irr/builtin/glsl/virtual_texturing/extensions.glsl>

#define instr_t uvec2
#define reg_t uint
#define params_t mat4x3
#define bxdf_eval_t vec3
#define eval_and_pdf_t vec4

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