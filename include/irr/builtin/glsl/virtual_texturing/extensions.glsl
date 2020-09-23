// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _IRR_BUILTIN_GLSL_VIRTUAL_TEXTURING_EXTENSIONS_INCLUDED_
#define _IRR_BUILTIN_GLSL_VIRTUAL_TEXTURING_EXTENSIONS_INCLUDED_

#extension GL_EXT_nonuniform_qualifier : enable

#ifdef IRR_GL_NV_gpu_shader5
#define IRR_GL_EXT_nonuniform_qualifier // TODO: we need to overhaul our GLSL preprocessing system to match what SPIRV-Cross actually does
#endif

#ifndef IRR_GL_EXT_nonuniform_qualifier
#error "SPIR-V Cross did not implement GL_KHR_shader_subgroup_ballot on GLSL yet!"
#endif

#endif