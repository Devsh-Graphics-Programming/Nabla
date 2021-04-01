// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_VIRTUAL_TEXTURING_EXTENSIONS_INCLUDED_
#define _NBL_BUILTIN_GLSL_VIRTUAL_TEXTURING_EXTENSIONS_INCLUDED_

#ifdef NBL_GL_EXT_nonuniform_qualifier
#extension GL_EXT_nonuniform_qualifier : enable
#else
#extension GL_KHR_shader_subgroup_ballot : enable
#endif

#endif