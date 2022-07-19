// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_BARYCENTRIC_EXTENSIONS_INCLUDED_
#define _NBL_BUILTIN_GLSL_BARYCENTRIC_EXTENSIONS_INCLUDED_

#ifdef NBL_GL_NV_fragment_shader_barycentric
#extension GL_NV_fragment_shader_barycentric : enable
#elif defined(NBL_GL_AMD_shader_explicit_vertex_parameter)
#extension GL_AMD_shader_explicit_vertex_parameter : enable
#endif

#endif