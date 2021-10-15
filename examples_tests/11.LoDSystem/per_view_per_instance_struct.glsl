// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifdef __cplusplus
#define mat4 nbl::core::matrix4SIMD
#endif
struct PerViewPerInstance_t
{
    mat4 mvp;
};
#ifdef __cplusplus
#undef mat4
#else
#define nbl_glsl_PerViewPerInstance_t PerViewPerInstance_t
#endif