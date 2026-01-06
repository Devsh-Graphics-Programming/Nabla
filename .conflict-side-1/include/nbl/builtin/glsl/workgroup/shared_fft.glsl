// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_WORKGROUP_SHARED_FFT_INCLUDED_
#define _NBL_BUILTIN_GLSL_WORKGROUP_SHARED_FFT_INCLUDED_


#include <nbl/builtin/glsl/macros.glsl>
#include <nbl/builtin/glsl/workgroup/basic.glsl>


// TODO: can we reduce it?
#define _NBL_GLSL_WORKGROUP_FFT_SHARED_SIZE_NEEDED_ (_NBL_GLSL_WORKGROUP_SIZE_*4)


#ifndef _NBL_GLSL_WORKGROUP_SIZE_LOG2_
    #if NBL_GLSL_IS_NOT_POT(_NBL_GLSL_WORKGROUP_SIZE_)
        #error "Radix2 FFT requires workgroup to be a Power of Two!"
    #endif
    #define _NBL_GLSL_WORKGROUP_SIZE_LOG2_ findMSB(_NBL_GLSL_WORKGROUP_SIZE_)
#endif


#endif