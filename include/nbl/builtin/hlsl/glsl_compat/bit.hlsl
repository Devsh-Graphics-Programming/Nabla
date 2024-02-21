// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_GLSL_BIT_INCLUDED_
#define _NBL_BUILTIN_HLSL_GLSL_BIT_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace glsl
{
namespace impl 
{

#ifdef __HLSL_VERSION

template<typename T, bool isSigned, bool isIntegral>
struct bitfieldExtract
{
    T operator()( T val, uint32_t offsetBits, uint32_t numBits )
    {
        if( isIntegral )
        {
            static_assert( false, "T must be an integral type" );
        }
        if( isSigned )
        {
            return spirv::bitFieldSExtract<T>( val, offsetBits, numBits );
        }
        return spirv::bitFieldUExtract<T>( val, offsetBits, numBits );
    }
};
    
}
#endif

}
}
}
}

#endif