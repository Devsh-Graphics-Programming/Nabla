// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_GLSL_COMPAT_CORE_INCLUDED_
#define _NBL_BUILTIN_HLSL_GLSL_COMPAT_CORE_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace glsl
{

#ifdef __HLSL_VERSION
/**
* Generic SPIR-V
*/
template<typename T>
T atomicAdd(NBL_REF_ARG(T) ptr, T value)
{
    return spirv::atomicAdd<T>(ptr, spv::ScopeDevice, spv::DecorationRelaxedPrecision, value);
}
template<typename T>
T atomicAnd(NBL_REF_ARG(T) ptr, T value)
{
    return spirv::atomicAnd<T>(ptr, spv::ScopeDevice, spv::DecorationRelaxedPrecision, value);
}
template<typename T>
T atomicOr(NBL_REF_ARG(T) ptr, T value)
{
    return spirv::atomicOr<T>(ptr, spv::ScopeDevice, spv::DecorationRelaxedPrecision, value);
}
template<typename T>
T atomicXor(NBL_REF_ARG(T) ptr, T value)
{
    return spirv::atomicXor<T>(ptr, spv::ScopeDevice, spv::DecorationRelaxedPrecision, value);
}
template<typename T>
T atomicMin(NBL_REF_ARG(T) ptr, T value)
{
    return spirv::atomicMin<T>(ptr, spv::ScopeDevice, spv::DecorationRelaxedPrecision, value);
}
template<typename T>
T atomicMax(NBL_REF_ARG(T) ptr, T value)
{
    return spirv::atomicMax<T>(ptr, spv::ScopeDevice, spv::DecorationRelaxedPrecision, value);
}
template<typename T>
T atomicExchange(NBL_REF_ARG(T) ptr, T value)
{
    return spirv::atomicExchange<T>(ptr, spv::ScopeDevice, spv::DecorationRelaxedPrecision, value);
}
template<typename T>
T atomicCompSwap(NBL_REF_ARG(T) ptr, T comparator, T value)
{
    return spirv::atomicCompSwap<T>(ptr, spv::ScopeDevice, spv::DecorationRelaxedPrecision, spv::DecorationRelaxedPrecision, value, comparator);
}

/**
 * GLSL extended math
 */
template<typename SquareMatrix> // NBL_REQUIRES() extents are square
SquareMatrix inverse(NBL_CONST_REF_ARG(SquareMatrix) mat)
{
    return spirv::matrixInverse(mat);
}


/**
 * For Vertex Shaders
 */
 // TODO: Extemely annoying that HLSL doesn't have references, so we can't transparently alias the variables as `&` :(
//void gl_Position() {spirv::}
uint32_t gl_VertexIndex() {return spirv::VertexIndex;}
uint32_t gl_InstanceIndex() {return spirv::InstanceIndex;}

/**
 * For Compute Shaders
 */

// TODO: Extemely annoying that HLSL doesn't have references, so we can't transparently alias the variables as `const&` :(
uint32_t3 gl_NumWorkGroups() {return spirv::NumWorkGroups;}
// TODO: DXC BUG prevents us from defining this!
uint32_t3 gl_WorkGroupSize();
uint32_t3 gl_WorkGroupID() {return spirv::WorkgroupId;}
uint32_t3 gl_LocalInvocationID() {return spirv::LocalInvocationId;}
uint32_t3 gl_GlobalInvocationID() {return spirv::GlobalInvocationId;}
uint32_t gl_LocalInvocationIndex() {return spirv::LocalInvocationIndex;}

void barrier() {
    spirv::controlBarrier(spv::ScopeWorkgroup, spv::ScopeWorkgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsWorkgroupMemoryMask);
}

/**
 * For Tessellation Control Shaders
 */
void tess_ctrl_barrier() {
    spirv::controlBarrier(spv::ScopeWorkgroup, spv::ScopeInvocation, 0);
}

void memoryBarrierShared() {
    spirv::memoryBarrier(spv::ScopeDevice, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsWorkgroupMemoryMask);
}

namespace impl 
{

template<typename T, bool isSigned, bool isIntegral>
struct bitfieldExtract {};

template<typename T, bool isSigned>
struct bitfieldExtract<T, isSigned, false>
{
    static T __call( T val, uint32_t offsetBits, uint32_t numBits )
    {
        bool nada;
        return val;
    }
};

template<typename T>
struct bitfieldExtract<T, true, true>
{
    static T __call( T val, uint32_t offsetBits, uint32_t numBits )
    {
        return spirv::bitFieldSExtract<T>( val, offsetBits, numBits );
    }
};

template<typename T>
struct bitfieldExtract<T, false, true>
{
    static T __call( T val, uint32_t offsetBits, uint32_t numBits )
    {
        return spirv::bitFieldUExtract<T>( val, offsetBits, numBits );
    } 
};

}

template<typename T>
T bitfieldExtract( T val, uint32_t offsetBits, uint32_t numBits )
{
    return impl::bitfieldExtract<T, is_signed<T>::value, is_integral<T>::value>::template  __call(val,offsetBits,numBits);
}

#else

template<typename T>
inline T bitfieldExtract( T val, uint32_t offsetBits, uint32_t numBits )
{
    return glm::bitfieldExtract<T>( val, int32_t( offsetBits ), int32_t( numBits ) );
}

#endif

}
}
}

#endif