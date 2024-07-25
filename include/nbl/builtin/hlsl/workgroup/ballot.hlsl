// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_BALLOT_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_BALLOT_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/subgroup/arithmetic_portability.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace workgroup
{

namespace impl
{
uint16_t getDWORD(uint16_t invocation)
{
    uint16_t dword = invocation>>5;
    return dword; // log2 of sizeof(uint32_t)*8
}

// essentially means 'how many DWORDs are needed to store ballots in bitfields, for each invocation of `itemCount`
uint16_t BallotDWORDCount(const uint16_t itemCount)
{
    return getDWORD(itemCount+31); // round up, in case all items don't fit in even number of DWORDs
}

// this silly thing exists only because we can't make the above `constexpr`
template<uint16_t ItemCount>
struct ballot_dword_count : integral_constant<uint16_t,((ItemCount+31)>>5)> {};

// Specialize on subgroup size
template<bool SubgroupSizeSmallerThanUint32>
struct ballot
{
    template<class Accessor>
    static void __call(const uint32_t4 subgroupBallot, const uint16_t subgroupInvocation, NBL_REF_ARG(Accessor) accessor)
    {
        if (glsl::subgroupElect())
        {
            const uint16_t SubgroupSizeLog2 = uint16_t(glsl::gl_SubgroupSizeLog2());
            const uint16_t subgroupID = uint16_t(glsl::gl_SubgroupID());
            const uint16_t shift = (subgroupID<<SubgroupSizeLog2)&uint16_t(0x31u);
            accessor.atomicOr(subgroupID>>(5-SubgroupSizeLog2),subgroupBallot[0]<<shift);
        }
    }
};
template<>
struct ballot<false>
{
    template<class Accessor>
    static void __call(const uint32_t4 subgroupBallot, const uint16_t subgroupInvocation, NBL_REF_ARG(Accessor) accessor)
    {
        const uint16_t SubgroupSizeLog2 = uint16_t(glsl::gl_SubgroupSizeLog2());
        const uint16_t subgroupID = uint16_t(glsl::gl_SubgroupID());
        const uint16_t destIx = subgroupID<<(SubgroupSizeLog2-5);

        const uint16_t UsefulComponents = uint16_t(0x1u)<<(SubgroupSizeLog2-5);
        if (subgroupInvocation<UsefulComponents)
            accessor.set(destIx+subgroupInvocation,subgroupBallot[subgroupInvocation]);
    }
};
}

/**
 * Simple ballot function.
 *
 * Each invocation provides a boolean value. Each value is represented by a 
 * single bit of a Uint. For example, if invocation index 5 supplies `value = true` 
 * then the Uint will be ...00100000
 * This way we can encode 32 invocations into a single Uint.
 *
 * All Uints are kept in contiguous accessor memory in an array (shared is best).
 * The size of that array is based on the ItemCount.
 *
 * For each group of 32 invocations, a DWORD is assigned to the array (i.e. a 32-bit value, in this case Uint).
 * For example, for a workgroup size 128, 4 DWORDs are needed.
 * For each invocation index, we can find its respective DWORD index in the accessor array 
 * by calling the getDWORD function.
 * 
 * TODO: try do it with 64bit ints instead? (requires modified/adapted accessor)
 * NOTE: Until we can detect exact signatures of functions, do everything as uint32_t.
 */
template<class Accessor>
void ballot(const bool value, NBL_REF_ARG(Accessor) accessor)
{
    const uint32_t4 bitfield = glsl::subgroupBallot(value);
    const uint16_t subgroupInvocation = uint16_t(glsl::gl_SubgroupInvocationID());

    if (glsl::gl_SubgroupSizeLog2()<4)
        impl::ballot<true>::template __call<Accessor>(bitfield,subgroupInvocation,accessor);
    else
        impl::ballot<false>::template __call<Accessor>(bitfield,subgroupInvocation,accessor);
}

template<class Accessor>
bool ballotBitExtract(const uint16_t index, NBL_REF_ARG(Accessor) accessor)
{
    assert(index<Volume());
    return bool(accessor.get(impl::getDWORD(index))&(1u<<(index&31u)));
}

/**
 * Once we have assigned ballots in the shared array, we can 
 * extract any invocation's ballot value using this function.
 */
template<class Accessor>
bool inverseBallot(NBL_REF_ARG(Accessor) accessor)
{
    return ballotBitExtract<Accessor>(SubgroupContiguousIndex(),accessor);
}

}
}
}
#endif