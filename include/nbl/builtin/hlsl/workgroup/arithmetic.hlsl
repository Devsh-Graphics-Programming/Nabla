// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_ARITHMETIC_INCLUDED_


#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/workgroup/ballot.hlsl"
#include "nbl/builtin/hlsl/workgroup/broadcast.hlsl"
#include "nbl/builtin/hlsl/workgroup/shared_scan.hlsl"


namespace nbl
{
namespace hlsl
{
namespace workgroup
{

// TODO: with Boost PP at some point
//#define NBL_ALIAS_CALL_OPERATOR_TO_STATIC_IMPL(OPTIONAL_TEMPLATE,RETURN_TYPE,/*tuples of argument types and names*/...)
//#define NBL_ALIAS_TEMPLATED_CALL_OPERATOR_TO_IMPL(TEMPLATE,RETURN_TYPE,/*tuples of argument types and names*/...)

template<class BinOp, uint16_t ItemCount>
struct reduction
{
    using type_t = typename BinOp::type_t;

    template<class Accessor>
    static type_t __call(NBL_CONST_REF_ARG(type_t) value, NBL_REF_ARG(Accessor) accessor)
    {
        impl::reduce<BinOp,ItemCount> fn;
        fn.template __call<Accessor>(value,accessor);
        accessor.workgroupExecutionAndMemoryBarrier();
        return Broadcast<type_t,Accessor>(fn.lastLevelScan,accessor,fn.lastInvocationInLevel);
    }
};

template<class BinOp, uint16_t ItemCount>
struct inclusive_scan
{
    using type_t = typename BinOp::type_t;

    template<class Accessor>
    static type_t __call(NBL_CONST_REF_ARG(type_t) value, NBL_REF_ARG(Accessor) accessor)
    {
        impl::scan<BinOp,false,ItemCount> fn;
        return fn.template __call<Accessor>(value,accessor);
    }
};

template<class BinOp, uint16_t ItemCount>
struct exclusive_scan
{
    using type_t = typename BinOp::type_t;

    template<class Accessor>
    static type_t __call(NBL_CONST_REF_ARG(type_t) value, NBL_REF_ARG(Accessor) accessor)
    {
        impl::scan<BinOp,true,ItemCount> fn;
        return fn.template __call<Accessor>(value,accessor);
    }
};

/**
 * Gives us the sum (reduction) of all ballots for the ItemCount bits of a workgroup.
 *
 * Only the first few invocations are used for performing the sum 
 * since we only have `1/32` amount of uints that we need 
 * to add together.
 * 
 * We add them all in the shared array index after the last DWORD 
 * that is used for the ballots. For example, if we have 128 workgroup size,
 * then the array index in which we accumulate the sum is `4` since 
 * indexes 0..3 are used for ballots.
 */ 
namespace impl
{
template<uint16_t ItemCount, class BallotAccessor, class ArithmeticAccessor, template<class,uint16_t> class op_t>
uint32_t ballotPolyCount(NBL_REF_ARG(BallotAccessor) ballotAccessor, NBL_REF_ARG(ArithmeticAccessor) arithmeticAccessor, NBL_REF_ARG(uint32_t) localBitfield)
{
    localBitfield = 0u;
    if (SubgroupContiguousIndex()<impl::BallotDWORDCount(ItemCount))
        localBitfield = ballotAccessor.get(SubgroupContiguousIndex());
    return op_t<plus<uint32_t>,impl::ballot_dword_count<ItemCount>::value>::template __call<ArithmeticAccessor>(countbits(localBitfield),arithmeticAccessor);
}
}

template<uint16_t ItemCount, class BallotAccessor, class ArithmeticAccessor>
uint16_t ballotBitCount(NBL_REF_ARG(BallotAccessor) ballotAccessor, NBL_REF_ARG(ArithmeticAccessor) arithmeticAccessor)
{
    uint32_t dummy;
    return uint16_t(impl::ballotPolyCount<ItemCount,BallotAccessor,ArithmeticAccessor,reduction>(ballotAccessor,arithmeticAccessor,dummy));
}

template<uint16_t ItemCount, class BallotAccessor, class ArithmeticAccessor>
uint16_t ballotInclusiveBitCount(NBL_REF_ARG(BallotAccessor) ballotAccessor, NBL_REF_ARG(ArithmeticAccessor) arithmeticAccessor)
{
    uint32_t localBitfield;
    uint32_t count = impl::ballotPolyCount<ItemCount,BallotAccessor,ArithmeticAccessor,exclusive_scan>(ballotAccessor,arithmeticAccessor,localBitfield);
    // only using part of the mask is on purpose, I'm only interested in LSB
    return uint16_t(countbits(glsl::gl_SubgroupLeMask()[0]&localBitfield)+count);
}

template<uint16_t ItemCount, class BallotAccessor, class ArithmeticAccessor>
uint16_t ballotExclusiveBitCount(NBL_REF_ARG(BallotAccessor) ballotAccessor, NBL_REF_ARG(ArithmeticAccessor) arithmeticAccessor)
{
    uint32_t localBitfield;
    uint32_t count = impl::ballotPolyCount<ItemCount,BallotAccessor,ArithmeticAccessor,exclusive_scan>(ballotAccessor,arithmeticAccessor,localBitfield);
    // only using part of the mask is on purpose, I'm only interested in LSB
    return uint16_t(countbits(glsl::gl_SubgroupLtMask()[0]&localBitfield)+count);
}

}
}
}

#endif