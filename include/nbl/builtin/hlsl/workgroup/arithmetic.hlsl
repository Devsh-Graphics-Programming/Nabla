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

template<class BinOp, uint16_t ItemCount, class device_capabilities=void>
struct reduction
{
    using type_t = typename BinOp::type_t;

    template<class Accessor>
    static type_t __call(NBL_CONST_REF_ARG(type_t) value, NBL_REF_ARG(Accessor) accessor)
    {
        impl::reduce<BinOp,ItemCount,device_capabilities> fn;
        fn.template __call<Accessor>(value,accessor);
        accessor.workgroupExecutionAndMemoryBarrier();
        return Broadcast<type_t,Accessor>(fn.lastLevelScan,accessor,fn.lastInvocationInLevel);
    }
};

template<class BinOp, uint16_t ItemCount, class device_capabilities=void>
struct inclusive_scan
{
    using type_t = typename BinOp::type_t;

    template<class Accessor>
    static type_t __call(NBL_CONST_REF_ARG(type_t) value, NBL_REF_ARG(Accessor) accessor)
    {
        impl::scan<BinOp,false,ItemCount,device_capabilities> fn;
        return fn.template __call<Accessor>(value,accessor);
    }
};

template<class BinOp, uint16_t ItemCount, class device_capabilities=void>
struct exclusive_scan
{
    using type_t = typename BinOp::type_t;

    template<class Accessor>
    static type_t __call(NBL_CONST_REF_ARG(type_t) value, NBL_REF_ARG(Accessor) accessor)
    {
        impl::scan<BinOp,true,ItemCount,device_capabilities> fn;
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
template<uint16_t ItemCount, class BallotAccessor>
uint16_t ballotCountedBitDWORD(NBL_REF_ARG(BallotAccessor) ballotAccessor)
{
    const uint32_t index = SubgroupContiguousIndex();
    static const uint16_t DWORDCount = impl::ballot_dword_count<ItemCount>::value;
    if (index<DWORDCount)
    {
        uint32_t bitfield = ballotAccessor.get(index);
        // strip unwanted bits from bitfield of the last item
        const uint16_t Remainder = ItemCount&31;
        if (Remainder!=0 && index==DWORDCount-1)
            bitfield &= (0x1u<<Remainder)-1;
        return uint16_t(countbits(bitfield));
    }
    return 0;
}

template<bool Exclusive, uint16_t ItemCount, class BallotAccessor, class ArithmeticAccessor, class device_capabilities>
uint16_t ballotScanBitCount(NBL_REF_ARG(BallotAccessor) ballotAccessor, NBL_REF_ARG(ArithmeticAccessor) arithmeticAccessor)
{
    const uint16_t subgroupIndex = SubgroupContiguousIndex();
    const uint16_t bitfieldIndex = getDWORD(subgroupIndex);
    const uint32_t localBitfield = ballotAccessor.get(bitfieldIndex);

    static const uint16_t DWORDCount = impl::ballot_dword_count<ItemCount>::value;
    uint32_t count = exclusive_scan<plus<uint32_t>,DWORDCount,device_capabilities>::template __call<ArithmeticAccessor>(
        ballotCountedBitDWORD<ItemCount,BallotAccessor>(ballotAccessor),
        arithmeticAccessor
    );
    arithmeticAccessor.workgroupExecutionAndMemoryBarrier();
    if (subgroupIndex<DWORDCount)
        arithmeticAccessor.set(subgroupIndex,count);
    arithmeticAccessor.workgroupExecutionAndMemoryBarrier();
    count = arithmeticAccessor.get(bitfieldIndex);
    return uint16_t(countbits(localBitfield&(Exclusive ? glsl::gl_SubgroupLtMask():glsl::gl_SubgroupLeMask())[getDWORD(uint16_t(glsl::gl_SubgroupInvocationID()))])+count);
}
}

template<uint16_t ItemCount, class BallotAccessor, class ArithmeticAccessor, class device_capabilities=void>
uint16_t ballotBitCount(NBL_REF_ARG(BallotAccessor) ballotAccessor, NBL_REF_ARG(ArithmeticAccessor) arithmeticAccessor)
{
    static const uint16_t DWORDCount = impl::ballot_dword_count<ItemCount>::value;
    return uint16_t(reduction<plus<uint32_t>,DWORDCount,device_capabilities>::template __call<ArithmeticAccessor>(
        impl::ballotCountedBitDWORD<ItemCount,BallotAccessor>(ballotAccessor),
        arithmeticAccessor
    ));
}

template<uint16_t ItemCount, class BallotAccessor, class ArithmeticAccessor, class device_capabilities=void>
uint16_t ballotInclusiveBitCount(NBL_REF_ARG(BallotAccessor) ballotAccessor, NBL_REF_ARG(ArithmeticAccessor) arithmeticAccessor)
{
    return impl::ballotScanBitCount<false,ItemCount,BallotAccessor,ArithmeticAccessor,device_capabilities>(ballotAccessor,arithmeticAccessor);
}

template<uint16_t ItemCount, class BallotAccessor, class ArithmeticAccessor, class device_capabilities=void>
uint16_t ballotExclusiveBitCount(NBL_REF_ARG(BallotAccessor) ballotAccessor, NBL_REF_ARG(ArithmeticAccessor) arithmeticAccessor)
{
    return impl::ballotScanBitCount<true,ItemCount,BallotAccessor,ArithmeticAccessor,device_capabilities>(ballotAccessor,arithmeticAccessor);
}

}
}
}

#endif