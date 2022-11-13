// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_ALGORITHM_INCLUDED_
#define _NBL_BUILTIN_HLSL_ALGORITHM_INCLUDED_

#include "binops.hlsl"

namespace nbl
{
namespace hlsl
{


namespace impl
{

// TODO: move this to some other header
bool isNPoT(const uint x)
{
    return x&(x-1);
}

template<class Accessor, class Comparator>
struct bound_t
{
    static bound_t<Accessor,Comparator> setup(uint begin, const uint end, const typename Accessor::value_type _value, const Comparator _comp)
    {
        bound_t<Accessor,Comparator> retval;
        retval.comp = _comp;
        retval.value = _value;
        retval.it = begin;
        retval.len = end-begin;
        return retval;
    }


    uint operator()(inout Accessor accessor)
    {
        if (isNPoT(len))
        {
            const uint newLen = 0x1u<<firstbithigh(len);
            const uint testPoint = it+(len-newLen);
            len = newLen;
            comp_step(accessor,testPoint);
        }
        while (len)
        {
            // could unroll 3 times or more
            iteration(accessor);
            iteration(accessor);
        }
        comp_step(accessor,it,it+1u);
        return it;
    }

    void iteration(inout Accessor accessor)
    {
        len >>= 1;
        const uint mid = it+len;
        comp_step(accessor,mid);
    }

    void comp_step(inout Accessor accessor, const uint testPoint, const uint rightBegin)
    {
        if (comp(accessor[testPoint],value))
            it = rightBegin;
    }
    void comp_step(inout Accessor accessor, const uint testPoint)
    {
        comp_step(accessor,testPoint,testPoint);
    }

    Comparator comp;
    typename Accessor::value_type value;
    uint it;
    uint len;
};

template<class Accessor, class Comparator>
struct lower_to_upper_comparator_transform_t
{
    bool operator()(const typename Accessor::value_type lhs, const typename Accessor::value_type rhs)
    {
        return !comp(rhs,lhs);
    }

    Comparator comp;
};

}


template<class Accessor, class Comparator>
uint lower_bound(inout Accessor accessor, const uint begin, const uint end, const typename Accessor::value_type value, const Comparator comp)
{
    impl::bound_t<Accessor,Comparator> implementation = impl::bound_t<Accessor,Comparator>::setup(begin,end,value,comp);
    return implementation(accessor);
}

template<class Accessor, class Comparator>
uint upper_bound(inout Accessor accessor, const uint begin, const uint end, const typename Accessor::value_type value, const Comparator comp)
{
    using TransformedComparator = impl::lower_to_upper_comparator_transform_t<Accessor,Comparator>;
    TransformedComparator transformedComparator;
    transformedComparator.comp = comp;
    return lower_bound<Accessor,TransformedComparator>(accessor,begin,end,value,transformedComparator);
}


namespace impl
{

// extra indirection due to https://github.com/microsoft/DirectXShaderCompiler/issues/4771
template<class Accessor, typename T>
uint lower_bound(inout Accessor accessor, const uint begin, const uint end, const T value)
{
    using Comparator = binops::comparator_lt_t<T>;
    Comparator comp;
    return nbl::hlsl::lower_bound<Accessor,Comparator>(accessor,begin,end,value,comp);
}
template<class Accessor, typename T>
uint upper_bound(inout Accessor accessor, const uint begin, const uint end, const T value)
{
    using Comparator = binops::comparator_lt_t<T>;
    Comparator comp;
    return nbl::hlsl::upper_bound<Accessor,Comparator>(accessor,begin,end,value,comp);
}

}

template<class Accessor>
uint lower_bound(inout Accessor accessor, const uint begin, const uint end, const typename Accessor::value_type value)
{
    return impl::lower_bound<Accessor,typename Accessor::value_type>(accessor,begin,end,value);
}
template<class Accessor>
uint upper_bound(inout Accessor accessor, const uint begin, const uint end, const typename Accessor::value_type value)
{
    return impl::upper_bound<Accessor,typename Accessor::value_type>(accessor,begin,end,value);
}

}
}

#endif
