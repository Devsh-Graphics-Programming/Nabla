// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_ALGORITHM_INCLUDED_
#define _NBL_BUILTIN_HLSL_ALGORITHM_INCLUDED_

#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/math/intutil.hlsl"

namespace nbl
{
namespace hlsl
{

namespace impl
{
#ifdef __HLSL_VERSION

    // TODO: use structs

    template<typename T>
    NBL_CONSTEXPR_FUNC void swap(NBL_REF_ARG(T) lhs, NBL_REF_ARG(T) rhs)
    {
        T tmp = lhs;
        lhs = rhs;
        rhs = tmp;
    }

    template<>
    NBL_CONSTEXPR_FUNC void swap(NBL_REF_ARG(uint16_t) lhs, NBL_REF_ARG(uint16_t) rhs)
    {
        lhs ^= rhs;
        rhs ^= lhs;
        lhs ^= rhs;
    }

    template<>
    NBL_CONSTEXPR_FUNC void swap(NBL_REF_ARG(uint32_t) lhs, NBL_REF_ARG(uint32_t) rhs)
    {
        lhs ^= rhs;
        rhs ^= lhs;
        lhs ^= rhs;
    }

    template<>
    NBL_CONSTEXPR_FUNC void swap(NBL_REF_ARG(uint64_t) lhs, NBL_REF_ARG(uint64_t) rhs)
    {
        lhs ^= rhs;
        rhs ^= lhs;
        lhs ^= rhs;
    }

    template<>
    NBL_CONSTEXPR_FUNC void swap(NBL_REF_ARG(int16_t) lhs, NBL_REF_ARG(int16_t) rhs)
    {
        lhs ^= rhs;
        rhs ^= lhs;
        lhs ^= rhs;
    }

    template<>
    NBL_CONSTEXPR_FUNC void swap(NBL_REF_ARG(int32_t) lhs, NBL_REF_ARG(int32_t) rhs)
    {
        lhs ^= rhs;
        rhs ^= lhs;
        lhs ^= rhs;
    }

    template<>
    NBL_CONSTEXPR_FUNC void swap(NBL_REF_ARG(int64_t) lhs, NBL_REF_ARG(int64_t) rhs)
    {
        lhs ^= rhs;
        rhs ^= lhs;
        lhs ^= rhs;
    }
#else
    template<typename T>
    NBL_CONSTEXPR_FUNC void swap(NBL_REF_ARG(T) lhs, NBL_REF_ARG(T) rhs)
    {
        std::swap(lhs, rhs);
    }
#endif
}

template<typename T>
NBL_CONSTEXPR_FUNC void swap(NBL_REF_ARG(T) lhs, NBL_REF_ARG(T) rhs)
{
    impl::swap<T>(lhs, rhs);
}


namespace impl
{

template<class Accessor, class Comparator>
struct bound_t
{
    static bound_t<Accessor,Comparator> setup(uint32_t begin, const uint32_t end, const typename Accessor::value_type _value, const Comparator _comp)
    {
        bound_t<Accessor,Comparator> retval;
        retval.comp = _comp;
        retval.value = _value;
        retval.it = begin;
        retval.len = end-begin;
        return retval;
    }


    uint32_t operator()(NBL_REF_ARG(Accessor) accessor)
    {
        if (isNPoT(len))
        {
            const uint32_t newLen = 0x1u<<nbl::hlsl::findMSB(len);
            const uint32_t testPoint = it+(len-newLen);
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

    void iteration(NBL_REF_ARG(Accessor) accessor)
    {
        len >>= 1;
        const uint32_t mid = it+len;
        comp_step(accessor,mid);
    }

    void comp_step(NBL_REF_ARG(Accessor) accessor, const uint32_t testPoint, const uint32_t rightBegin)
    {
        if (comp(accessor[testPoint],value))
            it = rightBegin;
    }
    void comp_step(NBL_REF_ARG(Accessor) accessor, const uint32_t testPoint)
    {
        comp_step(accessor,testPoint,testPoint);
    }

    Comparator comp;
    typename Accessor::value_type value;
    uint32_t it;
    uint32_t len;
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
uint32_t lower_bound(NBL_REF_ARG(Accessor) accessor, const uint32_t begin, const uint32_t end, const typename Accessor::value_type value, const Comparator comp)
{
    impl::bound_t<Accessor,Comparator> implementation = impl::bound_t<Accessor,Comparator>::setup(begin,end,value,comp);
    return implementation(accessor);
}

template<class Accessor, class Comparator>
uint32_t upper_bound(NBL_REF_ARG(Accessor) accessor, const uint32_t begin, const uint32_t end, const typename Accessor::value_type value, const Comparator comp)
{
    //using TransformedComparator = impl::lower_to_upper_comparator_transform_t<Accessor,Comparator>;
    //TransformedComparator transformedComparator;
    
    impl::lower_to_upper_comparator_transform_t<Accessor,Comparator> transformedComparator;
    transformedComparator.comp = comp;
    return lower_bound<Accessor,impl::lower_to_upper_comparator_transform_t<Accessor,Comparator> >(accessor,begin,end,value,transformedComparator);
}


namespace impl
{

// extra indirection due to https://github.com/microsoft/DirectXShaderCompiler/issues/4771
template<class Accessor, typename T>
uint32_t lower_bound(NBL_REF_ARG(Accessor) accessor, const uint32_t begin, const uint32_t end, const T value)
{
    //using Comparator = nbl::hlsl::less<T>;
    //Comparator comp;
    
    nbl::hlsl::less<T> comp;
    return nbl::hlsl::lower_bound<Accessor, nbl::hlsl::less<T> >(accessor,begin,end,value,comp);
}
template<class Accessor, typename T>
uint32_t upper_bound(NBL_REF_ARG(Accessor) accessor, const uint32_t begin, const uint32_t end, const T value)
{
    //using Comparator = nbl::hlsl::less<T>;
    //Comparator comp;
    
    nbl::hlsl::less<T> comp;
    return nbl::hlsl::upper_bound<Accessor, nbl::hlsl::less<T> >(accessor,begin,end,value,comp);
}

}

template<class Accessor>
uint32_t lower_bound(NBL_REF_ARG(Accessor) accessor, const uint32_t begin, const uint32_t end, const typename Accessor::value_type value)
{
    return impl::lower_bound<Accessor,typename Accessor::value_type>(accessor,begin,end,value);
}
template<class Accessor>
uint32_t upper_bound(NBL_REF_ARG(Accessor) accessor, const uint32_t begin, const uint32_t end, const typename Accessor::value_type value)
{
    return impl::upper_bound<Accessor,typename Accessor::value_type>(accessor,begin,end,value);
}


template<int begin, int end>
struct unrolled_for_range;
template<int end>
struct unrolled_for_range<end,end>
{
    template<typename F>
    static void __call(NBL_REF_ARG(F) f) {}
};
template<int begin, int end>
struct unrolled_for_range
{
    template<typename F>
    static void __call(NBL_REF_ARG(F) f)
    {
        f.template __call<begin>();
        unrolled_for_range<begin+1,end>::template __call<F>(f);
    }
};

}
}

#endif
