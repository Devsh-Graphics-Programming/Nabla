// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_TYPE_TRAITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_TYPE_TRAITS_INCLUDED_

// C++ headers
#ifndef __HLSL_VERSION
#include <type_traits>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
#endif

namespace nbl::hlsl::type_traits
{

#ifdef __HLSL_VERSION // HLSL

template<class T, T val>
struct integral_constant {
    static const T value = val;
    using value_type = T;
};

template <bool val>
struct bool_constant : integral_constant<bool, val> {};

template <bool C, class T, class F>
struct conditional 
{ 
    using type = T;
};

template <class T, class F>
struct conditional<false, T, F> 
{
    using type = F;
};

template<class A, class B>
struct is_same : bool_constant<false> {};

template<class A>
struct is_same<A,A> : bool_constant<true> {};

template<class T>
struct is_void : bool_constant<is_same<T, void>::value> {};

template<class T>
struct is_array : bool_constant<false> {};

template<class T, uint32_t count>
struct is_array<T[count]> : bool_constant<true>{};

// TODO: decide whether matricies and vectors qualify for these implementations
namespace impl
{

template<class T>
struct is_integral : bool_constant<
    is_same<T, bool>::value ||
    is_same<T, uint16_t>::value ||
    is_same<T, uint32_t>::value ||
    is_same<T, uint64_t>::value ||
    is_same<T, int16_t>::value ||
    is_same<T, int32_t>::value ||
    is_same<T, int64_t>::value
> {};


template<class T>
struct is_floating_point : bool_constant<
    is_same<T, half>::value ||
    is_same<T, float>::value ||
    is_same<T, double>::value
> {};

}

template<class T>
struct is_scalar : bool_constant<
    impl::is_integral<T>::value || 
    impl::is_floating_point<T>::value
> {};

#else // C++

template<class T, T val>
using integral_constant = std::integral_constant<T, val>;

template<bool val>
using bool_constant = std::bool_constant<val>;

template <bool C, class T, class F>
using conditional  = std::conditional<C, T, F>;

template<class A, class B>
using is_same = std::is_same<A, B>;

template<class T>
using is_void = std::is_void<T>;

template<class T>
using is_array = std::is_array<T>;

template<class T>
using is_scalar = std::is_scalar<T>;

#endif

// Overlapping definitions

template<class T>
struct is_vector : bool_constant<false> {};

template<class T>
struct is_matrix : bool_constant<false> {};

template<class T, uint32_t N>
struct is_vector<vector<T, N> > : bool_constant<true> {};

template<class T, uint32_t N, uint32_t M>
struct is_matrix<matrix<T, N, M> > : bool_constant<true> {};

}

#endif
