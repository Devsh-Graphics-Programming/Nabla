// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_TYPE_TRAITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_TYPE_TRAITS_INCLUDED_

// Since HLSL currently doesnt allow type aliases we declare them as seperate structs thus they are (WORKAROUND)s
/*
  // helper class
  template<class T, T v> struct integral_constant; (DONE)
 
  template<bool B>
    using bool_constant = integral_constant<bool, B>; (WORKDAROUND)
  using true_type  = bool_constant<true>; (WORKDAROUND)
  using false_type = bool_constant<false>; (WORKDAROUND)
 
  // primary type categories
  template<class T> struct is_void; (DONE)
  template<class T> struct is_null_pointer; (NOT-APPLICABLE)
  template<class T> struct is_integral; (DONE)
  template<class T> struct is_floating_point;  (DONE)
  template<class T> struct is_array; (DONE)
  template<class T> struct is_pointer; (TODO)
  template<class T> struct is_lvalue_reference; (TODO)
  template<class T> struct is_rvalue_reference; (TODO)
  template<class T> struct is_member_object_pointer; (TODO)
  template<class T> struct is_member_function_pointer; (TODO)
  template<class T> struct is_enum;  (NOT-APPLICABLE)
  template<class T> struct is_union; (NOT-APPLICABLE)
  template<class T> struct is_class; (NOT-APPLICABLE)
  template<class T> struct is_function; (NOT-APPLICABLE)
 
  // composite type categories
  template<class T> struct is_reference; (TODO)
  template<class T> struct is_arithmetic; (DONE)
  template<class T> struct is_fundamental; (TODO)
  template<class T> struct is_object; (NOT-APPLICABLE)
  template<class T> struct is_scalar; (DONE)
  template<class T> struct is_compound; (DONE)
  template<class T> struct is_member_pointer; (TODO)
 
  // type properties
  template<class T> struct is_const; (DONE)
  template<class T> struct is_volatile; (DONE)
  template<class T> struct is_trivial; (EVERYTHING IS)
  template<class T> struct is_trivially_copyable; (EVERYTHING IS)
  template<class T> struct is_standard_layout; (NOT-APPLICABLE)
  template<class T> struct is_empty; (DONE? sizeof(T) == 0)
  template<class T> struct is_polymorphic; (NOTHING IS)
  template<class T> struct is_abstract; (NOTHING IS)
  template<class T> struct is_final; (NOTHING IS)
  template<class T> struct is_aggregate; (DONE)
 
  template<class T> struct is_signed; (DONE)
  template<class T> struct is_unsigned; (DONE)
  template<class T> struct is_bounded_array;
  template<class T> struct is_unbounded_array;
  template<class T> struct is_scoped_enum; (NOT-APPLICABLE)

  // type property queries
  template<class T> struct alignment_of; (TODO)
  template<class T> struct rank; (TODO)
  template<class T, unsigned I = 0> struct extent; (TODO)
 
  // type relations
  template<class T, class U> struct is_same; (DONE)
  template<class Base, class Derived> struct is_base_of; (TODO)
  template<class From, class To> struct is_convertible; (TODO)
  template<class From, class To> struct is_nothrow_convertible; (NOT-APPLICABLE)
  template<class T, class U> struct is_layout_compatible; (TODO)
  template<class Base, class Derived> struct is_pointer_interconvertible_base_of; (NOT-APPLICABLE)
 
  NO FOLD EXPRESSIONS IN HLSL!
    template<class Fn, class... ArgTypes> struct is_invocable;
    template<class R, class Fn, class... ArgTypes> struct is_invocable_r;
    template<class Fn, class... ArgTypes> struct is_nothrow_invocable;
    template<class R, class Fn, class... ArgTypes> struct is_nothrow_invocable_r;
 
  // const-volatile modifications
  template<class T> struct remove_const; (TODO)
  template<class T> struct remove_volatile; (TODO)
  template<class T> struct remove_cv; (TODO)
  template<class T> struct add_const; (TODO)
  template<class T> struct add_volatile; (TODO)
  template<class T> struct add_cv; (TODO)

  // reference modifications
  template<class T> struct remove_reference; (TODO)
  template<class T> struct add_lvalue_reference; (TODO)
  template<class T> struct add_rvalue_reference; (TODO)

  // sign modifications
  template<class T> struct make_signed; (TODO)
  template<class T> struct make_unsigned; (TODO)
 
  // array modifications
  template<class T> struct remove_extent; (TODO)
  template<class T> struct remove_all_extents; (TODO)

  // pointer modifications
  template<class T> struct remove_pointer; (TODO)
  template<class T> struct add_pointer; (TODO)

  // other transformations
  template<class T> struct type_identity; (TODO)
  template<class T> struct remove_cvref; (TODO)
  template<class T> struct decay; (TODO)
  template<bool, class T = void> struct enable_if; (NOT-APPLICABLE)
  template<bool, class T, class F> struct conditional; (DONE)
  template<class... T> struct common_type; (NOT-APPLICABLE)
  template<class T, class U, template<class> class TQual, template<class> class UQual>
    struct basic_common_reference { }; (NOT-APPLICABLE)
  template<class... T> struct common_reference; (NOT-APPLICABLE)
  template<class T> struct underlying_type;
  template<class Fn, class... ArgTypes> struct invoke_result;
  template<class T> struct unwrap_reference; (TODO)
  template<class T> struct unwrap_ref_decay; (TODO)

  NO FOLD EXPRESSIONS IN HLSL!
    // logical operator traits
    template<class... B> struct conjunction;
    template<class... B> struct disjunction;
    template<class B> struct negation;
*/

// C++ headers
#ifndef __HLSL_VERSION
#include <type_traits>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
#endif

namespace nbl
{
namespace hlsl
{
namespace type_traits
{
namespace impl
{
    
template<template<class> class Trait, class T>
struct base_type_forwarder : Trait<T> {};

template<template<class> class Trait, class T, uint16_t N>
struct base_type_forwarder<Trait,vector<T,N> > : Trait<T> {};

template<template<class> class Trait, class T, uint16_t N, uint16_t M>
struct base_type_forwarder<Trait,matrix<T,N,M> > : Trait<T> {};

}

#if __HLSL_VERSION // HLSL

template<class T, T val>
struct integral_constant {
    static const T value = val;
    using value_type = T;
};

template <bool val>
struct bool_constant : integral_constant<bool, val> {};

struct true_type : bool_constant<true> {};
struct false_type : bool_constant<true> {};

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

namespace impl
{

template<class T>
struct is_unsigned : bool_constant<
    is_same<T, bool>::value ||
    is_same<T, uint16_t>::value ||
    is_same<T, uint32_t>::value ||
    is_same<T, uint64_t>::value
> {};

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

template<class T>
struct is_signed : bool_constant<
    is_same<T, int16_t>::value ||
    is_same<T, int32_t>::value ||
    is_same<T, int64_t>::value ||
    is_floating_point<T>::value
> {};

}

template<class T> 
struct is_unsigned : impl::base_type_forwarder<impl::is_unsigned, T> {};

template<class T> 
struct is_integral : impl::base_type_forwarder<impl::is_integral, T> {};

template<class T> 
struct is_floating_point : impl::base_type_forwarder<impl::is_floating_point, T> {};

template<class T> 
struct is_signed : impl::base_type_forwarder<impl::is_signed, T> {};

template<class T>
struct is_scalar : bool_constant<
    impl::is_integral<T>::value || 
    impl::is_floating_point<T>::value
> {};

template<class T>
struct is_const : bool_constant<false> {};

template<class T>
struct is_const<const T> : bool_constant<true> {};

template<class T>
struct is_volatile : bool_constant<false> {};

template<class T>
struct is_volatile<volatile T> : bool_constant<true> {};

template<class>
struct is_trivial : bool_constant<true> {};

template<class>
struct is_trivially_copyable : bool_constant<true> {};

// this implementation is fragile
template<class T>
struct is_empty : bool_constant<0==sizeof(T)> {};

template<class>
struct is_polymorphic : bool_constant<false> {};

template<class>
struct is_abstract : bool_constant<false> {};

template<class>
struct is_final : bool_constant<false> {};

template <class T>
struct is_fundamental : bool_constant<
    is_scalar<T>::value || 
    is_void<T>::value
> {};

template <class T>
struct is_compound : bool_constant<!is_fundamental<T>::value> {};

template <class T>
struct is_aggregate : is_compound<T> {};

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

template<class T>
struct is_signed : impl::base_type_forwarder<std::is_signed, T> {};

template<class T>
struct is_unsigned : impl::base_type_forwarder<std::is_unsigned, T> {};

template<class T>
struct is_integral : impl::base_type_forwarder<std::is_integral, T> {};

template<class T>
struct is_floating_point : impl::base_type_forwarder<std::is_floating_point, T> {};

template<class T>
using is_const = std::is_const<T>;

template<class T>
using is_volatile = std::is_volatile<T>;

template<class T>
using is_trivial = std::is_trivial<T>;

template<class T>
using is_trivially_copyable = std::is_trivially_copyable<T>;

template<class T> 
using is_empty = std::is_empty<T>;

template<class T> 
using is_polymorphic = std::is_polymorphic<T>;

template<class T> 
using is_abstract = std::is_abstract<T>;

template<class T> 
using is_final = std::is_final<T>;

template<class T>
using is_fundamental = std::is_fundamental<T>;

template<class T>
using is_compound = std::is_compound<T>;

template<class T>
using is_aggregate = std::is_aggregate<T>;

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
}
}
#endif
