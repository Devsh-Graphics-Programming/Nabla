// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_TYPE_TRAITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_TYPE_TRAITS_INCLUDED_

// C++ headers
#ifndef __HLSL_VERSION
#include <type_traits>
#endif


#include <nbl/builtin/hlsl/cpp_compat.hlsl>


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
  template<class T> struct is_fundamental; (DONE)

  template<class T> struct is_object; (TODO)
    C++ spec defines object as:
        void      is not an object
        int       is object
        int&      is not an object
        int*      is object
        int*&     is not an object
        cls       is object
        cls&      is not an object
        cls*      is object
        int()     is not an object
        int(*)()  is object
        int(&)()  is not an object
        basically !(is_reference || is_void || is_function)

  template<class T> struct is_scalar; (DONE)
  template<class T> struct is_compound; (DONE)
  template<class T> struct is_member_pointer; (TODO)
 
  // type properties
  template<class T> struct is_const; (DONE)
  template<class T> struct is_volatile; (DONE)
  template<class T> struct is_trivial; (EVERYTHING IS)
  template<class T> struct is_trivially_copyable; (EVERYTHING IS)
  template<class T> struct is_standard_layout; (APPLICABLE BUT IMPOSSIBLE TO TEST WITHOUT REFLECTION)
  template<class T> struct is_empty; (DONE? sizeof(T) == 0)
  template<class T> struct is_polymorphic; (NOTHING IS)
  template<class T> struct is_abstract; (NOTHING IS)
  template<class T> struct is_final; (NOTHING IS until they add the final keyword)
  template<class T> struct is_aggregate; (DONE)
 
  template<class T> struct is_signed; (DONE)
  template<class T> struct is_unsigned; (DONE)
  template<class T> struct is_bounded_array(DONE);
  template<class T> struct is_unbounded_array(DONE);
  template<class T> struct is_scoped_enum; (NOT-APPLICABLE)

  // type property queries
  template<class T> struct alignment_of; (SPECIALIZED FOR REGISTERED TYPES)
  template<class T> struct rank; (DONE)
  template<class T, unsigned I = 0> struct extent; (DONE)
 
  // type relations
  template<class T, class U> struct is_same; (DONE)
  template<class Base, class Derived> struct is_base_of; (TODO)
  template<class From, class To> struct is_convertible; (TODO)
  template<class From, class To> struct is_nothrow_convertible; (TODO: ALIAS OF is_convertible)
  template<class T, class U> struct is_layout_compatible; (TODO)
  template<class Base, class Derived> struct is_pointer_interconvertible_base_of; (NOT-APPLICABLE)
 
  NO FOLD EXPRESSIONS IN HLSL!
    template<class Fn, class... ArgTypes> struct is_invocable;
    template<class R, class Fn, class... ArgTypes> struct is_invocable_r;
    template<class Fn, class... ArgTypes> struct is_nothrow_invocable;
    template<class R, class Fn, class... ArgTypes> struct is_nothrow_invocable_r;
 
  // const-volatile modifications
  template<class T> struct remove_const; (DONE)
  template<class T> struct remove_volatile; (DONE)
  template<class T> struct remove_cv; (DONE)
  template<class T> struct add_const; (DONE)
  template<class T> struct add_volatile; (DONE)
  template<class T> struct add_cv; (DONE)

  // reference modifications
  template<class T> struct remove_reference; (DONE BUT DONT USE)
  template<class T> struct add_lvalue_reference; (DONE BUT DONT USE)
  template<class T> struct add_rvalue_reference; (TODO)

  // sign modifications
  template<class T> struct make_signed; (DONE)
  template<class T> struct make_unsigned; (DONE)
 
  // array modifications
  template<class T> struct remove_extent; (DONE)
  template<class T> struct remove_all_extents; (DONE)

  // pointer modifications
  template<class T> struct remove_pointer; (NOT-POSSIBLE UNTIL PSUEDO PTRS ARE IMPLEMENTED)
  template<class T> struct add_pointer; (NOT-POSSIBLE UNTIL PSUEDO PTRS ARE IMPLEMENTED)

  // other transformations
  template<class T> struct type_identity; (DONE)
  template<class T> struct remove_cvref; (DONE)
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
  template<class...> using void_t = void; (VARIADICS NOT SUPPORTED USE `make_void` INSTEAD)

  NO FOLD EXPRESSIONS IN HLSL!
    // logical operator traits
    template<class... B> struct conjunction;
    template<class... B> struct disjunction;
    template<class B> struct negation;
*/


namespace nbl
{
namespace hlsl
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

#ifdef __HLSL_VERSION // HLSL


#define decltype(expr) __decltype(expr)

template<class T>
struct type_identity 
{
    using type = T;
};

namespace impl
{
template<class> struct remove_reference;
}

template<class T> struct remove_const : type_identity<T> {};
template<class T> struct remove_const<const T> : type_identity<T> {};

template<class T> struct remove_volatile : type_identity<T> {};
template<class T> struct remove_volatile<volatile T> : type_identity<T> {};

template<class T> struct remove_cv : type_identity<T> {};
template<class T> struct remove_cv<const T> : type_identity<T> {};
template<class T> struct remove_cv<volatile T> : type_identity<T> {};
template<class T> struct remove_cv<const volatile T> : type_identity<T> {};

template<class T> struct remove_cvref : remove_cv<typename impl::remove_reference<T>::type> {};

template<class T> struct add_const : type_identity<const T> {};
template<class T> struct add_volatile : type_identity<volatile T> {};
template<class T> struct add_cv : type_identity<const volatile T> {};


template<class T, T val>
struct integral_constant {
    NBL_CONSTEXPR_STATIC_INLINE T value = val;
    using value_type = T;
};

template<bool val>
struct bool_constant : integral_constant<bool, val> {};

struct true_type : bool_constant<true> {};
struct false_type : bool_constant<false> {};

template<bool C, class T, class F>
struct conditional : type_identity<T> {};

template<class T, class F>
struct conditional<false, T, F>  : type_identity<F> {};

template<class A, class B>
struct is_same : bool_constant<false> {};

template<class A>
struct is_same<A,A> : bool_constant<true> {};

template<class T>
struct is_void : bool_constant<is_same<typename remove_cv<T>::type, void>::value> {};

template<class T>
struct is_bounded_array : bool_constant<false> {};

template<class T, uint32_t count>
struct is_bounded_array<T[count]> : bool_constant<true>{};

template<class T>
struct is_unbounded_array : bool_constant<false>{};

template<class T>
struct is_unbounded_array<T[]> : bool_constant<true>{};

template<class T>
struct is_array : bool_constant<is_bounded_array<T>::value || is_unbounded_array<T>::value> {};

namespace impl
{

// need this crutch because we can't make `#define typeid` work both on expression and types
template<typename T>
struct typeid_t;

template<typename T, T v>
struct function_info : type_identity<void> {};

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
struct is_unsigned : impl::base_type_forwarder<impl::is_unsigned, typename remove_cv<T>::type> {};

template<class T> 
struct is_integral : impl::base_type_forwarder<impl::is_integral, T> {};

template<class T> 
struct is_floating_point : impl::base_type_forwarder<impl::is_floating_point, typename remove_cv<T>::type> {};

template<class T> 
struct is_signed : impl::base_type_forwarder<impl::is_signed, typename remove_cv<T>::type> {};

template<class T>
struct is_scalar : bool_constant<
    impl::is_integral<typename remove_cv<T>::type>::value || 
    impl::is_floating_point<typename remove_cv<T>::type>::value
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

template<class T>
struct rank : integral_constant<uint64_t, 0> { };

template<class T, uint64_t N>
struct rank<T[N]> : integral_constant<uint64_t, 1 + rank<T>::value> { };

template<class T>
struct rank<T[]> : integral_constant<uint64_t, 1 + rank<T>::value> { };

template<class T, uint32_t I = 0> 
struct extent : integral_constant<uint64_t, 0> {};

template<class T, uint64_t N> 
struct extent<T[N], 0> : integral_constant<uint64_t, N> {};

template<class T, uint64_t N, uint32_t I> 
struct extent<T[N], I> : integral_constant<uint64_t,extent<T, I - 1>::value> {};

template<class T, uint32_t I> 
struct extent<T[], I> : integral_constant<uint64_t,extent<T, I - 1>::value> {};

template<bool B, class T = void>
struct enable_if {};
 
template<class T>
struct enable_if<true, T> : type_identity<T> {};

template<class T>
struct alignment_of;

template<class>
struct make_void { using type = void; };

// reference stuff needed for semantics 

// not for "human consumption"
// dxc doesnt actually allow variables to be references
// every reference is treated as having 'restrict' qualifier
// https://godbolt.org/z/dsj99fY96
namespace impl
{

template<typename,class=void>
struct is_reference : bool_constant<true> { };

template<typename T>
struct is_reference<T,typename make_void<T[1]>::type> : bool_constant<false> { };

template<class T, bool = is_reference<T>::value>
struct add_reference_helper : type_identity<T> {};

template<class T>
struct add_reference_helper<T, false>
{
    static T member[1];
    using type = decltype(member[0]);
};

template<class T>
struct add_lvalue_reference : add_reference_helper<T> {};

template<typename T>
T remove_reference_impl(T v)
{
    return v;
}

template<typename T, bool = is_reference<T>::value>
struct remove_reference_helper : type_identity<T> {};

template<typename T>
struct remove_reference_helper<T, true>
{
    static T member;
    using type = decltype(remove_reference_impl(member));
};

template<typename T>
struct remove_reference : remove_reference_helper<T> {};

}

template<class T> struct remove_extent : type_identity<T> {};
template<class T, uint32_t I> struct remove_extent<T[I]> : type_identity<T> {};
template<class T> struct remove_extent<T[]> : type_identity<T> {};

template <class T>
struct remove_all_extents : type_identity<T> {};

template <class T, uint32_t I>
struct remove_all_extents<T[I]> : type_identity<typename remove_all_extents<T>::type> {};

template <class T>
struct remove_all_extents<T[]> : type_identity<typename remove_all_extents<T>::type> {};

namespace impl
{

template<uint32_t sz> struct int_type :
    conditional<8==sz, int64_t, typename conditional<4==sz, int32_t, int16_t>::type>{};

template<uint32_t sz> struct uint_type : 
    type_identity<unsigned typename int_type<sz>::type> {};
}

template<class T>
struct make_signed : impl::int_type<sizeof(T)>
{
    _Static_assert(is_integral<T>::value && !is_same<typename remove_cv<T>::type, bool>::value,
        "make_signed<T> requires that T shall be a (possibly cv-qualified) "
        "integral type or enumeration but not a bool type.");
};

template<class T>
struct make_unsigned : impl::uint_type<sizeof(T)>
{
    _Static_assert(is_integral<T>::value && !is_same<typename remove_cv<T>::type, bool>::value, 
        "make_unsigned<T> requires that T shall be a (possibly cv-qualified) "
        "integral type or enumeration but not a bool type.");
};

#else // C++

template<class T, T val>
using integral_constant = std::integral_constant<T, val>;

template<bool val>
using bool_constant = std::bool_constant<val>;

using true_type   = std::true_type;
using false_type  = std::false_type;

template <bool C, class T, class F>
using conditional  = std::conditional<C, T, F>;

template<class A, class B>
using is_same = std::is_same<A, B>;

template<class T>
using is_void = std::is_void<T>;

template<class T>
using is_array = std::is_array<T>;

template<class T>
using is_bounded_array = std::is_bounded_array<T>;

template<class T>
using is_unbounded_array = std::is_unbounded_array<T>;

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

template<typename T>
using type_identity = std::type_identity<T>;

template<class T>
using rank = std::rank<T>;

template<class T, unsigned I = 0> 
using extent = std::extent<T, I>;

template<bool B, class T = void>
using enable_if = std::enable_if<B, T>;

template<class T>
using alignment_of = std::alignment_of<T>;

template<class T> using remove_const = std::remove_const<T>;
template<class T> using remove_volatile = std::remove_volatile<T>;
template<class T> using remove_cv = std::remove_cv<T>;
template<class T> using add_const = std::add_const<T>;
template<class T> using add_volatile = std::add_volatile<T>;
template<class T> using add_cv = std::add_cv<T>;

template<class T> using remove_extent = std::remove_extent<T>;
template<class T> using remove_all_extents = std::remove_all_extents<T>;

template<class T>
using make_signed = std::make_signed<T>;

template<class T>
using make_unsigned = std::make_unsigned<T>;

#endif

// Template Variables
template<typename A, typename B>
NBL_CONSTEXPR bool is_same_v = is_same<A,B>::value;

// Overlapping definitions
template<bool C, typename T, T A, T B>
struct conditional_value
{
    NBL_CONSTEXPR_STATIC_INLINE T value = C ? A : B;
};

template<class T>
struct is_vector : bool_constant<false> {};

template<class T>
struct is_matrix : bool_constant<false> {};

template<class T, uint32_t N>
struct is_vector<vector<T, N> > : bool_constant<true> {};

template<class T, uint32_t N, uint32_t M>
struct is_matrix<matrix<T, N, M> > : bool_constant<true> {};

template<class T>
NBL_CONSTEXPR bool is_matrix_v = is_matrix<T>::value;


template<typename T,bool=is_scalar<T>::value>
struct scalar_type
{
    using type = void;
};

template<typename T>
struct scalar_type<T,true>
{
    using type = T;
};

template<typename T, uint16_t N>
struct scalar_type<vector<T,N>,false>
{
    using type = T;
};

template<typename T, uint16_t N, uint16_t M>
struct scalar_type<matrix<T,N,M>,false>
{
    using type = T;
};

template<typename T>
using scalar_type_t = typename scalar_type<T>::type;


template<uint16_t bytesize>
struct unsigned_integer_of_size
{
    using type = void;
};

template<>
struct unsigned_integer_of_size<2>
{
    using type = uint16_t;
};

template<>
struct unsigned_integer_of_size<4>
{
    using type = uint32_t;
};

template<>
struct unsigned_integer_of_size<8>
{
    using type = uint64_t;
};

}
}

// deal with typetraits, for now we rely on Clang/DXC internal __decltype(), if it breaks we revert to commit e4ab38ca227b15b2c79641c39161f1f922b779a3
#ifdef __HLSL_VERSION

#define alignof(expr) ::nbl::hlsl::alignment_of<__decltype(expr)>::value

// shoudl really return a std::type_info like struct or something, but no `constexpr` and unsure whether its possible to have a `const static SomeStruct` makes it hard to do...
#define typeid(expr) (::nbl::hlsl::impl::typeid_t<__decltype(expr)>::value)

// Found a bug in Boost.Wave, try to avoid multi-line macros https://github.com/boostorg/wave/issues/195
#define NBL_IMPL_SPECIALIZE_TYPE_ID(T) namespace impl { template<> struct typeid_t<T> : integral_constant<uint32_t,__COUNTER__> {}; }

#define NBL_REGISTER_OBJ_TYPE(T,A) namespace nbl { namespace hlsl { NBL_IMPL_SPECIALIZE_TYPE_ID(T) \
    template<> struct alignment_of<T> : integral_constant<uint32_t,A> {}; \
    template<> struct alignment_of<const T> : integral_constant<uint32_t,A> {}; \
    template<> struct alignment_of<typename impl::add_lvalue_reference<T>::type> : integral_constant<uint32_t,A> {}; \
    template<> struct alignment_of<typename impl::add_lvalue_reference<const T>::type> : integral_constant<uint32_t,A> {}; \
}}

// TODO: find out how to do it such that we don't get duplicate definition if we use two function identifiers with same signature
#define NBL_REGISTER_FUN_TYPE(fn) namespace nbl { namespace hlsl { NBL_IMPL_SPECIALIZE_TYPE_ID(__decltype(fn)) }}
// TODO: ideally we'd like to call NBL_REGISTER_FUN_TYPE under the hood, but we can't right now. Also we have a bigger problem, the passing of the function identifier as the second template parameter doesn't work :(
/*
template<> \
struct function_info<__decltype(fn),fn> \
{ \
    using type = __decltype(fn); \
    static const uint32_t address = __COUNTER__; \
}; \
}}}}
*/

// builtins

#define NBL_REGISTER_MATRICES(T, A) \
    NBL_REGISTER_OBJ_TYPE(T, A) \
    NBL_REGISTER_OBJ_TYPE(T ## x4, A) \
    NBL_REGISTER_OBJ_TYPE(T ## x3, A) \
    NBL_REGISTER_OBJ_TYPE(T ## x2, A) \

#define NBL_REGISTER_TYPES_FOR_SCALAR(T) \
    NBL_REGISTER_OBJ_TYPE(T, sizeof(T)) \
    NBL_REGISTER_OBJ_TYPE(T ## 1, sizeof(T)) \
    NBL_REGISTER_MATRICES(T ## 2, sizeof(T)) \
    NBL_REGISTER_MATRICES(T ## 3, sizeof(T)) \
    NBL_REGISTER_MATRICES(T ## 4, sizeof(T))

NBL_REGISTER_TYPES_FOR_SCALAR(int16_t)
NBL_REGISTER_TYPES_FOR_SCALAR(int32_t)
NBL_REGISTER_TYPES_FOR_SCALAR(int64_t)

NBL_REGISTER_TYPES_FOR_SCALAR(uint16_t)
NBL_REGISTER_TYPES_FOR_SCALAR(uint32_t)
NBL_REGISTER_TYPES_FOR_SCALAR(uint64_t)

NBL_REGISTER_TYPES_FOR_SCALAR(bool)

NBL_REGISTER_TYPES_FOR_SCALAR(float16_t)
NBL_REGISTER_TYPES_FOR_SCALAR(float32_t)
NBL_REGISTER_TYPES_FOR_SCALAR(float64_t)

#undef NBL_REGISTER_MATRICES
#undef NBL_REGISTER_TYPES_FOR_SCALAR

#endif


#endif
