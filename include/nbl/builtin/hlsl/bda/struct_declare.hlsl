// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BDA_STRUCT_DECLARE_INCLUDED_
#define _NBL_BUILTIN_HLSL_BDA_STRUCT_DECLARE_INCLUDED_

#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/mpl.hlsl"
#ifdef  __HLSL_VERSION
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
#endif //  __HLSL_VERSION


namespace nbl
{
namespace hlsl
{
namespace bda
{
// silly utility traits
template<typename T>
struct member_count
{
    NBL_CONSTEXPR_STATIC_INLINE uint32_t value = 0;
};
template<typename T>
NBL_CONSTEXPR uint32_t member_count_v = member_count<T>::value;

template<typename T, int32_t MemberIx>
struct member_type;
template<typename T, int32_t MemberIx>
using member_type_t = typename member_type<T,MemberIx>::type;

// default alignment is the alignment of the type
template<typename T, int32_t MemberIx>
struct member_alignment
{
    NBL_CONSTEXPR_STATIC_INLINE uint32_t value = alignment_of_v<member_type_t<T,MemberIx> >;
};
template<typename T, int32_t MemberIx>
NBL_CONSTEXPR uint32_t member_alignment_v = member_alignment<T,MemberIx>::value;

// the default specialization of the offset assumes scalar layout
template<typename T, int32_t MemberIx>
struct member_offset
{
    // TODO: assert that the custom alignment is no less than the type's natural alignment?
    // first byte past previous member, rounded up to out alignment
    NBL_CONSTEXPR_STATIC_INLINE uint64_t value = mpl::align_up_v<member_offset<T,MemberIx-1>::value+size_of_v<member_type_t<T,MemberIx-1> >,member_alignment_v<T,MemberIx> >;
};
template<typename T>
struct member_offset<T,0>
{
    NBL_CONSTEXPR_STATIC_INLINE uint64_t value = 0;
};
template<typename T, int32_t MemberIx>
NBL_CONSTEXPR uint64_t member_offset_v = member_offset<T,MemberIx>::value;

// stuff needed to compute alignment of the struct properly
namespace impl
{
template<typename T, uint32_t N>
struct default_alignment
{
    NBL_CONSTEXPR_STATIC_INLINE uint32_t value = mpl::max_v<uint32_t,member_alignment_v<T,N-1>,default_alignment<T,N-1>::value>;
};
// le invalid values
template<typename T>
struct default_alignment<T,0>
{
    NBL_CONSTEXPR_STATIC_INLINE uint32_t value = 0;
};
template<typename T, typename MemberCount=member_count<T> >
NBL_CONSTEXPR uint32_t default_alignment_v = default_alignment<T,MemberCount::value>::value;
}
}
}
}

//! Need to gen identical struct in HLSL and C++, right now this tool can declare non-templated structs and full explicit specialized ones

//implementation details
#define NBL_HLSL_IMPL_DEFINE_STRUCT_GET_MEMBER_TYPE(identifier,...) __VA_ARGS__
#define NBL_HLSL_IMPL_DEFINE_STRUCT_GET_MEMBER_NAME(identifier,...) identifier
#define NBL_HLSL_IMPL_DEFINE_STRUCT_MEMBER_TYPE(r,IDENTIFIER,i,e) template<> \
struct ::nbl::hlsl::bda::member_type<NBL_EVAL IDENTIFIER,i> \
{ \
using type = NBL_HLSL_IMPL_DEFINE_STRUCT_GET_MEMBER_TYPE e; \
};

//! TODO: handle declarations for partial template specializations and non-specializations
#define NBL_HLSL_IMPL_DECLARE_STRUCT_MEMBER(identifier,...) __VA_ARGS__ identifier;
#ifdef __HLSL_VERSION
#define NBL_HLSL_IMPL_DEFINE_STRUCT_MEMBER(r,IDENTIFIER,i,e) [[vk::ext_decorate(spv::DecorationOffset,::nbl::hlsl::bda::member_offset_v<NBL_EVAL IDENTIFIER,i>)]] NBL_HLSL_IMPL_DECLARE_STRUCT_MEMBER e
#define NBL_HLSL_IMPL_DEFINE_STRUCT_MEMBER_REFERENCE(r,unused,i,e) ::nbl::hlsl::bda::__ref< \
	NBL_HLSL_IMPL_DEFINE_STRUCT_GET_MEMBER_TYPE e, \
	::nbl::hlsl::mpl::min_v<uint32_t,::nbl::hlsl::bda::member_alignment_v<__referenced_t,i>,alignment>, \
_restrict> NBL_HLSL_IMPL_DEFINE_STRUCT_GET_MEMBER_NAME e;
#define NBL_HLSL_IMPL_INIT_STRUCT_MEMBER_REFERENCE(r,unused,i,e) NBL_HLSL_IMPL_DEFINE_STRUCT_GET_MEMBER_NAME e .__init( \
	::nbl::hlsl::spirv::accessChain<NBL_HLSL_IMPL_DEFINE_STRUCT_GET_MEMBER_TYPE e>(base_t::ptr.value,i) \
);
#define NBL_HLSL_IMPL_DEFINE_STRUCT(IDENTIFIER,MEMBER_SEQ) NBL_EVAL IDENTIFIER \
{ \
BOOST_PP_SEQ_FOR_EACH_I(NBL_HLSL_IMPL_DEFINE_STRUCT_MEMBER,IDENTIFIER,MEMBER_SEQ) \
}; \
template<uint32_t alignment, bool _restrict> \
struct ::nbl::hlsl::bda::__ref<NBL_EVAL IDENTIFIER,alignment,_restrict> : ::nbl::hlsl::bda::__base_ref<NBL_EVAL IDENTIFIER,alignment,_restrict> \
{ \
	using __referenced_t = NBL_EVAL IDENTIFIER; \
    using base_t = __base_ref<__referenced_t,alignment,_restrict>; \
    using this_t = __ref<__referenced_t,alignment,_restrict>; \
\
	BOOST_PP_SEQ_FOR_EACH_I(NBL_HLSL_IMPL_DEFINE_STRUCT_MEMBER_REFERENCE,dummy,MEMBER_SEQ) \
\
	void __init(const ::nbl::hlsl::spirv::bda_pointer_t<__referenced_t> _ptr) \
	{ \
		base_t::__init(_ptr); \
		BOOST_PP_SEQ_FOR_EACH_I(NBL_HLSL_IMPL_INIT_STRUCT_MEMBER_REFERENCE,dummy,MEMBER_SEQ) \
	} \
}
#else
#define NBL_HLSL_IMPL_DEFINE_STRUCT_MEMBER(r,IDENTIFIER,i,e) alignas(::nbl::hlsl::bda::member_alignment_v<NBL_EVAL IDENTIFIER,i>) NBL_HLSL_IMPL_DECLARE_STRUCT_MEMBER e
#define NBL_HLSL_IMPL_DEFINE_STRUCT(IDENTIFIER,MEMBER_SEQ) alignas(::nbl::hlsl::alignment_of_v<NBL_EVAL IDENTIFIER >) NBL_EVAL IDENTIFIER \
{ \
BOOST_PP_SEQ_FOR_EACH_I(NBL_HLSL_IMPL_DEFINE_STRUCT_MEMBER,IDENTIFIER,MEMBER_SEQ) \
}
#endif

// some weird stuff to handle alignment
#define NBL_HLSL_IMPL_DEFINE_STRUCT_BEGIN(IDENTIFIER,MEMBER_SEQ) template<> \
struct ::nbl::hlsl::bda::member_count<NBL_EVAL IDENTIFIER > \
{ \
	NBL_CONSTEXPR_STATIC_INLINE uint32_t value = BOOST_PP_SEQ_SIZE(MEMBER_SEQ); \
}; \
BOOST_PP_SEQ_FOR_EACH_I(NBL_HLSL_IMPL_DEFINE_STRUCT_MEMBER_TYPE,IDENTIFIER,MEMBER_SEQ) \
template <> \
struct ::nbl::hlsl::alignment_of<NBL_EVAL IDENTIFIER > \
{
#define NBL_HLSL_IMPL_DEFINE_STRUCT_END(IDENTIFIER,MEMBER_SEQ,...) }; \
template<> \
struct ::nbl::hlsl::size_of<NBL_EVAL IDENTIFIER > \
{ \
	using type = NBL_EVAL IDENTIFIER; \
	NBL_CONSTEXPR_STATIC_INLINE uint32_t __last_member_ix_v = ::nbl::hlsl::bda::member_count_v<type>-1; \
	NBL_CONSTEXPR_STATIC_INLINE uint64_t __last_member_offset_v = ::nbl::hlsl::bda::member_offset_v<type, __last_member_ix_v>; \
	NBL_CONSTEXPR_STATIC_INLINE uint64_t __last_member_size_v = ::nbl::hlsl::size_of_v<::nbl::hlsl::bda::member_type_t<type, __last_member_ix_v> >; \
	NBL_CONSTEXPR_STATIC_INLINE uint32_t value = ::nbl::hlsl::mpl::align_up_v<__last_member_offset_v + __last_member_size_v, alignment_of_v<type > >; \
\
__VA_ARGS__ \
\
}; \
struct NBL_HLSL_IMPL_DEFINE_STRUCT(IDENTIFIER,MEMBER_SEQ)

#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/seq/size.hpp>
// MEMBER_SEQ is to be a sequence of variable name and type (identifier0,Type0)...(identifierN,TypeN) @see NBL_HLSL_IMPL_DEFINE_STRUCT_GET_MEMBER_TYPE
// the VA_ARGS is the struct alignment for alignas, usage example
// ```
// NBL_HLSL_DEFINE_STRUCT((MyStruct2),
//	((a, float32_t))
//	((b, int32_t))
//	((c, int32_t2)),
// 
// ... block of code for the methods ...
// 
// );
// ```
#define NBL_HLSL_DEFINE_STRUCT(IDENTIFIER,MEMBER_SEQ,...) NBL_HLSL_IMPL_DEFINE_STRUCT_BEGIN(IDENTIFIER,MEMBER_SEQ) \
	NBL_CONSTEXPR_STATIC_INLINE uint32_t value = ::nbl::hlsl::bda::impl::default_alignment_v<NBL_EVAL IDENTIFIER >; \
NBL_HLSL_IMPL_DEFINE_STRUCT_END(IDENTIFIER,MEMBER_SEQ,__VA_ARGS__)
// version allowing custom alignment on whole struct
#define NBL_HLSL_DEFINE_ALIGNAS_STRUCT(IDENTIFIER,ALIGNMENT,MEMBER_SEQ,...) NBL_HLSL_IMPL_DEFINE_STRUCT_BEGIN(IDENTIFIER,MEMBER_SEQ) \
	NBL_CONSTEXPR_STATIC_INLINE uint32_t value = ALIGNMENT; \
NBL_HLSL_IMPL_DEFINE_STRUCT_END(IDENTIFIER,MEMBER_SEQ,__VA_ARGS__)

#endif