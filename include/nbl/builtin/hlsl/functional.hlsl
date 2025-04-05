// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_FUNCTIONAL_INCLUDED_
#define _NBL_BUILTIN_HLSL_FUNCTIONAL_INCLUDED_


#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/limits.hlsl"


namespace nbl
{
namespace hlsl
{
#ifdef __HLSL_VERSION // HLSL
template<uint32_t StorageClass, typename T>
using __spv_ptr_t = spirv::pointer_t<StorageClass,T>;

template<uint32_t StorageClass, typename T>
[[vk::ext_instruction(spv::OpCopyObject)]]
__spv_ptr_t<StorageClass,T> addrof([[vk::ext_reference]] T v); 

template<uint32_t StorageClass, typename T>
struct reference_wrapper_base
{
    using spv_ptr_t = __spv_ptr_t<StorageClass,T>;
    spv_ptr_t ptr;

    void __init(spv_ptr_t _ptr)
    {
        ptr = _ptr;
    }

    T load()
    {
        return spirv::load<T,spv_ptr_t>(ptr);
    }

    void store(const T val)
    {
        spirv::store<T,spv_ptr_t>(ptr,val);
    }

    // TODO: use the same defaults as `glsl::atomicAnd`
    template<uint32_t memoryScope, uint32_t memorySemantics, typename S=T>
    // TODO: instead of `is_scalar_v` we need a test on whether the `spirv::atomicAnd` expression is callable
    enable_if_t<is_same_v<S,T>&&is_scalar_v<S>,T> atomicAnd(const T value)
    {
        return spirv::atomicAnd<T,spv_ptr_t>(ptr,memoryScope,memorySemantics,value);
    }
    // TODO: generate Or,Xor through a macro

    // TODO: comp swap is special, has an extra parameter
};
template<typename T>
struct reference_wrapper_base<spv::StorageClassPhysicalStorageBuffer,T>
{
    using spv_ptr_t = typename T::instead_use_nbl::hlsl::bda::__ref;

    // normally would have specializations of load and store
};

// we need to explicitly white-list storage classes due to
// https://github.com/microsoft/DirectXShaderCompiler/issues/6578#issuecomment-2297181671
template<uint32_t StorageClass, typename T>
struct reference_wrapper : enable_if_t<
    is_same_v<StorageClass,spv::StorageClassInput>||
    is_same_v<StorageClass,spv::StorageClassUniform>||
    is_same_v<StorageClass,spv::StorageClassWorkgroup>||
    is_same_v<StorageClass,spv::StorageClassPushConstant>||
    is_same_v<StorageClass,spv::StorageClassImage>||
    is_same_v<StorageClass,spv::StorageClassStorageBuffer>,
    reference_wrapper_base<StorageClass,T>
>
{
};
// TODO: generate atomic Add,Sub,Min,Max through partial template specializations on T
// TODO: partial specializations for T being a special SPIR-V type for image ops, etc.


#define ALIAS_STD(NAME,OP) template<typename T> struct NAME { \
    using type_t = T; \
    \
    T operator()(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs) \
    { \
        return lhs OP rhs; \
    }


#else // CPP


#define ALIAS_STD(NAME,OP) template<typename T> struct NAME : std::NAME<T> { \
    using type_t = T;

#endif

ALIAS_STD(bit_and,&)
    using scalar_t = typename scalar_type<T>::type;
    using bitfield_t = typename unsigned_integer_of_size<sizeof(scalar_t)>::type;

    static_assert(!is_floating_point<T>::value,"We cannot define the identity element properly, you can thank https://github.com/microsoft/DirectXShaderCompiler/issues/5868 !");
    NBL_CONSTEXPR_STATIC_INLINE T identity = ~bitfield_t(0); // TODO: need a `all_components<T>` (not in cpp_compat) which can create vectors and matrices with all members set to scalar
};

ALIAS_STD(bit_or,|)

    NBL_CONSTEXPR_STATIC_INLINE T identity = T(0);
};

ALIAS_STD(bit_xor,^)

    NBL_CONSTEXPR_STATIC_INLINE T identity = T(0);
};

ALIAS_STD(plus,+)

    NBL_CONSTEXPR_STATIC_INLINE T identity = T(0);
};

ALIAS_STD(minus,-)

    NBL_CONSTEXPR_STATIC_INLINE T identity = T(0);
};

ALIAS_STD(multiplies,*)

    NBL_CONSTEXPR_STATIC_INLINE T identity = T(1);
};

ALIAS_STD(divides,/)

    NBL_CONSTEXPR_STATIC_INLINE T identity = T(1);
};


ALIAS_STD(greater,>) };
ALIAS_STD(less,<) };
ALIAS_STD(greater_equal,>=) };
ALIAS_STD(less_equal,<=) };

#undef ALIAS_STD

// ------------------------ Compound assignment operators ----------------------

#define COMPOUND_ASSIGN(NAME) template<typename T> struct NAME##_assign { \
    using type_t = T; \
    using base_t = NAME <type_t>; \
    base_t baseOp; \
    \
    void operator()(NBL_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs) \
    { \
        lhs = baseOp(lhs, rhs); \
    }\
    NBL_CONSTEXPR_STATIC_INLINE T identity = base_t::identity; \
};

COMPOUND_ASSIGN(plus)
COMPOUND_ASSIGN(minus)         
COMPOUND_ASSIGN(multiplies) 
COMPOUND_ASSIGN(divides) 

#undef COMPOUND_ASSIGN

// ----------------- End of compound assignment ops ----------------

// Min, Max, and Ternary and Shift operators don't use ALIAS_STD because they don't exist in STD
// TODO: implement as mix(rhs<lhs,lhs,rhs) (SPIR-V intrinsic from the extended set & glm on C++)
template<typename T>
struct minimum
{
    using type_t = T;
    using scalar_t = typename scalar_type<T>::type;

    T operator()(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs)
    {
        return rhs<lhs ? rhs:lhs;
    }

    NBL_CONSTEXPR_STATIC_INLINE T identity = numeric_limits<scalar_t>::max; // TODO: `all_components<T>`
};

template<typename T>
struct maximum
{
    using type_t = T;
    using scalar_t = typename scalar_type<T>::type;

    T operator()(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs)
    {
        return lhs<rhs ? rhs:lhs;
    }

    NBL_CONSTEXPR_STATIC_INLINE T identity = numeric_limits<scalar_t>::lowest; // TODO: `all_components<T>`
};

template<typename T NBL_STRUCT_CONSTRAINABLE >
struct ternary_operator
{
    using type_t = T;

    NBL_CONSTEXPR_INLINE_FUNC T operator()(bool condition, NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs)
    {
        return condition ? lhs : rhs;
    }
};

template<typename T NBL_STRUCT_CONSTRAINABLE >
struct left_shift_operator
{
    using type_t = T;

    NBL_CONSTEXPR_INLINE_FUNC T operator()(NBL_CONST_REF_ARG(T) operand, NBL_CONST_REF_ARG(T) bits)
    {
        return operand << bits;
    }
};

template<typename T NBL_STRUCT_CONSTRAINABLE >
struct arithmetic_right_shift_operator
{
    using type_t = T;

    NBL_CONSTEXPR_INLINE_FUNC T operator()(NBL_CONST_REF_ARG(T) operand, NBL_CONST_REF_ARG(T) bits)
    {
        return operand >> bits;
    }
};

template<typename T NBL_STRUCT_CONSTRAINABLE >
struct logical_right_shift_operator
{
    using type_t = T;
    using unsigned_type_t = make_unsigned_t<T>;

    NBL_CONSTEXPR_INLINE_FUNC T operator()(NBL_CONST_REF_ARG(T) operand, NBL_CONST_REF_ARG(T) bits)
    {
        arithmetic_right_shift_operator<unsigned_type_t> arithmeticRightShift;
        return _static_cast<T>(arithmeticRightShift(_static_cast<unsigned_type_t>(operand), _static_cast<unsigned_type_t>(bits)));
    }
};



} //namespace nbl
} //namespace hlsl

#endif