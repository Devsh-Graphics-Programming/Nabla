// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_UTILS_ELEMENTWISE_INCLUDED_
#define _NBL_BUILTIN_HLSL_UTILS_ELEMENTWISE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/concepts/vector.hlsl>
#include <nbl/builtin/hlsl/concepts/matrix.hlsl>
#include <nbl/builtin/hlsl/array_accessors.hlsl>

namespace nbl
{
namespace hlsl
{
namespace utils
{

// Per-element testers: apply a binary predicate to every pair of scalar elements of `lhs` and `rhs`
// (scalars, vectors or matrices) and reduce the results to a single `bool`.
//
// `Pred` must be a struct with a non-templated `bool operator()(scalar_type, scalar_type) NBL_CONST_MEMBER_FUNC`,
// any parameters it needs (tolerances etc.) are its members.
namespace impl
{

template<typename T, typename Pred NBL_STRUCT_CONSTRAINABLE>
struct ElementwiseShortCircuit;

// Every specialization returns `stopOn` as soon as the predicate evaluates to `stopOn`, otherwise `!stopOn`.
// `stopOn=false` gives "all", `stopOn=true` gives "any".
template<typename Scalar, typename Pred>
NBL_PARTIAL_REQ_TOP(!concepts::Vectorial<Scalar> && !concepts::Matricial<Scalar>)
struct ElementwiseShortCircuit<Scalar, Pred NBL_PARTIAL_REQ_BOT(!concepts::Vectorial<Scalar> && !concepts::Matricial<Scalar>) >
{
    static bool __call(NBL_CONST_REF_ARG(Scalar) lhs, NBL_CONST_REF_ARG(Scalar) rhs, Pred pred, const bool stopOn)
    {
        return pred(lhs, rhs);
    }
};

template<typename Vectorial, typename Pred>
NBL_PARTIAL_REQ_TOP(concepts::Vectorial<Vectorial>)
struct ElementwiseShortCircuit<Vectorial, Pred NBL_PARTIAL_REQ_BOT(concepts::Vectorial<Vectorial>) >
{
    static bool __call(NBL_CONST_REF_ARG(Vectorial) lhs, NBL_CONST_REF_ARG(Vectorial) rhs, Pred pred, const bool stopOn)
    {
        using traits = hlsl::vector_traits<Vectorial>;
        array_get<Vectorial, typename traits::scalar_type> getter;
        for (uint32_t i = 0; i < traits::Dimension; ++i)
        {
            if (pred(getter(lhs, i), getter(rhs, i)) == stopOn)
                return stopOn;
        }
        return !stopOn;
    }
};

template<typename Matricial, typename Pred>
NBL_PARTIAL_REQ_TOP(concepts::Matricial<Matricial>)
struct ElementwiseShortCircuit<Matricial, Pred NBL_PARTIAL_REQ_BOT(concepts::Matricial<Matricial>) >
{
    static bool __call(NBL_CONST_REF_ARG(Matricial) lhs, NBL_CONST_REF_ARG(Matricial) rhs, Pred pred, const bool stopOn)
    {
        using traits = hlsl::matrix_traits<Matricial>;
        for (uint32_t i = 0; i < traits::RowCount; ++i)
        {
            if (ElementwiseShortCircuit<typename traits::row_type, Pred>::__call(lhs[i], rhs[i], pred, stopOn) == stopOn)
                return stopOn;
        }
        return !stopOn;
    }
};

}

// true if `pred` holds for every element pair
template<typename T, typename Pred>
bool elementwiseAll(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs, Pred pred)
{
    return impl::ElementwiseShortCircuit<T, Pred>::__call(lhs, rhs, pred, false);
}

// true if `pred` holds for at least one element pair
template<typename T, typename Pred>
bool elementwiseAny(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs, Pred pred)
{
    return impl::ElementwiseShortCircuit<T, Pred>::__call(lhs, rhs, pred, true);
}

// true if `pred` holds for no element pair
template<typename T, typename Pred>
bool elementwiseNone(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs, Pred pred)
{
    return !elementwiseAny<T, Pred>(lhs, rhs, pred);
}

}
}
}

#endif
