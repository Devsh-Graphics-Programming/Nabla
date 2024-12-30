// Copyright (C) 2024-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_VECTOR_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_VECTOR_INCLUDED_


#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>

namespace nbl
{
namespace hlsl
{
namespace concepts
{

template<typename T>
NBL_BOOL_CONCEPT Vector = is_vector<T>::value;

// declare concept
#define NBL_CONCEPT_NAME Vectorial
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)

NBL_CONCEPT_BEGIN(0)
NBL_CONCEPT_END
(
    ((NBL_CONCEPT_REQ_TYPE)(vector_traits<T>::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((vector_traits<T>::Dimension), ::nbl::hlsl::is_integral_v))
);

#include <nbl/builtin/hlsl/concepts/__end.hlsl>

}
}
}
#endif