// Copyright (C) 2024-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_MATRIX_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_MATRIX_INCLUDED_


#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/matrix_utils/matrix_traits.hlsl>

namespace nbl
{
namespace hlsl
{
namespace concepts
{

template<typename T>
NBL_BOOL_CONCEPT Matrix = is_matrix<T>::value;

#define NBL_CONCEPT_NAME Matricial
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
NBL_CONCEPT_BEGIN(0)
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(matrix_traits<T>::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(matrix_traits<T>::row_type))
    ((NBL_CONCEPT_REQ_TYPE)(matrix_traits<T>::transposed_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((matrix_traits<T>::RowCount), ::nbl::hlsl::is_integral_v))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((matrix_traits<T>::ColumnCount), ::nbl::hlsl::is_integral_v))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((matrix_traits<T>::Square), ::nbl::hlsl::is_same_v, bool))
);
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

}
}
}
#endif