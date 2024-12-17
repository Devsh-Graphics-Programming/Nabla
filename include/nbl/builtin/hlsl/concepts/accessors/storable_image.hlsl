// Copyright (C) 2024-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_STORABLE_IMAGE_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_STORABLE_IMAGE_INCLUDED_


#include <nbl/builtin/hlsl/concepts.hlsl>


namespace nbl
{
namespace hlsl
{
namespace concepts
{
namespace accessors
{
// declare concept
#define NBL_CONCEPT_NAME StorableImage
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)(int32_t)
#define NBL_CONCEPT_TPLT_PRM_NAMES (U)(T)(Dims)
// not the greatest syntax but works
#define NBL_CONCEPT_PARAM_0 (a,U)
#define NBL_CONCEPT_PARAM_1 (uv,vector<uint16_t,Dims>)
#define NBL_CONCEPT_PARAM_2 (layer,uint16_t)
#define NBL_CONCEPT_PARAM_3 (data,vector<T,4>)
// start concept
NBL_CONCEPT_BEGIN(4)
// need to be defined AFTER the cocnept begins
#define a NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define uv NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define layer NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define data NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR)(a.template set<T,Dims>(uv,layer,data)))
);
#undef data
#undef layer
#undef uv
#undef a
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
}
}
}
}
#endif