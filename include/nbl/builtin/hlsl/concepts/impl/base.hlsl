// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_IMPL_BASE_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_IMPL_BASE_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace concepts
{

//! Now diverge
#ifdef __cpp_concepts

// to define a concept using `concept Name = SomeContexprBoolCondition<T>;`
#define NBL_BOOL_CONCEPT concept

// to put right before the closing `>` of the primary template definition, otherwise `NBL_PARTIAL_REQUIRES` wont work on specializations
#define NBL_STRUCT_CONSTRAINABLE

#else

// to define a concept using `concept Name = SomeContexprBoolCondition<T>;`
#define NBL_BOOL_CONCEPT NBL_CONSTEXPR bool

// to put right before the closing `>` of the primary template definition, otherwise `NBL_PARTIAL_REQUIRES` wont work on specializations
#define NBL_STRUCT_CONSTRAINABLE ,typename __requires=void

#endif
}
}
}

#endif