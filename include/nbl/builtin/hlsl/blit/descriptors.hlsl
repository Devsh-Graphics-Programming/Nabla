// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BLIT_DESCRIPTORS_INCLUDED_
#define _NBL_BUILTIN_HLSL_BLIT_DESCRIPTORS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/type_traits.hlsl>

namespace nbl
{
namespace hlsl
{
namespace blit
{

// DXC is not happy with using for type-alias outside structs atm. Use typedef for now.
typedef type_traits::integral_constant<uint32_t, 0> BlitDescriptorSet;
typedef type_traits::integral_constant<uint32_t, 0> InImageBinding;
typedef type_traits::integral_constant<uint32_t, 1> OutImageBinding;
typedef type_traits::integral_constant<uint32_t, 2> StatisticsBinding;

typedef type_traits::integral_constant<uint32_t, 1> KernelWeightsDescriptorSet;
typedef type_traits::integral_constant<uint32_t, 0> KernelWeightsBinding;

}
}
}

#endif
