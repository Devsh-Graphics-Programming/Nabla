// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BLIT_FORMATS_ENCODE_INCLUDED_
#define _NBL_BUILTIN_HLSL_BLIT_FORMATS_ENCODE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>
// NOTE: Assumes E_FORMAT is available in <nbl/builtin/hlsl/EFormat.hlsl> even tho it's not there yet.
#include <nbl/builtin/hlsl/Eformat.hlsl>

namespace nbl
{
namespace hlsl
{
namespace blit
{

template <E_FORMAT format>
struct encode_output { using type = float4; };
template<> struct encode_output<EF_R32G32_SFLOAT> { using type = uint32_t4; };
template<> struct encode_output<EF_B10G11R11_UFLOAT_PACK32> { using type = uint32_t4; };

template<E_FORMAT Format>
typename encode_output<Format>::type encode(NBL_CONST_REF_ARG(float4) value)
{
	return value;
}

template<>
encode_output<E_FORMAT::EF_R32G32_SFLOAT>::type encode<E_FORMAT::EF_R32G32_SFLOAT>(NBL_CONST_REF_ARG(float4) value)
{
	return uint32_t4(asuint(value.r), asuint(value.g), 0u, 0u);
}

template<>
encode_output<E_FORMAT::EF_B10G11R11_UFLOAT_PACK32>::type encode<E_FORMAT::EF_B10G11R11_UFLOAT_PACK32>(NBL_CONST_REF_ARG(float4) value)
{
	return uint32_t4(ieee754::encode_ufloat<5, 5>(value.b), ieee754::encode_ufloat<6, 5>(value.g), ieee754::encode_ufloat<6, 5>(value.r), 0u);
}

}
}
}
#endif