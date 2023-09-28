// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BLIT_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BLIT_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/blit/formats_encode.hlsl>

namespace nbl
{
namespace hlsl
{
namespace blit
{

template <uint32_t AlphaBinCount>
struct AlphaStatistics
{
	uint32_t passedPixelCount;
	uint32_t histogram [AlphaBinCount];
};

template <uint32_t Dimension>
struct dim_to_in_texture { using type = void; };

template <>
struct dim_to_in_texture<1> { using type = Texture1DArray<float4>; };

template <>
struct dim_to_in_texture<2> { using type = Texture2DArray<float4>; };

template <>
struct dim_to_in_texture<3> { using type = Texture3D<float4>; };


template<uint32_t Dimension, bool RequiresSoftwareEncode>
struct dim_and_software_encode_to_out_texture {};

// These have to be specialized for both dimension and software encode
// instead of defining seperate trait for the element type because
// for some reason DXC complains when using inferred types for RWTexture
template <>
struct dim_and_software_encode_to_out_texture<1, true>
{
	using type = RWTexture1DArray<uint32_t4>;
};

template <>
struct dim_and_software_encode_to_out_texture<2, true>
{
	using type = RWTexture2DArray<uint32_t4>;
};

template <>
struct dim_and_software_encode_to_out_texture<3, true>
{
	using type = RWTexture3D<uint32_t4>;
};

template <>
struct dim_and_software_encode_to_out_texture<1, false>
{
	using type = RWTexture1DArray<float4>;
};

template <>
struct dim_and_software_encode_to_out_texture<2, false>
{
	using type = RWTexture2DArray<float4>;
};


template <>
struct dim_and_software_encode_to_out_texture<3, false>
{
	using type = RWTexture3D<float4>;
};

template <uint32_t dim>
struct load_coords_num {};

template <>
struct load_coords_num<1> { static const uint32_t value = 2; };

template <>
struct load_coords_num<2> { static const uint32_t value = 3; };

template <>
struct load_coords_num<3> { static const uint32_t value = 3; };

template <uint32_t dim>
vector<uint, load_coords_num<dim>::value> getIndexCoord(uint32_t3 coords, uint32_t layer)
{
	// NOTE: This assumes #define NBL_STATIC_ASSERT(C, M) _Static_assert(C, M) is defined in cpp_compat.hlsl
	NBL_STATIC_ASSERT(dim <= 3, "Unsupported");
}

template <>
uint32_t2 getIndexCoord<1>(NBL_CONST_REF_ARG(uint32_t3) coords, uint32_t layer)
{
	return uint32_t2(coords.x, layer);
}

template <>
uint32_t3 getIndexCoord<2>(NBL_CONST_REF_ARG(uint32_t3) coords, uint32_t layer)
{
	return uint32_t3(coords.xy, layer);
}

template <>
uint32_t3 getIndexCoord<3>(NBL_CONST_REF_ARG(uint32_t3) coords, uint32_t layer)
{
	return coords;
}

template <uint32_t BlitDimCount, typename InTexture>
float4 getData(NBL_CONST_REF_ARG(InTexture) inTexture, NBL_CONST_REF_ARG(uint32_t3) coords, uint32_t layer)
{
	vector<uint, load_coords_num<BlitDimCount>::value> indexCoord = getIndexCoord<BlitDimCount>(coords, layer);
	return inTexture[indexCoord];
}

template <uint32_t BlitDimCount, E_FORMAT Format, typename OutTexture>
void setData(NBL_REF_ARG(OutTexture) outTexture, NBL_CONST_REF_ARG(uint32_t3) coords, uint32_t layer, float4 value)
{
	vector<uint, load_coords_num<BlitDimCount>::value> indexCoord = getIndexCoord<BlitDimCount>(coords, layer);
	outTexture[indexCoord] = encode<Format>(value);
}


template <typename T, uint32_t N>
struct SharedAccessor
{
	T data[N];

	void set(uint32_t idx, NBL_CONST_REF_ARG(T) val) { data[idx] = val; }
	T get(uint32_t idx) { return data[idx]; }
	uint32_t size() { return N; }
};


}
}
}

#endif