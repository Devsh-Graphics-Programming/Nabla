// Copyright (C) 2018-2021 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_QUANT_NORMAL_CACHE_H_INCLUDED
#define __NBL_ASSET_C_QUANT_NORMAL_CACHE_H_INCLUDED


#include "nbl/asset/utils/CDirQuantCacheBase.h"


namespace nbl 
{
namespace asset 
{
	
namespace impl
{

struct NBL_API VectorUV
{
	inline VectorUV(const core::vectorSIMDf& absNormal)
	{
		const float rcpManhattanNorm = 1.f / (absNormal.x + absNormal.y + absNormal.z);
		u = absNormal.x * rcpManhattanNorm;
		v = absNormal.z * rcpManhattanNorm;
	}

	inline bool operator==(const VectorUV& other) const
	{
		return (u == other.u && v == other.v);
	}

	float u;
	float v;
};

struct NBL_API QuantNormalHash
{
	inline size_t operator()(const VectorUV& vec) const noexcept
	{
		static constexpr size_t primeNumber1 = 18446744073709551557ull;
		static constexpr size_t primeNumber2 = 4611686018427388273ull;
				
		return  ((static_cast<size_t>(static_cast<double>(vec.u)*(std::numeric_limits<size_t>::max)()) * primeNumber1) ^
			(static_cast<size_t>(static_cast<double>(vec.v)*(std::numeric_limits<size_t>::max)()) * primeNumber2));
	}
};

}


class NBL_API CQuantNormalCache : public CDirQuantCacheBase<impl::VectorUV,impl::QuantNormalHash,EF_A2B10G10R10_SNORM_PACK32,EF_R8G8B8_SNORM,EF_R16G16B16_SNORM>
{
		using Base = CDirQuantCacheBase<impl::VectorUV,impl::QuantNormalHash,EF_A2B10G10R10_SNORM_PACK32,EF_R8G8B8_SNORM,EF_R16G16B16_SNORM>;

	public:
		template<E_FORMAT CacheFormat>
		value_type_t<CacheFormat> quantize(core::vectorSIMDf normal)
		{
			normal.makeSafe3D();
			return Base::quantize<3u,CacheFormat>(normal);
		}
};

}
}
#endif