// Copyright (C) 2018-2021 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_QUANT_QUATERNION_CACHE_H_INCLUDED
#define __NBL_ASSET_C_QUANT_QUATERNION_CACHE_H_INCLUDED

#include "nbl/asset/utils/CDirQuantCacheBase.h"

namespace nbl
{
namespace asset
{
namespace impl
{
struct Projection
{
    inline Projection(const core::vectorSIMDf& absDir)
    {
        const float rcpManhattanNorm = 1.f / (absDir.x + absDir.y + absDir.z + absDir.w);
        x = absDir.x * rcpManhattanNorm;
        y = absDir.y * rcpManhattanNorm;
        z = absDir.z * rcpManhattanNorm;
    }

    inline bool operator==(const Projection& other) const
    {
        return (x == other.x && y == other.y && z == other.z);
    }

    float x;
    float y;
    float z;
};

struct QuantQuaternionHash
{
    inline size_t operator()(const Projection& vec) const noexcept
    {
        static constexpr size_t primeNumber1 = 18446744073709551557ull;
        static constexpr size_t primeNumber2 = 4611686018427388273ull;
        static constexpr size_t primeNumber3 = 10278296396886393151ull;

        return (static_cast<size_t>(static_cast<double>(vec.x) * (std::numeric_limits<size_t>::max)()) * primeNumber1) ^
            (static_cast<size_t>(static_cast<double>(vec.y) * (std::numeric_limits<size_t>::max)()) * primeNumber2) ^
            (static_cast<size_t>(static_cast<double>(vec.z) * (std::numeric_limits<size_t>::max)()) * primeNumber3);
    }
};

}

class CQuantQuaternionCache : public CDirQuantCacheBase<impl::Projection, impl::QuantQuaternionHash, EF_R8G8B8A8_SNORM, EF_R16G16B16A16_SNORM>
{
    using Base = CDirQuantCacheBase<impl::Projection, impl::QuantQuaternionHash, EF_R8G8B8A8_SNORM, EF_R16G16B16A16_SNORM>;

public:
    template<E_FORMAT CacheFormat>
    value_type_t<CacheFormat> quantize(const core::quaternion& quat)
    {
        return Base::quantize<4u, CacheFormat>(reinterpret_cast<const core::vectorSIMDf&>(quat));
    }
};

}
}
#endif