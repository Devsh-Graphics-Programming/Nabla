// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_GEOMETRY_LOADER_COMMON_H_INCLUDED_
#define _NBL_ASSET_S_GEOMETRY_LOADER_COMMON_H_INCLUDED_


#include "nbl/asset/ICPUPolygonGeometry.h"


namespace nbl::asset
{

class SGeometryLoaderCommon
{
    public:
        static inline IGeometry<ICPUBuffer>::SDataView createDataView(core::smart_refctd_ptr<ICPUBuffer>&& buffer, const size_t byteCount, const uint32_t stride, const E_FORMAT format)
        {
            if (!buffer || byteCount == 0ull)
                return {};

            return {
                .composed = {
                    .stride = stride,
                    .format = format,
                    .rangeFormat = IGeometryBase::getMatchingAABBFormat(format)
                },
                .src = {
                    .offset = 0ull,
                    .size = byteCount,
                    .buffer = std::move(buffer)
                }
            };
        }

        template<typename ValueType, E_FORMAT Format>
        static inline IGeometry<ICPUBuffer>::SDataView createAdoptedView(core::vector<ValueType>&& data)
        {
            if (data.empty())
                return {};

            auto backer = core::make_smart_refctd_ptr<core::adoption_memory_resource<core::vector<ValueType>>>(std::move(data));
            auto& storage = backer->getBacker();
            const size_t byteCount = storage.size() * sizeof(ValueType);
            auto* const ptr = storage.data();
            auto buffer = ICPUBuffer::create(
                { { byteCount }, ptr, core::smart_refctd_ptr<core::refctd_memory_resource>(std::move(backer)), alignof(ValueType) },
                core::adopt_memory);
            return createDataView(std::move(buffer), byteCount, static_cast<uint32_t>(sizeof(ValueType)), Format);
        }
};

}


#endif
