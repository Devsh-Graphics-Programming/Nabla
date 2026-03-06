// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_GEOMETRY_LOADER_COMMON_H_INCLUDED_
#define _NBL_ASSET_S_GEOMETRY_LOADER_COMMON_H_INCLUDED_


#include <concepts>
#include <ranges>
#include <type_traits>
#include <utility>

#include "nbl/asset/ICPUPolygonGeometry.h"


namespace nbl::asset
{

namespace impl
{

// Owns contiguous storage that can be adopted by the buffer. Views like std::span are rejected.
template<typename Storage>
concept AdoptedViewStorage =
    std::ranges::contiguous_range<std::remove_reference_t<Storage>> &&
    std::ranges::sized_range<std::remove_reference_t<Storage>> &&
    (!std::ranges::view<std::remove_cvref_t<Storage>>) &&
    requires(std::remove_reference_t<Storage>& storage)
    {
        typename std::ranges::range_value_t<std::remove_reference_t<Storage>>;
        { std::ranges::data(storage) } -> std::same_as<std::ranges::range_value_t<std::remove_reference_t<Storage>>*>;
    };

}

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

        template<E_FORMAT Format, impl::AdoptedViewStorage Storage>
        static inline IGeometry<ICPUBuffer>::SDataView createAdoptedView(Storage&& data)
        {
            using storage_t = std::remove_cvref_t<Storage>;
            using value_t = std::ranges::range_value_t<storage_t>;

            if (std::ranges::empty(data))
                return {};

            auto backer = core::make_smart_refctd_ptr<core::adoption_memory_resource<storage_t>>(std::forward<Storage>(data));
            auto& storage = backer->getBacker();
            const size_t byteCount = std::ranges::size(storage) * sizeof(value_t);
            auto buffer = ICPUBuffer::create(
                { { byteCount }, std::ranges::data(storage), core::smart_refctd_ptr<core::refctd_memory_resource>(std::move(backer)), alignof(value_t) },
                core::adopt_memory);
            return createDataView(std::move(buffer), byteCount, static_cast<uint32_t>(sizeof(value_t)), Format);
        }
};

}


#endif
