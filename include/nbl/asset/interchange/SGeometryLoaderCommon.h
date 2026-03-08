// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_GEOMETRY_LOADER_COMMON_H_INCLUDED_
#define _NBL_ASSET_S_GEOMETRY_LOADER_COMMON_H_INCLUDED_
#include <ranges>
#include <type_traits>
#include "nbl/asset/SBufferAdoption.h"
#include "nbl/asset/ICPUPolygonGeometry.h"
namespace nbl::asset
{
//! Shared geometry-loader helpers for adopting buffers and assembling formatted data views.
class SGeometryLoaderCommon
{
    public:
        //! Creates one formatted data view over an existing CPU buffer.
        static inline IGeometry<ICPUBuffer>::SDataView createDataView(core::smart_refctd_ptr<ICPUBuffer>&& buffer, const size_t byteCount, const uint32_t stride, const E_FORMAT format)
        {
            if (!buffer || byteCount == 0ull)
                return {};
            return {.composed = {.stride = stride, .format = format, .rangeFormat = IGeometryBase::getMatchingAABBFormat(format)}, .src = {.offset = 0ull, .size = byteCount, .buffer = std::move(buffer)}};
        }

        //! Adopts contiguous caller-owned storage into a CPU buffer and exposes it as a formatted data view.
        template<E_FORMAT Format, impl::AdoptedBufferStorage Storage>
        static inline IGeometry<ICPUBuffer>::SDataView createAdoptedView(Storage&& data)
        {
            using storage_t = std::remove_cvref_t<Storage>;
            using value_t = std::ranges::range_value_t<storage_t>;

            auto buffer = SBufferAdoption::create(std::forward<Storage>(data));
            if (!buffer)
                return {};
            return createDataView(std::move(buffer), buffer->getSize(), static_cast<uint32_t>(sizeof(value_t)), Format);
        }
};
}
#endif
