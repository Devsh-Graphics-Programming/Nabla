// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_GEOMETRY_LOADER_COMMON_H_INCLUDED_
#define _NBL_ASSET_S_GEOMETRY_LOADER_COMMON_H_INCLUDED_
#include <algorithm>
#include <cassert>
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
		//! Tracks the widest scalar component format and highest component index seen for one structured attribute.
		static inline void negotiateStructuredComponent(IGeometry<ICPUBuffer>::SDataViewBase& view, const E_FORMAT componentFormat, const uint8_t component)
		{
			assert(getFormatChannelCount(componentFormat) != 0u);
			if (getTexelOrBlockBytesize(componentFormat) > getTexelOrBlockBytesize(view.format))
				view.format = componentFormat;
			view.stride = std::max<uint32_t>(view.stride, component);
		}
		//! Finalizes one structured base view and invokes `onComponent(offset,stride,componentFormat)` per component slot.
		template<typename Fn>
		static inline void finalizeStructuredBaseView(IGeometry<ICPUBuffer>::SDataViewBase& view, Fn&& onComponent)
		{
			if (view.format == EF_UNKNOWN)
				return;
			const auto componentFormat = view.format;
			const auto componentCount = view.stride + 1u;
			view.format = getFormatWithChannelCount(componentFormat, componentCount);
			view.stride = getTexelOrBlockBytesize(view.format);
			for (uint32_t c = 0u; c < componentCount; ++c)
				onComponent(getTexelOrBlockBytesize(componentFormat) * c, view.stride, componentFormat);
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
