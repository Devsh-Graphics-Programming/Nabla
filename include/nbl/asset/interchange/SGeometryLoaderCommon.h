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
		//! Creates one owned data view with storage sized for `elementCount` items in `format`.
		static inline IGeometry<ICPUBuffer>::SDataView createOwnedView(const E_FORMAT format, const size_t elementCount)
		{
			if (format == EF_UNKNOWN || elementCount == 0ull)
				return {};
			const auto stride = getTexelOrBlockBytesize(format);
			auto buffer = ICPUBuffer::create({stride * elementCount});
			return buffer ? createDataView(std::move(buffer), stride * elementCount, stride, format) : IGeometry<ICPUBuffer>::SDataView{};
		}
		//! Finalizes one structured base view, calls `onComponent`, and allocates the resulting owned data view.
		template<typename Fn>
		static inline IGeometry<ICPUBuffer>::SDataView createStructuredView(IGeometry<ICPUBuffer>::SDataViewBase& view, const size_t elementCount, Fn&& onComponent)
		{
			if (view.format == EF_UNKNOWN)
				return {};
			finalizeStructuredBaseView(view, std::forward<Fn>(onComponent));
			return createOwnedView(view.format, elementCount);
		}
		//! Finalizes one structured view, appends per-component iterator bindings, rebases them against the allocated buffer, and passes the created view to `setter`.
		template<typename IteratorContainer, typename PushComponent, typename RebaseComponent, typename Setter>
		static inline void attachStructuredView(IGeometry<ICPUBuffer>::SDataViewBase& baseView, const size_t elementCount, IteratorContainer& iterators, PushComponent&& pushComponent, RebaseComponent&& rebaseComponent, Setter&& setter)
		{
			auto beginIx = iterators.size();
			auto view = createStructuredView(baseView, elementCount, [&](const size_t offset, const uint32_t stride, const E_FORMAT componentFormat) -> void { pushComponent(iterators, offset, stride, componentFormat); });
			if (!view)
				return;
			const auto basePtr = ptrdiff_t(view.src.buffer->getPointer()) + view.src.offset;
			for (const auto endIx = iterators.size(); beginIx != endIx; ++beginIx)
				rebaseComponent(iterators[beginIx], basePtr);
			setter(std::move(view));
		}
		//! Visits position, normal, and auxiliary attribute views for one polygon geometry.
		template<typename Visitor>
		static inline void visitVertexAttributeViews(const ICPUPolygonGeometry* geometry, Visitor&& visitor)
		{
			if (!geometry)
				return;
			visitor(geometry->getPositionView());
			visitor(geometry->getNormalView());
			for (const auto& view : geometry->getAuxAttributeViews())
				visitor(view);
		}
		//! Visits all views owned by one polygon geometry, including index and skeletal data.
		template<typename Visitor>
		static inline void visitGeometryViews(const ICPUPolygonGeometry* geometry, Visitor&& visitor)
		{
			if (!geometry)
				return;
			visitVertexAttributeViews(geometry, visitor);
			visitor(geometry->getIndexView());
			for (const auto& view : geometry->getJointWeightViews())
			{
				visitor(view.indices);
				visitor(view.weights);
			}
			if (const auto jointObb = geometry->getJointOBBView(); jointObb)
				visitor(*jointObb);
		}
		//! Stores one auxiliary view at `slot`, resizing the aux array as needed.
		static inline void setAuxViewAt(ICPUPolygonGeometry* geometry, const uint32_t slot, IGeometry<ICPUBuffer>::SDataView&& view)
		{
			if (!geometry || !view)
				return;
			auto* const auxViews = geometry->getAuxAttributeViews();
			if (auxViews->size() <= slot)
				auxViews->resize(slot + 1u);
			(*auxViews)[slot] = std::move(view);
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
