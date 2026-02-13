// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_GEOMETRY_MANIPULATOR_H_INCLUDED_
#define _NBL_ASSET_C_GEOMETRY_MANIPULATOR_H_INCLUDED_


#include "nbl/core/declarations.h"

#include "nbl/asset/ICPUPolygonGeometry.h"
#include "nbl/asset/utils/CQuantNormalCache.h"
#include "nbl/asset/utils/CQuantQuaternionCache.h"

namespace nbl::asset
{

class NBL_API2 CGeometryManipulator
{
	public:
		static inline void recomputeContentHash(const IGeometry<ICPUBuffer>::SDataView& view)
		{
			if (!view)
				return;
			view.src.buffer->setContentHash(view.src.buffer->computeContentHash());
		}

		static inline IGeometryBase::SAABBStorage computeRange(const IGeometry<ICPUBuffer>::SDataView& view)
		{
			if (!view || !view.composed.isFormatted())
				return {};
			auto it = reinterpret_cast<char*>(view.src.buffer->getPointer())+view.src.offset;
			[[maybe_unused]] const auto end = it+view.src.actualSize();
			auto addToAABB = [&](auto& aabb)->void
			{
				using aabb_t = std::remove_reference_t<decltype(aabb)>;
				for (uint64_t i=0; i!=view.getElementCount(); i++)
				{
					typename aabb_t::point_t pt;
					view.decodeElement(i,pt);
					aabb.addPoint(pt);
				}
			};
			IGeometryBase::SDataViewBase tmp = view.composed;
			tmp.resetRange();
			tmp.visitRange(addToAABB);
			return tmp.encodedDataRange;
		}

		static inline void recomputeRange(IGeometry<ICPUBuffer>::SDataView& view, const bool deduceRangeFormat=true)
		{
			if (!view || !view.composed.isFormatted())
				return;
			if (deduceRangeFormat)
				view.composed.rangeFormat = IGeometryBase::getMatchingAABBFormat(view.composed.format);
			view.composed.encodedDataRange = computeRange(view);
		}
};

// TODO: Utility in another header for GeometryCollection to compute AABBs, deal with skins (joints), etc.

} // end namespace nbl::asset
#endif
