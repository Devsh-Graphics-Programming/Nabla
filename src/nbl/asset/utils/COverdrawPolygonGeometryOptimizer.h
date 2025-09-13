// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_OVERDRAW_POLYGON_GEOMETRY_OPTIMIZER_H_INCLUDED_
#define _NBL_ASSET_C_OVERDRAW_POLYGON_GEOMETRY_OPTIMIZER_H_INCLUDED_


#include "nbl/asset/ICPUPolygonGeometry.h"


namespace nbl::asset
{
// Based on zeux's meshoptimizer (https://github.com/zeux/meshoptimizer) available under MIT license
class COverdrawPolygonGeometryOptimizer
{
		_NBL_STATIC_INLINE_CONSTEXPR size_t CACHE_SIZE = 16;

		struct ClusterSortData
		{
			uint32_t cluster;
			float dot;

			bool operator>(const ClusterSortData& other) const
			{
				// high product = possible occluder, render early
				return dot > other.dot;
			}
		};

		// private, undefined constructor
		COverdrawPolygonGeometryOptimizer() = delete;

	public:
		//! Creates new or modifies given mesh reordering indices to reduce pixel overdraw and vertex shader invocations.
		/**
		@param _inbuffer Input mesh buffer.
		@param _createNew Flag deciding whether to create new mesh (not modifying given one) or just optimize given mesh. Defaulted to true (i.e. create new).
		@param _threshold Indicates how much the overdraw optimizer can degrade vertex cache efficiency (1.05 = up to 5%) to reduce overdraw more efficiently. Defaulted to 1.05 (i.e. 5%).
		*/
		static void createOptimized(ICPUPolygonGeometry* out, const ICPUPolygonGeometry* in, float _threshold = 1.05f, const system::logger_opt_ptr logger = nullptr);

	private:
		template<typename IdxT>
		static size_t genHardBoundaries(uint32_t* _dst, const IdxT* _indices, size_t _idxCount, size_t _vtxCount);
		template<typename IdxT>
		static size_t genSoftBoundaries(uint32_t* _dst, const IdxT* _indices, size_t _idxCount, size_t _vtxCount, const uint32_t* _clusters, size_t _clusterCount, float _threshold);

		template<typename IdxT>
		static void calcSortData(ClusterSortData* _dst, const IdxT* _indices, size_t _idxCount, const core::vector<core::vectorSIMDf>& _positions, const uint32_t* _clusters, size_t _clusterCount);

		static size_t updateCache(uint32_t _a, uint32_t _b, uint32_t _c, size_t* _cacheTimestamps, size_t& _timestamp);
};

}
#endif