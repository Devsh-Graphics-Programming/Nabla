// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED_
#define _NBL_ASSET_C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED_


#include "nbl/asset/utils/CPolygonGeometryManipulator.h"

#include <functional>


namespace nbl::asset 
{

class CSmoothNormalGenerator
{
	public:
		CSmoothNormalGenerator() = delete;
		~CSmoothNormalGenerator() = delete;
#if 0
		static core::smart_refctd_ptr<ICPUMeshBuffer> calculateNormals(ICPUMeshBuffer* buffer, float epsilon, uint32_t normalAttrID, IMeshManipulator::VxCmpFunction function);

	private:
		class VertexHashMap
		{
		public:
			struct BucketBounds
			{
				core::vector<IMeshManipulator::SSNGVertexData>::iterator begin;
				core::vector<IMeshManipulator::SSNGVertexData>::iterator end;
			};

		public:
			VertexHashMap(size_t _vertexCount, uint32_t _hashTableMaxSize, float _cellSize);

			//inserts vertex into hash table
			void add(IMeshManipulator::SSNGVertexData&& vertex);

			//sorts hashtable and sets iterators at beginnings of bucktes
			void validate();

			//
			std::array<uint32_t, 8> getNeighboringCellHashes(const IMeshManipulator::SSNGVertexData& vertex);

			inline uint32_t getBucketCount() const { return buckets.size(); }
			inline BucketBounds getBucketBoundsById(uint32_t index) { return { buckets[index], buckets[index + 1] }; }
			BucketBounds getBucketBoundsByHash(uint32_t hash);

		private:
			static constexpr uint32_t invalidHash = 0xFFFFFFFF;

		private:
			//holds iterators pointing to beginning of each bucket, last iterator points to vertices.end()
			core::vector<core::vector<IMeshManipulator::SSNGVertexData>::iterator> buckets;
			core::vector<IMeshManipulator::SSNGVertexData> vertices;
			const uint32_t hashTableMaxSize;
			const float cellSize;

		private:
			uint32_t hash(const IMeshManipulator::SSNGVertexData& vertex) const;
			uint32_t hash(const core::vector3du32_SIMD& position) const;

		};

	private:
		static VertexHashMap setupData(const ICPUMeshBuffer* buffer, float epsilon);
		static void processConnectedVertices(ICPUMeshBuffer* buffer, VertexHashMap& vertices, float epsilon, uint32_t normalAttrID, IMeshManipulator::VxCmpFunction vxcmp);
#endif
};

}
#endif