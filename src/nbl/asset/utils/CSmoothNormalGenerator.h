// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED_
#define _NBL_ASSET_C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED_


#include "nbl/asset/utils/CPolygonGeometryManipulator.h"

namespace nbl::asset 
{

class CSmoothNormalGenerator
{
	public:
		CSmoothNormalGenerator() = delete;
		~CSmoothNormalGenerator() = delete;

		static core::smart_refctd_ptr<ICPUPolygonGeometry> calculateNormals(const ICPUPolygonGeometry* polygon, bool enableWelding, float epsilon, CPolygonGeometryManipulator::VxCmpFunction function);

	private:
		class VertexHashMap
		{
			public:
				struct BucketBounds
				{
					core::vector<CPolygonGeometryManipulator::SSNGVertexData>::iterator begin;
					core::vector<CPolygonGeometryManipulator::SSNGVertexData>::iterator end;
				};

			public:
				VertexHashMap(size_t _vertexCount, uint32_t _hashTableMaxSize, float _cellSize);

				//inserts vertex into hash table
				void add(CPolygonGeometryManipulator::SSNGVertexData&& vertex);

				//sorts hashtable and sets iterators at beginnings of bucktes
				void validate();

				inline uint32_t getVertexCount() const { return m_vertices.size(); }

				//
				std::array<uint32_t, 8> getNeighboringCellHashes(const CPolygonGeometryManipulator::SSNGVertexData& vertex);

				inline uint32_t getBucketCount() { return m_buckets.size(); }
				inline BucketBounds getBucketBoundsById(uint32_t index) const { return { m_buckets[index], m_buckets[index + 1] }; }
				BucketBounds getBucketBoundsByHash(uint32_t hash);

			private:
				static inline constexpr uint32_t invalidHash = 0xFFFFFFFF;
				static inline constexpr uint32_t primeNumber1 = 73856093;
				static inline constexpr uint32_t primeNumber2 = 19349663;
				static inline constexpr uint32_t primeNumber3 = 83492791;

				//holds iterators pointing to beginning of each bucket, last iterator points to m_vertices.end()
				core::vector<core::vector<CPolygonGeometryManipulator::SSNGVertexData>::iterator> m_buckets;
				core::vector<CPolygonGeometryManipulator::SSNGVertexData> m_vertices;
				const uint32_t m_hashTableMaxSize;
				const float m_cellSize;

				uint32_t hash(const CPolygonGeometryManipulator::SSNGVertexData& vertex) const;
				uint32_t hash(const hlsl::uint32_t3& position) const;

		};

	private:
		static VertexHashMap setupData(const ICPUPolygonGeometry* polygon, float epsilon);
		static core::smart_refctd_ptr<ICPUPolygonGeometry> processConnectedVertices(const ICPUPolygonGeometry* polygon, VertexHashMap& vertices, float epsilon, CPolygonGeometryManipulator::VxCmpFunction vxcmp);
		static core::smart_refctd_ptr<ICPUPolygonGeometry> weldVertices(const ICPUPolygonGeometry* polygon, VertexHashMap& vertices, float epsilon);
};

}
#endif