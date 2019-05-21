#ifndef __C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED__
#define __C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED__

#include <iostream>
#include <functional>
#include "irr/asset/ICPUMeshBuffer.h"
#include "irr/asset/CMeshManipulator.h"
#include "irr/core/math/irrMath.h"


namespace irr 
{	
namespace asset 
{

class CSmoothNormalGenerator
{
public:
	static asset::ICPUMeshBuffer* calculateNormals(asset::ICPUMeshBuffer* buffer, float epsilon, asset::E_VERTEX_ATTRIBUTE_ID normalAttrID, VxCmpFunction function);

	CSmoothNormalGenerator() = delete;
	~CSmoothNormalGenerator() = delete;

private:
	class VertexHashMap
	{
	public:
		struct BucketBounds
		{
			core::vector<SSNGVertexData>::iterator begin;
			core::vector<SSNGVertexData>::iterator end;
		};

	public:
		VertexHashMap(size_t _vertexCount, uint32_t _hashTableMaxSize, float _cellSize);

		//inserts vertex into hash table
		void add(SSNGVertexData&& vertex);

		//sorts hashtable and sets iterators at beginnings of bucktes
		void validate();

		//
		std::array<uint32_t, 8> getNeighboringCellHashes(const SSNGVertexData& vertex);

		inline uint32_t getBucketCount() const { return buckets.size(); }
		inline core::vector<SSNGVertexData>::iterator getBucketById(uint32_t index) { return buckets[index]; }
		BucketBounds getBucketBoundsByHash(uint32_t hash);

	public:
		static constexpr uint32_t invalidHash = 0xFFFFFFFF;

	private:
		//holds iterators pointing to beginning of each bucket, last iterator points to vertices.end()
		core::vector<core::vector<SSNGVertexData>::iterator> buckets;
		core::vector<SSNGVertexData> vertices;
		const uint32_t hashTableMaxSize;
		const float cellSize;

	private:
		uint32_t hash(const SSNGVertexData& vertex) const;
		uint32_t hash(const core::vector3du32_SIMD& position) const;

	};

private:
	static VertexHashMap setupData(asset::ICPUMeshBuffer* buffer, float epsilon);
	static void processConnectedVertices(asset::ICPUMeshBuffer* buffer, VertexHashMap& vertices, float epsilon, asset::E_VERTEX_ATTRIBUTE_ID normalAttrID, VxCmpFunction vxcmp);

};

//vec3 pos = vertexPos;
//vec3 voxelCoordFloat = pos / voxelSize - vec3(0.5);

//ivec3 leftBottomNear =	ivec3(voxelCoordFloat);
//ivec3 rightBottomNear =	leftBottomNear + ivec3(1,0,0);
//ivec3 leftTopNear =		leftBottomNear + ivec3(0,1,0);
//ivec3 rightTopNear =		rightBottomNear + ivec3(0,1,0);
//ivec3 leftBottomFar =		leftBottomNear + ivec3(0,0,1);
//ivec3 rightBottomFar =	rightBottomNear + ivec3(0,0,1);
//ivec3 leftTopFar =		leftTopNear + ivec3(0,0,1);
//ivec3 rightTopFar =		rightTopNear + ivec3(0,0,1);

}
}

#endif