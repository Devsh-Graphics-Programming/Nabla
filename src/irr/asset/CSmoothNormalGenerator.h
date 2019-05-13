#ifndef __C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED__
#define __C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED__

#include <iostream>
#include "irr/asset/ICPUMeshBuffer.h"
#include "irr/core/math/irrMath.h"

namespace irr 
{	
namespace asset 
{


class CSmoothNormalGenerator
{
public:
	static asset::ICPUMeshBuffer* calculateNormals(asset::ICPUMeshBuffer* buffer, float creaseAngle, float epsilon);
	static asset::ICPUMeshBuffer* calculateNormals_hash(asset::ICPUMeshBuffer* buffer, float creaseAngle, float epsilon);

	CSmoothNormalGenerator() = delete;
	~CSmoothNormalGenerator() = delete;

public:
	struct Vertex
	{
		uint32_t indexOffset;								//offset of the vertex into index buffer
		float wage;											//angle wage of the vertex
		core::vector3df_SIMD position;						//position of the vertex in 3D space
		core::vector3df_SIMD normal;						//normal to be smoothed
		core::vector3df_SIMD parentTriangleFaceNormal;
	};

private:
	class VertexHashMap
	{
	public:
		VertexHashMap(size_t _hashTableSize, float _cellSize);

		//similar to std::unordered_map::insert
		void add(const Vertex& vertex);

		inline size_t getTableSize() { return hashTableSize; }
		//returns array of vertices assigned to hash with value of parameter index
		inline core::vector<Vertex>& getBucket(size_t index) { return hashTable[index]; }

	private:
		core::vector<core::vector<Vertex>> hashTable;
		const size_t hashTableSize;
		const float cellSize;

	private:
		uint32_t hash(const Vertex& vertex) const;
	};

private:
	//I keep these for reference
	static core::vector<Vertex> setupData(asset::ICPUMeshBuffer* buffer, float creaseAngle);
	static void processConnectedVertices(asset::ICPUMeshBuffer* buffer, core::vector<Vertex>& vertices, float creaseAngle, float epsilon);

	static VertexHashMap setupData_hash(asset::ICPUMeshBuffer* buffer, float creaseAngle);
	static void processConnectedVertices_hash(asset::ICPUMeshBuffer* buffer, VertexHashMap& vertices, float creaseAngle, float epsilon);

};



}
}

#endif