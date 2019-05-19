#ifndef __C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED__
#define __C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED__

#include <iostream>
#include <functional>
#include "irr/asset/ICPUMeshBuffer.h"
#include "irr/core/math/irrMath.h"


namespace irr 
{	
namespace asset 
{
	
//vertex data needed for CSmoothNormalGenerator
struct SSNGVertexData
{
	uint32_t indexOffset;									//offset of the vertex into index buffer
	uint32_t hash;											//
	float wage;												//angle wage of the vertex
	core::vector4df_SIMD position;							//position of the vertex in 3D space
	core::vector3df_SIMD normal;							//normal to be smoothed
	core::vector3df_SIMD parentTriangleFaceNormal;			//
}; 

typedef std::function<bool(const SSNGVertexData&, const SSNGVertexData&, asset::ICPUMeshBuffer*)> VxCmpFunction;

//returns true if crease angle between parent triangles of vertices v0 and v1 is lower than 45 degrees 
bool defaultVxCmpFunction(const SSNGVertexData&, const SSNGVertexData&, asset::ICPUMeshBuffer*);

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
		VertexHashMap(size_t _vertexCount, uint32_t _hashTableMaxSize, float _cellSize);

		//inserts vertex into hash table
		void add(const SSNGVertexData& vertex);

		//sorts hashtable and sets iterators at beginnings of bucktes
		void validate();

		//small number of unused buckets still may occure 
		inline uint32_t getBucketCount() const { return buckets.size(); }
		inline core::vector<SSNGVertexData>::iterator getBucket(uint32_t index) { return buckets[index]; }

	private:
		//holds iterators pointing to beginning of each bucket, last iterator points to vertices.end()
		core::vector<core::vector<SSNGVertexData>::iterator> buckets;
		core::vector<SSNGVertexData> vertices;
		const uint32_t hashTableMaxSize;
		const float cellSize;

	private:
		uint32_t hash(const SSNGVertexData& vertexPosition) const;

	};

private:
	static VertexHashMap setupData(asset::ICPUMeshBuffer* buffer, float epsilon);
	static void processConnectedVertices(asset::ICPUMeshBuffer* buffer, VertexHashMap& vertices, float epsilon, asset::E_VERTEX_ATTRIBUTE_ID normalAttrID, VxCmpFunction vxcmp);

};



}
}

#endif