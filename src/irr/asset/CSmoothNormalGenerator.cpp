#include "CSmoothNormalGenerator.h"

#include <iostream>
#include <algorithm>

namespace irr
{
namespace asset
{

bool defaultVxCmpFunction(const SSNGVertexData& v0, const SSNGVertexData& v1, asset::ICPUMeshBuffer* buffer)
{
	static constexpr float cosOf45Deg = 0.70710678118f;
	return v0.parentTriangleFaceNormal.dotProductAsFloat(v1.parentTriangleFaceNormal) > cosOf45Deg;
}

//needed for std::upper_boud
static inline bool operator<(uint32_t lhs, const SSNGVertexData& rhs)
{
	return lhs < rhs.hash;
}

static inline bool compareVertexPosition(const core::vectorSIMDf& a, const core::vectorSIMDf& b, float epsilon)
{
	const core::vectorSIMDf difference = core::abs(b - a);
	return (difference.x <= epsilon && difference.y <= epsilon && difference.z <= epsilon);
}

static inline core::vector3df_SIMD getAngleWeight(const core::vector3df_SIMD & v1,
	const core::vector3df_SIMD & v2,
	const core::vector3df_SIMD & v3)
{
		// Calculate this triangle's weight for each of its three vertices
		// start by calculating the lengths of its sides
	const float a = v2.getDistanceFromSQAsFloat(v3);
	const float asqrt = sqrtf(a);
	const float b = v1.getDistanceFromSQAsFloat(v3);
	const float bsqrt = sqrtf(b);
	const float c = v1.getDistanceFromSQAsFloat(v2);
	const float csqrt = sqrtf(c);

		// use them to find the angle at each vertex
	return core::vector3df_SIMD(
		acosf((b + c - a) / (2.f * bsqrt * csqrt)),
		acosf((-b + c + a) / (2.f * asqrt * csqrt)),
		acosf((b - c + a) / (2.f * bsqrt * asqrt)));
}

asset::ICPUMeshBuffer* irr::asset::CSmoothNormalGenerator::calculateNormals(asset::ICPUMeshBuffer* buffer, float epsilon, asset::E_VERTEX_ATTRIBUTE_ID normalAttrID, VxCmpFunction vxcmp)
{
	//should i always trust RVO?
	VertexHashMap vertexArray = setupData(buffer, epsilon);
	processConnectedVertices(buffer, vertexArray, epsilon, normalAttrID, vxcmp);

	return buffer;
}

CSmoothNormalGenerator::VertexHashMap::VertexHashMap(size_t _vertexCount, uint32_t _hashTableMaxSize, float _cellSize)
	:hashTableMaxSize(_hashTableMaxSize), 
	cellSize(_cellSize)
{
	vertices.reserve(_vertexCount);
	buckets.reserve(_hashTableMaxSize+1);
}

uint32_t CSmoothNormalGenerator::VertexHashMap::hash(const SSNGVertexData& vertexPosition) const
{
	static constexpr uint32_t primeNumber1 = 73856093;
	static constexpr uint32_t primeNumber2 = 19349663;
	static constexpr uint32_t primeNumber3 = 83492791;

	const core::vector3df_SIMD position = vertexPosition.position / cellSize;

	return	(((uint32_t)position.x * primeNumber1) ^
		((uint32_t)position.y * primeNumber2) ^
		((uint32_t)position.z * primeNumber3)) & (hashTableMaxSize - 1);
}

void CSmoothNormalGenerator::VertexHashMap::add(const SSNGVertexData& vertex)
{
	const_cast<uint32_t&>(vertex.hash) = hash(vertex);
	vertices.push_back(vertex);
}

void CSmoothNormalGenerator::VertexHashMap::validate()
{
	std::sort(vertices.begin(), vertices.end(), [](SSNGVertexData& a, SSNGVertexData& b) { return a.hash < b.hash; });

	uint16_t prevHash = vertices[0].hash;
	core::vector<SSNGVertexData>::iterator prevBegin = vertices.begin();
	buckets.push_back(prevBegin);

	while (true)
	{
		core::vector<SSNGVertexData>::iterator next = std::upper_bound(prevBegin, vertices.end(), prevHash);

		buckets.push_back(next);

		if (next == vertices.end())
			break;

		prevBegin = next;
		prevHash = next->hash;
	}
}

CSmoothNormalGenerator::VertexHashMap CSmoothNormalGenerator::setupData(asset::ICPUMeshBuffer * buffer, float epsilon)
{
	const size_t idxCount = buffer->getIndexCount();
	const size_t vxCount = buffer->calcVertexCount();

	_IRR_DEBUG_BREAK_IF((idxCount % 3));
	_IRR_DEBUG_BREAK_IF((idxCount != vxCount));

	VertexHashMap vertices(vxCount, std::min(16u * 1024u, core::roundUpToPoT<unsigned int>(idxCount * 1.0f / 32.0f)), epsilon * 1.2f);

	core::vector3df_SIMD faceNormal;

	for (uint32_t i = 0; i < idxCount; i += 3)
	{
		//calculate face normal of parent triangle
		core::vectorSIMDf v1 = buffer->getPosition(i);
		core::vectorSIMDf v2 = buffer->getPosition(i + 1);
		core::vectorSIMDf v3 = buffer->getPosition(i + 2);

		faceNormal = core::cross(v2 - v1, v3 - v1);
		faceNormal = core::normalize(faceNormal);

		//set data for vertices
		core::vector3df_SIMD angleWages = getAngleWeight(v1, v2, v3);

		vertices.add({ i,		0,	angleWages.x,	v1,	faceNormal * angleWages.x,	faceNormal });
		vertices.add({ i + 1,	0,	angleWages.y,	v2,	faceNormal * angleWages.y,	faceNormal });
		vertices.add({ i + 2,	0,	angleWages.z,	v3,	faceNormal * angleWages.z,	faceNormal });
	}

	vertices.validate();

	return vertices;
}

void CSmoothNormalGenerator::processConnectedVertices(asset::ICPUMeshBuffer * buffer, VertexHashMap & vertexHashMap, float epsilon, asset::E_VERTEX_ATTRIBUTE_ID normalAttrID, VxCmpFunction vxcmp)
{
	core::vector<SSNGVertexData>::iterator bucketBegin;
	core::vector<SSNGVertexData>::iterator bucketEnd;

	for (uint32_t cell = 0; cell < vertexHashMap.getBucketCount() - 1; cell++)
	{
		bucketBegin = vertexHashMap.getBucket(cell);
		bucketEnd = vertexHashMap.getBucket(cell + 1);

		for (core::vector<SSNGVertexData>::iterator processedVertex = bucketBegin; processedVertex != bucketEnd; processedVertex++)
		{
			for (core::vector<SSNGVertexData>::iterator nextVertex = processedVertex+1; nextVertex != bucketEnd; nextVertex++)
			{
				if (compareVertexPosition(processedVertex->position, nextVertex->position, epsilon) &&
					vxcmp(*processedVertex, *nextVertex, buffer))
				{
					processedVertex->normal += nextVertex->parentTriangleFaceNormal * nextVertex->wage;
					nextVertex->normal += processedVertex->parentTriangleFaceNormal * processedVertex->wage;
				}
			}	

			processedVertex->normal = core::normalize(processedVertex->normal);
			buffer->setAttribute(processedVertex->normal, normalAttrID, processedVertex->indexOffset);
		}
	}
	

 }

}
}

