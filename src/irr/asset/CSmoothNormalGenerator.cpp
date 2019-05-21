#include "CSmoothNormalGenerator.h"

#include <iostream>
#include <algorithm>
#include <array>

namespace irr
{
namespace asset
{

bool defaultVxCmpFunction(const SSNGVertexData& v0, const SSNGVertexData& v1, asset::ICPUMeshBuffer* triangleSoupMeshBuffer)
{
	static constexpr float cosOf45Deg = 0.70710678118f;
	return v0.parentTriangleFaceNormal.dotProductAsFloat(v1.parentTriangleFaceNormal) > cosOf45Deg;
}

//needed for upper_bound vertex search
static inline bool operator<(uint32_t lhs, const SSNGVertexData& rhs)
{
	return lhs < rhs.hash;
}

//needed for lower_bound bucket search
static inline bool operator<(const core::vector<SSNGVertexData>::iterator& lhs, uint32_t rhs)
{
	return lhs->hash < rhs;
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
	VertexHashMap vertexArray = setupData(buffer, epsilon);
	processConnectedVertices(buffer, vertexArray, epsilon, normalAttrID, vxcmp);

	return buffer;
}

CSmoothNormalGenerator::VertexHashMap::VertexHashMap(size_t _vertexCount, uint32_t _hashTableMaxSize, float _cellSize)
	:hashTableMaxSize(_hashTableMaxSize), 
	cellSize(_cellSize)
{
	_IRR_DEBUG_BREAK_IF((!core::isPoT(_hashTableMaxSize)));

	vertices.reserve(_vertexCount);
	buckets.reserve(_hashTableMaxSize+1);
}

uint32_t CSmoothNormalGenerator::VertexHashMap::hash(const SSNGVertexData& vertex) const
{
	static constexpr uint32_t primeNumber1 = 73856093;
	static constexpr uint32_t primeNumber2 = 19349663;
	static constexpr uint32_t primeNumber3 = 83492791;

	const core::vector3df_SIMD position = vertex.position / cellSize;

	return	((static_cast<uint32_t>(position.x) * primeNumber1) ^
			 (static_cast<uint32_t>(position.y) * primeNumber2) ^
			 (static_cast<uint32_t>(position.z) * primeNumber3)) & (hashTableMaxSize - 1);
}

uint32_t CSmoothNormalGenerator::VertexHashMap::hash(const core::vector3du32_SIMD& position) const
{
	static constexpr uint32_t primeNumber1 = 73856093;
	static constexpr uint32_t primeNumber2 = 19349663;
	static constexpr uint32_t primeNumber3 = 83492791;

	return	((position.x * primeNumber1) ^
			 (position.y * primeNumber2) ^
			 (position.z * primeNumber3)) & (hashTableMaxSize - 1);
}

void CSmoothNormalGenerator::VertexHashMap::add(SSNGVertexData&& vertex)
{
	vertex.hash = hash(vertex);
	vertices.push_back(vertex);
}

CSmoothNormalGenerator::VertexHashMap::BucketBounds CSmoothNormalGenerator::VertexHashMap::getBucketBoundsByHash(uint32_t hash)
{
	auto begin = std::lower_bound(buckets.begin(), buckets.end()-1, hash);

	if (begin == buckets.end()-1)
		return { vertices.end(), vertices.end() };

	return { *begin, *(begin + 1) };

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

	VertexHashMap vertices(vxCount, std::min(16u * 1024u, core::roundUpToPoT<unsigned int>(idxCount * 1.0f / 32.0f)), epsilon != 0.0f ? epsilon * 1.00001f : 0.00001f);

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

		vertices.add({ i,		0,	angleWages.x,	v1,	core::vector3df_SIMD(0.0f),	faceNormal});
		vertices.add({ i + 1,	0,	angleWages.y,	v2,	core::vector3df_SIMD(0.0f),	faceNormal});
		vertices.add({ i + 2,	0,	angleWages.z,	v3,	core::vector3df_SIMD(0.0f),	faceNormal});
	}

	vertices.validate();

	return vertices;
}

void CSmoothNormalGenerator::processConnectedVertices(asset::ICPUMeshBuffer * buffer, VertexHashMap & vertexHashMap, float epsilon, asset::E_VERTEX_ATTRIBUTE_ID normalAttrID, VxCmpFunction vxcmp)
{
	

	for (uint32_t cell = 0; cell < vertexHashMap.getBucketCount() - 1; cell++)
	{
		core::vector<SSNGVertexData>::iterator firstVertexInProcessedBucket = vertexHashMap.getBucketById(cell);
		core::vector<SSNGVertexData>::iterator lastVertexInProcessedBucket = vertexHashMap.getBucketById(cell+1);

		for (core::vector<SSNGVertexData>::iterator processedVertex = firstVertexInProcessedBucket; processedVertex != lastVertexInProcessedBucket; processedVertex++)
		{
			std::array<uint32_t, 8> neighboringCells = vertexHashMap.getNeighboringCellHashes(*processedVertex);

			//iterate among all neighboring cells
			for (int i = 0; i < 8; i++)
			{
				VertexHashMap::BucketBounds bounds = vertexHashMap.getBucketBoundsByHash(neighboringCells[i]);
				for (; bounds.begin != bounds.end; bounds.begin++)
				{
					
					if (compareVertexPosition(processedVertex->position, bounds.begin->position, epsilon) &&
						vxcmp(*processedVertex, *bounds.begin, buffer))
					{
						processedVertex->normal += bounds.begin->parentTriangleFaceNormal * bounds.begin->wage;
					}
				}
			}

			processedVertex->normal = core::normalize(processedVertex->normal);
			buffer->setAttribute(processedVertex->normal, normalAttrID, processedVertex->indexOffset);
		}
	}
	

 }

std::array<uint32_t, 8> CSmoothNormalGenerator::VertexHashMap::getNeighboringCellHashes(const SSNGVertexData& vertex)
{
	static unsigned int a = 0;
	std::array<uint32_t, 8> neighborhood;

	core::vectorSIMDf cellFloatCoord = vertex.position / cellSize - core::vectorSIMDf(0.5f);
	core::vector3du32_SIMD neighbor = core::vector3du32_SIMD(static_cast<uint32_t>(cellFloatCoord.x), static_cast<uint32_t>(cellFloatCoord.y), static_cast<uint32_t>(cellFloatCoord.z));
	
	//left bottom near
	neighborhood[0] = hash(neighbor);

	//right bottom near
	neighbor = neighbor + core::vector3du32_SIMD(1, 0, 0);
	neighborhood[1] = hash(neighbor);

	//right bottom far
	neighbor = neighbor + core::vector3du32_SIMD(0, 0, 1);
	neighborhood[2] = hash(neighbor);

	//left bottom far
	neighbor = neighbor - core::vector3du32_SIMD(1, 0, 0);
	neighborhood[3] = hash(neighbor);

	//left top far
	neighbor = neighbor + core::vector3du32_SIMD(0, 1, 0);
	neighborhood[4] = hash(neighbor);

	//right top far
	neighbor = neighbor + core::vector3du32_SIMD(1, 0, 0);
	neighborhood[5] = hash(neighbor);

	//righ top near
	neighbor = neighbor - core::vector3du32_SIMD(0, 0, 1);
	neighborhood[6] = hash(neighbor);

	//left top near
	neighbor = neighbor - core::vector3du32_SIMD(1, 0, 0);
	neighborhood[7] = hash(neighbor);

	//erase duplicated hashes
	for (int i = 0; i < 8; i++)
	{
		uint32_t currHash = neighborhood[i];
		for (int j = i + 1; j < 8; j++)
		{
			if (neighborhood[j] == currHash)
				neighborhood[j] = invalidHash;
		}
	}
	return neighborhood;
}

}
}

/*for (core::vector<SSNGVertexData>::iterator nextVertex = processedVertex+1; nextVertex != bucketEnd; nextVertex++)
			{
				if (compareVertexPosition(processedVertex->position, nextVertex->position, epsilon) &&
					vxcmp(*processedVertex, *nextVertex, buffer))
				{
					processedVertex->normal += nextVertex->parentTriangleFaceNormal * nextVertex->wage;
					nextVertex->normal += processedVertex->parentTriangleFaceNormal * processedVertex->wage;
				}
			}	*/
