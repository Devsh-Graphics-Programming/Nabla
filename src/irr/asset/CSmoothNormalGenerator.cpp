#include "CSmoothNormalGenerator.h"

#include <iostream>
#include <algorithm>
#include <array>

namespace irr
{
namespace asset
{
static inline bool operator<(uint32_t lhs, const IMeshManipulator::SSNGVertexData& rhs)
{
	return lhs < rhs.hash;
}

static inline bool operator<(const IMeshManipulator::SSNGVertexData& lhs, uint32_t rhs)
{
	return lhs.hash < rhs;
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

asset::ICPUMeshBuffer* irr::asset::CSmoothNormalGenerator::calculateNormals(asset::ICPUMeshBuffer* buffer, float epsilon, asset::E_VERTEX_ATTRIBUTE_ID normalAttrID, IMeshManipulator::VxCmpFunction vxcmp)
{
	VertexHashMap vertexArray = setupData(buffer, epsilon);
	processConnectedVertices(buffer, vertexArray, epsilon, normalAttrID, vxcmp);

	return buffer;
}

CSmoothNormalGenerator::VertexHashMap::VertexHashMap(size_t _vertexCount, uint32_t _hashTableMaxSize, float _cellSize)
	:hashTableMaxSize(_hashTableMaxSize), 
	cellSize(_cellSize)
{
	assert((core::isPoT(hashTableMaxSize)));

	vertices.reserve(_vertexCount);
	buckets.reserve(_hashTableMaxSize+1);
}

uint32_t CSmoothNormalGenerator::VertexHashMap::hash(const IMeshManipulator::SSNGVertexData& vertex) const
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

void CSmoothNormalGenerator::VertexHashMap::add(IMeshManipulator::SSNGVertexData&& vertex)
{
	vertex.hash = hash(vertex);
	vertices.push_back(vertex);
}

CSmoothNormalGenerator::VertexHashMap::BucketBounds CSmoothNormalGenerator::VertexHashMap::getBucketBoundsByHash(uint32_t hash)
{
	if (hash == invalidHash)
		return { vertices.end(), vertices.end() };

	core::vector<IMeshManipulator::SSNGVertexData>::iterator begin = std::lower_bound(vertices.begin(), vertices.end(), hash);
	core::vector<IMeshManipulator::SSNGVertexData>::iterator end = std::upper_bound(vertices.begin(), vertices.end(), hash);

	//bucket missing
	if(begin == vertices.end())
		return { vertices.end(), vertices.end() };
	
	//bucket missing
	if (begin->hash != hash)
		return { vertices.end(), vertices.end() };

	return { begin, end };
}

void CSmoothNormalGenerator::VertexHashMap::validate()
{
	std::sort(vertices.begin(), vertices.end(), [](IMeshManipulator::SSNGVertexData& a, IMeshManipulator::SSNGVertexData& b) { return a.hash < b.hash; });

	uint16_t prevHash = vertices[0].hash;
	core::vector<IMeshManipulator::SSNGVertexData>::iterator prevBegin = vertices.begin();
	buckets.push_back(prevBegin);

	while (true)
	{
		core::vector<IMeshManipulator::SSNGVertexData>::iterator next = std::upper_bound(prevBegin, vertices.end(), prevHash);
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
	_IRR_DEBUG_BREAK_IF((idxCount % 3));

	VertexHashMap vertices(idxCount, std::min(16u * 1024u, core::roundUpToPoT<unsigned int>(idxCount * 1.0f/32.0f)), epsilon == 0.0f ? 0.00001f : epsilon * 1.00001f);

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

		vertices.add({ i,		0,	angleWages.x,	v1,		faceNormal});
		vertices.add({ i + 1,	0,	angleWages.y,	v2,		faceNormal});
		vertices.add({ i + 2,	0,	angleWages.z,	v3,		faceNormal});
	}

	vertices.validate();

	return vertices;
}

void CSmoothNormalGenerator::processConnectedVertices(asset::ICPUMeshBuffer * buffer, VertexHashMap & vertexHashMap, float epsilon, asset::E_VERTEX_ATTRIBUTE_ID normalAttrID, IMeshManipulator::VxCmpFunction vxcmp)
{
	for (uint32_t cell = 0; cell < vertexHashMap.getBucketCount() - 1; cell++)
	{
		VertexHashMap::BucketBounds processedBucket = vertexHashMap.getBucketBoundsById(cell);

		for (core::vector<IMeshManipulator::SSNGVertexData>::iterator processedVertex = processedBucket.begin; processedVertex != processedBucket.end; processedVertex++)
		{
			std::array<uint32_t, 8> neighboringCells = vertexHashMap.getNeighboringCellHashes(*processedVertex);
			core::vector3df_SIMD normal(0.0f);

			//iterate among all neighboring cells
			for (int i = 0; i < 8; i++)
			{
				VertexHashMap::BucketBounds bounds = vertexHashMap.getBucketBoundsByHash(neighboringCells[i]);
				for (; bounds.begin != bounds.end; bounds.begin++)
				{
					
					if (compareVertexPosition(processedVertex->position, bounds.begin->position, epsilon) &&
						vxcmp(*processedVertex, *bounds.begin, buffer))
					{
						normal += bounds.begin->parentTriangleFaceNormal * bounds.begin->wage;
					}
				}
			}

			normal = core::normalize(normal);
			buffer->setAttribute(normal, normalAttrID, processedVertex->indexOffset);
		}
	}
	

 }

std::array<uint32_t, 8> CSmoothNormalGenerator::VertexHashMap::getNeighboringCellHashes(const IMeshManipulator::SSNGVertexData& vertex)
{
	std::array<uint32_t, 8> neighbourhood;

	core::vectorSIMDf cellFloatCoord = vertex.position / cellSize - core::vectorSIMDf(0.5f);
	core::vector3du32_SIMD neighbor = core::vector3du32_SIMD(static_cast<uint32_t>(cellFloatCoord.x), static_cast<uint32_t>(cellFloatCoord.y), static_cast<uint32_t>(cellFloatCoord.z));
	
	//left bottom near
	neighbourhood[0] = hash(neighbor);

	//right bottom near
	neighbor = neighbor + core::vector3du32_SIMD(1, 0, 0);
	neighbourhood[1] = hash(neighbor);

	//right bottom far
	neighbor = neighbor + core::vector3du32_SIMD(0, 0, 1);
	neighbourhood[2] = hash(neighbor);

	//left bottom far
	neighbor = neighbor - core::vector3du32_SIMD(1, 0, 0);
	neighbourhood[3] = hash(neighbor);

	//left top far
	neighbor = neighbor + core::vector3du32_SIMD(0, 1, 0);
	neighbourhood[4] = hash(neighbor);

	//right top far
	neighbor = neighbor + core::vector3du32_SIMD(1, 0, 0);
	neighbourhood[5] = hash(neighbor);

	//righ top near
	neighbor = neighbor - core::vector3du32_SIMD(0, 0, 1);
	neighbourhood[6] = hash(neighbor);

	//left top near
	neighbor = neighbor - core::vector3du32_SIMD(1, 0, 0);
	neighbourhood[7] = hash(neighbor);

	//erase duplicated hashes
	for (int i = 0; i < 8; i++)
	{
		uint32_t currHash = neighbourhood[i];
		for (int j = i + 1; j < 8; j++)
		{
			if (neighbourhood[j] == currHash)
				neighbourhood[j] = invalidHash;
		}
	}
	return neighbourhood;
}

}
}