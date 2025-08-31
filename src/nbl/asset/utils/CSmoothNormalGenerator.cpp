// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "CSmoothNormalGenerator.h"

#include "nbl/core/declarations.h"

#include <algorithm>
#include <array>

namespace nbl
{
namespace asset
{
static bool operator<(uint32_t lhs, const CPolygonGeometryManipulator::SSNGVertexData& rhs)
{
	return lhs < rhs.hash;
}

static bool operator<(const CPolygonGeometryManipulator::SSNGVertexData& lhs, uint32_t rhs)
{
	return lhs.hash < rhs;
}

static bool compareVertexPosition(const hlsl::float32_t3& a, const hlsl::float32_t3& b, float epsilon)
{
	const hlsl::float32_t3 difference = abs(b - a);
	return (difference.x <= epsilon && difference.y <= epsilon && difference.z <= epsilon);
}

static hlsl::float32_t3 getAngleWeight(
	const hlsl::float32_t3& v1,
	const hlsl::float32_t3& v2,
	const hlsl::float32_t3& v3)
{
	auto distancesquared = [](const hlsl::float32_t3& v1, const hlsl::float32_t3& v2)
  {
    const auto diff = v1 - v2;
		return hlsl::dot(diff, diff);
  };
	// Calculate this triangle's weight for each of its three m_vertices
	// start by calculating the lengths of its sides
	const float a = distancesquared(v2, v3);
	const float asqrt = sqrt(a);
	const float b = distancesquared(v1,v3);
	const float bsqrt = sqrt(b);
	const float c = distancesquared(v1,v2);
	const float csqrt = sqrt(c);

	// use them to find the angle at each vertex
	return hlsl::float32_t3(
		acosf((b + c - a) / (2.f * bsqrt * csqrt)),
		acosf((-b + c + a) / (2.f * asqrt * csqrt)),
		acosf((b - c + a) / (2.f * bsqrt * asqrt)));
}

core::smart_refctd_ptr<ICPUPolygonGeometry> CSmoothNormalGenerator::calculateNormals(asset::ICPUPolygonGeometry* polygon, float epsilon, CPolygonGeometryManipulator::VxCmpFunction vxcmp)
{
	VertexHashMap vertexArray = setupData(polygon, epsilon);
	processConnectedVertices(polygon, vertexArray, epsilon,vxcmp);

	return core::smart_refctd_ptr<ICPUPolygonGeometry>(polygon);
}

CSmoothNormalGenerator::VertexHashMap::VertexHashMap(size_t _vertexCount, uint32_t _hashTableMaxSize, float _cellSize)
	:m_hashTableMaxSize(_hashTableMaxSize),
	m_cellSize(_cellSize)
{
	assert((core::isPoT(m_hashTableMaxSize)));

	m_vertices.reserve(_vertexCount);
	m_buckets.reserve(_hashTableMaxSize + 1);
}

uint32_t CSmoothNormalGenerator::VertexHashMap::hash(const CPolygonGeometryManipulator::SSNGVertexData & vertex) const
{
	const hlsl::float32_t3 position = vertex.position / m_cellSize;

	return	((static_cast<uint32_t>(position.x) * primeNumber1) ^
		(static_cast<uint32_t>(position.y) * primeNumber2) ^
		(static_cast<uint32_t>(position.z) * primeNumber3))& (m_hashTableMaxSize - 1);
}

uint32_t CSmoothNormalGenerator::VertexHashMap::hash(const hlsl::uint32_t3& position) const
{
	return	((position.x * primeNumber1) ^
		(position.y * primeNumber2) ^
		(position.z * primeNumber3))& (m_hashTableMaxSize - 1);
}

void CSmoothNormalGenerator::VertexHashMap::add(CPolygonGeometryManipulator::SSNGVertexData && vertex)
{
	vertex.hash = hash(vertex);
	m_vertices.push_back(vertex);
}

CSmoothNormalGenerator::VertexHashMap::BucketBounds CSmoothNormalGenerator::VertexHashMap::getBucketBoundsByHash(uint32_t hash)
{
	if (hash == invalidHash)
		return { m_vertices.end(), m_vertices.end() };

	core::vector<CPolygonGeometryManipulator::SSNGVertexData>::iterator begin = std::lower_bound(m_vertices.begin(), m_vertices.end(), hash);
	core::vector<CPolygonGeometryManipulator::SSNGVertexData>::iterator end = std::upper_bound(m_vertices.begin(), m_vertices.end(), hash);

	//bucket missing
	if (begin == m_vertices.end())
		return { m_vertices.end(), m_vertices.end() };

	//bucket missing
	if (begin->hash != hash)
		return { m_vertices.end(), m_vertices.end() };

	return { begin, end };
}

struct KeyAccessor
{
	_NBL_STATIC_INLINE_CONSTEXPR size_t key_bit_count = 32ull;

	template<auto bit_offset, auto radix_mask>
	inline decltype(radix_mask) operator()(const CPolygonGeometryManipulator::SSNGVertexData& item) const
	{
		return static_cast<decltype(radix_mask)>(item.hash>>static_cast<uint32_t>(bit_offset))&radix_mask;
	}
};
void CSmoothNormalGenerator::VertexHashMap::validate()
{
	const auto oldSize = m_vertices.size();
	m_vertices.resize(oldSize*2u);
	// TODO: maybe use counting sort (or big radix) and use the histogram directly for the m_buckets
	auto finalSortedOutput = core::radix_sort(m_vertices.data(),m_vertices.data()+oldSize,oldSize,KeyAccessor());
	// TODO: optimize out the erase
	if (finalSortedOutput!=m_vertices.data())
		m_vertices.erase(m_vertices.begin(),m_vertices.begin()+oldSize);
	else
		m_vertices.erase(m_vertices.begin()+oldSize,m_vertices.end());

	// TODO: are `m_buckets` even begin USED!?
	uint16_t prevHash = m_vertices[0].hash;
	core::vector<CPolygonGeometryManipulator::SSNGVertexData>::iterator prevBegin = m_vertices.begin();
	m_buckets.push_back(prevBegin);

	while (true)
	{
		core::vector<CPolygonGeometryManipulator::SSNGVertexData>::iterator next = std::upper_bound(prevBegin, m_vertices.end(), prevHash);
		m_buckets.push_back(next);

		if (next == m_vertices.end())
			break;

		prevBegin = next;
		prevHash = next->hash;
	}
}

CSmoothNormalGenerator::VertexHashMap CSmoothNormalGenerator::setupData(const asset::ICPUPolygonGeometry* polygon, float epsilon)
{
	const size_t idxCount = polygon->getPrimitiveCount() * 3;

	VertexHashMap vertices(idxCount, std::min(16u * 1024u, core::roundUpToPoT<unsigned int>(idxCount * 1.0f / 32.0f)), epsilon == 0.0f ? 0.00001f : epsilon * 1.00001f);

	for (uint32_t i = 0; i < idxCount; i += 3)
	{
		//calculate face normal of parent triangle
		hlsl::float32_t3 v1, v2, v3;
		polygon->getPositionView().decodeElement<hlsl::float32_t3>(i, v1);
		polygon->getPositionView().decodeElement<hlsl::float32_t3>(i + 1, v2);
		polygon->getPositionView().decodeElement<hlsl::float32_t3>(i + 2, v3);

		const auto faceNormal = normalize(cross(v3 - v1, v2 - v1));

		//set data for m_vertices
		const auto angleWages = getAngleWeight(v1, v2, v3);

		vertices.add({ i,	0,	angleWages.x,	v1,		faceNormal});
		vertices.add({ i + 1,	0,	angleWages.y,	v2,		faceNormal});
		vertices.add({ i + 2,	0,	angleWages.z,	v3,		faceNormal});
	}

	vertices.validate();

	return vertices;
}

void CSmoothNormalGenerator::processConnectedVertices(asset::ICPUPolygonGeometry* polygon, VertexHashMap& vertexHashMap, float epsilon, CPolygonGeometryManipulator::VxCmpFunction vxcmp)
{
	auto* normalPtr = reinterpret_cast<std::byte*>(polygon->getNormalPtr());
	auto normalStride = polygon->getNormalView().composed.stride;
	for (uint32_t cell = 0; cell < vertexHashMap.getBucketCount() - 1; cell++)
	{
		VertexHashMap::BucketBounds processedBucket = vertexHashMap.getBucketBoundsById(cell);

		for (core::vector<CPolygonGeometryManipulator::SSNGVertexData>::iterator processedVertex = processedBucket.begin; processedVertex != processedBucket.end; processedVertex++)
		{
			std::array<uint32_t, 8> neighboringCells = vertexHashMap.getNeighboringCellHashes(*processedVertex);
			hlsl::float32_t3 normal = processedVertex->parentTriangleFaceNormal * processedVertex->wage;

			//iterate among all neighboring cells
			for (int i = 0; i < 8; i++)
			{
				VertexHashMap::BucketBounds bounds = vertexHashMap.getBucketBoundsByHash(neighboringCells[i]);
				for (; bounds.begin != bounds.end; bounds.begin++)
				{
					if (processedVertex != bounds.begin)
						if (compareVertexPosition(processedVertex->position, bounds.begin->position, epsilon) &&
							vxcmp(*processedVertex, *bounds.begin, polygon))
						{
							//TODO: better mean calculation algorithm
							normal += bounds.begin->parentTriangleFaceNormal * bounds.begin->wage;
						}
				}
			}
			normal = normalize(normal);
			memcpy(normalPtr + (normalStride * processedVertex->index), &normal, sizeof(normal));
		}
	}
}

std::array<uint32_t, 8> CSmoothNormalGenerator::VertexHashMap::getNeighboringCellHashes(const CPolygonGeometryManipulator::SSNGVertexData & vertex)
{
	std::array<uint32_t, 8> neighbourhood;

	hlsl::float32_t3 cellFloatCoord = vertex.position / m_cellSize - hlsl::float32_t3(0.5f);
	hlsl::uint32_t3 neighbor = hlsl::uint32_t3(static_cast<uint32_t>(cellFloatCoord.x), static_cast<uint32_t>(cellFloatCoord.y), static_cast<uint32_t>(cellFloatCoord.z));

	//left bottom near
	neighbourhood[0] = hash(neighbor);

	//right bottom near
	neighbor = neighbor + hlsl::uint32_t3(1, 0, 0);
	neighbourhood[1] = hash(neighbor);

	//right bottom far
	neighbor = neighbor + hlsl::uint32_t3(0, 0, 1);
	neighbourhood[2] = hash(neighbor);

	//left bottom far
	neighbor = neighbor - hlsl::uint32_t3(1, 0, 0);
	neighbourhood[3] = hash(neighbor);

	//left top far
	neighbor = neighbor + hlsl::uint32_t3(0, 1, 0);
	neighbourhood[4] = hash(neighbor);

	//right top far
	neighbor = neighbor + hlsl::uint32_t3(1, 0, 0);
	neighbourhood[5] = hash(neighbor);

	//righ top near
	neighbor = neighbor - hlsl::uint32_t3(0, 0, 1);
	neighbourhood[6] = hash(neighbor);

	//left top near
	neighbor = neighbor - hlsl::uint32_t3(1, 0, 0);
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