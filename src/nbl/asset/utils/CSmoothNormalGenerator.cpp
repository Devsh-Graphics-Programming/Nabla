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

static bool isAttributeEqual(const ICPUPolygonGeometry::SDataView& view, uint32_t index1, uint32_t index2, float epsilon)
{
	if (!view) return true;
	const auto channelCount = getFormatChannelCount(view.composed.format);
	switch (view.composed.rangeFormat)
	{
    case IGeometryBase::EAABBFormat::U64:
    case IGeometryBase::EAABBFormat::U32:
    {
			hlsl::uint64_t4 val1, val2;
			view.decodeElement<hlsl::uint64_t4>(index1, val1);
			view.decodeElement<hlsl::uint64_t4>(index2, val2);
			for (auto channel_i = 0u; channel_i < channelCount; channel_i++)
				if (val1[channel_i] != val2[channel_i]) return false;
      break;
    }
    case IGeometryBase::EAABBFormat::S64:
    case IGeometryBase::EAABBFormat::S32:
    {
			hlsl::int64_t4 val1, val2;
			view.decodeElement<hlsl::int64_t4>(index1, val1);
			view.decodeElement<hlsl::int64_t4>(index2, val2);
			for (auto channel_i = 0u; channel_i < channelCount; channel_i++)
				if (val1[channel_i] != val2[channel_i]) return false;
      break;
    }
    default:
    {
			hlsl::float64_t4 val1, val2;
			view.decodeElement<hlsl::float64_t4>(index1, val1);
			view.decodeElement<hlsl::float64_t4>(index2, val2);
			for (auto channel_i = 0u; channel_i < channelCount; channel_i++)
			{
				const auto diff = abs(val1[channel_i] - val2[channel_i]);
				if (diff > epsilon) return false;
			}
			break;
    }
	}
	return true;
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

core::smart_refctd_ptr<ICPUPolygonGeometry> CSmoothNormalGenerator::calculateNormals(const asset::ICPUPolygonGeometry* polygon, bool enableWelding, float epsilon, CPolygonGeometryManipulator::VxCmpFunction vxcmp)
{
	VertexHashMap vertexArray = setupData(polygon, epsilon);
	const auto smoothPolygon = processConnectedVertices(polygon, vertexArray, epsilon,vxcmp);

	if (enableWelding)
	{
		return weldVertices(smoothPolygon.get(), vertexArray, epsilon);
	}
	return smoothPolygon;
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

	VertexHashMap vertices(idxCount, std::min(16u * 1024u, core::roundUpToPoT<unsigned int>(idxCount * 1.0f / 32.0f)), epsilon == 0.0f ? 0.00001f : epsilon * 2.f);

	for (uint32_t i = 0; i < idxCount; i += 3)
	{
		//calculate face normal of parent triangle
		hlsl::float32_t3 v1, v2, v3;
		polygon->getPositionView().decodeElement<hlsl::float32_t3>(i, v1);
		polygon->getPositionView().decodeElement<hlsl::float32_t3>(i + 1, v2);
		polygon->getPositionView().decodeElement<hlsl::float32_t3>(i + 2, v3);

		const auto faceNormal = normalize(cross(v2 - v1, v3 - v1));

		//set data for m_vertices
		const auto angleWages = getAngleWeight(v1, v2, v3);

		vertices.add({ i,	0,	angleWages.x,	v1,		faceNormal});
		vertices.add({ i + 1,	0,	angleWages.y,	v2,		faceNormal});
		vertices.add({ i + 2,	0,	angleWages.z,	v3,		faceNormal});
	}

	vertices.validate();

	return vertices;
}

core::smart_refctd_ptr<ICPUPolygonGeometry> CSmoothNormalGenerator::processConnectedVertices(const asset::ICPUPolygonGeometry* polygon, VertexHashMap& vertexHashMap, float epsilon, CPolygonGeometryManipulator::VxCmpFunction vxcmp)
{
  auto outPolygon = core::move_and_static_cast<ICPUPolygonGeometry>(polygon->clone(0u));
  static constexpr auto NormalFormat = EF_R32G32B32_SFLOAT;
  const auto normalFormatBytesize = asset::getTexelOrBlockBytesize(NormalFormat);
  auto normalBuf = ICPUBuffer::create({ normalFormatBytesize * outPolygon->getPositionView().getElementCount()});
  auto normalView = polygon->getNormalView();

  hlsl::shapes::AABB<4,hlsl::float32_t> aabb;
  aabb.maxVx = hlsl::float32_t4(1, 1, 1, 0.f);
  aabb.minVx = -aabb.maxVx;
  outPolygon->setNormalView({
    .composed = {
      .encodedDataRange = {.f32 = aabb},
      .stride = sizeof(hlsl::float32_t3),
      .format = NormalFormat,
      .rangeFormat = IGeometryBase::EAABBFormat::F32
    },
    .src = { .offset = 0, .size = normalBuf->getSize(), .buffer = std::move(normalBuf) },
   });

	auto* normalPtr = reinterpret_cast<std::byte*>(outPolygon->getNormalPtr());
	auto normalStride = outPolygon->getNormalView().composed.stride;

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

	CPolygonGeometryManipulator::recomputeContentHashes(outPolygon.get());

	return outPolygon;
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

core::smart_refctd_ptr<ICPUPolygonGeometry> CSmoothNormalGenerator::weldVertices(const ICPUPolygonGeometry* polygon, VertexHashMap& vertices, float epsilon)
{
	struct Group
	{
		uint32_t vertex_reference_index; // index to referenced vertex in the original polygon
	};
	core::vector<Group> groups; 
	groups.reserve(vertices.getVertexCount());

	core::vector<std::optional<uint32_t>> groupIndexes(vertices.getVertexCount());

	auto canJoinVertices = [&](uint32_t index1, uint32_t index2)-> bool
  {
    if (!isAttributeEqual(polygon->getPositionView(), index1, index2, epsilon))
			return false;
    if (!isAttributeEqual(polygon->getNormalView(), index1, index2, epsilon))
			return false;
		for (const auto& jointWeightView : polygon->getJointWeightViews())
		{
			if (!isAttributeEqual(jointWeightView.indices, index1, index2, epsilon)) return false;
			if (!isAttributeEqual(jointWeightView.weights, index1, index2, epsilon)) return false;
		}
		for (const auto& auxAttributeView : polygon->getAuxAttributeViews())
			if (!isAttributeEqual(auxAttributeView, index1, index2, epsilon)) return false;

		return true;
  };

	for (uint32_t cell = 0; cell < vertices.getBucketCount() - 1; cell++)
	{
		VertexHashMap::BucketBounds processedBucket = vertices.getBucketBoundsById(cell);

		for (core::vector<CPolygonGeometryManipulator::SSNGVertexData>::iterator processedVertex = processedBucket.begin; processedVertex != processedBucket.end; processedVertex++)
		{
			std::array<uint32_t, 8> neighboringCells = vertices.getNeighboringCellHashes(*processedVertex);
			hlsl::float32_t3 normal = processedVertex->parentTriangleFaceNormal * processedVertex->wage;

			auto& groupIndex = groupIndexes[processedVertex->index];

			//iterate among all neighboring cells
			for (int i = 0; i < 8; i++)
			{
				VertexHashMap::BucketBounds bounds = vertices.getBucketBoundsByHash(neighboringCells[i]);
				for (auto neighbourVertex_it = bounds.begin; neighbourVertex_it != bounds.end; neighbourVertex_it++)
				{
					const auto neighbourGroupIndex = groupIndexes[neighbourVertex_it->index];

					hlsl::float32_t3 normal1, normal2;
					polygon->getNormalView().decodeElement(processedVertex->index, normal1);
					polygon->getNormalView().decodeElement(neighbourVertex_it->index, normal2);

					hlsl::float32_t3 position1, position2;
					polygon->getPositionView().decodeElement(processedVertex->index, position1);
					polygon->getPositionView().decodeElement(neighbourVertex_it->index, position2);
					 
					// find the first group that this vertex can join
					if (processedVertex != neighbourVertex_it && neighbourGroupIndex && canJoinVertices(processedVertex->index, neighbourVertex_it->index))
					{
						groupIndex = neighbourGroupIndex;
						break;
					}
				}
			}
			if (!groupIndex)
			{
        // create new group if no group nearby that is compatible with this vertex
        groupIndex = groups.size();
        groups.push_back({ processedVertex->index});
			}
		}
	}

  auto outPolygon = core::move_and_static_cast<ICPUPolygonGeometry>(polygon->clone(0u));
	outPolygon->setIndexing(IPolygonGeometryBase::TriangleList());

	const uint32_t indexSize = (groups.size() < std::numeric_limits<uint16_t>::max()) ? sizeof(uint16_t) : sizeof(uint32_t);
  auto indexBuffer = ICPUBuffer::create({indexSize * groupIndexes.size(), IBuffer::EUF_INDEX_BUFFER_BIT});
  auto indexBufferPtr = reinterpret_cast<std::byte*>(indexBuffer->getPointer());
  auto indexView = ICPUPolygonGeometry::SDataView{
    .composed = {
      .stride = indexSize,
    },
    .src = {
      .offset = 0,
      .size = indexBuffer->getSize(),
      .buffer = std::move(indexBuffer)
    }
  };
	if (indexSize == 2)
	{
		indexView.composed.encodedDataRange.u16.minVx[0] = 0;
		indexView.composed.encodedDataRange.u16.maxVx[0] = groups.size() - 1;
		indexView.composed.format = EF_R16_UINT;
		indexView.composed.rangeFormat = IGeometryBase::EAABBFormat::U16;
	} else if (indexSize == 4)
	{
		indexView.composed.encodedDataRange.u32.minVx[0] = 0;
		indexView.composed.encodedDataRange.u32.maxVx[0] = groups.size() - 1;
		indexView.composed.format = EF_R32_UINT;
		indexView.composed.rangeFormat = IGeometryBase::EAABBFormat::U32;
	}

	for (auto index_i = 0u; index_i < groupIndexes.size(); index_i++)
	{
		if (indexSize == 2)
		{
			uint16_t index = *groupIndexes[index_i];
			memcpy(indexBufferPtr + indexSize * index_i, &index, sizeof(index));
		} else if (indexSize == 4)
		{
			uint32_t index = *groupIndexes[index_i];
			memcpy(indexBufferPtr + indexSize * index_i, &index, sizeof(index));
		}
	}
	outPolygon->setIndexView(std::move(indexView));


	using position_t = hlsl::float32_t3;
	constexpr auto PositionAttrSize = sizeof(position_t);
	auto positionBuffer = ICPUBuffer::create({ PositionAttrSize * groups.size(), IBuffer::EUF_NONE });
	auto outPositions = reinterpret_cast<position_t*>(positionBuffer->getPointer());
	const auto inPositions = reinterpret_cast<const position_t*>(polygon->getPositionView().getPointer());
	outPolygon->setPositionView({
		.composed = polygon->getPositionView().composed,
	  .src = {.offset = 0, .size = positionBuffer->getSize(), .buffer = std::move(positionBuffer)}
  });

	using normal_t = hlsl::float32_t3;
	constexpr auto NormalAttrSize = sizeof(normal_t);
	auto normalBuffer = ICPUBuffer::create({ NormalAttrSize * groups.size(), IBuffer::EUF_NONE });
	auto outNormals = reinterpret_cast<normal_t*>(normalBuffer->getPointer());
	const auto inNormals = reinterpret_cast<const normal_t*>(polygon->getNormalView().getPointer());
	outPolygon->setNormalView({
		.composed = polygon->getNormalView().composed,
	  .src = {.offset = 0, .size = normalBuffer->getSize(), .buffer = std::move(normalBuffer)}
  });

	auto createOutView = [&](const ICPUPolygonGeometry::SDataView& view)
  {
    auto buffer = ICPUBuffer::create({ view.composed.stride * groups.size(), view.src.buffer->getUsageFlags() });
		return ICPUPolygonGeometry::SDataView{
			.composed = view.composed,
			.src = {.offset = 0, .size = buffer->getSize(), .buffer = std::move(buffer)}
		};
  };

	const auto& inJointWeightViews = polygon->getJointWeightViews();
	auto* outJointWeightViews = outPolygon->getJointWeightViews();
	outJointWeightViews->resize(inJointWeightViews.size());
	for (auto jointWeightView_i = 0u; jointWeightView_i < inJointWeightViews.size(); jointWeightView_i++)
	{
		const auto& inJointWeightView = inJointWeightViews[jointWeightView_i];
		outJointWeightViews->operator[](jointWeightView_i).indices = createOutView(inJointWeightView.indices);
		outJointWeightViews->operator[](jointWeightView_i).weights = createOutView(inJointWeightView.weights);
	}

	const auto& inAuxAttributeViews = polygon->getAuxAttributeViews();
	auto* outAuxAttributeViews = outPolygon->getAuxAttributeViews();
	outAuxAttributeViews->resize(inAuxAttributeViews.size());
	for (auto auxAttributeView_i = 0u; auxAttributeView_i < inAuxAttributeViews.size(); auxAttributeView_i++)
	{
		const auto& inAuxAttributeView = inAuxAttributeViews[auxAttributeView_i];
		outAuxAttributeViews->operator[](auxAttributeView_i) = createOutView(inAuxAttributeView);
	}

	for (auto group_i = 0u; group_i < groups.size(); group_i++)
	{
		const auto srcIndex = groups[group_i].vertex_reference_index;
		outPositions[group_i] = inPositions[srcIndex];
		outNormals[group_i] = inPositions[srcIndex];

    for (uint64_t jointView_i = 0u; jointView_i < polygon->getJointWeightViews().size(); jointView_i++)
    {
      auto& inView = polygon->getJointWeightViews()[jointView_i];
      auto& outView = outPolygon->getJointWeightViews()->operator[](jointView_i);

      const std::byte* const inJointIndices = reinterpret_cast<const std::byte*>(inView.indices.getPointer());
      const auto jointIndexSize = inView.indices.composed.stride;
      std::byte* const outJointIndices = reinterpret_cast<std::byte*>(outView.indices.getPointer());
      memcpy(outJointIndices + group_i * jointIndexSize, inJointIndices + srcIndex * jointIndexSize, jointIndexSize);

      const std::byte* const inWeights = reinterpret_cast<const std::byte*>(inView.weights.getPointer());
      const auto jointWeightSize = inView.weights.composed.stride;
      std::byte* const outWeights = reinterpret_cast<std::byte*>(outView.weights.getPointer());
      memcpy(outWeights + group_i * jointWeightSize, inWeights + srcIndex * jointWeightSize, jointWeightSize);
    }

    for (auto auxView_i = 0u; auxView_i < polygon->getAuxAttributeViews().size(); auxView_i++)
    {
      auto& inView = polygon->getAuxAttributeViews()[auxView_i];
      auto& outView = outPolygon->getAuxAttributeViews()->operator[](auxView_i);
      const auto attrSize = inView.composed.stride;
      const std::byte* const inAuxs = reinterpret_cast<const std::byte*>(inView.getPointer());
      std::byte* const outAuxs = reinterpret_cast<std::byte*>(outView.getPointer());
      memcpy(outAuxs + group_i * attrSize, inAuxs + srcIndex * attrSize, attrSize);
    }
	}

  CPolygonGeometryManipulator::recomputeContentHashes(outPolygon.get());
  return outPolygon;

}
}
}