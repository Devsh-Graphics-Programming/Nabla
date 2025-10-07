#include "nbl/asset/utils/CVertexHashMap.h"

namespace nbl::asset {

CVertexHashMap::CVertexHashMap(size_t _vertexCount, uint32_t _hashTableMaxSize, float _cellSize) :
    m_sorter(createSorter(_vertexCount)),
    m_hashTableMaxSize(_hashTableMaxSize),
    m_cellSize(_cellSize)
{
  assert((core::isPoT(m_hashTableMaxSize)));

  m_vertices.reserve(_vertexCount);
}

uint32_t CVertexHashMap::hash(const VertexData& vertex) const
{
	const hlsl::float32_t3 position = vertex.position / m_cellSize;

	return	((static_cast<uint32_t>(position.x) * primeNumber1) ^
		(static_cast<uint32_t>(position.y) * primeNumber2) ^
		(static_cast<uint32_t>(position.z) * primeNumber3))& (m_hashTableMaxSize - 1);
}

uint32_t CVertexHashMap::hash(const hlsl::uint32_t3& position) const
{
	return	((position.x * primeNumber1) ^
		(position.y * primeNumber2) ^
		(position.z * primeNumber3))& (m_hashTableMaxSize - 1);
}

void CVertexHashMap::add(VertexData&& vertex)
{
	vertex.hash = hash(vertex);
	m_vertices.push_back(vertex);
}

CVertexHashMap::BucketBounds CVertexHashMap::getBucketBoundsByHash(uint32_t hash)
{
	if (hash == invalidHash)
		return { m_vertices.end(), m_vertices.end() };

	const auto skipListBound = std::visit([&](auto& sorter)
	{
	  auto hashBound = sorter.getHashBound(hash);
		return std::pair<collection_t::iterator, collection_t::iterator>(m_vertices.begin() + hashBound.first, m_vertices.begin() + hashBound.second);
	}, m_sorter);

  auto begin = std::lower_bound(
		skipListBound.first, 
		skipListBound.second, 
		hash,
    [](const VertexData& vertex, uint32_t hash)
    {
        return vertex.hash < hash;
    });

	auto end = std::upper_bound(
		skipListBound.first, 
		skipListBound.second, 
		hash, 
		[](uint32_t hash, const VertexData& vertex)
		{
			return hash < vertex.hash;
		});

	//bucket missing
	if (begin == m_vertices.end())
		return { m_vertices.end(), m_vertices.end() };

	//bucket missing
	if (begin->hash != hash)
		return { m_vertices.end(), m_vertices.end() };

	return { begin, end };
}

void CVertexHashMap::validate()
{
	const auto oldSize = m_vertices.size();
	m_vertices.resize(oldSize*2u);
	// TODO: maybe use counting sort (or big radix) and use the histogram directly for the m_buckets
	auto finalSortedOutput = std::visit( [&](auto& sorter) { return sorter(m_vertices.data(), m_vertices.data() + oldSize, oldSize, KeyAccessor()); },m_sorter );
	// TODO: optimize out the erase
	if (finalSortedOutput != m_vertices.data())
		m_vertices.erase(m_vertices.begin(), m_vertices.begin() + oldSize);
	else
		m_vertices.resize(oldSize);
}

uint8_t CVertexHashMap::getNeighboringCellHashes(uint32_t* outNeighbours, const VertexData& vertex)
{
	hlsl::float32_t3 cellFloatCoord = floor(vertex.position / m_cellSize - hlsl::float32_t3(0.5f));
	hlsl::uint32_t3 neighbor = hlsl::uint32_t3(static_cast<uint32_t>(cellFloatCoord.x), static_cast<uint32_t>(cellFloatCoord.y), static_cast<uint32_t>(cellFloatCoord.z));

	uint8_t neighbourCount = 0;

	//left bottom near
	outNeighbours[neighbourCount] = hash(neighbor);
	neighbourCount++;

	auto addUniqueNeighbour = [&neighbourCount, outNeighbours](uint32_t hashVal)
  {
    if (std::find(outNeighbours, outNeighbours + neighbourCount, hashVal) != outNeighbours + neighbourCount)
    {
			outNeighbours[neighbourCount] = hashVal;
			neighbourCount++;
    }
  };

	//right bottom near
	neighbor = neighbor + hlsl::uint32_t3(1, 0, 0);
	addUniqueNeighbour(hash(neighbor));

	//right bottom far
	neighbor = neighbor + hlsl::uint32_t3(0, 0, 1);
	addUniqueNeighbour(hash(neighbor));

	//left bottom far
	neighbor = neighbor - hlsl::uint32_t3(1, 0, 0);
	addUniqueNeighbour(hash(neighbor));

	//left top far
	neighbor = neighbor + hlsl::uint32_t3(0, 1, 0);
	addUniqueNeighbour(hash(neighbor));

	//right top far
	neighbor = neighbor + hlsl::uint32_t3(1, 0, 0);
	addUniqueNeighbour(hash(neighbor));

	//righ top near
	neighbor = neighbor - hlsl::uint32_t3(0, 0, 1);
	addUniqueNeighbour(hash(neighbor));

	//left top near
	neighbor = neighbor - hlsl::uint32_t3(1, 0, 0);
	addUniqueNeighbour(hash(neighbor));

	return neighbourCount;
}

}