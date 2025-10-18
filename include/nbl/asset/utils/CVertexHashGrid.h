#ifndef _NBL_ASSET_C_VERTEX_HASH_MAP_H_INCLUDED_
#define _NBL_ASSET_C_VERTEX_HASH_MAP_H_INCLUDED_

#include "nbl/core/declarations.h"

namespace nbl::asset
{

template <typename T>
concept HashGridVertexData = requires(T obj, T const cobj, uint32_t hash) {
		{ cobj.getHash() } -> std::same_as<uint32_t>;
		{ obj.setHash(hash) } -> std::same_as<void>;
		{ cobj.getPosition() } -> std::same_as<hlsl::float32_t3>;
};

template <HashGridVertexData VertexData>
class CVertexHashGrid
{
public:

	using vertex_data_t = VertexData;
	using collection_t = core::vector<VertexData>;
	struct BucketBounds
	{
		collection_t::const_iterator begin;
		collection_t::const_iterator end;
	};

	CVertexHashGrid(size_t _vertexCount, uint32_t _hashTableMaxSize, float _cellSize) :
		m_sorter(createSorter(_vertexCount)),
		m_hashTableMaxSize(_hashTableMaxSize),
		m_cellSize(_cellSize)
	{
		assert((core::isPoT(m_hashTableMaxSize)));

		m_vertices.reserve(_vertexCount);
	}

	//inserts vertex into hash table
	void add(VertexData&& vertex)
	{
		vertex.setHash(hash(vertex));
		m_vertices.push_back(vertex);
	}

	void validate()
	{
		const auto oldSize = m_vertices.size();
		m_vertices.resize(oldSize*2u);
		auto finalSortedOutput = std::visit( [&](auto& sorter) { return sorter(m_vertices.data(), m_vertices.data() + oldSize, oldSize, KeyAccessor()); },m_sorter );

		if (finalSortedOutput != m_vertices.data())
			m_vertices.erase(m_vertices.begin(), m_vertices.begin() + oldSize);
		else
			m_vertices.resize(oldSize);
	}

	const collection_t& vertices() const { return m_vertices; }

	collection_t& vertices(){ return m_vertices; }

	inline uint32_t getVertexCount() const { return m_vertices.size(); }

	template <typename Fn>
	void iterateBroadphaseCandidates(const VertexData& vertex, Fn fn) const
	{
		std::array<uint32_t, 8> neighboringCells;
		const auto cellCount = getNeighboringCellHashes(neighboringCells.data(), vertex);

		//iterate among all neighboring cells
		for (uint8_t i = 0; i < cellCount; i++)
		{
			const auto& neighborCell = neighboringCells[i];
			BucketBounds bounds = getBucketBoundsByHash(neighborCell);
			for (; bounds.begin != bounds.end; bounds.begin++)
			{
				const vertex_data_t& neighborVertex = *bounds.begin;
				if (&vertex != &neighborVertex)
					if (!fn(neighborVertex)) break;
			}
		}
		
	};

private:
	struct KeyAccessor
	{
		_NBL_STATIC_INLINE_CONSTEXPR size_t key_bit_count = 32ull;

		template<auto bit_offset, auto radix_mask>
		inline decltype(radix_mask) operator()(const VertexData& item) const
		{
			return static_cast<decltype(radix_mask)>(item.getHash() >> static_cast<uint32_t>(bit_offset)) & radix_mask;
		}
	};

	static constexpr uint32_t primeNumber1 = 73856093;
	static constexpr uint32_t primeNumber2 = 19349663;
	static constexpr uint32_t primeNumber3 = 83492791;

	static constexpr uint32_t invalidHash = 0xFFFFFFFF;

	using sorter_t = std::variant<
		core::LSBSorter<KeyAccessor::key_bit_count, uint16_t>,
		core::LSBSorter<KeyAccessor::key_bit_count, uint32_t>,
		core::LSBSorter<KeyAccessor::key_bit_count, size_t>>;
	sorter_t m_sorter;

	static sorter_t createSorter(size_t vertexCount)
	{
		if (vertexCount < (0x1ull << 16ull))
			return core::LSBSorter<KeyAccessor::key_bit_count, uint16_t>();
		if (vertexCount < (0x1ull << 32ull))
			return core::LSBSorter<KeyAccessor::key_bit_count, uint32_t>();
		return core::LSBSorter<KeyAccessor::key_bit_count, size_t>();
	}

	collection_t m_vertices;
	const uint32_t m_hashTableMaxSize;
	const float m_cellSize;

	uint32_t hash(const VertexData& vertex) const
	{
		const hlsl::float32_t3 position = floor(vertex.getPosition() / m_cellSize);

		return	((static_cast<uint32_t>(position.x) * primeNumber1) ^
			(static_cast<uint32_t>(position.y) * primeNumber2) ^
			(static_cast<uint32_t>(position.z) * primeNumber3))& (m_hashTableMaxSize - 1);
	}

	uint32_t hash(const hlsl::uint32_t3& position) const
	{
		return	((position.x * primeNumber1) ^
			(position.y * primeNumber2) ^
			(position.z * primeNumber3))& (m_hashTableMaxSize - 1);
	}

	uint8_t getNeighboringCellHashes(uint32_t* outNeighbors, const VertexData& vertex) const
	{
		hlsl::float32_t3 cellfloatcoord = floor(vertex.getPosition() / m_cellSize - hlsl::float32_t3(0.5));
		hlsl::uint32_t3 baseCoord = hlsl::uint32_t3(static_cast<uint32_t>(cellfloatcoord.x), static_cast<uint32_t>(cellfloatcoord.y), static_cast<uint32_t>(cellfloatcoord.z));

		uint8_t neighborCount = 0;

		outNeighbors[neighborCount] = hash(baseCoord);
		neighborCount++;

		auto addUniqueNeighbor = [&neighborCount, outNeighbors](uint32_t hashval)
		{
			if (std::find(outNeighbors, outNeighbors + neighborCount, hashval) == outNeighbors + neighborCount)
			{
				outNeighbors[neighborCount] = hashval;
				neighborCount++;
			}
		};

		addUniqueNeighbor(hash(baseCoord + hlsl::uint32_t3(0, 0, 1)));
		addUniqueNeighbor(hash(baseCoord + hlsl::uint32_t3(0, 1, 0)));
		addUniqueNeighbor(hash(baseCoord + hlsl::uint32_t3(1, 0, 0)));
		addUniqueNeighbor(hash(baseCoord + hlsl::uint32_t3(1, 1, 0)));
		addUniqueNeighbor(hash(baseCoord + hlsl::uint32_t3(1, 0, 1)));
		addUniqueNeighbor(hash(baseCoord + hlsl::uint32_t3(0, 1, 1)));
		addUniqueNeighbor(hash(baseCoord + hlsl::uint32_t3(1, 1, 1)));

		return neighborCount;
	}

	BucketBounds getBucketBoundsByHash(uint32_t hash) const
	{
		if (hash == invalidHash)
			return { m_vertices.end(), m_vertices.end() };

		const auto skipListBound = std::visit([&](auto& sorter)
		{
			auto hashBound = sorter.getHashBound(hash);
			return std::pair<collection_t::const_iterator, collection_t::const_iterator>(m_vertices.begin() + hashBound.first, m_vertices.begin() + hashBound.second);
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

		const auto beginIx = begin - m_vertices.begin();
		const auto endIx = end - m_vertices.begin();
		//bucket missing
		if (begin == end)
			return { m_vertices.end(), m_vertices.end() };

		//bucket missing
		if (begin->hash != hash)
			return { m_vertices.end(), m_vertices.end() };

		return { begin, end };
	}
};

}
#endif