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

template <typename Fn, typename T>
concept HashGridIteratorFn = HashGridVertexData<T> && requires(Fn && fn, T const cobj)
{
	// return whether hash grid should continue the iteration
	{ std::invoke(std::forward<Fn>(fn), cobj) } -> std::same_as<bool>;
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

	inline CVertexHashGrid(float cellSize, uint32_t hashTableMaxSizeLog2, size_t vertexCountReserve = 8192) :
		m_cellSize(cellSize),
		m_hashTableMaxSize(1llu << hashTableMaxSizeLog2),
		m_sorter(createSorter(vertexCountReserve))
	{
		m_vertices.reserve(vertexCountReserve);
	}

	//inserts vertex into hash table
	inline void add(VertexData&& vertex)
	{
		vertex.setHash(hash(vertex));
		m_vertices.push_back(std::move(vertex));
	}

	inline void bake()
	{
		auto scratchBuffer = collection_t(m_vertices.size());

		auto finalSortedOutput = std::visit( [&](auto& sorter)
		{
			return sorter(m_vertices.data(), scratchBuffer.data(), m_vertices.size(), KeyAccessor());
		}, m_sorter );

		if (finalSortedOutput != m_vertices.data())
			m_vertices = std::move(scratchBuffer);
	}

	inline const collection_t& vertices() const { return m_vertices; }

	inline uint32_t getVertexCount() const { return m_vertices.size(); }

	template <HashGridIteratorFn<VertexData> Fn>
	inline void forEachBroadphaseNeighborCandidates(const VertexData& vertex, Fn&& fn) const
	{
		std::array<uint32_t, 8> neighboringCells;
		const auto cellCount = getNeighboringCellHashes(neighboringCells.data(), vertex.getPosition());

		//iterate among all neighboring cells
		for (uint8_t i = 0; i < cellCount; i++)
		{
			const auto& neighborCell = neighboringCells[i];
			BucketBounds bounds = getBucketBoundsByHash(neighborCell);
			for (; bounds.begin != bounds.end; bounds.begin++)
			{
				const vertex_data_t& neighborVertex = *bounds.begin;
				if (&vertex != &neighborVertex)
					if (!std::invoke(std::forward<Fn>(fn), neighborVertex)) break;
			}
		}
	};

	template <HashGridIteratorFn<VertexData> Fn>
	inline void forEachBroadphaseNeighborCandidates(const hlsl::float32_t3& position, Fn&& fn) const
	{
		std::array<uint32_t, 8> neighboringCells;
		const auto cellCount = getNeighboringCellHashes(neighboringCells.data(), position);

		//iterate among all neighboring cells
		for (uint8_t i = 0; i < cellCount; i++)
		{
			const auto& neighborCell = neighboringCells[i];
			BucketBounds bounds = getBucketBoundsByHash(neighborCell);
			for (; bounds.begin != bounds.end; bounds.begin++)
			{
				const vertex_data_t& neighborVertex = *bounds.begin;
				if (!std::invoke(std::forward<Fn>(fn), neighborVertex)) break;
			}
		}
	};

private:
	struct KeyAccessor
	{
		constexpr static inline size_t key_bit_count = 32ull;

		template<auto bit_offset, auto radix_mask>
		inline decltype(radix_mask) operator()(const VertexData& item) const
		{
			return static_cast<decltype(radix_mask)>(item.getHash() >> static_cast<uint32_t>(bit_offset)) & radix_mask;
		}
	};

	static constexpr uint32_t primeNumber1 = 73856093;
	static constexpr uint32_t primeNumber2 = 19349663;
	static constexpr uint32_t primeNumber3 = 83492791;

	using sorter_t = std::variant<
		core::RadixLsbSorter<KeyAccessor::key_bit_count, uint16_t>,
		core::RadixLsbSorter<KeyAccessor::key_bit_count, uint32_t>,
		core::RadixLsbSorter<KeyAccessor::key_bit_count, size_t>>;
	sorter_t m_sorter;

	inline static sorter_t createSorter(size_t vertexCount)
	{
		if (vertexCount < (0x1ull << 16ull))
			return core::RadixLsbSorter<KeyAccessor::key_bit_count, uint16_t>();
		if (vertexCount < (0x1ull << 32ull))
			return core::RadixLsbSorter<KeyAccessor::key_bit_count, uint32_t>();
		return core::RadixLsbSorter<KeyAccessor::key_bit_count, size_t>();
	}

	collection_t m_vertices;
	const uint32_t m_hashTableMaxSize;
	const float m_cellSize;

	inline uint32_t hash(const VertexData& vertex) const
	{
		const hlsl::float32_t3 position = floor(vertex.getPosition() / m_cellSize);
		const auto position_uint32 = hlsl::uint32_t3(position.x, position.y, position.z);
		return hash(position_uint32);
	}

	inline uint32_t hash(const hlsl::uint32_t3& position) const
	{
		return	((position.x * primeNumber1) ^
			(position.y * primeNumber2) ^
			(position.z * primeNumber3))& (m_hashTableMaxSize - 1);
	}

	inline uint8_t getNeighboringCellHashes(uint32_t* outNeighbors, hlsl::float32_t3 position) const
	{
		// both 0.x and -0.x would be converted to 0 if we directly casting the position to unsigned integer. Causing the 0 to be crowded then the rest of the cells. So we use floor here to spread the vertex more uniformly.
		hlsl::float32_t3 cellfloatcoord = floor(position / m_cellSize - hlsl::float32_t3(0.5));
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

	inline BucketBounds getBucketBoundsByHash(uint32_t hash) const
	{
		const auto skipListBound = std::visit([&](auto& sorter)
		{
			auto hashBound = sorter.getMostSignificantRadixBound(hash);
			return std::pair<collection_t::const_iterator, collection_t::const_iterator>(m_vertices.begin() + hashBound.first, m_vertices.begin() + hashBound.second);
		}, m_sorter);

		auto begin = std::lower_bound(
			skipListBound.first, 
			skipListBound.second, 
			hash,
			[](const VertexData& vertex, uint32_t hash)
			{
				return vertex.getHash() < hash;
			});

		auto end = std::upper_bound(
			skipListBound.first, 
			skipListBound.second, 
			hash, 
			[](uint32_t hash, const VertexData& vertex)
			{
				return hash < vertex.getHash();
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