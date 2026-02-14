#include "nbl/core/hash/blake.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <future>
#include <thread>

extern "C"
{
#include "blake3_impl.h"
}

/*
	BLAKE3 is tree-based and explicitly designed for parallel processing. The tree mode
	(chunks and parent-node reduction) is part of the specification, so a parallel
	implementation can be done without changing hash semantics.

	Why this local implementation exists:
	- Nabla needs a multithreaded hash path integrated with its own runtime policy and
	  standard C++ threading.
	- Upstream C API exposes a single-threaded update path and an optional oneTBB path
	  (`blake3_hasher_update_tbb`) which requires building with `BLAKE3_USE_TBB`.
	- Here we keep the same algorithmic rules and final digest, while using only C++20
	  standard facilities (`std::async`, `std::thread`) and no oneTBB dependency.
	- The local helpers below are adapted from upstream tree-processing internals used
	  in `c/blake3.c` and the oneTBB integration path.

	Primary references:
	- BLAKE3 spec repository (paper): https://github.com/BLAKE3-team/BLAKE3-specs
	- C2SP BLAKE3 specification: https://c2sp.org/BLAKE3
	- Upstream BLAKE3 C API notes (`update_tbb`): https://github.com/BLAKE3-team/BLAKE3/blob/master/c/README.md
*/

namespace nbl::core
{

namespace
{

struct output_t
{
	uint32_t input_cv[8];
	uint64_t counter;
	uint8_t block[BLAKE3_BLOCK_LEN];
	uint8_t block_len;
	uint8_t flags;
};

INLINE void chunk_state_init_local(blake3_chunk_state* self, const uint32_t key[8], uint8_t flags)
{
	std::memcpy(self->cv, key, BLAKE3_KEY_LEN);
	self->chunk_counter = 0;
	std::memset(self->buf, 0, BLAKE3_BLOCK_LEN);
	self->buf_len = 0;
	self->blocks_compressed = 0;
	self->flags = flags;
}

INLINE void chunk_state_reset_local(blake3_chunk_state* self, const uint32_t key[8], uint64_t chunk_counter)
{
	std::memcpy(self->cv, key, BLAKE3_KEY_LEN);
	self->chunk_counter = chunk_counter;
	self->blocks_compressed = 0;
	std::memset(self->buf, 0, BLAKE3_BLOCK_LEN);
	self->buf_len = 0;
}

INLINE size_t chunk_state_len_local(const blake3_chunk_state* self)
{
	return (BLAKE3_BLOCK_LEN * static_cast<size_t>(self->blocks_compressed)) + static_cast<size_t>(self->buf_len);
}

INLINE size_t chunk_state_fill_buf_local(blake3_chunk_state* self, const uint8_t* input, size_t input_len)
{
	size_t take = BLAKE3_BLOCK_LEN - static_cast<size_t>(self->buf_len);
	if (take > input_len)
		take = input_len;
	auto* const dest = self->buf + static_cast<size_t>(self->buf_len);
	std::memcpy(dest, input, take);
	self->buf_len += static_cast<uint8_t>(take);
	return take;
}

INLINE uint8_t chunk_state_maybe_start_flag_local(const blake3_chunk_state* self)
{
	return self->blocks_compressed == 0 ? CHUNK_START : 0;
}

INLINE output_t make_output_local(const uint32_t input_cv[8], const uint8_t block[BLAKE3_BLOCK_LEN], uint8_t block_len, uint64_t counter, uint8_t flags)
{
	output_t ret = {};
	std::memcpy(ret.input_cv, input_cv, 32);
	std::memcpy(ret.block, block, BLAKE3_BLOCK_LEN);
	ret.block_len = block_len;
	ret.counter = counter;
	ret.flags = flags;
	return ret;
}

INLINE void output_chaining_value_local(const output_t* self, uint8_t cv[32])
{
	uint32_t cv_words[8];
	std::memcpy(cv_words, self->input_cv, 32);
	blake3_compress_in_place(cv_words, self->block, self->block_len, self->counter, self->flags);
	store_cv_words(cv, cv_words);
}

INLINE void chunk_state_update_local(blake3_chunk_state* self, const uint8_t* input, size_t input_len)
{
	if (self->buf_len > 0)
	{
		size_t take = chunk_state_fill_buf_local(self, input, input_len);
		input += take;
		input_len -= take;
		if (input_len > 0)
		{
			blake3_compress_in_place(
				self->cv,
				self->buf,
				BLAKE3_BLOCK_LEN,
				self->chunk_counter,
				self->flags | chunk_state_maybe_start_flag_local(self));
			self->blocks_compressed += 1;
			self->buf_len = 0;
			std::memset(self->buf, 0, BLAKE3_BLOCK_LEN);
		}
	}

	while (input_len > BLAKE3_BLOCK_LEN)
	{
		blake3_compress_in_place(
			self->cv,
			input,
			BLAKE3_BLOCK_LEN,
			self->chunk_counter,
			self->flags | chunk_state_maybe_start_flag_local(self));
		self->blocks_compressed += 1;
		input += BLAKE3_BLOCK_LEN;
		input_len -= BLAKE3_BLOCK_LEN;
	}

	(void)chunk_state_fill_buf_local(self, input, input_len);
}

INLINE output_t chunk_state_output_local(const blake3_chunk_state* self)
{
	const uint8_t block_flags = self->flags | chunk_state_maybe_start_flag_local(self) | CHUNK_END;
	return make_output_local(self->cv, self->buf, self->buf_len, self->chunk_counter, block_flags);
}

INLINE output_t parent_output_local(const uint8_t block[BLAKE3_BLOCK_LEN], const uint32_t key[8], uint8_t flags)
{
	return make_output_local(key, block, BLAKE3_BLOCK_LEN, 0, flags | PARENT);
}

INLINE size_t left_len_local(size_t content_len)
{
	const size_t full_chunks = (content_len - 1) / BLAKE3_CHUNK_LEN;
	return round_down_to_power_of_2(full_chunks) * BLAKE3_CHUNK_LEN;
}

INLINE size_t compress_chunks_parallel_local(
	const uint8_t* input,
	size_t input_len,
	const uint32_t key[8],
	uint64_t chunk_counter,
	uint8_t flags,
	uint8_t* out)
{
	const uint8_t* chunks_array[MAX_SIMD_DEGREE];
	size_t input_position = 0;
	size_t chunks_array_len = 0;
	while (input_len - input_position >= BLAKE3_CHUNK_LEN)
	{
		chunks_array[chunks_array_len] = &input[input_position];
		input_position += BLAKE3_CHUNK_LEN;
		chunks_array_len += 1;
	}

	blake3_hash_many(
		chunks_array,
		chunks_array_len,
		BLAKE3_CHUNK_LEN / BLAKE3_BLOCK_LEN,
		key,
		chunk_counter,
		true,
		flags,
		CHUNK_START,
		CHUNK_END,
		out);

	if (input_len > input_position)
	{
		const uint64_t counter = chunk_counter + static_cast<uint64_t>(chunks_array_len);
		blake3_chunk_state chunk_state = {};
		chunk_state_init_local(&chunk_state, key, flags);
		chunk_state.chunk_counter = counter;
		chunk_state_update_local(&chunk_state, &input[input_position], input_len - input_position);
		const auto output = chunk_state_output_local(&chunk_state);
		output_chaining_value_local(&output, &out[chunks_array_len * BLAKE3_OUT_LEN]);
		return chunks_array_len + 1;
	}

	return chunks_array_len;
}

INLINE size_t compress_parents_parallel_local(
	const uint8_t* child_chaining_values,
	size_t num_chaining_values,
	const uint32_t key[8],
	uint8_t flags,
	uint8_t* out)
{
	const uint8_t* parents_array[MAX_SIMD_DEGREE_OR_2];
	size_t parents_array_len = 0;
	while (num_chaining_values - (2 * parents_array_len) >= 2)
	{
		parents_array[parents_array_len] =
			&child_chaining_values[2 * parents_array_len * BLAKE3_OUT_LEN];
		parents_array_len += 1;
	}

	blake3_hash_many(
		parents_array,
		parents_array_len,
		1,
		key,
		0,
		false,
		flags | PARENT,
		0,
		0,
		out);

	if (num_chaining_values > 2 * parents_array_len)
	{
		std::memcpy(
			&out[parents_array_len * BLAKE3_OUT_LEN],
			&child_chaining_values[2 * parents_array_len * BLAKE3_OUT_LEN],
			BLAKE3_OUT_LEN);
		return parents_array_len + 1;
	}

	return parents_array_len;
}

constexpr size_t ParallelMinInputBytes = 1ull << 20;
constexpr size_t ParallelThreadGranularityBytes = 768ull << 10;
constexpr size_t ParallelSpawnMinSubtreeBytes = 512ull << 10;
constexpr uint32_t ParallelMaxThreads = 8u;
std::atomic_uint32_t g_parallelHashCalls = 0u;

class SParallelCallGuard final
{
	public:
		SParallelCallGuard() : m_active(g_parallelHashCalls.fetch_add(1u, std::memory_order_relaxed) + 1u)
		{
		}

		~SParallelCallGuard()
		{
			g_parallelHashCalls.fetch_sub(1u, std::memory_order_relaxed);
		}

		inline uint32_t activeCalls() const
		{
			return m_active;
		}

	private:
		uint32_t m_active = 1u;
};

size_t compress_subtree_wide_mt(
	const uint8_t* input,
	size_t input_len,
	const uint32_t key[8],
	uint64_t chunk_counter,
	uint8_t flags,
	uint8_t* out,
	uint32_t threadBudget);

INLINE void compress_subtree_to_parent_node_mt(
	const uint8_t* input,
	size_t input_len,
	const uint32_t key[8],
	uint64_t chunk_counter,
	uint8_t flags,
	uint8_t out[2 * BLAKE3_OUT_LEN],
	uint32_t threadBudget)
{
	uint8_t cv_array[MAX_SIMD_DEGREE_OR_2 * BLAKE3_OUT_LEN];
	size_t num_cvs = compress_subtree_wide_mt(input, input_len, key, chunk_counter, flags, cv_array, threadBudget);
	assert(num_cvs <= MAX_SIMD_DEGREE_OR_2);

#if MAX_SIMD_DEGREE_OR_2 > 2
	uint8_t out_array[MAX_SIMD_DEGREE_OR_2 * BLAKE3_OUT_LEN / 2];
	while (num_cvs > 2)
	{
		num_cvs = compress_parents_parallel_local(cv_array, num_cvs, key, flags, out_array);
		std::memcpy(cv_array, out_array, num_cvs * BLAKE3_OUT_LEN);
	}
#endif

	std::memcpy(out, cv_array, 2 * BLAKE3_OUT_LEN);
}

size_t compress_subtree_wide_mt(
	const uint8_t* input,
	size_t input_len,
	const uint32_t key[8],
	uint64_t chunk_counter,
	uint8_t flags,
	uint8_t* out,
	uint32_t threadBudget)
{
	if (input_len <= blake3_simd_degree() * BLAKE3_CHUNK_LEN)
		return compress_chunks_parallel_local(input, input_len, key, chunk_counter, flags, out);

	const size_t left_input_len = left_len_local(input_len);
	const size_t right_input_len = input_len - left_input_len;
	const uint8_t* const right_input = &input[left_input_len];
	const uint64_t right_chunk_counter = chunk_counter + static_cast<uint64_t>(left_input_len / BLAKE3_CHUNK_LEN);

	uint8_t cv_array[2 * MAX_SIMD_DEGREE_OR_2 * BLAKE3_OUT_LEN];
	size_t degree = blake3_simd_degree();
	if (left_input_len > BLAKE3_CHUNK_LEN && degree == 1)
		degree = 2;
	uint8_t* const right_cvs = &cv_array[degree * BLAKE3_OUT_LEN];

	size_t left_n = 0;
	size_t right_n = 0;
	bool spawned = false;
	if (
		threadBudget > 1u &&
		left_input_len >= ParallelSpawnMinSubtreeBytes &&
		right_input_len >= ParallelSpawnMinSubtreeBytes)
	{
		try
		{
			uint32_t leftBudget = threadBudget / 2u;
			if (leftBudget == 0u)
				leftBudget = 1u;
			uint32_t rightBudget = threadBudget - leftBudget;
			if (rightBudget == 0u)
				rightBudget = 1u;

			auto rightFuture = std::async(std::launch::async, [right_input, right_input_len, key, right_chunk_counter, flags, right_cvs, rightBudget]() -> size_t
			{
				return compress_subtree_wide_mt(right_input, right_input_len, key, right_chunk_counter, flags, right_cvs, rightBudget);
			});
			left_n = compress_subtree_wide_mt(input, left_input_len, key, chunk_counter, flags, cv_array, leftBudget);
			right_n = rightFuture.get();
			spawned = true;
		}
		catch (...)
		{
			spawned = false;
		}
	}

	if (!spawned)
	{
		left_n = compress_subtree_wide_mt(input, left_input_len, key, chunk_counter, flags, cv_array, 1u);
		right_n = compress_subtree_wide_mt(right_input, right_input_len, key, right_chunk_counter, flags, right_cvs, 1u);
	}

	if (left_n == 1)
	{
		std::memcpy(out, cv_array, 2 * BLAKE3_OUT_LEN);
		return 2;
	}

	const size_t num_chaining_values = left_n + right_n;
	return compress_parents_parallel_local(cv_array, num_chaining_values, key, flags, out);
}

INLINE void hasher_merge_cv_stack_local(::blake3_hasher* self, uint64_t total_len)
{
	const size_t post_merge_stack_len = static_cast<size_t>(popcnt(total_len));
	while (self->cv_stack_len > post_merge_stack_len)
	{
		auto* const parent_node = &self->cv_stack[(self->cv_stack_len - 2) * BLAKE3_OUT_LEN];
		const auto output = parent_output_local(parent_node, self->key, self->chunk.flags);
		output_chaining_value_local(&output, parent_node);
		self->cv_stack_len -= 1;
	}
}

INLINE void hasher_push_cv_local(::blake3_hasher* self, uint8_t new_cv[BLAKE3_OUT_LEN], uint64_t chunk_counter)
{
	hasher_merge_cv_stack_local(self, chunk_counter);
	std::memcpy(&self->cv_stack[self->cv_stack_len * BLAKE3_OUT_LEN], new_cv, BLAKE3_OUT_LEN);
	self->cv_stack_len += 1;
}

void hasher_update_parallel(::blake3_hasher* self, const uint8_t* input_bytes, size_t input_len, uint32_t threadBudget)
{
	if (input_len == 0)
		return;

	if (chunk_state_len_local(&self->chunk) > 0)
	{
		size_t take = BLAKE3_CHUNK_LEN - chunk_state_len_local(&self->chunk);
		if (take > input_len)
			take = input_len;
		chunk_state_update_local(&self->chunk, input_bytes, take);
		input_bytes += take;
		input_len -= take;
		if (input_len > 0)
		{
			const auto output = chunk_state_output_local(&self->chunk);
			uint8_t chunk_cv[BLAKE3_OUT_LEN];
			output_chaining_value_local(&output, chunk_cv);
			hasher_push_cv_local(self, chunk_cv, self->chunk.chunk_counter);
			chunk_state_reset_local(&self->chunk, self->key, self->chunk.chunk_counter + 1);
		}
		else
		{
			return;
		}
	}

	while (input_len > BLAKE3_CHUNK_LEN)
	{
		size_t subtree_len = round_down_to_power_of_2(input_len);
		const uint64_t count_so_far = self->chunk.chunk_counter * BLAKE3_CHUNK_LEN;
		while ((((uint64_t)(subtree_len - 1)) & count_so_far) != 0)
			subtree_len /= 2;

		const uint64_t subtree_chunks = subtree_len / BLAKE3_CHUNK_LEN;
		if (subtree_len <= BLAKE3_CHUNK_LEN)
		{
			blake3_chunk_state chunk_state = {};
			chunk_state_init_local(&chunk_state, self->key, self->chunk.flags);
			chunk_state.chunk_counter = self->chunk.chunk_counter;
			chunk_state_update_local(&chunk_state, input_bytes, subtree_len);
			const auto output = chunk_state_output_local(&chunk_state);
			uint8_t cv[BLAKE3_OUT_LEN];
			output_chaining_value_local(&output, cv);
			hasher_push_cv_local(self, cv, chunk_state.chunk_counter);
		}
		else
		{
			uint8_t cv_pair[2 * BLAKE3_OUT_LEN];
			compress_subtree_to_parent_node_mt(
				input_bytes,
				subtree_len,
				self->key,
				self->chunk.chunk_counter,
				self->chunk.flags,
				cv_pair,
				threadBudget);
			hasher_push_cv_local(self, cv_pair, self->chunk.chunk_counter);
			hasher_push_cv_local(self, &cv_pair[BLAKE3_OUT_LEN], self->chunk.chunk_counter + (subtree_chunks / 2));
		}
		self->chunk.chunk_counter += subtree_chunks;
		input_bytes += subtree_len;
		input_len -= subtree_len;
	}

	if (input_len > 0)
	{
		chunk_state_update_local(&self->chunk, input_bytes, input_len);
		hasher_merge_cv_stack_local(self, self->chunk.chunk_counter);
	}
}

INLINE uint32_t pick_parallel_budget(const size_t bytes)
{
	const uint32_t hw = std::thread::hardware_concurrency();
	if (hw <= 1u || bytes < ParallelMinInputBytes)
		return 1u;

	const uint32_t maxBySize = static_cast<uint32_t>(std::max<size_t>(1ull, bytes / ParallelThreadGranularityBytes));
	uint32_t budget = std::min<uint32_t>(hw, ParallelMaxThreads);
	budget = std::min<uint32_t>(budget, maxBySize);
	return std::max<uint32_t>(1u, budget);
}

}

blake3_hasher::blake3_hasher()
{
	::blake3_hasher_init(&m_state);
}

blake3_hasher& blake3_hasher::update(const void* data, const size_t bytes)
{
	::blake3_hasher_update(&m_state, data, bytes);
	return *this;
}

void blake3_hasher::reset()
{
	::blake3_hasher_reset(&m_state);
}

blake3_hasher::operator blake3_hash_t() const
{
	blake3_hash_t retval = {};
	::blake3_hasher_finalize(&m_state, retval.data, sizeof(retval));
	return retval;
}

blake3_hash_t blake3_hash_buffer(const void* data, size_t bytes)
{
	if (!data || bytes == 0ull)
		return static_cast<blake3_hash_t>(blake3_hasher{});

	uint32_t threadBudget = pick_parallel_budget(bytes);
	if (threadBudget <= 1u)
	{
		blake3_hasher hasher;
		hasher.update(data, bytes);
		return static_cast<blake3_hash_t>(hasher);
	}

	SParallelCallGuard guard;
	const uint32_t activeCalls = std::max<uint32_t>(1u, guard.activeCalls());
	const uint32_t hw = std::max<uint32_t>(1u, std::thread::hardware_concurrency());
	const uint32_t hwShare = std::max<uint32_t>(1u, hw / activeCalls);
	threadBudget = std::min(threadBudget, hwShare);
	if (threadBudget <= 1u)
	{
		blake3_hasher hasher;
		hasher.update(data, bytes);
		return static_cast<blake3_hash_t>(hasher);
	}

	::blake3_hasher hasherState = {};
	::blake3_hasher_init(&hasherState);
	hasher_update_parallel(&hasherState, reinterpret_cast<const uint8_t*>(data), bytes, threadBudget);
	blake3_hash_t retval = {};
	::blake3_hasher_finalize(&hasherState, retval.data, sizeof(retval));
	return retval;
}

blake3_hash_t blake3_hash_buffer_sequential(const void* data, size_t bytes)
{
	if (!data || bytes == 0ull)
		return static_cast<blake3_hash_t>(blake3_hasher{});

	blake3_hasher hasher;
	hasher.update(data, bytes);
	return static_cast<blake3_hash_t>(hasher);
}

}
