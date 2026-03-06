#include "nbl/core/hash/blake.h"

#include <cassert>
#include <cstring>

namespace nbl::core
{

blake3_hasher::blake3_hasher()
{
	::blake3_hasher_init(&m_state);
}

blake3_hasher& blake3_hasher::update(const void* data, const size_t bytes)
{
	if (bytes == 0ull)
		return *this;

	assert(data != nullptr);
	if (!data)
		return *this;

	::blake3_hasher_update(&m_state, data, bytes);
	return *this;
}

void blake3_hasher::reset()
{
	::blake3_hasher_init(&m_state);
}

blake3_hasher::operator blake3_hash_t() const
{
	blake3_hash_t retval = {};
	::blake3_hasher stateCopy = m_state;
	::blake3_hasher_finalize(&stateCopy, retval.data, BLAKE3_OUT_LEN);
	return retval;
}

blake3_hash_t blake3_hash_buffer_sequential(const void* data, size_t bytes)
{
	if (!data && bytes != 0ull)
		return {};

	::blake3_hasher hasher = {};
	::blake3_hasher_init(&hasher);
	if (bytes != 0ull)
		::blake3_hasher_update(&hasher, data, bytes);

	blake3_hash_t retval = {};
	::blake3_hasher_finalize(&hasher, retval.data, BLAKE3_OUT_LEN);
	return retval;
}

blake3_hash_t blake3_hash_buffer(const void* data, size_t bytes)
{
	return blake3_hash_buffer_sequential(data, bytes);
}

}
