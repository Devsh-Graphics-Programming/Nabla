// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_CORE_HASH_BLAKE3_H_INCLUDED_
#define _NBL_CORE_HASH_BLAKE3_H_INCLUDED_

#include "blake3.h"

namespace nbl::core
{
struct blake3_hash_t
{
	inline bool operator==(const blake3_hash_t&) const = default;

	uint8_t data[BLAKE3_OUT_LEN];
};

template<typename T>
inline void blake3_hasher_update(blake3_hasher& self, const T& input)
{
	::blake3_hasher_update(&self,&input,sizeof(input));
}

inline blake3_hash_t blake3_hasher_finalize(blake3_hasher& self)
{
	blake3_hash_t retval;
	::blake3_hasher_finalize(&self,retval.data,sizeof(retval));
	return retval;
}
}


namespace std
{
template<>
struct hash<nbl::core::blake3_hash_t>
{
	inline size_t operator()(const nbl::core::blake3_hash_t& blake3) const
	{
		auto* as_p_uint64_t = reinterpret_cast<const size_t*>(blake3.data);
		size_t retval = as_p_uint64_t[0];
		for (auto i=1; i<BLAKE3_OUT_LEN; i++)
			retval ^= as_p_uint64_t[i] + 0x9e3779b97f4a7c15ull + (retval << 6) + (retval >> 2);
		return retval;
	}
};
}

#endif // _NBL_CORE_HASH_BLAKE3_H_INCLUDED_