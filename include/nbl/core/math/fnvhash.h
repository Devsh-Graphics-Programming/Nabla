// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_FNV_HASH_H_INCLUDED__
#define __NBL_CORE_FNV_HASH_H_INCLUDED__

#include <stdint.h>
#include <nbl/core/definitions.h>

namespace nbl::core
{

constexpr uint64_t FNV_SEED = 14695981039346656037ull;
constexpr uint64_t FNV_PRIME = 1099511628211ull;

template <typename T, typename U>
NBL_FORCE_INLINE uint64_t fnv_hash_base(const T& value) {
    const U* buf = reinterpret_cast<const U*>(&value);
    uint64_t hash_value = FNV_SEED;
    for (uint64_t i = 0; i < (sizeof(T) / sizeof(U)); i++) {
        hash_value ^= buf[i];
        hash_value *= FNV_PRIME;
    }
    return hash_value;
}

// Calculate Fowler–Noll–Vo hash of arbitrary data
// Note that it's UB when T is struct with padding bytes
template <typename T>
NBL_FORCE_INLINE uint64_t fnv_hash(const T& value) {
    if constexpr (is_aligned_to(sizeof(T), 8))
        return fnv_hash_base<T, uint64_t>(value);

    if constexpr (is_aligned_to(sizeof(T), 4))
        return fnv_hash_base<T, uint32_t>(value);

    if constexpr (is_aligned_to(sizeof(T), 2))
        return fnv_hash_base<T, uint16_t>(value);

    return fnv_hash_base<T, uint8_t>(value);
}

}

#endif