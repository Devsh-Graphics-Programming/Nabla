// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_XXHASH256_H_INCLUDED__
#define __NBL_CORE_XXHASH256_H_INCLUDED__

#include <cstdint>
#include <cstring>

namespace nbl::core
{

/*
   xxHash256 - A fast checksum algorithm
   Copyright (C) 2012, Yann Collet & Maciej Adamczyk.
   BSD 2-Clause License (http://www.opensource.org/licenses/bsd-license.php)

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

       * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following disclaimer
   in the documentation and/or other materials provided with the
   distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

************************************************************************
   This file contains a super-fast hash function for checksumming
   purposes, designed for large (1 KB++) inputs.
   Known limitations:
    * they use 64-bit math and will not be so fast on 32-bit machines
    * on architectures that disallow unaligned memory access, the input
      must be 8-byte aligned. Aligning it on other architectures
      is not needed, but will improve performance.
    * it produces different results on big and small endian.

   Changelog:
    v0: initial release
    v1: the strong hash is faster and stronger, it obsoletes the fast one
*/

//! Super-fast function for checksuming purposes. Designed for large (>1KB) inputs.
/** @param[in] input Pointer to data being the input for hasing algorithm.
@param[in] len Size in bytes of data pointed by `input`.
@param[out] out Pointer to 8byte memory to which result will be written.
*/
inline void XXHash_256(const void* input, size_t len, uint64_t* out)
{
	//**************************************
	// Macros
	//**************************************
	#define _rotl(x,r) ((x << r) | (x >> (64 - r)))

	//**************************************
	// Constants
	//**************************************
    const uint64_t PRIME = 11400714819323198393ULL;

    const uint8_t* p = (uint8_t*)input;
    const uint8_t* const bEnd = p + len;
    uint64_t v1 = len * PRIME;
    uint64_t v2 = v1;
    uint64_t v3 = v1;
    uint64_t v4 = v1;

    const size_t big_loop_step = 4 * 4 * sizeof(uint64_t);
    const size_t small_loop_step = 4 * sizeof(uint64_t);
    // Set the big loop limit early enough, so the well-mixing small loop can be executed twice after it
    const uint8_t* const big_loop_limit   = bEnd - big_loop_step - 2 * small_loop_step;
    const uint8_t* const small_loop_limit = bEnd - small_loop_step;

    while (p < big_loop_limit)
    {
        v1 = _rotl(v1, 29) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v2 = _rotl(v2, 31) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v3 = _rotl(v3, 33) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v4 = _rotl(v4, 35) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v1 += v2 *= PRIME;
        v1 = _rotl(v1, 29) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v2 = _rotl(v2, 31) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v3 = _rotl(v3, 33) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v4 = _rotl(v4, 35) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v2 += v3 *= PRIME;
        v1 = _rotl(v1, 29) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v2 = _rotl(v2, 31) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v3 = _rotl(v3, 33) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v4 = _rotl(v4, 35) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v3 += v4 *= PRIME;
        v1 = _rotl(v1, 29) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v2 = _rotl(v2, 31) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v3 = _rotl(v3, 33) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v4 = _rotl(v4, 35) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v4 += v1 *= PRIME;
    }

    while (p < small_loop_limit)
    {
        v1 = _rotl(v1, 29) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v2 += v1 *= PRIME;
        v2 = _rotl(v2, 31) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v3 += v2 *= PRIME;
        v3 = _rotl(v3, 33) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v4 += v3 *= PRIME;
        v4 = _rotl(v4, 35) + (*(uint64_t*)p); p+=sizeof(uint64_t);
        v1 += v4 *= PRIME;
    }
#undef _rotl
    size_t leftOverBytes = bEnd - p;
    memcpy(out, p, leftOverBytes);
    for (uint8_t* leftOverZeroP = reinterpret_cast<uint8_t*>(out)+leftOverBytes; leftOverZeroP<reinterpret_cast<uint8_t*>(out+4); leftOverZeroP++)
        *leftOverZeroP = 0;


    out[0] += v1;
    out[1] += v2;
    out[2] += v3;
	out[3] += v4;
}


}

#endif
