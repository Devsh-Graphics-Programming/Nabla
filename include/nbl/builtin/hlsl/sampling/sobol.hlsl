// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SAMPLING_SOBOL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_SOBOL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/macros.h>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

// without acceleration
struct RowMajorSobolMatrix
{
   inline uint16_t operator()(const uint16_t sampleIx)
   {
      // max number bits set is 16, if want to mask then need to be 5 apart
      const uint16_t mask = 0b10000100001u;
      uint16_t       val  = _static_cast<uint16_t>(spirv::bitCount<uint32_t>(rows[0] & sampleIx));
      val |= _static_cast<uint16_t>(spirv::bitCount<uint32_t>(rows[5] & sampleIx)) << 5;
      val |= _static_cast<uint16_t>(spirv::bitCount<uint32_t>(rows[10] & sampleIx)) << 10;
      val |= _static_cast<uint16_t>(spirv::bitCount<uint32_t>(rows[15] & sampleIx)) << 15;
      val &= mask;
      NBL_UNROLL
      for (uint16_t i = 1; i < 5; i++)
      {
         uint16_t tmp = _static_cast<uint16_t>(spirv::bitCount<uint32_t>(rows[i] & sampleIx));
         tmp |= _static_cast<uint16_t>(spirv::bitCount<uint32_t>(rows[5 + i] & sampleIx)) << 5;
         tmp |= _static_cast<uint16_t>(spirv::bitCount<uint32_t>(rows[10 + i] & sampleIx)) << 10;
         val |= (tmp & mask) << i;
      }
      return val;
   }

   uint16_t rows[16];
};

// bitcount is expensive, its full 32 bit and needs shifting and masking anyway
struct ColMajorSobolMatrix
{
   inline uint16_t operator()(const uint16_t sampleIx)
   {
      // abuse 2s completement to get a 0xffffu mask when `sampleIx` bit is on
      uint16_t val = cols[0] & ((sampleIx & _static_cast<uint16_t>(0x1u)) - _static_cast<uint16_t>(2));
      NBL_UNROLL
      for (uint16_t i = 1; i < 16; i++)
      {
         const uint16_t mask = _static_cast<uint16_t>(0x1u) << i;
         val |= cols[i] & ((sampleIx & mask) - _static_cast<uint16_t>(mask + 1));
      }
      return val;
   }

   uint16_t cols[16];
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
