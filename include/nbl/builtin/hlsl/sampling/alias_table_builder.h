// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_ALIAS_TABLE_BUILDER_H_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_ALIAS_TABLE_BUILDER_H_INCLUDED_

#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <nbl/builtin/hlsl/sampling/alias_table.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

// Builds the alias table from an array of non-negative weights. Output vectors are resized to N, which is also returned.
template<typename T>
struct AliasTableBuilder
{
   static uint32_t build(std::span<const T> weights, std::vector<T>& outProbability, std::vector<uint32_t>& outAlias, std::vector<T>& outPdf)
   {
      const uint32_t N = static_cast<uint32_t>(weights.size());

      outProbability.resize(N);
      outAlias.resize(N);
      outPdf.resize(N);
      std::vector<uint32_t> workspace(N);

      T totalWeight = T(0);
      for (uint32_t i = 0; i < N; i++)
         totalWeight += weights[i];

      const T rcpTotalWeight = T(1) / totalWeight;

      // Compute PDFs, scaled probabilities, and partition into small/large in one pass
      uint32_t smallEnd   = 0u;
      uint32_t largeBegin = N;
      for (uint32_t i = 0; i < N; i++)
      {
         outPdf[i]         = weights[i] * rcpTotalWeight;
         outProbability[i] = outPdf[i] * T(N);

         if (outProbability[i] < T(1))
            workspace[smallEnd++] = i;
         else
            workspace[--largeBegin] = i;
      }

      // Pair small and large entries
      while (smallEnd > 0u && largeBegin < N)
      {
         const uint32_t s = workspace[--smallEnd];
         const uint32_t l = workspace[largeBegin];

         outAlias[s] = l;
         // outProbability[s] already holds the correct probability for bin s

         outProbability[l] -= (T(1) - outProbability[s]);

         if (outProbability[l] < T(1))
         {
            // l became small: pop from large, push to small
            largeBegin++;
            workspace[smallEnd++] = l;
         }
         // else l stays in large (don't pop, reuse next iteration)
      }

      // Remaining entries (floating point rounding artifacts)
      while (smallEnd > 0u)
      {
         const uint32_t s  = workspace[--smallEnd];
         outProbability[s] = T(1);
         outAlias[s]       = s;
      }
      while (largeBegin < N)
      {
         const uint32_t l  = workspace[largeBegin++];
         outProbability[l] = T(1);
         outAlias[l]       = l;
      }

      return N;
   }

   // Pack (target, stayProb) into a single 32-bit word with Log2N bits for
   // target and (32 - Log2N) bits for the unorm-quantized threshold. Used by
   // every packed variant; each packX() below calls this on a per-entry basis.
   template<uint32_t Log2N>
   static uint32_t packWord(uint32_t target, T stayProb)
   {
      const uint32_t targetMask = (Log2N == 32u) ? ~0u : ((1u << Log2N) - 1u);
      const T        clamped    = stayProb < T(0) ? T(0) : (stayProb > T(1) ? T(1) : stayProb);
      const uint32_t unormMax   = (Log2N == 0u) ? ~0u : ((~0u) >> Log2N);
      const uint32_t probUnorm  = static_cast<uint32_t>(std::round(static_cast<double>(clamped) * static_cast<double>(unormMax)));
      return (target & targetMask) | (probUnorm << Log2N);
   }

   // Variant A, pack SoA outputs into an array of 4 B packed words. The
   // pdf[] array is consumed directly by the sampler as a second accessor.
   // outWords must be pre-allocated to N uint32_t entries.
   template<uint32_t Log2N>
   static void packA(std::span<const T> probability, std::span<const uint32_t> alias, uint32_t* outWords)
   {
      const uint32_t N = static_cast<uint32_t>(probability.size());
      for (uint32_t i = 0; i < N; i++)
         outWords[i] = packWord<Log2N>(alias[i], probability[i]);
   }

   // Variant B, pack SoA outputs into 8 B entries { packedWord, ownPdf }.
   // The pdf[] array is *also* passed to the sampler (same contents as ownPdf
   // column, but tapped independently with a 4 B fetch when the sample aliases).
   // outEntries must be pre-allocated to N entries.
   template<uint32_t Log2N>
   static void packB(std::span<const T> probability, std::span<const uint32_t> alias, std::span<const T> pdf,
      PackedAliasEntryB<T>* outEntries)
   {
      const uint32_t N = static_cast<uint32_t>(probability.size());
      for (uint32_t i = 0; i < N; i++)
      {
         outEntries[i].packedWord = packWord<Log2N>(alias[i], probability[i]);
         outEntries[i].ownPdf     = pdf[i];
      }
   }
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
