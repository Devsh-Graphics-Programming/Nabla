// Copyright (C) 2009-2012 Christian Stehno
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_NORMAL_QUANTIZATION_H_INCLUDED__
#define __IRR_NORMAL_QUANTIZATION_H_INCLUDED__

#include "irr/core/math/glslFunctions.tcc"
#include <vector>
#include <fstream>
#include <iterator>
#include <algorithm>

namespace irr
{
namespace asset
{
	struct QuantizationCacheEntryBase
	{
		core::vectorSIMDf key;

		inline bool operator<(const QuantizationCacheEntryBase& other) const
		{
			if (key.Z<other.key.Z)
				return true;
			else if (key.Z == other.key.Z)
			{
				if (key.Y<other.key.Y)
					return true;
				else if (key.Y == other.key.Y)
					return key.X<other.key.X;
				else
					return false;
			}
			else
				return false;
		}
	};
	static_assert(sizeof(QuantizationCacheEntryBase) == 16u);

    struct QuantizationCacheEntry2_10_10_10 : QuantizationCacheEntryBase
    {
        uint32_t value;
    };

	using QuantizationCacheEntry8_8_8 = QuantizationCacheEntry2_10_10_10;

	struct QuantizationCacheEntry16_16_16 : QuantizationCacheEntryBase
	{
		uint64_t value;
	};

	using QuantizationCacheEntryHalfFloat = QuantizationCacheEntry16_16_16;

	// defined in CMeshManipulator.cpp
    extern core::vector<QuantizationCacheEntry2_10_10_10>   normalCacheFor2_10_10_10Quant;
	extern core::vector<QuantizationCacheEntry8_8_8>        normalCacheFor8_8_8Quant;
	extern core::vector<QuantizationCacheEntry16_16_16>     normalCacheFor16_16_16Quant;
	extern core::vector<QuantizationCacheEntryHalfFloat>    normalCacheForHalfFloatQuant;

    inline core::vectorSIMDf findBestFit(const uint32_t& bits, const core::vectorSIMDf& normal)
    {
        core::vectorSIMDf fittingVector = normal;
        fittingVector.makeSafe3D();
        fittingVector = abs(fittingVector);

		// precise normalize
		auto vectorForDots = fittingVector.preciseDivision(length(fittingVector));

        float maxNormalComp;
        core::vectorSIMDf corners[4];
        core::vectorSIMDf floorOffset;
        if (fittingVector.X>fittingVector.Y)
        {
            maxNormalComp = fittingVector.X;
            corners[1].set(0,1,0);
            corners[2].set(0,0,1);
            corners[3].set(0,1,1);
            floorOffset.set(0.499f,0.f,0.f);
        }
        else
        {
            maxNormalComp = fittingVector.Y;
            corners[1].set(1,0,0);
            corners[2].set(0,0,1);
            corners[3].set(1,0,1);
            floorOffset.set(0.f,0.499f,0.f);
        }
        //second round
        if (fittingVector.Z>maxNormalComp)
        {
            maxNormalComp = fittingVector.Z;
            corners[1].set(1,0,0);
            corners[2].set(0,1,0);
            corners[3].set(1,1,0);
            floorOffset.set(0.f,0.f,0.499f);
        }
		
		//max component of 3d normal cannot be less than sqrt(1/3)
        if (maxNormalComp<=0.577f) //max component of 3d normal cannot be less than sqrt(1/3)
		{
			_IRR_DEBUG_BREAK_IF(true);
            return core::vectorSIMDf(0.f);
		}


        fittingVector = fittingVector.preciseDivision(core::vectorSIMDf(maxNormalComp));


        const uint32_t cubeHalfSize = (0x1u<<(bits-1u))-1u;
		const core::vectorSIMDf cubeHalfSize3D = core::vectorSIMDf(cubeHalfSize);
		core::vectorSIMDf bestFit;
		float closestTo1 = -1.f;
		auto evaluateFit = [&](const core::vectorSIMDf& newFit) -> void
		{
			auto newFitLen = core::length(newFit);
			auto dp = core::dot<core::vectorSIMDf>(newFit,vectorForDots).preciseDivision(newFitLen);
			if (dp[0] > closestTo1)
			{
				closestTo1 = dp[0];
				bestFit = newFit;
			}
		};
		for (uint32_t n=cubeHalfSize; n>0u; n--)
		{
            //we'd use float addition in the interest of speed, to increment the loop
            //but adding a small number to a large one loses precision, so multiplication preferrable
            core::vectorSIMDf bottomFit = core::floor(fittingVector*float(n)+floorOffset);
			for (uint32_t i=0u; i<4u; i++)
			{
				core::vectorSIMDf bottomFitTmp = bottomFit;
				if (i)
					bottomFitTmp += corners[i];
                if ((bottomFitTmp>cubeHalfSize3D).any())
					continue;
				evaluateFit(bottomFitTmp);
			}
		}

		return bestFit;
    }

	inline uint32_t quantizeNormal2_10_10_10(const core::vectorSIMDf &normal)
	{
        QuantizationCacheEntry2_10_10_10 dummySearchVal;
        dummySearchVal.key = normal;
        auto found = std::lower_bound(normalCacheFor2_10_10_10Quant.begin(),normalCacheFor2_10_10_10Quant.end(),dummySearchVal);
        if (found!=normalCacheFor2_10_10_10Quant.end()&&(found->key==normal).all())
        {
            return found->value;
        }

		constexpr uint32_t quantizationBits = 10u;
		const auto xorflag = core::vectorSIMDu32((0x1u<<quantizationBits)-1u);
        core::vectorSIMDf fit = findBestFit(quantizationBits, normal);
		auto negativeMask = normal < core::vectorSIMDf(0.f);
		auto absIntFit = core::vectorSIMDu32(core::abs(fit))^core::mix(core::vectorSIMDu32(0u),core::vectorSIMDu32(xorflag),negativeMask);
		auto snormVec = (absIntFit+core::mix(core::vectorSIMDu32(0u),core::vectorSIMDu32(1u),negativeMask))&xorflag;
        
        uint32_t bestFit = snormVec[0]|(snormVec[1]<<quantizationBits)|(snormVec[2]<<(quantizationBits*2u));

		dummySearchVal.value = bestFit;
        normalCacheFor2_10_10_10Quant.insert(found,dummySearchVal);


	    return bestFit;
	}

	inline uint32_t quantizeNormal888(const core::vectorSIMDf &normal)
	{
		QuantizationCacheEntry8_8_8 dummySearchVal;
		dummySearchVal.key = normal;
		auto found = std::lower_bound(normalCacheFor8_8_8Quant.begin(), normalCacheFor8_8_8Quant.end(), dummySearchVal);
		if (found != normalCacheFor8_8_8Quant.end() && (found->key == normal).all())
		{
			return found->value;
		}
		
		constexpr uint32_t quantizationBits = 8u;
		const auto xorflag = core::vectorSIMDu32((0x1u<<quantizationBits)-1u);
        core::vectorSIMDf fit = findBestFit(quantizationBits, normal);
		auto negativeMask = normal < core::vectorSIMDf(0.f);
		auto absIntFit = core::vectorSIMDu32(core::abs(fit))^core::mix(core::vectorSIMDu32(0u),xorflag,negativeMask);
		auto snormVec = (absIntFit+core::mix(core::vectorSIMDu32(0u),core::vectorSIMDu32(1u),negativeMask))&xorflag;
        
        uint32_t bestFit = snormVec[0]|(snormVec[1]<<quantizationBits)|(snormVec[2]<<(quantizationBits*2u));

		dummySearchVal.value = bestFit;
		normalCacheFor8_8_8Quant.insert(found, dummySearchVal);

	    return bestFit;
	}

	inline uint64_t quantizeNormal16_16_16(const core::vectorSIMDf& normal)
	{
		QuantizationCacheEntry16_16_16 dummySearchVal;
		dummySearchVal.key = normal;
		auto found = std::lower_bound(normalCacheFor16_16_16Quant.begin(), normalCacheFor16_16_16Quant.end(), dummySearchVal);
		if (found != normalCacheFor16_16_16Quant.end() && (found->key == normal).all())
		{
			return found->value;
		}

		uint16_t bestFit[4]{0u,0u,0u,0u};

		constexpr uint32_t quantizationBits = 10u;
		const auto xorflag = core::vectorSIMDu32((0x1u<<quantizationBits)-1u);
        core::vectorSIMDf fit = findBestFit(quantizationBits, normal);
		auto negativeMask = normal < core::vectorSIMDf(0.f);
		auto absIntFit = core::vectorSIMDu32(core::abs(fit))^core::mix(core::vectorSIMDu32(0u),core::vectorSIMDu32(xorflag),negativeMask);
		auto snormVec = (absIntFit+core::mix(core::vectorSIMDu32(0u),core::vectorSIMDu32(1u),negativeMask))&xorflag;
        
		bestFit[0] = snormVec[0];
		bestFit[1] = snormVec[1];
		bestFit[2] = snormVec[2];

		dummySearchVal.value = *reinterpret_cast<uint64_t*>(bestFit);
		normalCacheFor16_16_16Quant.insert(found, dummySearchVal);

		return *reinterpret_cast<uint64_t*>(bestFit);
	}

	inline uint64_t quantizeNormalHalfFloat(const core::vectorSIMDf& normal)
	{
		QuantizationCacheEntryHalfFloat dummySearchVal;
		dummySearchVal.key = normal;
		auto found = std::lower_bound(normalCacheForHalfFloatQuant.begin(), normalCacheForHalfFloatQuant.end(), dummySearchVal);
		if (found != normalCacheForHalfFloatQuant.end() && (found->key == normal).all())
		{
			return found->value;
		}

		uint16_t bestFit[4] {
			core::Float16Compressor::compress(normal.x),
			core::Float16Compressor::compress(normal.y),
			core::Float16Compressor::compress(normal.z),
			0u
		};

		dummySearchVal.value = *reinterpret_cast<uint64_t*>(bestFit);
		normalCacheForHalfFloatQuant.insert(found, dummySearchVal);

		return *reinterpret_cast<uint64_t*>(bestFit);
	}

} // end namespace scene
} // end namespace irr


#endif
