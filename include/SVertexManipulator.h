// Copyright (C) 2009-2012 Christian Stehno
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __S_VERTEX_MANIPULATOR_H_INCLUDED__
#define __S_VERTEX_MANIPULATOR_H_INCLUDED__

#include "vectorSIMD.h"
#include "irr/video/IGPUMeshBuffer.h"
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
        core::vectorSIMDf vectorForDots(fittingVector);
        vectorForDots /= vectorForDots.getLength(); //precise normalize
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
        if (maxNormalComp<=0.577f) //max component of 3d normal cannot be less than sqrt(1/3)
            return core::vectorSIMDf(0.f);

        fittingVector /= maxNormalComp;


        uint32_t cubeHalfSize = (0x1u<<(bits-1))-1;
        float closestTo1 = -1.f;
        core::vectorSIMDf bestFit = fittingVector;
        for (uint32_t n=1; n<=cubeHalfSize; n++)
        {
            //we'd use float addition in the interest of speed, to increment the loop
            //but adding a small number to a large one loses precision, so multiplication preferrable
            core::vectorSIMDf bottomFit = fittingVector*float(n);
            bottomFit += floorOffset;
            bottomFit = floor(bottomFit);
            for (uint32_t i=0; i<4; i++)
            {
                core::vectorSIMDf bottomFitTmp = bottomFit;
                if (i)
                {
                    bottomFitTmp += corners[i];
                    if ((bottomFitTmp>core::vectorSIMDf(cubeHalfSize)).any())
                        continue;
                }

                float bottomFitLen = bottomFitTmp.getLengthAsFloat();//more precise normalization
                float dp = bottomFitTmp.dotProductAsFloat(vectorForDots);
                if (dp>closestTo1*bottomFitLen)
                {
                    closestTo1 = dp/bottomFitLen;
                    bestFit = bottomFitTmp;
                }
            }
        }

        return core::min_(bestFit,core::vectorSIMDf(cubeHalfSize))+0.01f;
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

        core::vectorSIMDf fit = findBestFit(10u, normal);
        const uint32_t xorflag = (0x1u<<10)-1;
        uint32_t bestFit = ((uint32_t(fit.X)^(normal.X<0.f ? xorflag:0))+(normal.X<0.f ? 1:0))&xorflag;
        bestFit |= (((uint32_t(fit.Y)^(normal.Y<0.f ? xorflag:0))+(normal.Y<0.f ? 1:0))&xorflag)<<10;
        bestFit |= (((uint32_t(fit.Z)^(normal.Z<0.f ? xorflag:0))+(normal.Z<0.f ? 1:0))&xorflag)<<20;
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

        uint8_t bestFit[4] {0u,0u,0u,0u};

        core::vectorSIMDf fit = findBestFit(8u,normal);
        const uint32_t xorflag = (0x1u<<8)-1;
        bestFit[0] = (uint32_t(fit.X)^(normal.X<0.f ? xorflag:0))+(normal.X<0.f ? 1:0);
        bestFit[1] = (uint32_t(fit.Y)^(normal.Y<0.f ? xorflag:0))+(normal.Y<0.f ? 1:0);
        bestFit[2] = (uint32_t(fit.Z)^(normal.Z<0.f ? xorflag:0))+(normal.Z<0.f ? 1:0);

		dummySearchVal.value = *reinterpret_cast<uint32_t*>(bestFit);
		normalCacheFor8_8_8Quant.insert(found, dummySearchVal);

	    return *reinterpret_cast<uint32_t*>(bestFit);
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

		core::vectorSIMDf fit = findBestFit(16u, normal);
		const uint32_t xorflag = (1u<<16)-1;
		bestFit[0] = (uint32_t(fit.x) ^ (normal.x < 0.f ? xorflag : 0u)) + (normal.x < 0.f ? 1u : 0u);
		bestFit[1] = (uint32_t(fit.y) ^ (normal.y < 0.f ? xorflag : 0u)) + (normal.y < 0.f ? 1u : 0u);
		bestFit[2] = (uint32_t(fit.z) ^ (normal.z < 0.f ? xorflag : 0u)) + (normal.z < 0.f ? 1u : 0u);

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
	/*
        ECT_FLOAT=0,
        ECT_HALF_FLOAT,
        ECT_DOUBLE_IN_FLOAT_OUT,
        ECT_UNSIGNED_INT_10F_11F_11F_REV,
        //INTEGER FORMS
        ECT_NORMALIZED_INT_2_10_10_10_REV,
        ECT_NORMALIZED_UNSIGNED_INT_2_10_10_10_REV,
        ECT_NORMALIZED_BYTE,
        ECT_NORMALIZED_UNSIGNED_BYTE,
        ECT_NORMALIZED_SHORT,
        ECT_NORMALIZED_UNSIGNED_SHORT,
        ECT_NORMALIZED_INT,
        ECT_NORMALIZED_UNSIGNED_INT,
        ECT_INT_2_10_10_10_REV,
        ECT_UNSIGNED_INT_2_10_10_10_REV,
        ECT_BYTE,
        ECT_UNSIGNED_BYTE,
        ECT_SHORT,
        ECT_UNSIGNED_SHORT,
        ECT_INT,
        ECT_UNSIGNED_INT,
        ECT_INTEGER_INT_2_10_10_10_REV,
        ECT_INTEGER_UNSIGNED_INT_2_10_10_10_REV,
        ECT_INTEGER_BYTE,
        ECT_INTEGER_UNSIGNED_BYTE,
        ECT_INTEGER_SHORT,
        ECT_INTEGER_UNSIGNED_SHORT,
        ECT_INTEGER_INT,
        ECT_INTEGER_UNSIGNED_INT,*/

} // end namespace scene
} // end namespace irr


#endif
