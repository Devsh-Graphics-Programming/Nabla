// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/utils/CQuantNormalCache.h"

#include "os.h"

#include "parallel-hashmap/parallel_hashmap/phmap_dump.h"

namespace nbl
{
namespace asset
{

core::vectorSIMDf CQuantNormalCache::findBestFit(const uint32_t& bits, const core::vectorSIMDf& normal) const
{
	core::vectorSIMDf fittingVector = normal;

	// precise normalize
	auto vectorForDots = fittingVector.preciseDivision(length(fittingVector));

	float maxNormalComp;
	core::vectorSIMDf corners[4];
	core::vectorSIMDf floorOffset;
	if (fittingVector.X > fittingVector.Y)
	{
		maxNormalComp = fittingVector.X;
		corners[1].set(0, 1, 0);
		corners[2].set(0, 0, 1);
		corners[3].set(0, 1, 1);
		floorOffset.set(0.499f, 0.f, 0.f);
	}
	else
	{
		maxNormalComp = fittingVector.Y;
		corners[1].set(1, 0, 0);
		corners[2].set(0, 0, 1);
		corners[3].set(1, 0, 1);
		floorOffset.set(0.f, 0.499f, 0.f);
	}
	//second round
	if (fittingVector.Z > maxNormalComp)
	{
		maxNormalComp = fittingVector.Z;
		corners[1].set(1, 0, 0);
		corners[2].set(0, 1, 0);
		corners[3].set(1, 1, 0);
		floorOffset.set(0.f, 0.f, 0.499f);
	}

	//max component of 3d normal cannot be less than sqrt(1/3)
	if (maxNormalComp <= 0.577f) //max component of 3d normal cannot be less than sqrt(1/3)
	{
		_NBL_DEBUG_BREAK_IF(true);
		return core::vectorSIMDf(0.f);
	}


	fittingVector = fittingVector.preciseDivision(core::vectorSIMDf(maxNormalComp));


	const uint32_t cubeHalfSize = (0x1u << (bits - 1u)) - 1u;
	const core::vectorSIMDf cubeHalfSize3D = core::vectorSIMDf(cubeHalfSize);
	core::vectorSIMDf bestFit;
	float closestTo1 = -1.f;
	auto evaluateFit = [&](const core::vectorSIMDf& newFit) -> void
	{
		auto newFitLen = core::length(newFit);
		auto dp = core::dot<core::vectorSIMDf>(newFit, vectorForDots).preciseDivision(newFitLen);
		if (dp[0] > closestTo1)
		{
			closestTo1 = dp[0];
			bestFit = newFit;
		}
	};
	for (uint32_t n = cubeHalfSize; n > 0u; n--)
	{
		//we'd use float addition in the interest of speed, to increment the loop
		//but adding a small number to a large one loses precision, so multiplication preferrable
		core::vectorSIMDf bottomFit = core::floor(fittingVector * float(n) + floorOffset);
		for (uint32_t i = 0u; i < 4u; i++)
		{
			core::vectorSIMDf bottomFitTmp = bottomFit;
			if (i)
				bottomFitTmp += corners[i];
			if ((bottomFitTmp > cubeHalfSize3D).any())
				continue;
			evaluateFit(bottomFitTmp);
		}
	}

	return bestFit;
}

bool CQuantNormalCache::saveCacheToBuffer(const E_CACHE_TYPE type, SBufferBinding<ICPUBuffer>& buffer)
{
	const uint64_t bufferSize = buffer.buffer.get()->getSize();
	const uint64_t offset = buffer.offset;

	if (bufferSize + offset > getSerializedCacheSizeInBytes(type))
	{
		os::Printer::log("Cannot save cache to buffer - not enough space", ELL_ERROR);
		return false;
	}

	switch (type)
	{
	case E_CACHE_TYPE::ECT_2_10_10_10:
	{
		CWriteBufferWrap buffWrap(buffer);
		return normalCacheFor2_10_10_10Quant.dump(buffWrap);
	}
	case E_CACHE_TYPE::ECT_8_8_8:
	{
		CWriteBufferWrap buffWrap(buffer);
		return normalCacheFor8_8_8Quant.dump(buffWrap);
	}
	case E_CACHE_TYPE::ECT_16_16_16:
	{
		CWriteBufferWrap buffWrap(buffer);
		return normalCacheFor16_16_16Quant.dump(buffWrap);
	}
	}

	return false;
}

bool CQuantNormalCache::validateSerializedCache(E_CACHE_TYPE type, const SBufferRange<ICPUBuffer>& buffer)
{
	if (buffer.buffer.get()->getSize() == 0 || buffer.size == 0)
		return true;
	
	const size_t buffSize = buffer.buffer.get()->getSize();
	const uint8_t* buffPtr = static_cast<uint8_t*>(buffer.buffer.get()->getPointer());
	const size_t size = *reinterpret_cast<const size_t*>(buffPtr + buffer.offset);
	const size_t capacity = *reinterpret_cast<const size_t*>(buffPtr + buffer.offset + sizeof(size_t));
	const uint8_t* const bufferRangeEnd = buffPtr + buffer.offset + buffer.size;

	if (bufferRangeEnd > buffPtr + buffSize)
	{
		os::Printer::log("cannot read from this buffer - invalid range", ELL_ERROR);
		return false;
	}

	if (size == 0)
		return true;

	size_t expectedCacheSize = sizeof(size_t) * 2 + 17;
	expectedCacheSize += (type == E_CACHE_TYPE::ECT_16_16_16) ? 17 * capacity : 13 * capacity;

	if ((buffer.offset + buffer.size) == expectedCacheSize)
		return true;

	os::Printer::log("cannot read from this buffer - invalid data", ELL_ERROR);
	return false;
}

}
}