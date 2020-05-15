#include "irr/asset/CQuantNormalCache.h"
#include "parallel-hashmap/parallel_hashmap/phmap_dump.h"

namespace irr
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
		_IRR_DEBUG_BREAK_IF(true);
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

bool CQuantNormalCache::saveCacheToBuffer(const E_QUANT_NORM_CACHE_TYPE type, SBufferBinding<ICPUBuffer>& buffer)
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
	case E_QUANT_NORM_CACHE_TYPE::Q_2_10_10_10:
	{
		CWriteBufferWrap buffWrap(buffer);
		return normalCacheFor2_10_10_10Quant.dump(buffWrap);
	}
	case E_QUANT_NORM_CACHE_TYPE::Q_8_8_8:
	{
		CWriteBufferWrap buffWrap(buffer);
		return normalCacheFor8_8_8Quant.dump(buffWrap);
	}
	case E_QUANT_NORM_CACHE_TYPE::Q_16_16_16:
	{
		CWriteBufferWrap buffWrap(buffer);
		return normalCacheFor16_16_16Quant.dump(buffWrap);
	}
	}

	return false;
}

}
}