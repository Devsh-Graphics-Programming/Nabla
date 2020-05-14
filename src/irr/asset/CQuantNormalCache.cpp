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

	if (bufferSize + offset > getCacheSizeInBytes(type))
	{
		os::Printer::log("Cannot save cache to buffer - not enough space", ELL_ERROR);
		return false;
	}

	uint8_t* buffPointer = static_cast<uint8_t*>(buffer.buffer.get()->getPointer()) + offset;

	switch (type)
	{
	case E_QUANT_NORM_CACHE_TYPE::Q_2_10_10_10:
	{
		auto cache = getCache2_10_10_10();

		for (auto it = cache.begin(); it != cache.end(); it++)
		{
			memcpy(buffPointer, &(it->first), sizeof(CQuantNormalCache::VectorUV));
			buffPointer += sizeof(CQuantNormalCache::VectorUV);

			memcpy(buffPointer, &(it->second), sizeof(uint32_t));
			buffPointer += sizeof(uint32_t);
		}

		return true;
	}
	case E_QUANT_NORM_CACHE_TYPE::Q_8_8_8:
	{
		auto cache = getCache8_8_8();

		for (auto it = cache.begin(); it != cache.end(); it++)
		{
			memcpy(buffPointer, &(it->first), sizeof(CQuantNormalCache::VectorUV));
			buffPointer += sizeof(CQuantNormalCache::VectorUV);

			memcpy(buffPointer, &(it->second), sizeof(CQuantNormalCache::Vector8u));
			buffPointer += sizeof(CQuantNormalCache::Vector8u);
		}

		return true;
	}
	case E_QUANT_NORM_CACHE_TYPE::Q_16_16_16:
	{
		auto cache = getCache16_16_16();

		for (auto it = cache.begin(); it != cache.end(); it++)
		{
			memcpy(buffPointer, &(it->first), sizeof(CQuantNormalCache::VectorUV));
			buffPointer += sizeof(CQuantNormalCache::VectorUV);

			memcpy(buffPointer, &(it->second), sizeof(CQuantNormalCache::Vector16u));
			buffPointer += sizeof(CQuantNormalCache::Vector16u);
		}

		return true;
	}
	}

	return false;
}

}
}