#ifndef C_QUANT_NORMAL_CACHE_H_INCLUDED
#define C_QUANT_NORMAL_CACHE_H_INCLUDED

#include "irrlicht.h"

namespace irr 
{
namespace asset 
{

enum class E_QUANT_NORM_CACHE_TYPE
{
	Q_2_10_10_10,
	Q_8_8_8,
	Q_16_16_16
};

class CQuantNormalCache
{ 
public:
	struct VectorUV
	{
		float u;
		float v;

		inline bool operator==(const VectorUV& other) const
		{
			return (u == other.u && v == other.v);
		}
	};

	struct Vector16u
	{
		uint16_t x;
		uint16_t y;
		uint16_t z;
	};

	struct Vector8u
	{
		uint8_t x;
		uint8_t y;
		uint8_t z;
	};

private:
	struct QuantNormalHash
	{
		inline size_t operator()(const VectorUV& vec) const noexcept
		{
			static constexpr size_t primeNumber1 = 18446744073709551557;
			static constexpr size_t primeNumber2 = 4611686018427388273;

			return  ((static_cast<size_t>(static_cast<double>(vec.u)* std::numeric_limits<size_t>::max()) * primeNumber1) ^
				(static_cast<size_t>(static_cast<double>(vec.v)* std::numeric_limits<size_t>::max()) * primeNumber2));
		}
	};

	struct QuantNormalEqualTo
	{
		inline bool operator()(const core::vectorSIMDf& lval, const core::vectorSIMDf& rval) const noexcept
		{
			return (lval == rval).all();
		}

		inline bool operator()(const VectorUV& lval, const VectorUV& rval) const noexcept
		{
			return (lval.u == rval.u && lval.v == rval.v);
		}
	};

	template<E_QUANT_NORM_CACHE_TYPE CacheType>
	struct vector_for_cache;

	template<> 
	struct vector_for_cache<E_QUANT_NORM_CACHE_TYPE::Q_2_10_10_10>
	{
		typedef uint32_t type;
		typedef uint32_t cachedVecType;
	};

	template<> 
	struct vector_for_cache<E_QUANT_NORM_CACHE_TYPE::Q_8_8_8>
	{
		typedef uint32_t type;
		typedef Vector8u cachedVecType;
	};


	template<> 
	struct vector_for_cache<E_QUANT_NORM_CACHE_TYPE::Q_16_16_16>
	{
		typedef uint64_t type;
		typedef Vector16u cachedVecType;
	};

	template<E_QUANT_NORM_CACHE_TYPE CacheType>
	using vector_for_cache_t = typename vector_for_cache<CacheType>::type;

	template<E_QUANT_NORM_CACHE_TYPE CacheType>
	using cached_vector_t = typename vector_for_cache<CacheType>::cachedVecType;

public:
	template<E_QUANT_NORM_CACHE_TYPE CacheType>
	vector_for_cache_t<CacheType> quantizeNormal(const core::vectorSIMDf& normal)
	{
		uint32_t quantizationBits = 0;
		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(CacheType == E_QUANT_NORM_CACHE_TYPE::Q_2_10_10_10)
		{
			quantizationBits = 10u;
		}
		IRR_PSEUDO_IF_CONSTEXPR_END
		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(CacheType == E_QUANT_NORM_CACHE_TYPE::Q_8_8_8)
		{
			quantizationBits = 8u;
		}
		IRR_PSEUDO_IF_CONSTEXPR_END
		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(CacheType == E_QUANT_NORM_CACHE_TYPE::Q_16_16_16)
		{
			quantizationBits = 16u;
		}
		IRR_PSEUDO_IF_CONSTEXPR_END

		const auto xorflag = core::vectorSIMDu32((0x1u << quantizationBits) - 1u);
		const auto negativeMask = normal < core::vectorSIMDf(0.0f);

		core::vectorSIMDf absNormal = normal;
		absNormal.makeSafe3D();
		absNormal = abs(absNormal);

		const VectorUV uvMappedNormal = mapToBarycentric(absNormal);

		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(CacheType == E_QUANT_NORM_CACHE_TYPE::Q_2_10_10_10)
		{
			auto found = normalCacheFor2_10_10_10Quant.find(uvMappedNormal);
			if (found != normalCacheFor2_10_10_10Quant.end() && (found->first == uvMappedNormal))
			{
				const uint32_t absVec = found->second;
				const auto vec = core::vectorSIMDu32(absVec, absVec >> quantizationBits, absVec >> (quantizationBits * 2)) & xorflag;

				return restoreSign<uint32_t>(vec, xorflag, negativeMask, quantizationBits);
			}
		}
		IRR_PSEUDO_IF_CONSTEXPR_END
		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(CacheType == E_QUANT_NORM_CACHE_TYPE::Q_8_8_8)
		{
			auto found = normalCacheFor8_8_8Quant.find(uvMappedNormal);
			if (found != normalCacheFor8_8_8Quant.end() && (found->first == uvMappedNormal))
			{
				const auto absVec = core::vectorSIMDu32(found->second.x, found->second.y, found->second.z);

				return restoreSign<uint32_t>(absVec, xorflag, negativeMask, quantizationBits);
			}
		}
		IRR_PSEUDO_IF_CONSTEXPR_END
		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(CacheType == E_QUANT_NORM_CACHE_TYPE::Q_16_16_16)
		{
			auto found = normalCacheFor16_16_16Quant.find(uvMappedNormal);
			if (found != normalCacheFor16_16_16Quant.end() && (found->first == uvMappedNormal))
			{
				const auto absVec = core::vectorSIMDu32(found->second.x, found->second.y, found->second.z);

				return restoreSign<uint64_t>(absVec, xorflag, negativeMask, quantizationBits);
			}
		}

		
		core::vectorSIMDf fit = findBestFit(quantizationBits, absNormal);

		auto absIntFit = core::vectorSIMDu32(core::abs(fit)) & xorflag;

		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(CacheType == E_QUANT_NORM_CACHE_TYPE::Q_2_10_10_10)
		{
			uint32_t absBestFit = absIntFit[0] | (absIntFit[1] << quantizationBits) | (absIntFit[2] << (quantizationBits * 2u));
			normalCacheFor2_10_10_10Quant.insert(std::make_pair(uvMappedNormal, absBestFit));
		}
		IRR_PSEUDO_IF_CONSTEXPR_END
		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(CacheType == E_QUANT_NORM_CACHE_TYPE::Q_8_8_8)
		{
			Vector8u bestFit = { absIntFit[0], absIntFit[1], absIntFit[2] };
			normalCacheFor8_8_8Quant.insert(std::make_pair(uvMappedNormal, bestFit));
		}
		IRR_PSEUDO_IF_CONSTEXPR_END
		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(CacheType == E_QUANT_NORM_CACHE_TYPE::Q_16_16_16)
		{
			Vector16u bestFit = { absIntFit[0], absIntFit[1], absIntFit[2] };
			normalCacheFor16_16_16Quant.insert(std::make_pair(uvMappedNormal, bestFit));
		}
		IRR_PSEUDO_IF_CONSTEXPR_END

		return restoreSign<vector_for_cache_t<CacheType>>(absIntFit, xorflag, negativeMask, quantizationBits);
	}

	template<E_QUANT_NORM_CACHE_TYPE CacheType>
	void insertIntoCache(const VectorUV key, cached_vector_t<CacheType> vector)
	{
		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(CacheType == E_QUANT_NORM_CACHE_TYPE::Q_2_10_10_10)
		{
			normalCacheFor2_10_10_10Quant.insert(std::make_pair(key, vector));
		}
		IRR_PSEUDO_IF_CONSTEXPR_END
		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(CacheType == E_QUANT_NORM_CACHE_TYPE::Q_8_8_8)
		{
			normalCacheFor8_8_8Quant.insert(std::make_pair(key, vector));
		}
		IRR_PSEUDO_IF_CONSTEXPR_END
		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(CacheType == E_QUANT_NORM_CACHE_TYPE::Q_16_16_16)
		{
			normalCacheFor16_16_16Quant.insert(std::make_pair(key, vector));
		}
		IRR_PSEUDO_IF_CONSTEXPR_END
	}

	//!
	template<E_QUANT_NORM_CACHE_TYPE CacheType>
	bool loadNormalQuantCacheFromBuffer(const SBufferRange<ICPUBuffer>& buffer, bool replaceCurrentContents = false)
	{
		//additional validation would be nice imo..

		const uint64_t bufferSize = buffer.buffer.get()->getSize();
		const uint64_t offset = buffer.offset;
		const size_t cacheElementSize = getCacheElementSize(CacheType);

		uint8_t* buffPointer = static_cast<uint8_t*>(buffer.buffer.get()->getPointer());
		const uint8_t* const bufferRangeEnd = buffPointer + offset + buffer.size;

		if (bufferRangeEnd > buffPointer + bufferSize)
		{
			os::Printer::log("cannot read from this buffer - invalid range", ELL_ERROR);
			return false;
		}

		if (replaceCurrentContents)
		{
			IRR_PSEUDO_IF_CONSTEXPR_BEGIN(CacheType == E_QUANT_NORM_CACHE_TYPE::Q_2_10_10_10)
			{
				normalCacheFor2_10_10_10Quant.clear();
			}
			IRR_PSEUDO_IF_CONSTEXPR_END
				IRR_PSEUDO_IF_CONSTEXPR_BEGIN(CacheType == E_QUANT_NORM_CACHE_TYPE::Q_8_8_8)
			{
				normalCacheFor8_8_8Quant.clear();
			}
			IRR_PSEUDO_IF_CONSTEXPR_END
				IRR_PSEUDO_IF_CONSTEXPR_BEGIN(CacheType == E_QUANT_NORM_CACHE_TYPE::Q_16_16_16)
			{
				normalCacheFor16_16_16Quant.clear();
			}
			IRR_PSEUDO_IF_CONSTEXPR_END
		}

		const size_t quantVecSize = sizeof(cached_vector_t<CacheType>);

		buffPointer += offset;
		while (buffPointer < bufferRangeEnd)
		{
			CQuantNormalCache::VectorUV key{ *reinterpret_cast<float*>(buffPointer),* reinterpret_cast<float*>(buffPointer + sizeof(float)) };
			buffPointer += sizeof(CQuantNormalCache::VectorUV);

			cached_vector_t<CacheType> vec;
			memcpy(&vec, buffPointer, quantVecSize);
			buffPointer += quantVecSize;

			insertIntoCache<CacheType>(key, vec);
		}

		return true;
	}

	//!
	bool saveCacheToBuffer(const E_QUANT_NORM_CACHE_TYPE type, SBufferBinding<ICPUBuffer>& buffer);

	inline size_t getCacheSizeInBytes(E_QUANT_NORM_CACHE_TYPE type) const
	{
		constexpr size_t normCacheElementSize2_10_10_10 = sizeof(VectorUV) + sizeof(uint32_t);
		constexpr size_t normCacheElementSize8_8_8      = sizeof(VectorUV) + sizeof(Vector8u);
		constexpr size_t normCacheElementSize16_16_16   = sizeof(VectorUV) + sizeof(Vector16u);

		switch (type)
		{
		case E_QUANT_NORM_CACHE_TYPE::Q_2_10_10_10:
			return normalCacheFor2_10_10_10Quant.size() * normCacheElementSize2_10_10_10;

		case E_QUANT_NORM_CACHE_TYPE::Q_8_8_8:
			return normalCacheFor8_8_8Quant.size() * normCacheElementSize8_8_8;

		case E_QUANT_NORM_CACHE_TYPE::Q_16_16_16:
			return normalCacheFor16_16_16Quant.size() * normCacheElementSize16_16_16;
		}
	}

	inline size_t getCacheElementSize(E_QUANT_NORM_CACHE_TYPE type) const
	{
		switch (type)
		{
		case E_QUANT_NORM_CACHE_TYPE::Q_2_10_10_10:
			return sizeof(VectorUV) + sizeof(uint32_t);

		case E_QUANT_NORM_CACHE_TYPE::Q_8_8_8:
			return sizeof(VectorUV) + sizeof(Vector8u);

		case E_QUANT_NORM_CACHE_TYPE::Q_16_16_16:
			return sizeof(VectorUV) + sizeof(Vector16u);
		}
	}

	inline auto& getCache2_10_10_10() { return normalCacheFor2_10_10_10Quant; }
	inline auto& getCache8_8_8()      { return normalCacheFor8_8_8Quant; }
	inline auto& getCache16_16_16()   { return normalCacheFor16_16_16Quant; }

private:
	inline VectorUV mapToBarycentric(const core::vectorSIMDf& vec) const
	{
		//normal to A = [1,0,0], B = [0,1,0], C = [0,0,1] triangle
		static const core::vector3df_SIMD n(0.577f, 0.577f, 0.577f);

		//point of intersection of vec and triangle - ( n dot A ) / (n dot vec) ) * vec
		const float r = 0.577f / core::dot(n, vec).x;

		//[0, 1, 0] + u * [1, -1, 0] + v * [0, -1, 1] = P, so u = Px and v = Pz
		return { r * vec.x, r * vec.z };
	}

	template <typename T>
	inline T restoreSign(const core::vectorSIMDu32& vec, const core::vectorSIMDu32& xorflag, const core::vector4db_SIMD& negativeMask, const uint32_t quantizationBits) const 
	{
		static_assert(std::is_same<T, uint32_t>::value || std::is_same<T, uint64_t>::value, "Type of returned value should be uint32_t or uint64_t.");

		auto restoredAsVec = core::vectorSIMDu32(vec) ^ core::mix(core::vectorSIMDu32(0u), xorflag, negativeMask);
		restoredAsVec = (restoredAsVec + core::mix(core::vectorSIMDu32(0u), core::vectorSIMDu32(1u), negativeMask)) & xorflag;

		T restoredAsInt = restoredAsVec[0] | (restoredAsVec[1] << quantizationBits) | (restoredAsVec[2] << (quantizationBits * 2u));

		return restoredAsInt;
	}

	core::vectorSIMDf findBestFit(const uint32_t& bits, const core::vectorSIMDf& normal) const;

private:
	core::unordered_map<VectorUV, uint32_t, QuantNormalHash, QuantNormalEqualTo> normalCacheFor2_10_10_10Quant;
	core::unordered_map<VectorUV, Vector8u, QuantNormalHash, QuantNormalEqualTo> normalCacheFor8_8_8Quant;
	core::unordered_map<VectorUV, Vector16u, QuantNormalHash, QuantNormalEqualTo> normalCacheFor16_16_16Quant;
};

}
}
#endif