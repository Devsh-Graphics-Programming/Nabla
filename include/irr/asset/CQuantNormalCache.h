// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_QUANT_NORMAL_CACHE_H_INCLUDED
#define __NBL_ASSET_C_QUANT_NORMAL_CACHE_H_INCLUDED


#include "irr/core/core.h"

#include "irr/system/system.h"
#include "IReadFile.h"
#include "IWriteFile.h"
#include "IFileSystem.h"

#include "irr/asset/ICPUBuffer.h"


#include "parallel-hashmap/parallel_hashmap/phmap_dump.h"


#include <iostream>
#include <limits>


namespace irr 
{
namespace asset 
{

class CQuantNormalCache
{ 
	public:
		enum class E_CACHE_TYPE
		{
			ECT_2_10_10_10,
			ECT_8_8_8,
			ECT_16_16_16
		};

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
				static constexpr size_t primeNumber1 = 18446744073709551557ull;
				static constexpr size_t primeNumber2 = 4611686018427388273ull;
				
				return  ((static_cast<size_t>(static_cast<double>(vec.u)*(std::numeric_limits<size_t>::max)()) * primeNumber1) ^
					(static_cast<size_t>(static_cast<double>(vec.v)*(std::numeric_limits<size_t>::max)()) * primeNumber2));
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

		template<E_CACHE_TYPE CacheType>
		struct vector_for_cache;
		template<E_CACHE_TYPE CacheType>
		using vector_for_cache_t = typename vector_for_cache<CacheType>::type;

		template<E_CACHE_TYPE CacheType>
		using cached_vector_t = typename vector_for_cache<CacheType>::cachedVecType;

	public:
		template<E_CACHE_TYPE CacheType>
		vector_for_cache_t<CacheType> quantizeNormal(const core::vectorSIMDf& normal)
		{
			uint32_t quantizationBits = 0;
			if constexpr(CacheType == E_CACHE_TYPE::ECT_2_10_10_10)
			{
				quantizationBits = 10u;
			}
		
			if constexpr(CacheType == E_CACHE_TYPE::ECT_8_8_8)
			{
				quantizationBits = 8u;
			}
		
			if constexpr(CacheType == E_CACHE_TYPE::ECT_16_16_16)
			{
				quantizationBits = 16u;
			}
		

			const auto xorflag = core::vectorSIMDu32((0x1u << quantizationBits) - 1u);
			const auto negativeMask = normal < core::vectorSIMDf(0.0f);

			core::vectorSIMDf absNormal = normal;
			absNormal.makeSafe3D();
			absNormal = abs(absNormal);

			const VectorUV uvMappedNormal = mapToBarycentric(absNormal);

			if constexpr(CacheType == E_CACHE_TYPE::ECT_2_10_10_10)
			{
				auto found = normalCacheFor2_10_10_10Quant.find(uvMappedNormal);
				if (found != normalCacheFor2_10_10_10Quant.end() && (found->first == uvMappedNormal))
				{
					const uint32_t absVec = found->second;
					const auto vec = core::vectorSIMDu32(absVec, absVec >> quantizationBits, absVec >> (quantizationBits * 2)) & xorflag;

					return restoreSign<uint32_t>(vec, xorflag, negativeMask, quantizationBits);
				}
			}
		
			if constexpr(CacheType == E_CACHE_TYPE::ECT_8_8_8)
			{
				auto found = normalCacheFor8_8_8Quant.find(uvMappedNormal);
				if (found != normalCacheFor8_8_8Quant.end() && (found->first == uvMappedNormal))
				{
					const auto absVec = core::vectorSIMDu32(found->second.x, found->second.y, found->second.z);

					return restoreSign<uint32_t>(absVec, xorflag, negativeMask, quantizationBits);
				}
			}
		
			if constexpr(CacheType == E_CACHE_TYPE::ECT_16_16_16)
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

			if constexpr(CacheType == E_CACHE_TYPE::ECT_2_10_10_10)
			{
				uint32_t absBestFit = absIntFit[0] | (absIntFit[1] << quantizationBits) | (absIntFit[2] << (quantizationBits * 2u));
				normalCacheFor2_10_10_10Quant.insert(std::make_pair(uvMappedNormal, absBestFit));
			}
		
			if constexpr(CacheType == E_CACHE_TYPE::ECT_8_8_8)
			{
				Vector8u bestFit = { absIntFit[0], absIntFit[1], absIntFit[2] };
				normalCacheFor8_8_8Quant.insert(std::make_pair(uvMappedNormal, bestFit));
			}
		
			if constexpr(CacheType == E_CACHE_TYPE::ECT_16_16_16)
			{
				Vector16u bestFit = { absIntFit[0], absIntFit[1], absIntFit[2] };
				normalCacheFor16_16_16Quant.insert(std::make_pair(uvMappedNormal, bestFit));
			}
		

			return restoreSign<vector_for_cache_t<CacheType>>(absIntFit, xorflag, negativeMask, quantizationBits);
		}

		template<E_CACHE_TYPE CacheType>
		void insertIntoCache(const VectorUV key, cached_vector_t<CacheType> vector)
		{
			if constexpr(CacheType == E_CACHE_TYPE::ECT_2_10_10_10)
			{
				normalCacheFor2_10_10_10Quant.insert(std::make_pair(key, vector));
			}
		
			if constexpr(CacheType == E_CACHE_TYPE::ECT_8_8_8)
			{
				normalCacheFor8_8_8Quant.insert(std::make_pair(key, vector));
			}
		
			if constexpr(CacheType == E_CACHE_TYPE::ECT_16_16_16)
			{
				normalCacheFor16_16_16Quant.insert(std::make_pair(key, vector));
			}
		
		}

		//!
		template<E_CACHE_TYPE CacheType>
		inline bool loadNormalQuantCacheFromBuffer(const SBufferRange<ICPUBuffer>& buffer, bool replaceCurrentContents = true)
		{
			//additional validation would be nice imo..
			if (!validateSerializedCache(CacheType, buffer))
				return false;

			core::unordered_map<VectorUV, vector_for_cache_t<CacheType>, QuantNormalHash, QuantNormalEqualTo> tmp;
			if (!replaceCurrentContents)
			{
				if constexpr (CacheType == E_CACHE_TYPE::ECT_2_10_10_10)
					tmp.swap(normalCacheFor2_10_10_10Quant);
				else if (CacheType == E_CACHE_TYPE::ECT_8_8_8)
					tmp.swap(normalCacheFor8_8_8Quant);
				else if (CacheType == E_CACHE_TYPE::ECT_16_16_16)
					tmp.swap(normalCacheFor16_16_16Quant);
			}
			
			CReadBufferWrap buffWrap(buffer);
		
			bool loadingSucceed;
			if constexpr (CacheType == E_CACHE_TYPE::ECT_2_10_10_10)
				loadingSucceed = normalCacheFor2_10_10_10Quant.load(buffWrap);
			else if (CacheType == E_CACHE_TYPE::ECT_8_8_8)
				loadingSucceed = normalCacheFor8_8_8Quant.load(buffWrap);
			else if (CacheType == E_CACHE_TYPE::ECT_16_16_16)
				loadingSucceed = normalCacheFor16_16_16Quant.load(buffWrap);

			if (!replaceCurrentContents)
			{
				if constexpr (CacheType == E_CACHE_TYPE::ECT_2_10_10_10)
					normalCacheFor2_10_10_10Quant.merge(tmp);
				else if (CacheType == E_CACHE_TYPE::ECT_8_8_8)
					normalCacheFor8_8_8Quant.merge(tmp);
				else if (CacheType == E_CACHE_TYPE::ECT_16_16_16)
					normalCacheFor16_16_16Quant.merge(tmp);
			}

			return loadingSucceed;
		}

		//!
		template<E_CACHE_TYPE CacheType>
		inline bool loadNormalQuantCacheFromFile(io::IReadFile* file, bool replaceCurrentContents = false)
		{
			if (!file)
				return false;

			asset::SBufferRange<asset::ICPUBuffer> bufferRange;
			bufferRange.offset = 0;
			bufferRange.size = file->getSize();
			bufferRange.buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(file->getSize());

			file->read(bufferRange.buffer->getPointer(), bufferRange.size);

			return loadNormalQuantCacheFromBuffer<CacheType>(bufferRange, replaceCurrentContents);
		}

		//!
		template<E_CACHE_TYPE CacheType>
		inline bool loadNormalQuantCacheFromFile(io::IFileSystem* fs, const std::string& path, bool replaceCurrentContents = false)
		{
			auto file = core::smart_refctd_ptr<io::IReadFile>(fs->createAndOpenFile(path.c_str()),core::dont_grab);
			return loadNormalQuantCacheFromFile<CacheType>(file.get(),replaceCurrentContents);
		}

		//!
		bool saveCacheToBuffer(const E_CACHE_TYPE type, SBufferBinding<ICPUBuffer>& buffer);

		//!
		inline bool saveCacheToFile(const E_CACHE_TYPE type, io::IWriteFile* file)
		{
			if (!file)
				return false;

			asset::SBufferBinding<asset::ICPUBuffer> bufferBinding;
			bufferBinding.offset = 0;
			bufferBinding.buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(getSerializedCacheSizeInBytes(type));
		
			saveCacheToBuffer(type, bufferBinding);
			file->write(bufferBinding.buffer->getPointer(), bufferBinding.buffer->getSize());
			return true;
		}
		//!
		inline bool saveCacheToFile(const E_CACHE_TYPE type, io::IFileSystem* fs, const std::string& path)
		{
			auto file = core::smart_refctd_ptr<io::IWriteFile>(fs->createAndWriteFile(path.c_str()));
			return saveCacheToFile(type,file.get());
		}

		//!
		inline size_t getSerializedCacheSizeInBytes(E_CACHE_TYPE type) const
		{
				//there is no way to access values of `Group::kWidth` and `sizeof(slot_type)` outside of phmap::flat_hash_map so these needs to be hardcoded
			//sizeof(_size) + sizeof(_capacity) + Group::kWidth + 1
			size_t cacheSize = sizeof(size_t) * 2 + 17;

			switch (type)
			{
			case E_CACHE_TYPE::ECT_2_10_10_10:
				//sizeof(slot_type) * capacity_ + capacity_
				cacheSize += 13 * normalCacheFor2_10_10_10Quant.capacity();
				return cacheSize;

			case E_CACHE_TYPE::ECT_8_8_8:
				cacheSize += 13 * normalCacheFor8_8_8Quant.capacity();
				return cacheSize;

			case E_CACHE_TYPE::ECT_16_16_16:
				cacheSize += 17 * normalCacheFor16_16_16Quant.capacity();
				return cacheSize;

			}
		}

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

	private:
		bool validateSerializedCache(E_CACHE_TYPE type, const SBufferRange<ICPUBuffer>& buffer);

		class CReadBufferWrap
		{
		public:
			CReadBufferWrap(const SBufferRange<ICPUBuffer>& _buffer)
				:buffer(_buffer)
			{
				buffPtr = static_cast<uint8_t*>(buffer.buffer.get()->getPointer());
			}

			bool load(char* p, size_t sz)
			{
				//TODO: safety
				memcpy(p, buffPtr, sz);
				buffPtr += sz;

				return true;
			}

			template<typename V>
			typename std::enable_if<phmap::type_traits_internal::IsTriviallyCopyable<V>::value, bool>::type
				load(V* v)
			{
				//TODO: safety
				memcpy(reinterpret_cast<uint8_t*>(v), buffPtr, sizeof(V));
				buffPtr += sizeof(V);

				return true;
			}

		private:
			const SBufferRange<ICPUBuffer>& buffer;
			uint8_t* buffPtr;

		};

		class CWriteBufferWrap
		{
		public:
			CWriteBufferWrap(SBufferBinding<ICPUBuffer>& _buffer)
				:buffer(_buffer)
			{
				bufferPtr = static_cast<uint8_t*>(buffer.buffer.get()->getPointer());
			}

			bool dump(const char* p, size_t sz)
			{
				memcpy(bufferPtr, p, sz);
				bufferPtr += sz;

				return true;
			}

			template<typename V>
			typename std::enable_if<phmap::type_traits_internal::IsTriviallyCopyable<V>::value, bool>::type
				dump(const V& v)
			{
				memcpy(bufferPtr, reinterpret_cast<const uint8_t*>(&v), sizeof(V));
				bufferPtr += sizeof(V);

				return true;
			}

		private:
			SBufferBinding<ICPUBuffer>& buffer;
			uint8_t* bufferPtr;

		};
};

// because GCC is a special boy
template<> 
struct CQuantNormalCache::vector_for_cache<CQuantNormalCache::E_CACHE_TYPE::ECT_2_10_10_10>
{
	typedef uint32_t type;
	typedef uint32_t cachedVecType;
};

template<> 
struct CQuantNormalCache::vector_for_cache<CQuantNormalCache::E_CACHE_TYPE::ECT_8_8_8>
{
	typedef uint32_t type;
	typedef Vector8u cachedVecType;
};


template<> 
struct CQuantNormalCache::vector_for_cache<CQuantNormalCache::E_CACHE_TYPE::ECT_16_16_16>
{
	typedef uint64_t type;
	typedef Vector16u cachedVecType;
};

}
}
#endif