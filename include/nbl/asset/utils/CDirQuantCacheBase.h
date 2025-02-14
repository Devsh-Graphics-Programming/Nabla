// Copyright (C) 2018-2021 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_DIR_QUANT_CACHE_BASE_H_INCLUDED
#define __NBL_ASSET_C_DIR_QUANT_CACHE_BASE_H_INCLUDED


#include <iostream>
#include <limits>
#include <cmath>

#include "parallel-hashmap/parallel_hashmap/phmap_dump.h"


#include "nbl/core/declarations.h"

#include "nbl/system/declarations.h"

#include "nbl/asset/format/EFormat.h"
#include "nbl/asset/ICPUBuffer.h"
#include "nbl/asset/utils/phmap_serialization.h"
#include "nbl/asset/utils/phmap_deserialization.h"
#include "nbl/builtin/hlsl/cpp_compat/vector.hlsl"


namespace nbl 
{
namespace asset 
{

namespace impl
{

class CDirQuantCacheBase
{
	public:
		struct alignas(uint8_t) Vector8u3
		{
			public:
				_NBL_STATIC_INLINE_CONSTEXPR uint32_t quantizationBits = 7u;
				
				Vector8u3() : x(0u),y(0u),z(0u) {}
				Vector8u3(const Vector8u3&) = default;
				explicit Vector8u3(const hlsl::uint32_t4& val)
				{
					operator=(val);
				}

				Vector8u3& operator=(const Vector8u3&) = default;
				Vector8u3& operator=(const hlsl::uint32_t4& val)
				{
					x = val.x;
					y = val.y;
					z = val.z;
					return *this;
				}

				inline hlsl::uint32_t4 getValue() const
				{
					return hlsl::uint32_t4(x,y,z,0);
				}

			private:
				uint8_t x;
				uint8_t y;
				uint8_t z;
		};
		struct alignas(uint32_t) Vector8u4
		{
			public:
				_NBL_STATIC_INLINE_CONSTEXPR uint32_t quantizationBits = 7u;
				
				Vector8u4() : x(0u),y(0u),z(0u),w(0u) {}
				Vector8u4(const Vector8u4&) = default;
				explicit Vector8u4(const hlsl::uint32_t4& val)
				{
					operator=(val);
				}

				Vector8u4& operator=(const Vector8u4&) = default;
				Vector8u4& operator=(const hlsl::uint32_t4& val)
				{
					x = val.x;
					y = val.y;
					z = val.z;
					w = val.w;
					return *this;
				}

				inline hlsl::uint32_t4 getValue() const
				{
					return hlsl::uint32_t4(x,y,z,w);
				}
				
			private:
				uint8_t x;
				uint8_t y;
				uint8_t z;
				uint8_t w;
		};

		struct alignas(uint32_t) Vector1010102
		{
			public:
				_NBL_STATIC_INLINE_CONSTEXPR uint32_t quantizationBits = 9u;

				Vector1010102() : storage(0u) {}
				Vector1010102(const Vector1010102&) = default;
				explicit Vector1010102(const hlsl::uint32_t4& val)
				{
					operator=(val);
				}

				Vector1010102& operator=(const Vector1010102&) = default;
				Vector1010102& operator=(const hlsl::uint32_t4& val)
				{
					constexpr auto storageBits = quantizationBits+1u;
					storage = val.x|(val.y<<storageBits)|(val.z<<(storageBits*2u));
					return *this;
				}

				inline bool operator<(const Vector1010102& other) const
				{
					return storage<other.storage;
				}
				inline bool operator==(const Vector1010102& other) const
				{
					return storage==other.storage;
				}

				inline hlsl::uint32_t4 getValue() const
				{
					constexpr auto storageBits = quantizationBits+1u;
					const hlsl::uint32_t4 mask((0x1u<<storageBits)-1u);
					return hlsl::uint32_t4(storage,storage>>storageBits,storage>>(storageBits*2u),0)&mask;
				}
				
			private:
				uint32_t storage;
		};


		struct alignas(uint16_t) Vector16u3
		{
			public:
				_NBL_STATIC_INLINE_CONSTEXPR uint32_t quantizationBits = 15u;
				
				Vector16u3() : x(0u),y(0u),z(0u) {}
				Vector16u3(const Vector16u3&) = default;
				explicit Vector16u3(const hlsl::uint32_t4& val)
				{
					operator=(val);
				}

				Vector16u3& operator=(const Vector16u3&) = default;
				Vector16u3& operator=(const hlsl::uint32_t4& val)
				{
					x = val.x;
					y = val.y;
					z = val.z;
					return *this;
				}

				inline hlsl::uint32_t4 getValue() const
				{
					return hlsl::uint32_t4(x,y,z,0);
				}
				
			private:
				uint16_t x;
				uint16_t y;
				uint16_t z;
		};
		struct alignas(uint64_t) Vector16u4
		{
			public:
				_NBL_STATIC_INLINE_CONSTEXPR uint32_t quantizationBits = 15u;

				Vector16u4() : x(0u),y(0u),z(0u),w(0u) {}
				Vector16u4(const Vector16u4&) = default;
				explicit Vector16u4(const hlsl::uint32_t4& val)
				{
					operator=(val);
				}

				Vector16u4& operator=(const Vector16u4&) = default;
				Vector16u4& operator=(const hlsl::uint32_t4& val)
				{
					x = val.x;
					y = val.y;
					z = val.z;
					w = val.w;
					return *this;
				}

				inline hlsl::uint32_t4 getValue() const
				{
					return hlsl::uint32_t4(x,y,z,w);
				}
				
			private:
				uint16_t x;
				uint16_t y;
				uint16_t z;
				uint16_t w;
		};


		template<E_FORMAT CacheFormat>
		struct value_type;
};

template<> 
struct CDirQuantCacheBase::value_type<EF_R8G8B8_SNORM>
{
	typedef Vector8u3 type;
};
template<> 
struct CDirQuantCacheBase::value_type<EF_R8G8B8A8_SNORM>
{
	typedef Vector8u4 type;
};

template<> 
struct CDirQuantCacheBase::value_type<EF_A2B10G10R10_SNORM_PACK32>
{
	typedef Vector1010102 type;
};

template<> 
struct CDirQuantCacheBase::value_type<EF_R16G16B16_SNORM>
{
	typedef Vector16u3 type;
};
template<> 
struct CDirQuantCacheBase::value_type<EF_R16G16B16A16_SNORM>
{
	typedef Vector16u4 type;
};

}


template<typename Key, class Hash, E_FORMAT... Formats>
class CDirQuantCacheBase : public impl::CDirQuantCacheBase
{ 
	public:
		template<E_FORMAT CacheFormat>
		using value_type_t = typename impl::CDirQuantCacheBase::value_type<CacheFormat>::type;

		template<E_FORMAT CacheFormat>
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t quantization_bits_v = value_type_t<CacheFormat>::quantizationBits;

		template<E_FORMAT CacheFormat>
		struct cache_type
		{
			using type = core::unordered_map<Key,value_type_t<CacheFormat>,Hash>;
		};
		template<E_FORMAT CacheFormat>
		using cache_type_t = typename cache_type<CacheFormat>::type;

		template<E_FORMAT CacheFormat>
		inline void insertIntoCache(const Key& key, const value_type_t<CacheFormat>& value)
		{
			std::get<cache_type_t<CacheFormat>>(cache).insert(std::make_pair(key,value));		
		}

		//!
		template<E_FORMAT CacheFormat>
		inline bool loadCacheFromBuffer(const SBufferRange<const ICPUBuffer>& buffer, bool replaceCurrentContents = true)
		{
			//additional validation would be nice imo..
			if (!validateSerializedCache<CacheFormat>(buffer))
				return false;

			auto& particularCache = std::get<cache_type_t<CacheFormat>>(cache);
			cache_type_t<CacheFormat> backup;

			if (!replaceCurrentContents)
				backup.swap(particularCache);
			
			CBufferPhmapInputArchive buffWrap(buffer);
			bool loadingSuccess = particularCache.load(buffWrap);

			if (!replaceCurrentContents || !loadingSuccess)
				particularCache.merge(std::move(backup));

			return loadingSuccess;
		}

		//!
		template<E_FORMAT CacheFormat>
		inline bool loadCacheFromFile(system::IFile* file, bool replaceCurrentContents = false)
		{
			if (!file)
				return false;

			auto buffer = asset::ICPUBuffer::create({ file->getSize() });

			system::IFile::success_t succ;
			file->read(succ, buffer->getPointer(), 0, file->getSize());
			if (!succ)
				return false;

			asset::SBufferRange<const asset::ICPUBuffer> bufferRange;
			bufferRange.offset = 0;
			bufferRange.size = file->getSize();
			bufferRange.buffer = std::move(buffer);
			return loadCacheFromBuffer<CacheFormat>(bufferRange, replaceCurrentContents);
		}

		//!
		template<E_FORMAT CacheFormat>
		inline bool loadCacheFromFile(nbl::system::ISystem* system, const system::path& path, bool replaceCurrentContents = false)
		{
			system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
			system->createFile(future,path,nbl::system::IFile::ECF_READ);
			if (auto file=future.acquire())
				return loadCacheFromFile<CacheFormat>(file->get(),replaceCurrentContents);
			return false;
		}

		//!
		template<E_FORMAT CacheFormat>
		inline bool saveCacheToBuffer(SBufferRange<ICPUBuffer>& buffer)
		{
			const uint64_t bufferSize = buffer.buffer.get()->getSize();
			const uint64_t offset = buffer.offset;

			if (bufferSize+offset>getSerializedCacheSizeInBytes<CacheFormat>())
				return false;

			CBufferPhmapOutputArchive buffWrap(buffer);
			return std::get<cache_type_t<CacheFormat>>(cache).dump(buffWrap);
		}

		//!
		template<E_FORMAT CacheFormat>
		inline bool saveCacheToFile(system::IFile* file)
		{
			if (!file)
				return false;

			asset::SBufferRange<asset::ICPUBuffer> bufferRange;
			bufferRange.offset = 0;
			bufferRange.size = getSerializedCacheSizeInBytes<CacheFormat>();
			bufferRange.buffer = asset::ICPUBuffer::create({ bufferRange.size });
		
			saveCacheToBuffer<CacheFormat>(bufferRange);

			system::IFile::success_t succ;
			file->write(succ,bufferRange.buffer->getPointer(), 0, bufferRange.buffer->getSize());
			return bool(succ);
		}

		//!
		template<E_FORMAT CacheFormat>
		inline bool saveCacheToFile(nbl::system::ISystem* system, const system::path& path)
		{
			system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
			system->createFile(future, path, nbl::system::IFile::ECF_WRITE);
			if (auto file=future.acquire())
				return bool(saveCacheToFile<CacheFormat>(file->get()));
			return false;
		}

		//!
		template<E_FORMAT CacheFormat>
		inline size_t getSerializedCacheSizeInBytes()
		{
			return getSerializedCacheSizeInBytes_impl<CacheFormat>(std::get<cache_type_t<CacheFormat>>(cache).capacity());
		}

	protected:
		std::tuple<cache_type_t<Formats>...> cache;
		
		template<uint32_t dimensions, E_FORMAT CacheFormat>
		value_type_t<CacheFormat> quantize(const hlsl::float32_t4& value)
		{
			const auto negativeMask = value < hlsl::float32_t4(0.0f);

			const hlsl::float32_t4 absValue = abs(value);
			const auto key = Key(absValue);

			constexpr auto quantizationBits = quantization_bits_v<CacheFormat>;
			value_type_t<CacheFormat> quantized;
			{
				auto& particularCache = std::get<cache_type_t<CacheFormat>>(cache);
				auto found = particularCache.find(key);
				if (found != particularCache.end() && (found->first == key))
					quantized = found->second;
				else
				{
					const hlsl::float32_t4 fit = findBestFit<dimensions,quantizationBits>(absValue);

					quantized = hlsl::uint32_t4(hlsl::abs(fit));
					insertIntoCache<CacheFormat>(key,quantized);
				}
			}

			const hlsl::uint32_t4 xorflag((0x1u<<(quantizationBits+1u))-1u);
			auto restoredAsVec = quantized.getValue()^hlsl::mix(hlsl::uint32_t4(0u),xorflag,negativeMask);
			restoredAsVec += hlsl::mix(hlsl::uint32_t4(0u),hlsl::uint32_t4(1u),negativeMask);
			return value_type_t<CacheFormat>(restoredAsVec&xorflag);
		}

		template<uint32_t dimensions, uint32_t quantizationBits>
		static inline hlsl::float32_t4 findBestFit(const hlsl::float32_t4& value)
		{
			static_assert(dimensions>1u,"No point");
			static_assert(dimensions<=4u,"High Dimensions are Hard!");
			// precise normalize
			const auto vectorForDots = value / length(value);

			//
			hlsl::float32_t4 fittingVector;
			hlsl::float32_t4 floorOffset;
			constexpr uint32_t cornerCount = (0x1u<<(dimensions-1u))-1u;
			hlsl::float32_t4 corners[cornerCount] = {};
			{
				uint32_t maxDirCompIndex = 0u;
				for (auto i=1u; i<dimensions; i++)
				if (value[i]>value[maxDirCompIndex])
					maxDirCompIndex = i;
				//
				const float maxDirectionComp = value[maxDirCompIndex];
				//max component of 3d normal cannot be less than sqrt(1/D)
				if (maxDirectionComp < std::sqrtf(0.9998f / float(dimensions)))
				{
					_NBL_DEBUG_BREAK_IF(true);
					return hlsl::float32_t4(0.f);
				}
				fittingVector = value / hlsl::float32_t4(maxDirectionComp);
				floorOffset[maxDirCompIndex] = 0.499f;
				const uint32_t localCorner[7][3] = {
					{1,0,0},
					{0,1,0},
					{1,1,0},
					{0,0,1},
					{1,0,1},
					{0,1,1},
					{1,1,1}
				};
				for (auto corn=0u; corn<cornerCount; corn++)
				{
					const auto* coordIt = localCorner[corn];
					for (auto i=0; i<dimensions; i++)
					if (i!=maxDirCompIndex)
						corners[corn][i] = *(coordIt++);
				}
			}

			hlsl::float32_t4 bestFit;
			float closestTo1 = -1.f;
			auto evaluateFit = [&](const hlsl::float32_t4& newFit) -> void
			{
				auto newFitLen = hlsl::length(newFit);
				const float dp = hlsl::dot<hlsl::float32_t4>(newFit,vectorForDots).preciseDivision(newFitLen)[0];
				if (dp > closestTo1)
				{
					closestTo1 = dp;
					bestFit = newFit;
				}
			};

			constexpr uint32_t cubeHalfSize = (0x1u << quantizationBits) - 1u;
			const hlsl::float32_t4 cubeHalfSizeND = hlsl::float32_t4(cubeHalfSize);
			for (uint32_t n=cubeHalfSize; n>0u; n--)
			{
				//we'd use float addition in the interest of speed, to increment the loop
				//but adding a small number to a large one loses precision, so multiplication preferrable
				hlsl::float32_t4 bottomFit = hlsl::floor(fittingVector*float(n)+floorOffset);
				if ((bottomFit<=cubeHalfSizeND).all())
					evaluateFit(bottomFit);
				for (auto i=0u; i<cornerCount; i++)
				{
					auto bottomFitTmp = bottomFit+corners[i];
					if ((bottomFitTmp<=cubeHalfSizeND).all())
						evaluateFit(bottomFitTmp);
				}
			}

			return bestFit;
		}
		
		template<E_FORMAT CacheFormat>
		static inline size_t getSerializedCacheSizeInBytes_impl(size_t capacity)
		{
			return 1u+sizeof(size_t)*2u+phmap::priv::Group::kWidth+(sizeof(typename cache_type_t<CacheFormat>::slot_type)+1u)*capacity;
		}
		template<E_FORMAT CacheFormat>
		static inline bool validateSerializedCache(const SBufferRange<const ICPUBuffer>& buffer)
		{
			if (buffer.offset+buffer.size>buffer.buffer.get()->getSize() || buffer.size<=sizeof(size_t)*2ull)
				return false;
			
			const uint8_t* buffPtr = static_cast<const uint8_t*>(buffer.buffer.get()->getPointer())+buffer.offset;

			const size_t size = reinterpret_cast<const size_t*>(buffPtr)[0];
			const size_t capacity = reinterpret_cast<const size_t*>(buffPtr)[1];
			if (size == 0)
				return true;

			if (buffer.size-sizeof(size_t)*2ull<getSerializedCacheSizeInBytes_impl<CacheFormat>(capacity))
				return false;

			return false;
		}
};

}
}
#endif