// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_ANIMATION_LIBRARY_H_INCLUDED__
#define __NBL_ASSET_I_ANIMATION_LIBRARY_H_INCLUDED__

#include "nbl/macros.h"

#include "nbl/core/declarations.h"
#include "nbl/asset/utils/CQuantQuaternionCache.h"

namespace nbl
{
namespace asset
{

//! Class which holds the keyframes for animations.
/** An IAnimationLibrary is nothing more than a Structure of Arrays style collection of
* named keyframe ranges.
*/
template <class BufferType>
class IAnimationLibrary : public virtual core::IReferenceCounted
{
	public:
		using keyframe_t = uint32_t;
		using animation_t = uint32_t;
		using timestamp_t = uint32_t;

		struct alignas(8) Keyframe
		{
				Keyframe() : scale(0ull) // TODO: initialize scale to 1.f
				{
					translation[2] = translation[1] = translation[0] = 0.f;
					quat = core::vectorSIMDu32(128u,128u,128u,255u); // should be (0,0,0,1) encoded
				}
				Keyframe(const core::vectorSIMDf& _scale, const core::quaternion& _quat, const CQuantQuaternionCache* quantCache, const core::vectorSIMDf& _translation)
				{
					std::copy(_translation.pointer,_translation.pointer+3,translation);
					quat = quantCache->template quantize<decltype(quat)>(_quat);
					// TODO: encode to RGB18E7S3
					//scale = ;
				}

				inline core::quaternion getRotation() const
				{
					const void* _pix[4] = {&quat,nullptr,nullptr,nullptr};
					double out[4];
					decodePixels<EF_R8G8B8A8_SNORM,double>(_pix,out,0u,0u);
					auto q = core::normalize(core::vectorSIMDf(out[0],out[1],out[2],out[3]));
					return reinterpret_cast<const core::quaternion*>(&q)[0];
				}

				inline core::vectorSIMDf getScale() const
				{
					return core::vectorSIMDf(0.f/0.f); // TODO: decode from RGB18E7S3
				}

			private:
				float translation[3];
				CQuantQuaternionCache::Vector8u4 quat;
				uint64_t scale;
		};
		struct alignas(8) Animation
		{
				enum E_INTERPOLATION_MODE : uint32_t
				{
					EIM_NEAREST=0u,
					EIM_LINEAR=1u<<30u,
					EIM_CUBIC=2u<<30u,
					EIM_MASK=3u<<30u
				};

				inline Animation()
				{
					data[0] = data[1] = 0xffffffffu;
				}
				inline Animation(const keyframe_t keyframeOffset, const uint32_t keyframeCount, const E_INTERPOLATION_MODE interpolation)
				{
					data[0] = keyframeOffset;
					assert(keyframeCount<0x4fffffffu);
					data[1] = keyframeCount|interpolation;
				}

				inline keyframe_t getKeyframeOffset() const
				{
					return data[0];
				}
				inline auto getTimestampOffset() const
				{
					return getKeyframeOffset();
				}
				inline uint32_t getKeyframeCount() const
				{
					return data[1]&(~EIM_MASK);
				}
				inline E_INTERPOLATION_MODE getInterpolationMode() const
				{
					return data[1]&EIM_MASK;
				}

			private:
				uint32_t data[2];
		};

		//
		inline const SBufferBinding<const BufferType>& getKeyframeStorageBinding() const
		{
			return reinterpret_cast<const SBufferBinding<const BufferType>&>(m_keyframeStorageBinding);
		}
		inline const SBufferBinding<const BufferType>& getTimestampStorageBinding() const
		{
			return reinterpret_cast<const SBufferBinding<const BufferType>&>(m_timestampStorageBinding);
		}

		//
		inline const SBufferRange<const BufferType>& getAnimationStorageRange() const
		{
			return reinterpret_cast<const SBufferRange<const BufferType>&>(m_animationStorageRange);
		}

		//
		inline uint32_t getAnimationCapacity() const
		{
			return m_animationStorageRange.actualSize()/sizeof(Animation);
		}

		inline animation_t getAnimationOffsetFromName(const char* animationName) const
		{
			m_temporaryString = animationName;
			auto found = m_nameToAnimation.find(0xffffffffu);
			m_temporaryString = nullptr;
			if (found != m_nameToAnimation.end())
				return found->second;
			return getAnimationCapacity();
		}


	protected:
		IAnimationLibrary(SBufferBinding<BufferType>&& _keyframeStorageBinding, SBufferBinding<BufferType>&& _timestampStorageBinding, uint32_t _keyframeCount, SBufferRange<BufferType>&& _animationStorageRange) :
			m_stringPool(), m_temporaryString(nullptr), m_nameToAnimation(StringComparator(&m_stringPool,&m_temporaryString)),
			m_keyframeStorageBinding(std::move(_keyframeStorageBinding)), m_timestampStorageBinding(std::move(_timestampStorageBinding)),
			m_animationStorageRange(std::move(_animationStorageRange)), m_keyframeCount(_keyframeCount)
		{
			assert(m_keyframeStorageBinding.buffer && m_keyframeStorageBinding.offset+sizeof(Keyframe)*m_keyframeCount<=m_keyframeStorageBinding.buffer->getSize());
			assert(m_timestampStorageBinding.buffer && m_timestampStorageBinding.offset+sizeof(timestamp_t)*m_keyframeCount<=m_timestampStorageBinding.buffer->getSize());
			
			if (!m_animationStorageRange.isValid())
				return;
			assert((m_animationStorageRange.offset%sizeof(Animation)==0u) && m_animationStorageRange.actualSize() >= sizeof(Animation));
		}
		virtual ~IAnimationLibrary()
		{
			clearAnimationNames();
		}

		template <typename>
		friend struct IAnimationLibrary;
		
		// map must contain one `const char*` per bone
		template<class OtherBufferType>
		inline void setAnimationNames(const IAnimationLibrary<OtherBufferType>* other)
		{
			m_stringPool = other->m_stringPool;
			m_nameToAnimation.clear();
			m_nameToAnimation.insert(other->m_nameToAnimation.begin(),other->m_nameToAnimation.end());
		}
		//
		template<typename NameIterator, typename OffsetIterator>
		inline void addAnimationNames(NameIterator nameBegin, NameIterator nameEnd, OffsetIterator offsetBegin)
		{
			// size the pool
			size_t extraChars = 0ull;
			for (auto it=nameBegin; it!=nameEnd; it++)
			{
				const auto nameLen = strlen(*it);
				if (nameLen)
					extraChars += nameLen+1ull;
			}
			size_t stringPoolEnd = m_stringPool.size();
			m_stringPool.resize(stringPoolEnd+extraChars);
				
			// useless names
			if (extraChars==0ull)
				return;
				
			auto offsetIt = offsetBegin;
			for (auto it=nameBegin; it!=nameEnd; it++,offsetIt++)
			{
				const char* inName = *it;
				const size_t newNameBegin = stringPoolEnd;
				while (*inName) { m_stringPool[stringPoolEnd++] = *(inName++); }
				if (stringPoolEnd!=newNameBegin)
				{
					m_stringPool[stringPoolEnd++] = 0;
					m_nameToAnimation.emplace(newNameBegin,*offsetIt);
				}
			}
		}
		//
		inline void clearAnimationNames()
		{
			m_nameToAnimation.clear();
			m_stringPool.clear();
		}

		struct StringComparator
		{
				StringComparator(const core::vector<char>* const _stringPool, const char* const* _temporaryString) : stringPool(_stringPool), temporaryString(_temporaryString) {}

				inline bool operator()(const uint32_t _lhs, const uint32_t _rhs) const
				{
					const char* lhs = _lhs!=0xffffffffu ? (stringPool->data()+_lhs):(*temporaryString);
					const char* rhs = _rhs!=0xffffffffu ? (stringPool->data()+_rhs):(*temporaryString);
					return strcmp(lhs,rhs)<0;
				}

			private:
				const core::vector<char>* stringPool;
				const char* const* temporaryString;
		};
		core::vector<char> m_stringPool;
		mutable const char* m_temporaryString;
		core::map<uint32_t,animation_t,StringComparator> m_nameToAnimation;

		SBufferBinding<BufferType> m_keyframeStorageBinding,m_timestampStorageBinding;
		SBufferRange<BufferType> m_animationStorageRange;
		uint32_t m_keyframeCount;
};

} // end namespace asset
} // end namespace nbl

#endif

