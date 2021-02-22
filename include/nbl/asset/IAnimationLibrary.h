// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_ANIMATION_LIBRARY_H_INCLUDED__
#define __NBL_ASSET_I_ANIMATION_LIBRARY_H_INCLUDED__

#include "nbl/macros.h"

#include "nbl/core/core.h"
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
		struct alignas(8) Keyframe
		{
				Keyframe() : quat(), scale(0ull)
				{
					translation[2] = translation[1] = translation[0] = 0.f;
				}
				Keyframe(const core::vectorSIMDf& _scale, const core::quaternion& _quat, const CQuantQuaternionCache* quantCache, const core::vectorSIMDf& _translation)
				{
					std::copy(translation,_translation.pointer,3u);
					quat = quantCache->quantize(_quat);
					// TODO: encode to RGB18E7S3
					//scale = ;
				}

				/*
				inline core::quaternion getRotation() const
				{
					return quat.getValue(); // TODO: decode from RGBA8_SNORM and normalize
				}
				*/

				inline core::vectorSIMDf getScale() const
				{
					return core::vectorSIMDf(0.f/0.f); // TODO: decode from RGB18E7S3
				}

			private:
				float translation[3];
				CDirQuantCacheBase::Vector8u4 quat;
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
				inline uint32_t getKeyframeOffset() const
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

		inline const auto& getNameToAnimationMap() const
		{
			return m_nameToAnimation;
		}
		inline uint32_t getAnimationOffsetFromName(const char* animationName) const
		{
			auto found = m_nameToAnimation.find(animationName);
			if (found != m_nameToAnimation.end())
				return found->second;
			return getAnimationCapacity();
		}

		inline uint32_t getAnimationCapacity() const
		{
			return m_animationStorageRange.size()/sizeof(Animation);
		}

		inline const SBufferRange<const BufferType>& getKeyframeStorageRange() const
		{
			return reinterpret_cast<const SBufferRange<const BufferType>*>(m_keyframeStorageRange);
		}
		inline const SBufferRange<const BufferType>& getAnimationStorageRange() const
		{
			return reinterpret_cast<const SBufferRange<const BufferType>*>(m_animationStorageRange);
		}


	protected:
		IAnimationLibrary(SBufferRange<BufferType>&& _keyframeStorageRange, SBufferRange<BufferType>&& _animationStorageRange) :
			m_stringPool(), m_nameToAnimation(StringComparator(&m_stringPool)), m_keyframeStorageRange(std::move(_keyframeStorageRange)),
			m_animationStorageRange(std::move(_animationStorageRange))
		{
			assert(m_keyframeStorageRange.isValid() && (m_keyframeStorageRange.offset%sizeof(Keyframe)==0u) && m_keyframeStorageRange.size>=sizeof(Keyframe));
			assert(m_animationStorageRange.isValid() && (m_animationStorageRange.offset%sizeof(Animation)==0u) && m_animationStorageRange.size>=sizeof(Animation));
		}
		virtual ~IAnimationLibrary()
		{
			m_nameToAnimation.clear();
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
		inline clearAnimationNames()
		{
			m_nameToAnimation.clear();
			m_stringPool.clear();
		}

		struct StringComparator
		{
				StringComparator(const core::vector<char>* const _stringPool) : stringPool(_stringPool) {}

				inline bool operator()(const uint32_t lhs, const uint32_t rhs) const
				{
					return strcmp(stringPool.data()+lhs,stringPool.data()+rhs)<0;
				}

			private:
				core::vector<char>* const stringPool;
		};
		core::vector<char> m_stringPool;
		core::map<uint32_t,uint32_t,StringComparator> m_nameToAnimation;

		SBufferRange<BufferType> m_keyframeStorageRange,m_animationStorageRange;
};

} // end namespace asset
} // end namespace nbl

#endif

