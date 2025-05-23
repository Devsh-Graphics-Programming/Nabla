// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_ANIMATION_LIBRARY_H_INCLUDED_
#define _NBL_ASSET_I_CPU_ANIMATION_LIBRARY_H_INCLUDED_

#include "nbl/asset/IAnimationLibrary.h"
#include "nbl/asset/ICPUBuffer.h"

namespace nbl::asset
{

class ICPUAnimationLibrary final : public IAnimationLibrary<ICPUBuffer>, public IAsset
{
	public:
		using base_t = IAnimationLibrary<ICPUBuffer>;

		template<typename... Args>
		inline ICPUAnimationLibrary(Args&&... args) : base_t(std::forward<Args>(args)...) {}

		//
		inline const SBufferBinding<ICPUBuffer>& getKeyframeStorageBinding() const
		{
			return m_keyframeStorageBinding;
		}
		inline const SBufferBinding<ICPUBuffer>& getTimestampStorageBinding() const
		{
			return m_timestampStorageBinding;
		}

		//
		inline const SBufferRange<ICPUBuffer>& getAnimationStorageRange() const
		{
			return m_animationStorageRange;
		}

		//!
		inline const Keyframe& getKeyframe(uint32_t keyframeOffset) const
		{
			const uint8_t* ptr = reinterpret_cast<const uint8_t*>(m_keyframeStorageBinding.buffer->getPointer());
			return reinterpret_cast<const Keyframe*>(ptr+m_keyframeStorageBinding.offset)[keyframeOffset];
		}
		inline Keyframe& getKeyframe(uint32_t keyframeOffset)
		{
			assert(isMutable());
			return const_cast<Keyframe&>(const_cast<const ICPUAnimationLibrary*>(this)->getKeyframe(keyframeOffset));
		}
		//!
		inline const timestamp_t& getTimestamp(uint32_t keyframeOffset) const
		{
			const uint8_t* ptr = reinterpret_cast<const uint8_t*>(m_timestampStorageBinding.buffer->getPointer());
			return reinterpret_cast<const timestamp_t*>(ptr+m_timestampStorageBinding.offset)[keyframeOffset];
		}
		inline timestamp_t& getTimestamp(uint32_t keyframeOffset)
		{
			assert(isMutable());
			return const_cast<timestamp_t&>(const_cast<const ICPUAnimationLibrary*>(this)->getTimestamp(keyframeOffset));
		}

		//!
		inline const Animation& getAnimation(uint32_t animationOffset) const
		{
			const uint8_t* ptr = reinterpret_cast<const uint8_t*>(m_animationStorageRange.buffer->getPointer());
			return reinterpret_cast<const Animation*>(ptr+m_animationStorageRange.offset)[animationOffset];
		}
		inline Animation& getAnimation(uint32_t animationOffset)
		{
			assert(isMutable());
			return const_cast<Animation&>(const_cast<const ICPUAnimationLibrary*>(this)->getAnimation(animationOffset));
		}

		//!
		template<typename NameIterator, typename OffsetIterator>
		inline void addAnimationNames(NameIterator nameBegin, NameIterator nameEnd, OffsetIterator offsetBegin)
		{
			base_t::addAnimationNames(nameBegin,nameEnd,offsetBegin);
		}
		inline void clearAnimationNames()
		{
			base_t::clearAnimationNames();
		}

		core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
		{
			SBufferBinding<ICPUBuffer> _keyframeStorageBinding = {m_keyframeStorageBinding.offset,_depth>0u ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_keyframeStorageBinding.buffer->clone(_depth-1u)):m_keyframeStorageBinding.buffer};
			SBufferBinding<ICPUBuffer> _timestampStorageBinding = {m_timestampStorageBinding.offset,_depth>0u ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_timestampStorageBinding.buffer->clone(_depth-1u)):m_timestampStorageBinding.buffer};

			SBufferRange<ICPUBuffer> _animationStorageRange = {m_animationStorageRange.offset,m_animationStorageRange.size,_depth>0u&&m_animationStorageRange.buffer ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_animationStorageRange.buffer->clone(_depth-1u)):core::smart_refctd_ptr(m_animationStorageRange.buffer)};

 			auto cp = core::make_smart_refctd_ptr<ICPUAnimationLibrary>(std::move(_keyframeStorageBinding),std::move(_timestampStorageBinding),m_keyframeCount,std::move(_animationStorageRange));
			cp->setAnimationNames(this);

			return cp;
		}

		constexpr static inline auto AssetType = ET_ANIMATION_LIBRARY;
		inline E_TYPE getAssetType() const override { return AssetType; }

    inline core::unordered_set<const IAsset*> computeDependants() const override
		{
			return { m_keyframeStorageBinding.buffer.get(), m_timestampStorageBinding.buffer.get(), m_animationStorageRange.buffer.get() };
		}

  private:

    template <typename Self>
      requires(std::same_as<std::remove_cv_t<Self>, ICPUAnimationLibrary>)
    static auto computeDependantsImpl(Self* self) {
        using asset_ptr_t = std::conditional_t<std::is_const_v<Self>, const IAsset*, IAsset*>;
        return core::unordered_set<asset_ptr_t>{ self->m_keyframeStorageBinding.buffer.get(), self->m_timestampStorageBinding.buffer.get(), self->m_animationStorageRange.buffer.get() };
    }
};

}
#endif
