// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_CPU_ANIMATION_LIBRARY_H_INCLUDED__
#define __NBL_ASSET_I_CPU_ANIMATION_LIBRARY_H_INCLUDED__

#include "nbl/asset/IAnimationLibrary.h"
#include "nbl/asset/ICPUBuffer.h"

namespace nbl
{
namespace asset
{
class ICPUAnimationLibrary final : public IAnimationLibrary<ICPUBuffer>, /*TODO: public BlobSerializable, */ public IAsset
{
public:
    using base_t = IAnimationLibrary<ICPUBuffer>;

    template<typename... Args>
    inline ICPUAnimationLibrary(Args&&... args)
        : base_t(std::forward<Args>(args)...) {}

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
        return reinterpret_cast<const Keyframe*>(ptr + m_keyframeStorageBinding.offset)[keyframeOffset];
    }
    inline Keyframe& getKeyframe(uint32_t keyframeOffset)
    {
        assert(!isImmutable_debug());
        return const_cast<Keyframe&>(const_cast<const ICPUAnimationLibrary*>(this)->getKeyframe(keyframeOffset));
    }
    //!
    inline const timestamp_t& getTimestamp(uint32_t keyframeOffset) const
    {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(m_timestampStorageBinding.buffer->getPointer());
        return reinterpret_cast<const timestamp_t*>(ptr + m_timestampStorageBinding.offset)[keyframeOffset];
    }
    inline timestamp_t& getTimestamp(uint32_t keyframeOffset)
    {
        assert(!isImmutable_debug());
        return const_cast<timestamp_t&>(const_cast<const ICPUAnimationLibrary*>(this)->getTimestamp(keyframeOffset));
    }

    //!
    inline const Animation& getAnimation(uint32_t animationOffset) const
    {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(m_animationStorageRange.buffer->getPointer());
        return reinterpret_cast<const Animation*>(ptr + m_animationStorageRange.offset)[animationOffset];
    }
    inline Animation& getAnimation(uint32_t animationOffset)
    {
        assert(!isImmutable_debug());
        return const_cast<Animation&>(const_cast<const ICPUAnimationLibrary*>(this)->getAnimation(animationOffset));
    }

    //!
    template<typename NameIterator, typename OffsetIterator>
    inline void addAnimationNames(NameIterator nameBegin, NameIterator nameEnd, OffsetIterator offsetBegin)
    {
        base_t::addAnimationNames(nameBegin, nameEnd, offsetBegin);
    }
    inline void clearAnimationNames()
    {
        base_t::clearAnimationNames();
    }

    //! Serializes animation library to blob for *.nbl file format.
    /** @param _stackPtr Optional pointer to stack memory to write blob on. If _stackPtr==NULL, sufficient amount of memory will be allocated.
			@param _stackSize Size of stack memory pointed by _stackPtr.
			@returns Pointer to memory on which blob was written.
		* TODO
		virtual void* serializeToBlob(void* _stackPtr = NULL, const size_t& _stackSize = 0) const override
		{
			return CorrespondingBlobTypeFor<ICPUAnimationLibrary>::type::createAndTryOnStack(this, _stackPtr, _stackSize);
		}
		*/

    core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
    {
        SBufferBinding<ICPUBuffer> _keyframeStorageBinding = {m_keyframeStorageBinding.offset, _depth > 0u ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_keyframeStorageBinding.buffer->clone(_depth - 1u)) : m_keyframeStorageBinding.buffer};
        SBufferBinding<ICPUBuffer> _timestampStorageBinding = {m_timestampStorageBinding.offset, _depth > 0u ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_timestampStorageBinding.buffer->clone(_depth - 1u)) : m_timestampStorageBinding.buffer};

        SBufferRange<ICPUBuffer> _animationStorageRange = {m_animationStorageRange.offset, m_animationStorageRange.size, _depth > 0u && m_animationStorageRange.buffer ? core::smart_refctd_ptr_static_cast<ICPUBuffer>(m_animationStorageRange.buffer->clone(_depth - 1u)) : m_animationStorageRange.buffer};

        auto cp = core::make_smart_refctd_ptr<ICPUAnimationLibrary>(std::move(_keyframeStorageBinding), std::move(_timestampStorageBinding), m_keyframeCount, std::move(_animationStorageRange));
        clone_common(cp.get());
        cp->setAnimationNames(this);

        return cp;
    }

    virtual void convertToDummyObject(uint32_t referenceLevelsBelowToConvert = 0u) override
    {
        convertToDummyObject_common(referenceLevelsBelowToConvert);

        if(referenceLevelsBelowToConvert)
        {
            m_keyframeStorageBinding.buffer->convertToDummyObject(referenceLevelsBelowToConvert - 1u);
            m_timestampStorageBinding.buffer->convertToDummyObject(referenceLevelsBelowToConvert - 1u);
            m_animationStorageRange.buffer->convertToDummyObject(referenceLevelsBelowToConvert - 1u);
        }
    }

    _NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_ANIMATION_LIBRARY;
    inline E_TYPE getAssetType() const override { return AssetType; }

    virtual size_t conservativeSizeEstimate() const override
    {
        size_t estimate = sizeof(SBufferBinding<ICPUBuffer>) * 2ull;
        estimate += sizeof(SBufferRange<ICPUBuffer>);
        estimate += sizeof(uint32_t);
        estimate += m_stringPool.size();
        estimate += m_nameToAnimation.size() * sizeof(std::pair<uint32_t, uint32_t>);
        // do we add other things to the size estimate?
        return estimate;
    }

    bool canBeRestoredFrom(const IAsset* _other) const override
    {
        auto other = static_cast<const ICPUAnimationLibrary*>(_other);
        if(m_keyframeCount != other->m_keyframeCount)
            return false;

        if(m_keyframeStorageBinding.offset != other->m_keyframeStorageBinding.offset)
            return false;
        if(m_keyframeStorageBinding.buffer->canBeRestoredFrom(other->m_keyframeStorageBinding.buffer.get()))
            return false;
        if(m_timestampStorageBinding.offset != other->m_timestampStorageBinding.offset)
            return false;
        if(m_timestampStorageBinding.buffer->canBeRestoredFrom(other->m_timestampStorageBinding.buffer.get()))
            return false;

        if(m_animationStorageRange.offset != other->m_animationStorageRange.offset)
            return false;
        if(m_animationStorageRange.size != other->m_animationStorageRange.size)
            return false;
        if((!m_animationStorageRange.buffer) != (!other->m_animationStorageRange.buffer))
            return false;
        if(m_animationStorageRange.buffer && !m_animationStorageRange.buffer->canBeRestoredFrom(other->m_animationStorageRange.buffer.get()))
            return false;

        return true;
    }

protected:
    void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
    {
        auto* other = static_cast<ICPUAnimationLibrary*>(_other);

        if(_levelsBelow)
        {
            --_levelsBelow;

            if(m_keyframeStorageBinding.buffer)
                restoreFromDummy_impl_call(m_keyframeStorageBinding.buffer.get(), other->m_keyframeStorageBinding.buffer.get(), _levelsBelow);
            if(m_timestampStorageBinding.buffer)
                restoreFromDummy_impl_call(m_timestampStorageBinding.buffer.get(), other->m_timestampStorageBinding.buffer.get(), _levelsBelow);

            if(m_animationStorageRange.buffer)
                restoreFromDummy_impl_call(m_animationStorageRange.buffer.get(), other->m_animationStorageRange.buffer.get(), _levelsBelow);
        }
    }

    bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override
    {
        --_levelsBelow;
        if(m_keyframeStorageBinding.buffer && m_keyframeStorageBinding.buffer->isAnyDependencyDummy(_levelsBelow))
            return true;
        if(m_timestampStorageBinding.buffer && m_timestampStorageBinding.buffer->isAnyDependencyDummy(_levelsBelow))
            return true;

        return m_animationStorageRange.buffer && m_animationStorageRange.buffer->isAnyDependencyDummy(_levelsBelow);
    }
};

}
}

#endif
