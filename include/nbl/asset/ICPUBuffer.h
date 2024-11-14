// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_BUFFER_H_INCLUDED_
#define _NBL_ASSET_I_CPU_BUFFER_H_INCLUDED_

#include <type_traits>

#include "nbl/asset/IBuffer.h"
#include "nbl/asset/IAsset.h"
#include "nbl/asset/IPreHashed.h"

namespace nbl::asset
{

//! One of CPU class-object representing an Asset
/**
    One of Assets used for storage of large arrays, so that storage can be decoupled
    from other objects such as meshbuffers, images, animations and shader source/bytecode.

    @see IAsset
*/
class ICPUBuffer final : public asset::IBuffer, public IPreHashed
{
    public:
        struct SCreationParams : asset::IBuffer::SCreationParams
        {
            size_t size;
            void* data = nullptr;
            size_t alignment = _NBL_SIMD_ALIGNMENT;
            std::pmr::memory_resource* memoryResource = nullptr;

            SCreationParams& operator =(const asset::IBuffer::SCreationParams& rhs)
            {
                static_cast<asset::IBuffer::SCreationParams&>(*this) = rhs;
                return *this;
            }
        };

        ICPUBuffer(size_t size, void* data, std::pmr::memory_resource* memoryResource, size_t alignment, bool adopt_memory) :
            asset::IBuffer({ size, EUF_TRANSFER_DST_BIT }), m_data(data), m_mem_resource(memoryResource), m_alignment(alignment), m_adopt_memory(adopt_memory) {}

        //! allocates uninitialized memory, copies `data` into allocation if `!data` not nullptr
        core::smart_refctd_ptr<ICPUBuffer> static create(const SCreationParams& params) {
            std::pmr::memory_resource* memoryResource = params.memoryResource;
            if (!params.memoryResource)
                memoryResource = std::pmr::get_default_resource();

            auto data = memoryResource->allocate(params.size, params.alignment);
            if (!data)
                return nullptr;

            if (params.data)
                memcpy(data, params.data, params.size);

            return core::make_smart_refctd_ptr<ICPUBuffer>(params.size, data, memoryResource, params.alignment, false);
        }

        //! does not allocate memory, adopts the `data` pointer, no copies done
        core::smart_refctd_ptr<ICPUBuffer> static create(const SCreationParams& params, core::adopt_memory_t) {
            std::pmr::memory_resource* memoryResource;
            if (!params.memoryResource)
                memoryResource = std::pmr::get_default_resource();
            return core::make_smart_refctd_ptr<ICPUBuffer>(params.size, params.data, memoryResource, params.alignment, true);
        }

        core::smart_refctd_ptr<IAsset> clone(uint32_t = ~0u) const override final
        {
            auto cp = create({ .size = m_creationParams.size, .data = m_data, .alignment = m_alignment });
            memcpy(cp->getPointer(), m_data, m_creationParams.size);
            cp->setContentHash(getContentHash());
            return cp;
        }

        constexpr static inline auto AssetType = ET_BUFFER;
        inline IAsset::E_TYPE getAssetType() const override final { return AssetType; }

        inline size_t getDependantCount() const override { return 0; }

        //
        inline core::blake3_hash_t computeContentHash() const override
        {
            core::blake3_hasher hasher;
            if (m_data)
                hasher.update(m_data, m_creationParams.size);
            return static_cast<core::blake3_hash_t>(hasher);
        }

        inline bool missingContent() const override { return !m_data; }

        //! Returns pointer to data.
        const void* getPointer() const { return m_data; }
        void* getPointer()
        {
            assert(isMutable());
            return m_data;
        }

        inline core::bitflag<E_USAGE_FLAGS> getUsageFlags() const
        {
            return m_creationParams.usage;
        }
        inline bool setUsageFlags(core::bitflag<E_USAGE_FLAGS> _usage)
        {
            assert(isMutable());
            m_creationParams.usage = _usage;
            return true;
        }
        inline bool addUsageFlags(core::bitflag<E_USAGE_FLAGS> _usage)
        {
            assert(isMutable());
            m_creationParams.usage |= _usage;
            return true;
        }

protected:
    inline IAsset* getDependant_impl(const size_t ix) override
    {
        return nullptr;
    }

    inline void discardContent_impl() override
    {
        return freeData();
    }

    // REMEMBER TO CALL FROM DTOR!
    virtual inline void freeData()
    {
        if (!m_adopt_memory && m_data)
            m_mem_resource->deallocate(m_data, m_creationParams.size, m_alignment);
        m_data = nullptr;
        m_creationParams.size = 0ull;
    }

    void* m_data;
    std::pmr::memory_resource* m_mem_resource;
    size_t m_alignment;
    bool m_adopt_memory;
};

} // end namespace nbl::asset

#endif