// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_BUFFER_H_INCLUDED_
#define _NBL_ASSET_I_CPU_BUFFER_H_INCLUDED_

#include <type_traits>

#include "nbl/asset/IBuffer.h"
#include "nbl/asset/IAsset.h"
#include "nbl/asset/IPreHashed.h"

#include "nbl/core/alloc/refctd_memory_resource.h"

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
            void* data = nullptr;
            core::smart_refctd_ptr<core::refctd_memory_resource> memoryResource = nullptr;
            size_t alignment = _NBL_SIMD_ALIGNMENT;

            SCreationParams& operator =(const asset::IBuffer::SCreationParams& rhs)
            {
                static_cast<asset::IBuffer::SCreationParams&>(*this) = rhs;
                return *this;
            }
        };

        //! allocates uninitialized memory, copies `data` into allocation if `!data` not nullptr
        core::smart_refctd_ptr<ICPUBuffer> static create(SCreationParams&& params)
        {
            if (!params.memoryResource)
                params.memoryResource = core::getDefaultMemoryResource();

            auto data = params.memoryResource->allocate(params.size, params.alignment);
            if (!data)
                return nullptr;
            if (params.data)
                memcpy(data, params.data, params.size);
            params.data = data;

            return core::smart_refctd_ptr<ICPUBuffer>(new ICPUBuffer(std::move(params)), core::dont_grab);
        }

        //! does not allocate memory, adopts the `data` pointer, no copies done
        core::smart_refctd_ptr<ICPUBuffer> static create(SCreationParams&& params, core::adopt_memory_t)
        {
            if (!params.data)
                return nullptr;
            if (!params.memoryResource)
                params.memoryResource = core::getDefaultMemoryResource();
            return core::smart_refctd_ptr<ICPUBuffer>(new ICPUBuffer(std::move(params)), core::dont_grab);
        }

        core::smart_refctd_ptr<IAsset> clone(uint32_t = ~0u) const override final
        {
            auto cp = create({ { m_creationParams.size }, m_data, nullptr, m_alignment });
            memcpy(cp->getPointer(), m_data, m_creationParams.size);
            cp->setContentHash(getContentHash());
            return cp;
        }

        constexpr static inline auto AssetType = ET_BUFFER;
        inline IAsset::E_TYPE getAssetType() const override final { return AssetType; }

        inline size_t getDependantCount() const override { return 0; }

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
        if (m_data)
            m_mem_resource->deallocate(m_data, m_creationParams.size, m_alignment);
        m_data = nullptr;
        m_mem_resource = nullptr;
        m_creationParams.size = 0ull;
    }

private:
    ICPUBuffer(SCreationParams&& params) :
        asset::IBuffer({ params.size, EUF_TRANSFER_DST_BIT }), m_data(params.data),
        m_mem_resource(params.memoryResource), m_alignment(params.alignment) {}

    ~ICPUBuffer() override {
        discardContent_impl();
    }

    void* m_data;
    core::smart_refctd_ptr<core::refctd_memory_resource> m_mem_resource;
    size_t m_alignment;
};

} // end namespace nbl::asset

#endif