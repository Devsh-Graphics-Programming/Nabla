#ifndef __NBL_I_DESCRIPTOR_POOL_H_INCLUDED__
#define __NBL_I_DESCRIPTOR_POOL_H_INCLUDED__


#include "nbl/core/IReferenceCounted.h"
#include "nbl/core/StorageTrivializer.h"

#include "nbl/asset/IDescriptorSetLayout.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class IGPUImageView;
class IGPUSampler;
class IGPUBufferView;
class IGPUDescriptorSetLayout;
class IGPUDescriptorSet;

class NBL_API IDescriptorPool : public core::IReferenceCounted, public IBackendObject
{
    public:
        enum E_CREATE_FLAGS : uint32_t
        {
            ECF_NONE = 0x00u,
            ECF_FREE_DESCRIPTOR_SET_BIT = 0x01,
            ECF_UPDATE_AFTER_BIND_BIT = 0x02,
            ECF_HOST_ONLY_BIT_VALVE = 0x04
        };

        struct SDescriptorPoolSize
        {
            asset::E_DESCRIPTOR_TYPE type;
            uint32_t count;
        };

        struct SDescriptorOffsets
        {
            SDescriptorOffsets()
            {
                // The default constructor should initiailze all the offsets to an invalid value (~0u) because ~IGPUDescriptorSet relies on it to
                // know which descriptors are present in the set and hence should be destroyed.
                std::fill_n(data, asset::EDT_COUNT, ~0u);
            }

            uint32_t data[asset::EDT_COUNT];
        };

    friend class IGPUDescriptorSet;

    private:
        // TODO(achal): This needs to get removed.
        struct SCombinedImageSampler
        {
            core::smart_refctd_ptr<video::IGPUImageView> view;
            core::smart_refctd_ptr<video::IGPUSampler> sampler;
        };

    public:
        explicit IDescriptorPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const IDescriptorPool::E_CREATE_FLAGS flags, uint32_t _maxSets, const uint32_t poolSizeCount, const IDescriptorPool::SDescriptorPoolSize* poolSizes) : IBackendObject(std::move(dev)), m_maxSets(_maxSets), m_flags(flags)
        {
            std::fill_n(m_maxDescriptorCount, asset::EDT_COUNT, 0u);

            for (auto i = 0; i < poolSizeCount; ++i)
                m_maxDescriptorCount[poolSizes[i].type] += poolSizes[i].count;

            for (auto i = 0; i < asset::EDT_COUNT; ++i)
            {
                if (m_maxDescriptorCount[i] > 0)
                {
                    if (m_flags & ECF_FREE_DESCRIPTOR_SET_BIT)
                    {
                        m_generalAllocatorReservedSpace[i] = std::make_unique<uint8_t[]>(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(1u, m_maxDescriptorCount[i], 1u));
                        m_generalAllocators[i] = core::GeneralpurposeAddressAllocator<uint32_t>(m_generalAllocatorReservedSpace[i].get(), 0u, 0u, 1u, m_maxDescriptorCount[i], 1u);
                    }
                    else
                    {
                        m_linearAllocators[i] = core::LinearAddressAllocator<uint32_t>(nullptr, 0u, 0u, 1u, m_maxDescriptorCount[i]);
                    }
                }
            }

            // Initialize the storages.
            m_combinedImageSamplerStorage = std::make_unique<core::StorageTrivializer<SCombinedImageSampler>[]>(m_maxDescriptorCount[asset::EDT_COMBINED_IMAGE_SAMPLER]);
            m_storageImageStorage = std::make_unique<core::StorageTrivializer<core::smart_refctd_ptr<IGPUImageView>>[]>(m_maxDescriptorCount[asset::EDT_STORAGE_IMAGE] + m_maxDescriptorCount[asset::EDT_INPUT_ATTACHMENT]);
            m_UBO_SSBOStorage = std::make_unique<core::StorageTrivializer<core::smart_refctd_ptr<IGPUBuffer>>[]>(m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER] + m_maxDescriptorCount[asset::EDT_STORAGE_BUFFER] + m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER_DYNAMIC] + m_maxDescriptorCount[asset::EDT_STORAGE_BUFFER_DYNAMIC]);
            m_UTB_STBStorage = std::make_unique<core::StorageTrivializer<core::smart_refctd_ptr<IGPUBufferView>>[]>(m_maxDescriptorCount[asset::EDT_UNIFORM_TEXEL_BUFFER] + m_maxDescriptorCount[asset::EDT_STORAGE_BUFFER_DYNAMIC]);
            m_accelerationStructureStorage = std::make_unique<core::StorageTrivializer<core::smart_refctd_ptr<IGPUAccelerationStructure>>[]>(m_maxDescriptorCount[asset::EDT_ACCELERATION_STRUCTURE]);
        }

        ~IDescriptorPool() {}

        // Returns the offset into the pool's descriptor storage. These offsets will be combined
        // later with base memory addresses to get the actual memory adress where we put the core::smart_refctd_ptr<const IDescriptor>.
        SDescriptorOffsets allocateDescriptors(const IGPUDescriptorSetLayout* layout);

        bool freeDescriptorSets(const uint32_t descriptorSetCount, IGPUDescriptorSet* const* const descriptorSets);

        inline uint32_t getCapacity() const { return m_maxSets; }

    protected:
        uint32_t m_maxSets;

        virtual bool freeDescriptorSets_impl(const uint32_t descriptorSetCount, IGPUDescriptorSet* const* const descriptorSets)
        {
            // We don't need to do anything on the GL backend, perhaps, so No-OP.
            return true;
        };

    private:
        const IDescriptorPool::E_CREATE_FLAGS m_flags;
        uint32_t m_maxDescriptorCount[asset::EDT_COUNT];
        union
        {
            core::LinearAddressAllocator<uint32_t> m_linearAllocators[asset::EDT_COUNT];
            core::GeneralpurposeAddressAllocator<uint32_t> m_generalAllocators[asset::EDT_COUNT];
        };
        std::unique_ptr<uint8_t[]> m_generalAllocatorReservedSpace[asset::EDT_COUNT];

        std::unique_ptr<core::StorageTrivializer<SCombinedImageSampler>[]> m_combinedImageSamplerStorage;
        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<IGPUImageView>>[]> m_storageImageStorage; // storage image | input attachment
        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<IGPUBuffer>>[]> m_UBO_SSBOStorage; // ubo | ssbo | ubo dynamic | ssbo dynamic
        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<IGPUBufferView>>[]> m_UTB_STBStorage; // utb | stb
        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<IGPUAccelerationStructure>>[]> m_accelerationStructureStorage;
};

}

#endif