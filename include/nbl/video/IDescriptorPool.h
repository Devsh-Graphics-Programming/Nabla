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
class IGPUDescriptorSet;
class IGPUDescriptorSetLayout;

class NBL_API IDescriptorPool : public core::IReferenceCounted, public IBackendObject
{
    friend class IGPUDescriptorSet;

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
            asset::IDescriptor::E_TYPE type;
            uint32_t count;
        };

        struct SDescriptorOffsets
        {
            SDescriptorOffsets()
            {
                // The default constructor should initiailze all the offsets to an invalid value (~0u) because ~IGPUDescriptorSet relies on it to
                // know which descriptors are present in the set and hence should be destroyed.
                std::fill_n(data, static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT) + 1, ~0u);
            }

            uint32_t data[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT) + 1];
        };

        inline void createDescriptorSets(uint32_t count, const IGPUDescriptorSetLayout* const* layouts, core::smart_refctd_ptr<IGPUDescriptorSet>* output)
        {
            for (uint32_t i = 0u; i < count; ++i)
            {
                auto layout = core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(layouts[i]);
                output[i++] = createDescriptorSet(std::move(layout));
            }
        }

        // Returns the offset into the pool's descriptor storage. These offsets will be combined
        // later with base memory addresses to get the actual memory address where we put the core::smart_refctd_ptr<const IDescriptor>.
        SDescriptorOffsets allocateDescriptorOffsets(const IGPUDescriptorSetLayout* layout);
        
        core::smart_refctd_ptr<IGPUDescriptorSet> createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout);
        bool freeDescriptorSets(const uint32_t descriptorSetCount, IGPUDescriptorSet* const* const descriptorSets);

        inline uint32_t getCapacity() const { return m_maxSets; }

    protected:
        explicit IDescriptorPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const IDescriptorPool::E_CREATE_FLAGS flags, uint32_t _maxSets, const uint32_t poolSizeCount, const IDescriptorPool::SDescriptorPoolSize* poolSizes) : IBackendObject(std::move(dev)), m_maxSets(_maxSets), m_flags(flags)
        {
            std::fill_n(m_maxDescriptorCount, static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT), 0u);

            for (auto i = 0; i < poolSizeCount; ++i)
                m_maxDescriptorCount[static_cast<uint32_t>(poolSizes[i].type)] += poolSizes[i].count;

            for (auto i = 0; i < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++i)
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

            // For (possibly) mutable samplers.
            if (m_flags & ECF_FREE_DESCRIPTOR_SET_BIT)
            {
                m_generalAllocatorReservedSpace[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)] = std::make_unique<uint8_t[]>(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(1u, m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)], 1u));
                m_generalAllocators[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)] = core::GeneralpurposeAddressAllocator<uint32_t>(m_generalAllocatorReservedSpace[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)].get(), 0u, 0u, 1u, m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)], 1u);
            }
            else
            {
                m_linearAllocators[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)] = core::LinearAddressAllocator<uint32_t>(nullptr, 0u, 0u, 1u, m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)]);
            }

            // Initialize the storages.
            m_textureStorage = std::make_unique<core::StorageTrivializer<core::smart_refctd_ptr<video::IGPUImageView>>[]>(m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)]);
            m_mutableSamplerStorage = std::make_unique<core::StorageTrivializer<core::smart_refctd_ptr<video::IGPUSampler>>[]>(m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)]);
            m_storageImageStorage = std::make_unique<core::StorageTrivializer<core::smart_refctd_ptr<IGPUImageView>>[]>(m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE)] + m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_INPUT_ATTACHMENT)]);
            m_UBO_SSBOStorage = std::make_unique<core::StorageTrivializer<core::smart_refctd_ptr<IGPUBuffer>>[]>(m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER)] + m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER)] + m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC)] + m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC)]);
            m_UTB_STBStorage = std::make_unique<core::StorageTrivializer<core::smart_refctd_ptr<IGPUBufferView>>[]>(m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER)] + m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC)]);
            m_accelerationStructureStorage = std::make_unique<core::StorageTrivializer<core::smart_refctd_ptr<IGPUAccelerationStructure>>[]>(m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_ACCELERATION_STRUCTURE)]);
        }

        virtual ~IDescriptorPool() {}

        virtual core::smart_refctd_ptr<IGPUDescriptorSet> createDescriptorSet_impl(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout, SDescriptorOffsets&& offsets) = 0;
        virtual bool freeDescriptorSets_impl(const uint32_t descriptorSetCount, IGPUDescriptorSet* const* const descriptorSets) = 0;

        uint32_t m_maxSets;

    private:
        const IDescriptorPool::E_CREATE_FLAGS m_flags;
        uint32_t m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)];
        union
        {
            core::LinearAddressAllocator<uint32_t> m_linearAllocators[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)+1];
            core::GeneralpurposeAddressAllocator<uint32_t> m_generalAllocators[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)+1];
        };
        std::unique_ptr<uint8_t[]> m_generalAllocatorReservedSpace[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)+1];

        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<video::IGPUImageView>>[]> m_textureStorage;
        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<video::IGPUSampler>>[]> m_mutableSamplerStorage;
        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<IGPUImageView>>[]> m_storageImageStorage; // storage image | input attachment
        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<IGPUBuffer>>[]> m_UBO_SSBOStorage; // ubo | ssbo | ubo dynamic | ssbo dynamic
        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<IGPUBufferView>>[]> m_UTB_STBStorage; // utb | stb
        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<IGPUAccelerationStructure>>[]> m_accelerationStructureStorage;
};

}

#endif