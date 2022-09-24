#ifndef __NBL_I_DESCRIPTOR_POOL_H_INCLUDED__
#define __NBL_I_DESCRIPTOR_POOL_H_INCLUDED__


#include "nbl/core/IReferenceCounted.h"

#include "nbl/asset/IDescriptorSetLayout.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class IGPUImageView;
class IGPUSampler;
class IGPUBufferView;

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

    private:
        struct SCombinedImageSampler
        {
            core::smart_refctd_ptr<video::IGPUImageView> view;
            core::smart_refctd_ptr<video::IGPUSampler> sampler;
        };

        // This construct makes it so that we don't trigger T's constructors and destructors.
        // TODO(achal): Should we move it to the core namespace?
        template <typename T>
        struct alignas(T) StorageTrivializer
        {
            uint8_t storage[sizeof(T)];
        };

    public:
        explicit IDescriptorPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const IDescriptorPool::E_CREATE_FLAGS flags, uint32_t _maxSets, const uint32_t poolSizeCount, const IDescriptorPool::SDescriptorPoolSize* poolSizes) : IBackendObject(std::move(dev)), m_maxSets(_maxSets), m_flags(flags)
        {
            memset(m_maxDescriptorCount, 0, asset::EDT_COUNT * sizeof(uint32_t));

            for (auto i = 0; i < poolSizeCount; ++i)
                m_maxDescriptorCount[poolSizes[i].type]++;

            for (auto i = 0; i < asset::EDT_COUNT; ++i)
            {
                if (m_maxDescriptorCount[i] > 0)
                {
                    if (m_flags & IDescriptorPool::ECF_FREE_DESCRIPTOR_SET_BIT)
                    {
                        m_generalAllocatorReservedSpace[i] = _NBL_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(core::smart_refctd_ptr<asset::IDescriptor>), m_maxDescriptorCount[i] * sizeof(void*), 1u), _NBL_SIMD_ALIGNMENT);
                        m_generalAllocators[i] = core::GeneralpurposeAddressAllocator<uint32_t>(m_generalAllocatorReservedSpace[i], 0u, 0u, alignof(core::smart_refctd_ptr<asset::IDescriptor>), m_maxDescriptorCount[i] * sizeof(void*), 1u);
                    }
                    else
                    {
                        m_linearAllocators[i] = core::LinearAddressAllocator<uint32_t>(nullptr, 0u, 0u, alignof(uint32_t), m_maxDescriptorCount[i]*sizeof(void*));
                    }
                }
            }

            // Initialize the storages.
            m_combinedImageSamplerStorage = std::make_unique<StorageTrivializer<SCombinedImageSampler>[]>(m_maxDescriptorCount[asset::EDT_COMBINED_IMAGE_SAMPLER]);
            m_storageImageStorage = std::make_unique<StorageTrivializer<core::smart_refctd_ptr<IGPUImageView>>[]>(m_maxDescriptorCount[asset::EDT_STORAGE_IMAGE] + m_maxDescriptorCount[asset::EDT_INPUT_ATTACHMENT]);
            m_UBO_SSBOStorage = std::make_unique<StorageTrivializer<core::smart_refctd_ptr<IGPUBuffer>>[]>(m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER] + m_maxDescriptorCount[asset::EDT_STORAGE_BUFFER] + m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER_DYNAMIC] + m_maxDescriptorCount[asset::EDT_STORAGE_BUFFER_DYNAMIC]);
            m_UTB_STBStorage = std::make_unique<StorageTrivializer<core::smart_refctd_ptr<IGPUBufferView>>[]>(m_maxDescriptorCount[asset::EDT_UNIFORM_TEXEL_BUFFER] + m_maxDescriptorCount[asset::EDT_STORAGE_BUFFER_DYNAMIC]);
            m_accelerationStructureStorage = std::make_unique<StorageTrivializer<core::smart_refctd_ptr<IGPUAccelerationStructure>>[]>(m_maxDescriptorCount[asset::EDT_ACCELERATION_STRUCTURE]);
        }

        ~IDescriptorPool()
        {
            for (auto i = 0; i < asset::EDT_COUNT; ++i)
            {
                if ((m_flags & IDescriptorPool::ECF_FREE_DESCRIPTOR_SET_BIT) && (m_maxDescriptorCount[i] > 0))
                    _NBL_ALIGNED_FREE(m_generalAllocatorReservedSpace[i]);
            }
        }

        uint32_t getCapacity() const { return m_maxSets; }

    protected:
        uint32_t m_maxSets;

    private:
        IDescriptorPool::E_CREATE_FLAGS m_flags;
        uint32_t m_maxDescriptorCount[asset::EDT_COUNT];
        core::LinearAddressAllocator<uint32_t> m_linearAllocators[asset::EDT_COUNT];
        core::GeneralpurposeAddressAllocator<uint32_t> m_generalAllocators[asset::EDT_COUNT];
        void* m_generalAllocatorReservedSpace[asset::EDT_COUNT];

        std::unique_ptr<StorageTrivializer<SCombinedImageSampler>[]> m_combinedImageSamplerStorage;
        std::unique_ptr<StorageTrivializer<core::smart_refctd_ptr<IGPUImageView>>[]> m_storageImageStorage; // storage image | input attachment
        std::unique_ptr<StorageTrivializer<core::smart_refctd_ptr<IGPUBuffer>>[]> m_UBO_SSBOStorage; // ubo | ssbo | ubo dynamic | ssbo dynamic
        std::unique_ptr<StorageTrivializer<core::smart_refctd_ptr<IGPUBufferView>>[]> m_UTB_STBStorage; // utb | stb
        std::unique_ptr<StorageTrivializer<core::smart_refctd_ptr<IGPUAccelerationStructure>>[]> m_accelerationStructureStorage;
};

}

#endif