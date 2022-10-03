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
                        m_generalAllocatorReservedSpace[i] = _NBL_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(core::smart_refctd_ptr<asset::IDescriptor>), m_maxDescriptorCount[i] * sizeof(void*), 1u), _NBL_SIMD_ALIGNMENT);
                        m_generalAllocators[i] = core::GeneralpurposeAddressAllocator<uint32_t>(m_generalAllocatorReservedSpace[i], 0u, 0u, 1u, m_maxDescriptorCount[i] * sizeof(void*), 1u);
                    }
                    else
                    {
                        m_linearAllocators[i] = core::LinearAddressAllocator<uint32_t>(nullptr, 0u, 0u, 1u, m_maxDescriptorCount[i]*sizeof(void*));
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

        ~IDescriptorPool()
        {
            for (auto i = 0; i < asset::EDT_COUNT; ++i)
            {
                if ((m_flags & IDescriptorPool::ECF_FREE_DESCRIPTOR_SET_BIT) && (m_maxDescriptorCount[i] > 0))
                    _NBL_ALIGNED_FREE(m_generalAllocatorReservedSpace[i]);
            }
        }

        // This will return the offset into the pool's descriptor storage. These offsets will be combined
        // later with base memory addresses to get the actual memory adress where we put the core::smart_refctd_ptr<const IDescriptor>.
        uint32_t allocateDescriptors(const asset::E_DESCRIPTOR_TYPE type, const uint32_t count)
        {
            const uint32_t bytesToAllocate = count * sizeof(void*);

            uint32_t offset;
            uint32_t invalidAddress;
            if (m_flags & ECF_FREE_DESCRIPTOR_SET_BIT)
            {
                offset = m_generalAllocators[type].alloc_addr(bytesToAllocate, 1u);
                invalidAddress = core::GeneralpurposeAddressAllocator<uint32_t>::invalid_address;
            }
            else
            {
                offset = m_linearAllocators[type].alloc_addr(bytesToAllocate, 1u);
                invalidAddress = core::LinearAddressAllocator<uint32_t>::invalid_address;
            }

            return (offset == invalidAddress) ? ~0u : offset;
        }

        void freeDescriptors(const uint32_t count, void* descriptors, const asset::E_DESCRIPTOR_TYPE type)
        {
            assert(m_flags & ECF_FREE_DESCRIPTOR_SET_BIT);
            uint32_t allocatedAddr = static_cast<uint32_t>(reinterpret_cast<uint8_t*>(descriptors) - getDescriptorMemoryBaseAddress(type));
            m_generalAllocators[type].free_addr(allocatedAddr, count*sizeof(core::smart_refctd_ptr<const asset::IDescriptor>));
        }

        inline uint8_t* getDescriptorMemoryBaseAddress(const asset::E_DESCRIPTOR_TYPE type) const
        {
            uint8_t* baseAddress;
            switch (type)
            {
            case asset::EDT_COMBINED_IMAGE_SAMPLER:
                baseAddress = reinterpret_cast<uint8_t*>(m_combinedImageSamplerStorage.get());
                break;
            case asset::EDT_STORAGE_IMAGE:
                baseAddress = reinterpret_cast<uint8_t*>(m_storageImageStorage.get());
                break;
            case asset::EDT_UNIFORM_TEXEL_BUFFER:
                baseAddress = reinterpret_cast<uint8_t*>(m_UTB_STBStorage.get());
                break;
            case asset::EDT_STORAGE_TEXEL_BUFFER:
                baseAddress = reinterpret_cast<uint8_t*>(m_UTB_STBStorage.get()) + m_maxDescriptorCount[asset::EDT_UNIFORM_TEXEL_BUFFER] * sizeof(void*);
                break;
            case asset::EDT_UNIFORM_BUFFER:
                baseAddress = reinterpret_cast<uint8_t*>(m_UBO_SSBOStorage.get());
                break;
            case asset::EDT_STORAGE_BUFFER:
                baseAddress = reinterpret_cast<uint8_t*>(m_UBO_SSBOStorage.get()) + m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER] * sizeof(void*);
                break;
            case asset::EDT_UNIFORM_BUFFER_DYNAMIC:
                baseAddress = reinterpret_cast<uint8_t*>(m_UBO_SSBOStorage.get()) + (m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER] + m_maxDescriptorCount[asset::EDT_STORAGE_BUFFER]) * sizeof(void*);
                break;
            case asset::EDT_STORAGE_BUFFER_DYNAMIC:
                baseAddress = reinterpret_cast<uint8_t*>(m_UBO_SSBOStorage.get()) + (m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER] + m_maxDescriptorCount[asset::EDT_STORAGE_BUFFER] + m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER_DYNAMIC]) * sizeof(void*);
                break;
            case asset::EDT_INPUT_ATTACHMENT:
                baseAddress = reinterpret_cast<uint8_t*>(m_storageImageStorage.get()) + m_maxDescriptorCount[asset::EDT_STORAGE_IMAGE] * sizeof(void*);
                break;
            case asset::EDT_ACCELERATION_STRUCTURE:
                baseAddress = reinterpret_cast<uint8_t*>(m_accelerationStructureStorage.get());
                break;
            default:
                assert(!"Invalid code path.");
                return nullptr;
            }

            return baseAddress;
        }

        inline bool allowsFreeingDescriptorSets() const
        {
            return (m_flags & IDescriptorPool::ECF_FREE_DESCRIPTOR_SET_BIT);
        }

        inline uint32_t getCapacity() const { return m_maxSets; }

    protected:
        uint32_t m_maxSets;

    private:
        IDescriptorPool::E_CREATE_FLAGS m_flags;
        uint32_t m_maxDescriptorCount[asset::EDT_COUNT];
        core::LinearAddressAllocator<uint32_t> m_linearAllocators[asset::EDT_COUNT];
        core::GeneralpurposeAddressAllocator<uint32_t> m_generalAllocators[asset::EDT_COUNT];
        void* m_generalAllocatorReservedSpace[asset::EDT_COUNT];

        std::unique_ptr<core::StorageTrivializer<SCombinedImageSampler>[]> m_combinedImageSamplerStorage;
        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<IGPUImageView>>[]> m_storageImageStorage; // storage image | input attachment
        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<IGPUBuffer>>[]> m_UBO_SSBOStorage; // ubo | ssbo | ubo dynamic | ssbo dynamic
        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<IGPUBufferView>>[]> m_UTB_STBStorage; // utb | stb
        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<IGPUAccelerationStructure>>[]> m_accelerationStructureStorage;
};

}

#endif