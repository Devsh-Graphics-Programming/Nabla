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

class IDescriptorPool : public core::IReferenceCounted, public IBackendObject
{
    public:
        enum E_CREATE_FLAGS : uint32_t
        {
            ECF_NONE = 0x00u,
            ECF_FREE_DESCRIPTOR_SET_BIT = 0x01,
            ECF_UPDATE_AFTER_BIND_BIT = 0x02,
            ECF_HOST_ONLY_BIT_VALVE = 0x04
        };

        struct SCreateInfo
        {
            core::bitflag<E_CREATE_FLAGS> flags = ECF_NONE;
            uint32_t maxSets = 0;
            uint32_t maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)] = { 0 };
        };

        struct SStorageOffsets
        {
            static constexpr inline uint32_t Invalid = ~0u;

            SStorageOffsets()
            {
                // The default constructor should initiailze all the offsets to Invalid because other parts of the codebase relies on it to
                // know which descriptors are present in the set and hence should be destroyed, or which set in the pool is non-zombie.
                std::fill_n(data, static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT) + 2, Invalid);
            }

            inline uint32_t getDescriptorOffset(const asset::IDescriptor::E_TYPE type) const
            {
                const uint32_t idx = static_cast<uint32_t>(type);
                assert(idx < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT));
                return data[idx];
            }

            inline uint32_t getMutableSamplerOffset() const { return data[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)]; }

            inline uint32_t getSetOffset() const { return data[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT) + 1]; }
            inline uint32_t& getSetOffset() { return data[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT) + 1]; }

            uint32_t data[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT) + 2];
        };

        inline core::smart_refctd_ptr<IGPUDescriptorSet> createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout)
        {
            core::smart_refctd_ptr<IGPUDescriptorSet> set;
            const bool result = createDescriptorSets(1, &layout.get(), &set);
            if (result)
                return set;
            else
                return nullptr;
        }

        uint32_t createDescriptorSets(uint32_t count, const IGPUDescriptorSetLayout* const* layouts, core::smart_refctd_ptr<IGPUDescriptorSet>* output);

        bool reset();

        inline uint32_t getCapacity() const { return m_creationParameters.maxSets; }
        inline bool allowsFreeing() const { return m_creationParameters.flags.hasFlags(ECF_FREE_DESCRIPTOR_SET_BIT); }

    protected:
        explicit IDescriptorPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreateInfo&& createInfo)
            : IBackendObject(std::move(dev)), m_creationParameters(std::move(createInfo))
        {
            for (auto i = 0; i < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++i)
                m_descriptorAllocators[i] = std::make_unique<allocator_state_t>(m_creationParameters.maxDescriptorCount[i], m_creationParameters.flags.hasFlags(ECF_FREE_DESCRIPTOR_SET_BIT));

            // For mutable samplers. We don't know if there will be mutable samplers in sets allocated by this pool when we create the pool.
            m_descriptorAllocators[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)] = std::make_unique<allocator_state_t>(m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)], m_creationParameters.flags.hasFlags(ECF_FREE_DESCRIPTOR_SET_BIT));

            // Initialize the storages.
            m_textureStorage = std::make_unique<core::StorageTrivializer<core::smart_refctd_ptr<video::IGPUImageView>>[]>(m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)]);
            m_mutableSamplerStorage = std::make_unique<core::StorageTrivializer<core::smart_refctd_ptr<video::IGPUSampler>>[]>(m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)]);
            m_storageImageStorage = std::make_unique<core::StorageTrivializer<core::smart_refctd_ptr<IGPUImageView>>[]>(m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE)] + m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_INPUT_ATTACHMENT)]);
            m_UBO_SSBOStorage = std::make_unique<core::StorageTrivializer<core::smart_refctd_ptr<IGPUBuffer>>[]>(m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER)] + m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER)] + m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC)] + m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC)]);
            m_UTB_STBStorage = std::make_unique<core::StorageTrivializer<core::smart_refctd_ptr<IGPUBufferView>>[]>(m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER)] + m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC)]);
            m_accelerationStructureStorage = std::make_unique<core::StorageTrivializer<core::smart_refctd_ptr<IGPUAccelerationStructure>>[]>(m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_ACCELERATION_STRUCTURE)]);

            m_allocatedDescriptorSets = std::make_unique<IGPUDescriptorSet* []>(m_creationParameters.maxSets);
            std::fill_n(m_allocatedDescriptorSets.get(), m_creationParameters.maxSets, nullptr);

            m_descriptorSetAllocatorReservedSpace = std::make_unique<uint8_t[]>(core::IteratablePoolAddressAllocator<uint32_t>::reserved_size(1, m_creationParameters.maxSets, 1));
            m_descriptorSetAllocator = core::IteratablePoolAddressAllocator<uint32_t>(m_descriptorSetAllocatorReservedSpace.get(), 0, 0, 1, m_creationParameters.maxSets, 1);
        }

        virtual ~IDescriptorPool()
        {
            assert(m_descriptorSetAllocator.get_allocated_size() == 0);
        }

        virtual bool createDescriptorSets_impl(uint32_t count, const IGPUDescriptorSetLayout* const* layouts, SStorageOffsets* const offsets, core::smart_refctd_ptr<IGPUDescriptorSet>* output) = 0;

        virtual bool reset_impl() = 0;

    private:
        inline core::smart_refctd_ptr<asset::IDescriptor>* getDescriptorStorage(const asset::IDescriptor::E_TYPE type) const
        {
            core::smart_refctd_ptr<asset::IDescriptor>* baseAddress;
            switch (type)
            {
            case asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_textureStorage.get());
                break;
            case asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_storageImageStorage.get());
                break;
            case asset::IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_UTB_STBStorage.get());
                break;
            case asset::IDescriptor::E_TYPE::ET_STORAGE_TEXEL_BUFFER:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_UTB_STBStorage.get()) + m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER)];
                break;
            case asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_UBO_SSBOStorage.get());
                break;
            case asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_UBO_SSBOStorage.get()) + m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER)];
                break;
            case asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_UBO_SSBOStorage.get()) + (m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER)] + m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER)]);
                break;
            case asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_UBO_SSBOStorage.get()) + (m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER)] + m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER)] + m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC)]);
                break;
            case asset::IDescriptor::E_TYPE::ET_INPUT_ATTACHMENT:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_storageImageStorage.get()) + m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE)];
                break;
            case asset::IDescriptor::E_TYPE::ET_ACCELERATION_STRUCTURE:
                baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_accelerationStructureStorage.get());
                break;
            default:
                assert(!"Invalid code path.");
                return nullptr;
            }

            return baseAddress;
        }

        inline core::smart_refctd_ptr<IGPUSampler>* getMutableSamplerStorage() const
        {
            return reinterpret_cast<core::smart_refctd_ptr<IGPUSampler>*>(m_mutableSamplerStorage.get());
        }

        friend class IGPUDescriptorSet;
        // Returns the offset into the pool's descriptor storage. These offsets will be combined
        // later with base memory addresses to get the actual memory address where we put the core::smart_refctd_ptr<const IDescriptor>.
        bool allocateStorageOffsets(SStorageOffsets& offsets, const IGPUDescriptorSetLayout* layout);
        void rewindLastStorageAllocations(const uint32_t count, const SStorageOffsets* offsets, const IGPUDescriptorSetLayout *const *const layouts);

        void deleteSetStorage(IGPUDescriptorSet*& set);

        struct allocator_state_t
        {
            allocator_state_t(const uint32_t maxDescriptorCount, const bool useGeneralAllocator)
            {
                if (maxDescriptorCount == 0)
                    return;

                if (useGeneralAllocator)
                {
                    generalAllocatorReservedSpace = std::make_unique<uint8_t[]>(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(1u, maxDescriptorCount, 1u));
                    generalAllocator = core::GeneralpurposeAddressAllocator<uint32_t>(generalAllocatorReservedSpace.get(), 0u, 0u, 1u, maxDescriptorCount, 1u);
                }
                else
                {
                    linearAllocator = core::LinearAddressAllocator<uint32_t>(nullptr, 0u, 0u, 1u, maxDescriptorCount);
                }
            }

            ~allocator_state_t() {}

            inline uint32_t allocate(const uint32_t count)
            {
                if (generalAllocatorReservedSpace)
                    return generalAllocator.alloc_addr(count, 1u);
                else
                    return linearAllocator.alloc_addr(count, 1u);
            }

            inline void free(const uint32_t allocatedOffset, const uint32_t count)
            {
                assert(generalAllocatorReservedSpace);
                generalAllocator.free_addr(allocatedOffset, count);
            }

            inline void reset()
            {
                if (generalAllocatorReservedSpace)
                    generalAllocator.reset();
                else
                    linearAllocator.reset();
            }

            inline uint32_t getAllocatedDescriptorCount() const
            {
                if (generalAllocatorReservedSpace)
                    return generalAllocator.get_allocated_size();
                else
                    return linearAllocator.get_allocated_size();
            }

            union
            {
                core::LinearAddressAllocator<uint32_t> linearAllocator;
                core::GeneralpurposeAddressAllocator<uint32_t> generalAllocator;
            };
            std::unique_ptr<uint8_t[]> generalAllocatorReservedSpace = nullptr;
        };
        std::unique_ptr<allocator_state_t> m_descriptorAllocators[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT) + 1];

        const SCreateInfo m_creationParameters;

        core::IteratablePoolAddressAllocator<uint32_t> m_descriptorSetAllocator;
        std::unique_ptr<uint8_t[]> m_descriptorSetAllocatorReservedSpace = nullptr;
        std::unique_ptr<IGPUDescriptorSet* []> m_allocatedDescriptorSets = nullptr; // This array might be sparse.

        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<video::IGPUImageView>>[]> m_textureStorage;
        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<video::IGPUSampler>>[]> m_mutableSamplerStorage;
        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<IGPUImageView>>[]> m_storageImageStorage; // storage image | input attachment
        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<IGPUBuffer>>[]> m_UBO_SSBOStorage; // ubo | ssbo | ubo dynamic | ssbo dynamic
        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<IGPUBufferView>>[]> m_UTB_STBStorage; // utb | stb
        std::unique_ptr<core::StorageTrivializer<core::smart_refctd_ptr<IGPUAccelerationStructure>>[]> m_accelerationStructureStorage;
};

}

#endif