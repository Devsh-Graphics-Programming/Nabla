#include "nbl/video/IDescriptorPool.h"
#include "nbl/video/IGPUDescriptorSetLayout.h"

namespace nbl::video
{

IDescriptorPool::IDescriptorPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const SCreateInfo& createInfo)
    : IBackendObject(std::move(dev)), m_creationParameters(createInfo), m_logger(getOriginDevice()->getPhysicalDevice()->getDebugCallback() ? getOriginDevice()->getPhysicalDevice()->getDebugCallback()->getLogger():nullptr)
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

    m_allocatedDescriptorSets = std::make_unique<IGPUDescriptorSet * []>(m_creationParameters.maxSets);
    std::fill_n(m_allocatedDescriptorSets.get(), m_creationParameters.maxSets, nullptr);

    m_descriptorSetAllocatorReservedSpace = std::make_unique<uint8_t[]>(core::IteratablePoolAddressAllocator<uint32_t>::reserved_size(1, m_creationParameters.maxSets, 1));
    m_descriptorSetAllocator = core::IteratablePoolAddressAllocator<uint32_t>(m_descriptorSetAllocatorReservedSpace.get(), 0, 0, 1, m_creationParameters.maxSets, 1);
}

uint32_t IDescriptorPool::createDescriptorSets(uint32_t count, const IGPUDescriptorSetLayout* const* layouts, core::smart_refctd_ptr<IGPUDescriptorSet>* output)
{
    core::vector<uint32_t> reverseMap(count,count);

    core::vector<const IGPUDescriptorSetLayout*> repackedLayouts;
    core::vector<SStorageOffsets> offsets;
    repackedLayouts.reserve(count);
    offsets.reserve(count);
    for (uint32_t i=0u; i<count; ++i)
    {
        if (!layouts[i])
        {
            m_logger.log("Null descriptor set layout found at index %u, skipping!", system::ILogger::ELL_PERFORMANCE, i);
            continue;
        }

        if (!isCompatibleDevicewise(layouts[i]))
        {
            m_logger.log("Device-Incompatible descriptor set layout found at index %u.", system::ILogger::ELL_ERROR, i);
            continue;
        }
        
        if (!allocateStorageOffsets(offsets.emplace_back(),layouts[i]))
        {
            m_logger.log("Failed to allocate descriptor or descriptor set offsets in the pool's storage for descriptor set layout at index %u.", system::ILogger::ELL_WARNING, i);
            offsets.pop_back();
            continue;
        }
        reverseMap[i] = repackedLayouts.size();
        repackedLayouts.push_back(layouts[i]);
    }
    auto successCount = offsets.size();

    const bool creationSuccess = createDescriptorSets_impl(successCount,repackedLayouts.data(),offsets.data(),output);
    if (creationSuccess)
    {
        for (uint32_t i=0u; i<successCount; ++i)
            m_allocatedDescriptorSets[offsets[i].getSetOffset()] = output[i].get();
        // shuffle the outputs
        for (int32_t i=count-1; i>=0; i--)
            output[i] = reverseMap[i]<count ? output[reverseMap[i]]:nullptr;
    }
    else
    {
        // Free the allocated offsets for all the successfully allocated descriptor sets and the offset of the descriptor sets themselves.
        rewindLastStorageAllocations(successCount,offsets.data(),layouts);
        std::fill_n(output,count,nullptr);
        successCount = 0;
    }

    assert(count >= successCount);
    return successCount;
}

bool IDescriptorPool::reset()
{
    // something else except for the allocated sets needs to be holding onto the pool, so that
    // we don't have an implicit call to `~IDescriptorPool` on `this` before we get out of the current stackframe  
    assert(getReferenceCount() > m_descriptorSetAllocator.get_allocated_size());

    const auto& compilerIsRetarded = m_descriptorSetAllocator;
    for (const uint32_t setIndex : compilerIsRetarded)
        deleteSetStorage(setIndex);

    // the `reset` doesn't deallocate anything, but it rearranges the free address stack in order to
    // not make the future allocated addresses dependent on what happened before a reset
    assert(m_descriptorSetAllocator.get_allocated_size() == 0);
    m_descriptorSetAllocator.reset();

    // see the comment at the first assert
    assert(getReferenceCount() >= 1);
    return reset_impl();
}

bool IDescriptorPool::allocateStorageOffsets(SStorageOffsets& offsets, const IGPUDescriptorSetLayout* layout)
{
    bool success = true;

    for (uint32_t i = 0u; i <= static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++i)
    {
        uint32_t count = 0u;
        uint32_t maxCount = 0u;
        if (i == static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT))
        {
            count = layout->getTotalMutableSamplerCount();
            maxCount = m_creationParameters.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)];
        }
        else
        {
            const auto type = static_cast<asset::IDescriptor::E_TYPE>(i);
            count = layout->getTotalDescriptorCount(type);
            maxCount = m_creationParameters.maxDescriptorCount[i];
        }

        if (count == 0u)
            continue;

        offsets.data[i] = m_descriptorAllocators[i]->allocate(count);
        if (offsets.data[i] >= maxCount)
        {
            success = false;
            break;
        }
    }

    if (success)
    {
        // Allocate offset into the pool's m_allocatedDescriptorSets
        offsets.getSetOffset() = m_descriptorSetAllocator.alloc_addr(1u, 1u);
        if (offsets.getSetOffset() == m_descriptorSetAllocator.invalid_address)
            success = false;
    }

    if (!success)
        rewindLastStorageAllocations(1, &offsets, &layout);

    return success;
}

void IDescriptorPool::rewindLastStorageAllocations(const uint32_t count, const SStorageOffsets* offsets, const IGPUDescriptorSetLayout *const *const layouts)
{
    for (uint32_t j = 0; j < count; ++j)
    {
        if (offsets[j].getSetOffset() != SStorageOffsets::Invalid)
            m_descriptorSetAllocator.free_addr(offsets[j].getSetOffset(), 1u);
    }

    // Order of iteration important, once we find the lowest allocated offset for a type we can skip all other allocations in the case of linear allocator.
    for (uint32_t i = 0; i <= static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++i)
    {
        for (uint32_t j = 0; j < count; ++j)
        {
            if (offsets[j].data[i] != SStorageOffsets::Invalid)
            {
                if (allowsFreeing())
                {
                    m_descriptorAllocators[i]->free(offsets[j].data[i], layouts[j]->getTotalDescriptorCount(static_cast<asset::IDescriptor::E_TYPE>(i)));
                }
                else
                {
                    // First allocated offset will be the lowest.
                    m_descriptorAllocators[i]->linearAllocator.reset(offsets->data[i]);
                    break;
                }
            }
        }
    }
}

void IDescriptorPool::deleteSetStorage(const uint32_t setIndex)
{
    auto* set = m_allocatedDescriptorSets[setIndex];

    assert(set);
    assert(!set->isZombie());

    for (auto i = 0u; i < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++i)
    {
        const auto type = static_cast<asset::IDescriptor::E_TYPE>(i);
        const uint32_t allocatedOffset = set->m_storageOffsets.getDescriptorOffset(type);

        // There is no descriptor of such type in the set.
        if (allocatedOffset == ~0u)
            continue;

        const uint32_t count = set->getLayout()->getTotalDescriptorCount(type);
        assert(count != 0u);

        std::destroy_n(getDescriptorStorage(type) + allocatedOffset, count);

        if (allowsFreeing())
            m_descriptorAllocators[i]->free(allocatedOffset, count);
    }

    const auto mutableSamplerCount = set->getLayout()->getTotalMutableSamplerCount();
    if (mutableSamplerCount > 0)
    {
        const uint32_t allocatedOffset = set->m_storageOffsets.getMutableSamplerOffset();
        assert(allocatedOffset != ~0u);

        std::destroy_n(getMutableSamplerStorage() + allocatedOffset, mutableSamplerCount);

        if (allowsFreeing())
            m_descriptorAllocators[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)]->free(allocatedOffset, mutableSamplerCount);
    }

    m_descriptorSetAllocator.free_addr(set->m_storageOffsets.getSetOffset(), 1);
    if ((m_descriptorSetAllocator.get_allocated_size() == 0) && !allowsFreeing())
        reset();
    
    // Order is important because we don't want first nullify the pool (which will destroy the pool for the last surviving DS) because ~IDescriptorPool
    // checks if all the allocated DS have been set to nullptr.
    m_allocatedDescriptorSets[setIndex] = nullptr;
    set->m_pool = nullptr;
}

}