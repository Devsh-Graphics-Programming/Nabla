#include "nbl/video/IGPUDescriptorSet.h"

#include "nbl/video/IDescriptorPool.h"

namespace nbl::video
{

void IGPUDescriptorSet::allocateDescriptors()
{
    auto& offsets = m_descriptorStorageOffsets;

    for (uint32_t i = 0u; i < asset::EDT_COUNT; ++i)
    {
        const auto type = static_cast<asset::E_DESCRIPTOR_TYPE>(i);
        const auto count = getLayout()->getTotalDescriptorCount(type);
        if (count == 0ull)
            continue;

        if (m_pool->m_flags & IDescriptorPool::ECF_FREE_DESCRIPTOR_SET_BIT)
            offsets.data[type] = m_pool->m_generalAllocators[type].alloc_addr(count, 1u);
        else
            offsets.data[type] = m_pool->m_linearAllocators[type].alloc_addr(count, 1u);

        assert((offsets.data[type] < m_pool->m_maxDescriptorCount[i]) && "PANIC: Allocation failed. This shoudn't have happened! Check your descriptor pool.");
    }

    const auto mutableSamplerCount = getLayout()->getTotalMutableSamplerCount();
    if (mutableSamplerCount != 0ull)
    {
        if (m_pool->m_flags & IDescriptorPool::ECF_FREE_DESCRIPTOR_SET_BIT)
            offsets.data[asset::EDT_COUNT] = m_pool->m_generalAllocators[asset::EDT_COUNT].alloc_addr(mutableSamplerCount, 1u);
        else
            offsets.data[asset::EDT_COUNT] = m_pool->m_linearAllocators[asset::EDT_COUNT].alloc_addr(mutableSamplerCount, 1u);
    }
}

core::smart_refctd_ptr<asset::IDescriptor>* IGPUDescriptorSet::getDescriptorStorage(const asset::E_DESCRIPTOR_TYPE type) const
{
    core::smart_refctd_ptr<asset::IDescriptor>* baseAddress;
    switch (type)
    {
    case asset::EDT_COMBINED_IMAGE_SAMPLER:
        baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_textureStorage.get());
        break;
    case asset::EDT_STORAGE_IMAGE:
        baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_storageImageStorage.get());
        break;
    case asset::EDT_UNIFORM_TEXEL_BUFFER:
        baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UTB_STBStorage.get());
        break;
    case asset::EDT_STORAGE_TEXEL_BUFFER:
        baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UTB_STBStorage.get()) + m_pool->m_maxDescriptorCount[asset::EDT_UNIFORM_TEXEL_BUFFER];
        break;
    case asset::EDT_UNIFORM_BUFFER:
        baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UBO_SSBOStorage.get());
        break;
    case asset::EDT_STORAGE_BUFFER:
        baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UBO_SSBOStorage.get()) + m_pool->m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER];
        break;
    case asset::EDT_UNIFORM_BUFFER_DYNAMIC:
        baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UBO_SSBOStorage.get()) + (m_pool->m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER] + m_pool->m_maxDescriptorCount[asset::EDT_STORAGE_BUFFER]);
        break;
    case asset::EDT_STORAGE_BUFFER_DYNAMIC:
        baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UBO_SSBOStorage.get()) + (m_pool->m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER] + m_pool->m_maxDescriptorCount[asset::EDT_STORAGE_BUFFER] + m_pool->m_maxDescriptorCount[asset::EDT_UNIFORM_BUFFER_DYNAMIC]);
        break;
    case asset::EDT_INPUT_ATTACHMENT:
        baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_storageImageStorage.get()) + m_pool->m_maxDescriptorCount[asset::EDT_STORAGE_IMAGE];
        break;
    case asset::EDT_ACCELERATION_STRUCTURE:
        baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_accelerationStructureStorage.get());
        break;
    default:
        assert(!"Invalid code path.");
        return nullptr;
    }

    return baseAddress;
}

core::smart_refctd_ptr<IGPUSampler>* IGPUDescriptorSet::getMutableSamplerStorage() const
{
    return reinterpret_cast<core::smart_refctd_ptr<IGPUSampler>*>(m_pool->m_mutableSamplerStorage.get());
}

}