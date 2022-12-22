#include "nbl/video/IGPUDescriptorSet.h"

#include "nbl/video/IDescriptorPool.h"

namespace nbl::video
{

IGPUDescriptorSet::IGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout, core::smart_refctd_ptr<IDescriptorPool>&& pool, IDescriptorPool::SDescriptorOffsets&& offsets)
    : base_t(std::move(layout)), IBackendObject(std::move(core::smart_refctd_ptr<const ILogicalDevice>(pool->getOriginDevice()))), m_version(0ull), m_pool(std::move(pool)), m_descriptorStorageOffsets(std::move(offsets))
{
    for (auto i = 0u; i < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++i)
    {
        // There is no descriptor of such type in the set.
        if (m_descriptorStorageOffsets.data[i] == ~0u)
            continue;

        const auto type = static_cast<asset::IDescriptor::E_TYPE>(i);

        // Default-construct the core::smart_refctd_ptr<IDescriptor>s because even if the user didn't update the descriptor set with ILogicalDevice::updateDescriptorSet we
        // won't have uninitialized memory and destruction wouldn't crash in ~IGPUDescriptorSet.
        std::uninitialized_default_construct_n(getDescriptorStorage(type) + m_descriptorStorageOffsets.data[i], m_layout->getTotalDescriptorCount(type));
    }

    const auto mutableSamplerCount = m_layout->getTotalMutableSamplerCount();
    if (mutableSamplerCount > 0)
        std::uninitialized_default_construct_n(getMutableSamplerStorage() + m_descriptorStorageOffsets.data[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT)], mutableSamplerCount);
}

core::smart_refctd_ptr<asset::IDescriptor>* IGPUDescriptorSet::getDescriptorStorage(const asset::IDescriptor::E_TYPE type) const
{
    core::smart_refctd_ptr<asset::IDescriptor>* baseAddress;
    switch (type)
    {
    case asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER:
        baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_textureStorage.get());
        break;
    case asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE:
        baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_storageImageStorage.get());
        break;
    case asset::IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER:
        baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UTB_STBStorage.get());
        break;
    case asset::IDescriptor::E_TYPE::ET_STORAGE_TEXEL_BUFFER:
        baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UTB_STBStorage.get()) + m_pool->m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER)];
        break;
    case asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER:
        baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UBO_SSBOStorage.get());
        break;
    case asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER:
        baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UBO_SSBOStorage.get()) + m_pool->m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER)];
        break;
    case asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC:
        baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UBO_SSBOStorage.get()) + (m_pool->m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER)] + m_pool->m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER)]);
        break;
    case asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC:
        baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_UBO_SSBOStorage.get()) + (m_pool->m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER)] + m_pool->m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER)] + m_pool->m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC)]);
        break;
    case asset::IDescriptor::E_TYPE::ET_INPUT_ATTACHMENT:
        baseAddress = reinterpret_cast<core::smart_refctd_ptr<asset::IDescriptor>*>(m_pool->m_storageImageStorage.get()) + m_pool->m_maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE)];
        break;
    case asset::IDescriptor::E_TYPE::ET_ACCELERATION_STRUCTURE:
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