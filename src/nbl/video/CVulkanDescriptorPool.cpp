#include "nbl/video/CVulkanDescriptorPool.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanDescriptorPool::~CVulkanDescriptorPool()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyDescriptorPool(vulkanDevice->getInternalObject(), m_descriptorPool, nullptr);
}

void CVulkanDescriptorPool::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

	if(vkSetDebugUtilsObjectNameEXT == 0) return;

    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
	VkDebugUtilsObjectNameInfoEXT nameInfo = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr};
	nameInfo.objectType = VK_OBJECT_TYPE_DESCRIPTOR_POOL;
	nameInfo.objectHandle = reinterpret_cast<uint64_t>(getInternalObject());
	nameInfo.pObjectName = getObjectDebugName();
	vkSetDebugUtilsObjectNameEXT(vulkanDevice->getInternalObject(), &nameInfo);
}

core::smart_refctd_ptr<IGPUDescriptorSet> CVulkanDescriptorPool::createDescriptorSet_impl(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout)
{
    if (layout->getAPIType() != EAT_VULKAN)
        return nullptr;

    VkDescriptorSetAllocateInfo vk_allocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    vk_allocateInfo.pNext = nullptr; // pNext must be NULL or a pointer to a valid instance of VkDescriptorSetVariableDescriptorCountAllocateInfo

    vk_allocateInfo.descriptorPool = m_descriptorPool;
    vk_allocateInfo.descriptorSetCount = 1u;

    VkDescriptorSetLayout vk_dsLayout = IBackendObject::device_compatibility_cast<const CVulkanDescriptorSetLayout*>(layout.get(), getOriginDevice())->getInternalObject();
    vk_allocateInfo.pSetLayouts = &vk_dsLayout;

    VkDescriptorSet vk_descriptorSet;

    const auto* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    if (vk->vk.vkAllocateDescriptorSets(vulkanDevice->getInternalObject(), &vk_allocateInfo, &vk_descriptorSet) == VK_SUCCESS)
        return core::make_smart_refctd_ptr<CVulkanDescriptorSet>(core::smart_refctd_ptr<const CVulkanLogicalDevice>(vulkanDevice), std::move(layout), core::smart_refctd_ptr<IDescriptorPool>(this), vk_descriptorSet);

    return nullptr;
}

void CVulkanDescriptorPool::updateDescriptorSets_impl(uint32_t descriptorWriteCount, const IGPUDescriptorSet::SWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const IGPUDescriptorSet::SCopyDescriptorSet* pDescriptorCopies)
{
    constexpr uint32_t MAX_DESCRIPTOR_ARRAY_COUNT = 256u;

    core::vector<VkWriteDescriptorSet> vk_writeDescriptorSets(descriptorWriteCount);

    uint32_t bufferInfoOffset = 0u;
    core::vector<VkDescriptorBufferInfo >vk_bufferInfos(descriptorWriteCount * MAX_DESCRIPTOR_ARRAY_COUNT);

    uint32_t imageInfoOffset = 0u;
    core::vector<VkDescriptorImageInfo> vk_imageInfos(descriptorWriteCount * MAX_DESCRIPTOR_ARRAY_COUNT);

    uint32_t bufferViewOffset = 0u;
    core::vector<VkBufferView> vk_bufferViews(descriptorWriteCount * MAX_DESCRIPTOR_ARRAY_COUNT);

    core::vector<VkWriteDescriptorSetAccelerationStructureKHR> vk_writeDescriptorSetAS(descriptorWriteCount);

    uint32_t accelerationStructuresOffset = 0u;
    core::vector<VkAccelerationStructureKHR> vk_accelerationStructures(descriptorWriteCount * MAX_DESCRIPTOR_ARRAY_COUNT);

    for (uint32_t i = 0u; i < descriptorWriteCount; ++i)
    {
        vk_writeDescriptorSets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        vk_writeDescriptorSets[i].pNext = nullptr; // Each pNext member of any structure (including this one) in the pNext chain must be either NULL or a pointer to a valid instance of VkWriteDescriptorSetAccelerationStructureKHR, VkWriteDescriptorSetAccelerationStructureNV, or VkWriteDescriptorSetInlineUniformBlockEXT

        const IGPUDescriptorSetLayout* layout = pDescriptorWrites[i].dstSet->getLayout();
        if (layout->getAPIType() != EAT_VULKAN)
            continue;

        const CVulkanDescriptorSet* vulkanDescriptorSet = static_cast<const CVulkanDescriptorSet*>(pDescriptorWrites[i].dstSet);
        vk_writeDescriptorSets[i].dstSet = vulkanDescriptorSet->getInternalObject();

        vk_writeDescriptorSets[i].dstBinding = pDescriptorWrites[i].binding;
        vk_writeDescriptorSets[i].dstArrayElement = pDescriptorWrites[i].arrayElement;
        vk_writeDescriptorSets[i].descriptorType = getVkDescriptorTypeFromDescriptorType(pDescriptorWrites[i].descriptorType);
        vk_writeDescriptorSets[i].descriptorCount = pDescriptorWrites[i].count;

        assert(pDescriptorWrites[i].count <= MAX_DESCRIPTOR_ARRAY_COUNT);
        assert(pDescriptorWrites[i].info[0].desc);

        switch (pDescriptorWrites[i].info->desc->getTypeCategory())
        {
        case asset::IDescriptor::EC_BUFFER:
        {
            VkDescriptorBufferInfo dummyInfo = {};
            dummyInfo.buffer = static_cast<const CVulkanBuffer*>(pDescriptorWrites[i].info[0].desc.get())->getInternalObject();
            dummyInfo.offset = pDescriptorWrites[i].info[0].info.buffer.offset;
            dummyInfo.range = pDescriptorWrites[i].info[0].info.buffer.size;

            for (uint32_t j = 0u; j < pDescriptorWrites[i].count; ++j)
            {
                if (pDescriptorWrites[i].info[j].info.buffer.size)
                {
                    vk_bufferInfos[bufferInfoOffset + j].buffer = static_cast<const CVulkanBuffer*>(pDescriptorWrites[i].info[j].desc.get())->getInternalObject();
                    vk_bufferInfos[bufferInfoOffset + j].offset = pDescriptorWrites[i].info[j].info.buffer.offset;
                    vk_bufferInfos[bufferInfoOffset + j].range = pDescriptorWrites[i].info[j].info.buffer.size;
                }
                else
                {
                    vk_bufferInfos[bufferInfoOffset + j] = dummyInfo;
                }
            }

            vk_writeDescriptorSets[i].pBufferInfo = vk_bufferInfos.data() + bufferInfoOffset;
            bufferInfoOffset += pDescriptorWrites[i].count;
        } break;

        case asset::IDescriptor::EC_IMAGE:
        {
            const auto& firstDescWriteImageInfo = pDescriptorWrites[i].info[0].info.image;

            VkDescriptorImageInfo dummyInfo = { VK_NULL_HANDLE };
            if (firstDescWriteImageInfo.sampler && (firstDescWriteImageInfo.sampler->getAPIType() == EAT_VULKAN))
                dummyInfo.sampler = static_cast<const CVulkanSampler*>(firstDescWriteImageInfo.sampler.get())->getInternalObject();
            dummyInfo.imageView = static_cast<const CVulkanImageView*>(pDescriptorWrites[i].info[0].desc.get())->getInternalObject();
            dummyInfo.imageLayout = static_cast<VkImageLayout>(pDescriptorWrites[i].info[0].info.image.imageLayout);

            for (uint32_t j = 0u; j < pDescriptorWrites[i].count; ++j)
            {
                const auto& descriptorWriteImageInfo = pDescriptorWrites[i].info[j].info.image;
                if (descriptorWriteImageInfo.imageLayout != asset::IImage::EL_UNDEFINED)
                {
                    VkSampler vk_sampler = VK_NULL_HANDLE;
                    if (descriptorWriteImageInfo.sampler && (descriptorWriteImageInfo.sampler->getAPIType() == EAT_VULKAN))
                        vk_sampler = static_cast<const CVulkanSampler*>(descriptorWriteImageInfo.sampler.get())->getInternalObject();

                    VkImageView vk_imageView = static_cast<const CVulkanImageView*>(pDescriptorWrites[i].info[j].desc.get())->getInternalObject();

                    vk_imageInfos[imageInfoOffset + j].sampler = vk_sampler;
                    vk_imageInfos[imageInfoOffset + j].imageView = vk_imageView;
                    vk_imageInfos[imageInfoOffset + j].imageLayout = static_cast<VkImageLayout>(descriptorWriteImageInfo.imageLayout);
                }
                else
                {
                    vk_imageInfos[imageInfoOffset + j] = dummyInfo;
                }
            }

            vk_writeDescriptorSets[i].pImageInfo = vk_imageInfos.data() + imageInfoOffset;
            imageInfoOffset += pDescriptorWrites[i].count;
        } break;

        case asset::IDescriptor::EC_BUFFER_VIEW:
        {
            VkBufferView dummyBufferView = static_cast<const CVulkanBufferView*>(pDescriptorWrites[i].info[0].desc.get())->getInternalObject();

            for (uint32_t j = 0u; j < pDescriptorWrites[i].count; ++j)
            {
                if (pDescriptorWrites[i].info[j].info.buffer.size)
                {
                    vk_bufferViews[bufferViewOffset + j] = static_cast<const CVulkanBufferView*>(pDescriptorWrites[i].info[j].desc.get())->getInternalObject();
                }
                else
                {
                    vk_bufferViews[bufferViewOffset + j] = dummyBufferView;
                }
            }

            vk_writeDescriptorSets[i].pTexelBufferView = vk_bufferViews.data() + bufferViewOffset;
            bufferViewOffset += pDescriptorWrites[i].count;
        } break;

        case asset::IDescriptor::EC_ACCELERATION_STRUCTURE:
        {
            // Get WriteAS
            auto& writeAS = vk_writeDescriptorSetAS[i];
            writeAS = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, nullptr };
            // Fill Write AS
            for (uint32_t j = 0u; j < pDescriptorWrites[i].count; ++j)
            {
                VkAccelerationStructureKHR vk_accelerationStructure = static_cast<const CVulkanAccelerationStructure*>(pDescriptorWrites[i].info[j].desc.get())->getInternalObject();
                vk_accelerationStructures[j + accelerationStructuresOffset] = vk_accelerationStructure;
            }

            writeAS.accelerationStructureCount = pDescriptorWrites[i].count;
            writeAS.pAccelerationStructures = &vk_accelerationStructures[accelerationStructuresOffset];

            // Give Write AS to writeDescriptor.pNext
            vk_writeDescriptorSets[i].pNext = &writeAS;

            accelerationStructuresOffset += pDescriptorWrites[i].count;
        } break;

        default:
            assert(!"Don't know what to do with this value!");
        }
    }

    core::vector<VkCopyDescriptorSet> vk_copyDescriptorSets(descriptorCopyCount);

    for (uint32_t i = 0u; i < descriptorCopyCount; ++i)
    {
        vk_copyDescriptorSets[i].sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
        vk_copyDescriptorSets[i].pNext = nullptr; // pNext must be NULL
        vk_copyDescriptorSets[i].srcSet = static_cast<const CVulkanDescriptorSet*>(pDescriptorCopies[i].srcSet)->getInternalObject();
        vk_copyDescriptorSets[i].srcBinding = pDescriptorCopies[i].srcBinding;
        vk_copyDescriptorSets[i].srcArrayElement = pDescriptorCopies[i].srcArrayElement;
        vk_copyDescriptorSets[i].dstSet = static_cast<const CVulkanDescriptorSet*>(pDescriptorCopies[i].dstSet)->getInternalObject();
        vk_copyDescriptorSets[i].dstBinding = pDescriptorCopies[i].dstBinding;
        vk_copyDescriptorSets[i].dstArrayElement = pDescriptorCopies[i].dstArrayElement;
        vk_copyDescriptorSets[i].descriptorCount = pDescriptorCopies[i].count;
    }

    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkUpdateDescriptorSets(vulkanDevice->getInternalObject(), descriptorWriteCount, vk_writeDescriptorSets.data(), descriptorCopyCount, vk_copyDescriptorSets.data());
}

bool CVulkanDescriptorPool::freeDescriptorSets_impl(const uint32_t descriptorSetCount, IGPUDescriptorSet* const* const descriptorSets)
{
    constexpr auto MaxDescriptorSetCount = 4u;
    assert(descriptorSetCount <= MaxDescriptorSetCount);
    VkDescriptorSet vk_descriptorSets[MaxDescriptorSetCount];

    for (auto i = 0; i < descriptorSetCount; ++i)
    {
        if (descriptorSets[i]->getAPIType() != EAT_VULKAN)
            return false;

        vk_descriptorSets[i] = static_cast<CVulkanDescriptorSet*>(descriptorSets[i])->getInternalObject();
    }

    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    return vk->vk.vkFreeDescriptorSets(vulkanDevice->getInternalObject(), m_descriptorPool, descriptorSetCount, vk_descriptorSets) == VK_SUCCESS;
}

}