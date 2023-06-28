#ifndef _NBL_VIDEO_C_VULKAN_LOGICAL_DEVICE_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_LOGICAL_DEVICE_H_INCLUDED_


#include "nbl/core/containers/CMemoryPool.h"

#include <algorithm>

#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/CVulkanCommon.h"
#include "nbl/video/CVulkanDeviceFunctionTable.h"
#include "nbl/video/CVulkanSwapchain.h"
#include "nbl/video/CVulkanQueue.h"
#include "nbl/video/CVulkanRenderpass.h"
#include "nbl/video/CVulkanImageView.h"
#include "nbl/video/CVulkanFramebuffer.h"
#include "nbl/video/CVulkanSemaphore.h"
#include "nbl/video/CVulkanShader.h"
#include "nbl/video/CVulkanSpecializedShader.h"
#include "nbl/video/CVulkanCommandPool.h"
#include "nbl/video/CVulkanDescriptorSetLayout.h"
#include "nbl/video/CVulkanSampler.h"
#include "nbl/video/CVulkanPipelineLayout.h"
#include "nbl/video/CVulkanPipelineCache.h"
#include "nbl/video/CVulkanComputePipeline.h"
#include "nbl/video/CVulkanDescriptorPool.h"
#include "nbl/video/CVulkanDescriptorSet.h"
#include "nbl/video/CVulkanMemoryAllocation.h"
#include "nbl/video/CVulkanBuffer.h"
#include "nbl/video/CVulkanBufferView.h"
#include "nbl/video/CVulkanImage.h"
#include "nbl/video/CVulkanDeferredOperation.h"
#include "nbl/video/CVulkanAccelerationStructure.h"
#include "nbl/video/CVulkanGraphicsPipeline.h"
#include "nbl/video/CVulkanRenderpassIndependentPipeline.h"


namespace nbl::video
{

class CVulkanCommandBuffer;

class CVulkanLogicalDevice final : public ILogicalDevice
{
    public:
        // in the future we'll make proper Vulkan allocators and RAII free functions to pass into Vulkan API calls
        using memory_pool_mt_t = core::CMemoryPool<core::PoolAddressAllocator<uint32_t>,core::default_aligned_allocator,true,uint32_t>;
        
        CVulkanLogicalDevice(core::smart_refctd_ptr<const IAPIConnection>&& api, renderdoc_api_t* const rdoc, const IPhysicalDevice* const physicalDevice, const VkDevice vkdev, const VkInstance vkinst, const SCreationParams& params);

        // sync sutff
        inline IQueue::RESULT waitIdle() const override
        {
            return CVulkanQueue::getResultFrom(m_devf.vk.vkDeviceWaitIdle(m_vkdev));
        }
            
        core::smart_refctd_ptr<ISemaphore> createSemaphore(const uint64_t initialValue) override;
        WAIT_RESULT waitForSemaphores(const uint32_t count, const SSemaphoreWaitInfo* const infos, const bool waitAll, const uint64_t timeout) override;
            
        core::smart_refctd_ptr<IEvent> createEvent(const IEvent::CREATE_FLAGS flags) override;
              
        core::smart_refctd_ptr<IDeferredOperation> createDeferredOperation() override;

        // memory  stuff
        SAllocation allocate(const SAllocateInfo& info) override;
        bool flushMappedMemoryRanges_impl(const core::SRange<const MappedMemoryRange>& ranges) override;
        bool invalidateMappedMemoryRanges_impl(const core::SRange<const MappedMemoryRange>& ranges) override;

        // memory binding
        bool bindBufferMemory_impl(const uint32_t count, const SBindBufferMemoryInfo* pInfos) override;
        bool bindImageMemory_impl(const uint32_t count, const SBindImageMemoryInfo* pInfos) override;

        // descriptor creation
        core::smart_refctd_ptr<IGPUBuffer> createBuffer_impl(IGPUBuffer::SCreationParams&& creationParams) override;
        core::smart_refctd_ptr<IGPUBufferView> createBufferView_impl(const asset::SBufferRange<const IGPUBuffer>& underlying, const asset::E_FORMAT _fmt) override;
        core::smart_refctd_ptr<IGPUImage> createImage_impl(IGPUImage::SCreationParams&& params) override;
        core::smart_refctd_ptr<IGPUImageView> createImageView_impl(IGPUImageView::SCreationParams&& params) override;
        core::smart_refctd_ptr<IGPUSampler> createSampler(const IGPUSampler::SParams& _params) override;
        core::smart_refctd_ptr<IGPUAccelerationStructure> createAccelerationStructure_impl(IGPUAccelerationStructure::SCreationParams&& params) override;

        // shaders
        core::smart_refctd_ptr<IGPUShader> createShader(core::smart_refctd_ptr<asset::ICPUShader>&& cpushader, const asset::ISPIRVOptimizer* optimizer) override;
        inline core::smart_refctd_ptr<IGPUSpecializedShader> createSpecializedShader_impl(const IGPUShader* _unspecialized, const asset::ISpecializedShader::SInfo& specInfo) override
        {
            return core::make_smart_refctd_ptr<CVulkanSpecializedShader>(core::smart_refctd_ptr<const CVulkanLogicalDevice>(this),core::smart_refctd_ptr<const CVulkanShader>(static_cast<const CVulkanShader*>(_unspecialized)),specInfo);
        }

        // layouts
        core::smart_refctd_ptr<IGPUDescriptorSetLayout> createDescriptorSetLayout_impl(const core::SRange<const IGPUDescriptorSetLayout::SBinding>& bindings, const uint32_t maxSamplersCount) override;
        core::smart_refctd_ptr<IGPUPipelineLayout> createPipelineLayout_impl(
            const core::SRange<const asset::SPushConstantRange>& pcRanges,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3
        ) override;

        // descriptor sets
        core::smart_refctd_ptr<IDescriptorPool> createDescriptorPool_impl(const IDescriptorPool::SCreateInfo& createInfo) override;














        // commandpool creation
        inline core::smart_refctd_ptr<IGPUCommandPool> createCommandPool(const uint32_t familyIndex, const core::bitflag<IGPUCommandPool::CREATE_FLAGS> flags) override
        {
            VkCommandPoolCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
            vk_createInfo.pNext = nullptr; // pNext must be NULL
            vk_createInfo.flags = static_cast<VkCommandPoolCreateFlags>(flags.value);
            vk_createInfo.queueFamilyIndex = familyIndex;

            VkCommandPool vk_commandPool = VK_NULL_HANDLE;
            if (m_devf.vk.vkCreateCommandPool(m_vkdev, &vk_createInfo, nullptr, &vk_commandPool) == VK_SUCCESS)
                return core::make_smart_refctd_ptr<CVulkanCommandPool>(core::smart_refctd_ptr<const CVulkanLogicalDevice>(this),flags,familyIndex,vk_commandPool);

            return nullptr;
        }

     
    // TODO: validation and factor out to `_impl`
    core::smart_refctd_ptr<IGPURenderpass> createRenderpass(const IGPURenderpass::SCreationParams& params) override
    {
        // Nothing useful in pNext, didn't implement VRS yet
        VkRenderPassCreateInfo2 createInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2,nullptr };
        createInfo.flags = static_cast<VkRenderPassCreateFlags>(0u); // No flags are supported by us (there exists QCOM stuff only)
        createInfo.attachmentCount = params.attachmentCount;

        // TODO reduce number of allocations/get rid of vectors
        core::vector<VkAttachmentDescription2> attachments(createInfo.attachmentCount,{VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,nullptr});
        for (uint32_t i=0u; i<attachments.size(); ++i)
        {
            const auto& att = params.attachments[i];
            auto& vkatt = attachments[i];
            vkatt.flags = att.mayAlias ? VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT:0u;
            vkatt.format = getVkFormatFromFormat(att.format);
            vkatt.samples = static_cast<VkSampleCountFlagBits>(att.samples);
            vkatt.loadOp = static_cast<VkAttachmentLoadOp>(att.loadOp);
            vkatt.storeOp = static_cast<VkAttachmentStoreOp>(att.storeOp);

            // Todo(achal): Do we want these??
            vkatt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            vkatt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

            vkatt.initialLayout = getVkImageLayoutFromImageLayout(att.initialLayout);
            vkatt.finalLayout = getVkImageLayoutFromImageLayout(att.finalLayout);
        }
        createInfo.pAttachments = attachments.data();

        createInfo.subpassCount = params.subpassCount;
        core::vector<VkSubpassDescription2> vk_subpasses(createInfo.subpassCount,{VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2,nullptr});
        
        constexpr uint32_t MemSz = 1u << 12;
        constexpr uint32_t MaxAttachmentRefs = MemSz / sizeof(VkAttachmentReference);
        VkAttachmentReference2 vk_attRefs[MaxAttachmentRefs]; // TODO: initialize properly
        uint32_t preserveAttRefs[MaxAttachmentRefs];

        uint32_t totalAttRefCount = 0u;
        uint32_t totalPreserveCount = 0u;

        auto fillUpVkAttachmentRefHandles = [&vk_attRefs, &totalAttRefCount](const uint32_t count, const auto* srcRef, uint32_t& dstCount, auto*& dstRef)
        {
            for (uint32_t j = 0u; j < count; ++j)
            {
                vk_attRefs[totalAttRefCount + j].attachment = srcRef[j].attachment;
                vk_attRefs[totalAttRefCount + j].layout = getVkImageLayoutFromImageLayout(srcRef[j].layout);
            }

            dstRef = srcRef ? vk_attRefs + totalAttRefCount : nullptr;
            dstCount = count;
            totalAttRefCount += count;
        };

        for (uint32_t i = 0u; i < params.subpassCount; ++i)
        {
            auto& vk_subpass = vk_subpasses[i];
            const auto& subpass = params.subpasses[i];

            vk_subpass.flags = static_cast<VkSubpassDescriptionFlags>(subpass.flags);
            vk_subpass.pipelineBindPoint = static_cast<VkPipelineBindPoint>(subpass.pipelineBindPoint);

            // Copy over input attachments for this subpass
            fillUpVkAttachmentRefHandles(subpass.inputAttachmentCount, subpass.inputAttachments,
                vk_subpass.inputAttachmentCount, vk_subpass.pInputAttachments);

            // Copy over color attachments for this subpass
            fillUpVkAttachmentRefHandles(subpass.colorAttachmentCount, subpass.colorAttachments,
                vk_subpass.colorAttachmentCount, vk_subpass.pColorAttachments);

            // Copy over resolve attachments for this subpass
            vk_subpass.pResolveAttachments = nullptr;
            if (subpass.resolveAttachments)
            {
                uint32_t unused;
                fillUpVkAttachmentRefHandles(subpass.colorAttachmentCount, subpass.resolveAttachments, unused, vk_subpass.pResolveAttachments);
            }

            // Copy over depth-stencil attachment for this subpass
            vk_subpass.pDepthStencilAttachment = nullptr;
            if (subpass.depthStencilAttachment)
            {
                uint32_t unused;
                fillUpVkAttachmentRefHandles(1u, subpass.depthStencilAttachment, unused, vk_subpass.pDepthStencilAttachment);
            }

            // Copy over attachments that need to be preserved for this subpass
            vk_subpass.preserveAttachmentCount = subpass.preserveAttachmentCount;
            vk_subpass.pPreserveAttachments = nullptr;
            if (subpass.preserveAttachments)
            {
                for (uint32_t j = 0u; j < subpass.preserveAttachmentCount; ++j)
                    preserveAttRefs[totalPreserveCount + j] = subpass.preserveAttachments[j];

                vk_subpass.pPreserveAttachments = preserveAttRefs + totalPreserveCount;
                totalPreserveCount += subpass.preserveAttachmentCount;
            }
        }
        assert(totalAttRefCount <= MaxAttachmentRefs);
        assert(totalPreserveCount <= MaxAttachmentRefs);

        createInfo.pSubpasses = vk_subpasses.data();

        createInfo.dependencyCount = params.dependencyCount;
        core::vector<VkSubpassDependency2> deps(createInfo.dependencyCount,{VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2,nullptr});
        for (uint32_t i = 0u; i < deps.size(); ++i)
        {
            const auto& dep = params.dependencies[i];
            auto& vkdep = deps[i];

            vkdep.srcSubpass = dep.srcSubpass;
            vkdep.dstSubpass = dep.dstSubpass;
            vkdep.srcStageMask = getVkPipelineStageFlagsFromPipelineStageFlags(dep.srcStageMask);
            vkdep.dstStageMask = getVkPipelineStageFlagsFromPipelineStageFlags(dep.dstStageMask);
            vkdep.srcAccessMask = getVkAccessFlagsFromAccessFlags(dep.srcAccessMask);
            vkdep.dstAccessMask = getVkAccessFlagsFromAccessFlags(dep.dstAccessMask);
            vkdep.dependencyFlags = static_cast<VkDependencyFlags>(dep.dependencyFlags);
        }
        createInfo.pDependencies = deps.data();

        constexpr auto MaxMultiviewViewCount = IGPURenderpass::SCreationParams::MaxMultiviewViewCount;
        uint32_t viewMasks[MaxMultiviewViewCount] = {0u};
        createInfo.pCorrelatedViewMasks = viewMasks;
        // group up
        for (auto i=0u; i<MaxMultiviewViewCount; i++)
        if (params.viewCorrelationGroup[i]<MaxMultiviewViewCount)
            viewMasks[i] |= 0x1u<<i;
        // compact
        createInfo.correlatedViewMaskCount = 0u;
        for (auto i=0u; i<MaxMultiviewViewCount; i++)
        if (i!=createInfo.correlatedViewMaskCount)
            viewMasks[createInfo.correlatedViewMaskCount++] = viewMasks[i];

        VkRenderPass vk_renderpass;
        if (m_devf.vk.vkCreateRenderPass2(m_vkdev, &createInfo, nullptr, &vk_renderpass) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanRenderpass>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this), params, vk_renderpass);
        }
        else
        {
            return nullptr;
        }
    }

    void updateDescriptorSets_impl(uint32_t descriptorWriteCount, const IGPUDescriptorSet::SWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const IGPUDescriptorSet::SCopyDescriptorSet* pDescriptorCopies)
    {
        core::vector<VkWriteDescriptorSet> vk_writeDescriptorSets(descriptorWriteCount);
        core::vector<VkWriteDescriptorSetAccelerationStructureKHR> vk_writeDescriptorSetAS(descriptorWriteCount);

        core::vector<VkDescriptorBufferInfo> vk_bufferInfos;
        core::vector<VkDescriptorImageInfo> vk_imageInfos;
        core::vector<VkBufferView> vk_bufferViews;
        core::vector<VkAccelerationStructureKHR> vk_accelerationStructures;

        for (uint32_t i = 0u; i < descriptorWriteCount; ++i)
        {
            vk_writeDescriptorSets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            vk_writeDescriptorSets[i].pNext = nullptr; // Each pNext member of any structure (including this one) in the pNext chain must be either NULL or a pointer to a valid instance of VkWriteDescriptorSetAccelerationStructureKHR, VkWriteDescriptorSetAccelerationStructureNV, or VkWriteDescriptorSetInlineUniformBlockEXT

            const CVulkanDescriptorSet* vulkanDescriptorSet = static_cast<const CVulkanDescriptorSet*>(pDescriptorWrites[i].dstSet);
            vk_writeDescriptorSets[i].dstSet = vulkanDescriptorSet->getInternalObject();

            vk_writeDescriptorSets[i].dstBinding = pDescriptorWrites[i].binding;
            vk_writeDescriptorSets[i].dstArrayElement = pDescriptorWrites[i].arrayElement;
            vk_writeDescriptorSets[i].descriptorType = getVkDescriptorTypeFromDescriptorType(pDescriptorWrites[i].descriptorType);
            vk_writeDescriptorSets[i].descriptorCount = pDescriptorWrites[i].count;

            const auto bindingWriteCount = pDescriptorWrites[i].count;

            switch (pDescriptorWrites[i].info->desc->getTypeCategory())
            {
            case asset::IDescriptor::EC_BUFFER:
            {
                vk_writeDescriptorSets[i].pBufferInfo = reinterpret_cast<VkDescriptorBufferInfo*>(vk_bufferInfos.size());
                vk_bufferInfos.resize(vk_bufferInfos.size() + bindingWriteCount);
            } break;

            case asset::IDescriptor::EC_IMAGE:
            {
                vk_writeDescriptorSets[i].pImageInfo = reinterpret_cast<VkDescriptorImageInfo*>(vk_imageInfos.size());
                vk_imageInfos.resize(vk_imageInfos.size() + bindingWriteCount);
            } break;

            case asset::IDescriptor::EC_BUFFER_VIEW:
            {
                vk_writeDescriptorSets[i].pTexelBufferView = reinterpret_cast<VkBufferView*>(vk_bufferViews.size());
                vk_bufferViews.resize(vk_bufferViews.size() + bindingWriteCount);
            } break;

            case asset::IDescriptor::EC_ACCELERATION_STRUCTURE:
            {
                auto& writeAS = vk_writeDescriptorSetAS[i];
                writeAS = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, nullptr };
                writeAS.accelerationStructureCount = bindingWriteCount;
                vk_writeDescriptorSets[i].pNext = &writeAS;

                writeAS.pAccelerationStructures = reinterpret_cast<VkAccelerationStructureKHR*>(vk_accelerationStructures.size());
                vk_accelerationStructures.resize(vk_accelerationStructures.size() + bindingWriteCount);
            } break;

            default:
                assert(!"Invalid code path.");
            }
        }

        for (uint32_t i = 0u; i < descriptorWriteCount; ++i)
        {
            switch (pDescriptorWrites[i].info->desc->getTypeCategory())
            {
            case asset::IDescriptor::E_CATEGORY::EC_BUFFER:
            {
                vk_writeDescriptorSets[i].pBufferInfo = reinterpret_cast<size_t>(vk_writeDescriptorSets[i].pBufferInfo) + vk_bufferInfos.data();

                const auto* infoSrc = pDescriptorWrites[i].info;
                auto* infoDst = const_cast<VkDescriptorBufferInfo*>(vk_writeDescriptorSets[i].pBufferInfo);
                for (uint32_t j = 0; j < pDescriptorWrites[i].count; ++j, ++infoSrc, ++infoDst)
                {
                    infoDst->buffer = static_cast<const CVulkanBuffer*>(infoSrc->desc.get())->getInternalObject();
                    infoDst->offset = infoSrc->info.buffer.offset;
                    infoDst->range = infoSrc->info.buffer.size;
                }
            } break;

            case asset::IDescriptor::E_CATEGORY::EC_IMAGE:
            {
                vk_writeDescriptorSets[i].pImageInfo = reinterpret_cast<size_t>(vk_writeDescriptorSets[i].pImageInfo) + vk_imageInfos.data();

                const auto* infoSrc = pDescriptorWrites[i].info;
                auto* infoDst = const_cast<VkDescriptorImageInfo*>(vk_writeDescriptorSets[i].pImageInfo);

                for (uint32_t j = 0; j < pDescriptorWrites[i].count; ++j, ++infoSrc, ++infoDst)
                {
                    VkSampler vk_sampler = infoSrc->info.image.sampler ? static_cast<const CVulkanSampler*>(infoSrc->info.image.sampler.get())->getInternalObject() : VK_NULL_HANDLE;

                    infoDst->sampler = vk_sampler;
                    infoDst->imageView = static_cast<const CVulkanImageView*>(infoSrc->desc.get())->getInternalObject();
                    infoDst->imageLayout = getVkImageLayoutFromImageLayout(infoSrc->info.image.imageLayout);
                }
            } break;

            case asset::IDescriptor::E_CATEGORY::EC_BUFFER_VIEW:
            {
                vk_writeDescriptorSets[i].pTexelBufferView = reinterpret_cast<size_t>(vk_writeDescriptorSets[i].pTexelBufferView) + vk_bufferViews.data();

                const auto* infoSrc = pDescriptorWrites[i].info;
                auto* infoDst = const_cast<VkBufferView*>(vk_writeDescriptorSets[i].pTexelBufferView);
                for (uint32_t j = 0u; j < pDescriptorWrites[i].count; ++j, ++infoSrc, ++infoDst)
                    *infoDst = static_cast<const CVulkanBufferView*>(infoSrc->desc.get())->getInternalObject();
            } break;

            case asset::IDescriptor::E_CATEGORY::EC_ACCELERATION_STRUCTURE:
            {
                vk_writeDescriptorSetAS[i].pAccelerationStructures = reinterpret_cast<size_t>(vk_writeDescriptorSetAS[i].pAccelerationStructures) + vk_accelerationStructures.data();

                const auto* infoSrc = pDescriptorWrites[i].info;
                auto* infoDst = const_cast<VkAccelerationStructureKHR*>(vk_writeDescriptorSetAS[i].pAccelerationStructures);
                for (uint32_t j = 0u; j < pDescriptorWrites[i].count; ++j, ++infoSrc, ++infoDst)
                    *infoDst = static_cast<const CVulkanAccelerationStructure*>(infoSrc->desc.get())->getInternalObject();
            } break;

            default:
                assert(!"Invalid code path.");
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

        m_devf.vk.vkUpdateDescriptorSets(m_vkdev, descriptorWriteCount, vk_writeDescriptorSets.data(), descriptorCopyCount, vk_copyDescriptorSets.data());
    }

    uint64_t getBufferDeviceAddress(IGPUBuffer* buffer) override
    {
        constexpr uint64_t invalid_address = ~0ull;
        CVulkanBuffer* vulkanBuffer = IBackendObject::device_compatibility_cast<CVulkanBuffer*>(buffer, this);
        if (!vulkanBuffer)
        {
            // TODO: log error
            assert(false);
            return invalid_address;
        }

        if (!buffer->getCreationParams().usage.hasFlags(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT))
        {
            // TODO: log error: Buffer should've been created with EUF_SHADER_DEVICE_ADDRESS_BIT
            assert(false);
            return invalid_address;
        }

        if (!m_enabledFeatures.bufferDeviceAddress)
        {
            // TODO: log error: bufferDeviceAddress extension is not enbaled
            assert(false);
            return invalid_address;
        }

        VkBufferDeviceAddressInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        info.buffer = vulkanBuffer->getInternalObject();
        return m_devf.vk.vkGetBufferDeviceAddress(m_vkdev, &info);
    }

            
    core::smart_refctd_ptr<IQueryPool> createQueryPool(IQueryPool::SCreationParams&& params) override;
    
    bool getQueryPoolResults(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, size_t dataSize, void * pData, uint64_t stride, core::bitflag<IQueryPool::E_QUERY_RESULTS_FLAGS> flags) override;

    bool buildAccelerationStructures(
        core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation,
        const core::SRange<IGPUAccelerationStructure::HostBuildGeometryInfo>& pInfos,
        IGPUAccelerationStructure::BuildRangeInfo* const* ppBuildRangeInfos) override;

    bool copyAccelerationStructure(core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation, const IGPUAccelerationStructure::CopyInfo& copyInfo) override;
    
    bool copyAccelerationStructureToMemory(core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation, const IGPUAccelerationStructure::HostCopyToMemoryInfo& copyInfo) override;

    bool copyAccelerationStructureFromMemory(core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation, const IGPUAccelerationStructure::HostCopyFromMemoryInfo& copyInfo) override;

    IGPUAccelerationStructure::BuildSizes getAccelerationStructureBuildSizes(const IGPUAccelerationStructure::HostBuildGeometryInfo& pBuildInfo, const uint32_t* pMaxPrimitiveCounts) override;

    IGPUAccelerationStructure::BuildSizes getAccelerationStructureBuildSizes(const IGPUAccelerationStructure::DeviceBuildGeometryInfo& pBuildInfo, const uint32_t* pMaxPrimitiveCounts) override;

    inline memory_pool_mt_t& getMemoryPoolForDeferredOperations()
    {
        return m_deferred_op_mempool;
    }

    const CVulkanDeviceFunctionTable* getFunctionTable() const { return &m_devf; }

    inline const void* getNativeHandle() const {return &m_vkdev;}
    VkDevice getInternalObject() const {return m_vkdev;}

protected:
    bool createCommandBuffers_impl(IGPUCommandPool* cmdPool, IGPUCommandBuffer::E_LEVEL level,
        uint32_t count, core::smart_refctd_ptr<IGPUCommandBuffer>* outCmdBufs) override;

    bool freeCommandBuffers_impl(IGPUCommandBuffer** _cmdbufs, uint32_t _count) override
    {
        return false;
    }

    core::smart_refctd_ptr<IGPUFramebuffer> createFramebuffer_impl(IGPUFramebuffer::SCreationParams&& params) override
    {
        // This flag isn't supported until Vulkan 1.2
        // assert(!(m_params.flags & ECF_IMAGELESS_BIT));

        constexpr uint32_t MemSize = 1u << 12;
        constexpr uint32_t MaxAttachments = MemSize / sizeof(VkImageView);

        VkImageView vk_attachments[MaxAttachments];
        uint32_t attachmentCount = 0u;
        for (uint32_t i = 0u; i < params.attachmentCount; ++i)
        {
            if (params.attachments[i]->getAPIType() == EAT_VULKAN)
            {
                vk_attachments[i] = IBackendObject::device_compatibility_cast<const CVulkanImageView*>(params.attachments[i].get(), this)->getInternalObject();
                ++attachmentCount;
            }
        }
        assert(attachmentCount <= MaxAttachments);

        VkFramebufferCreateInfo createInfo = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        createInfo.pNext = nullptr;
        createInfo.flags = static_cast<VkFramebufferCreateFlags>(params.flags);

        if (params.renderpass->getAPIType() != EAT_VULKAN)
            return nullptr;

        createInfo.renderPass = IBackendObject::device_compatibility_cast<const CVulkanRenderpass*>(params.renderpass.get(), this)->getInternalObject();
        createInfo.attachmentCount = attachmentCount;
        createInfo.pAttachments = vk_attachments;
        createInfo.width = params.width;
        createInfo.height = params.height;
        createInfo.layers = params.layers;

        VkFramebuffer vk_framebuffer;
        if (m_devf.vk.vkCreateFramebuffer(m_vkdev, &createInfo, nullptr, &vk_framebuffer) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanFramebuffer>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this), std::move(params), vk_framebuffer);
        }
        else
        {
            return nullptr;
        }
    }

    core::smart_refctd_ptr<IGPUSpecializedShader> createSpecializedShader_impl(
        const IGPUShader* _unspecialized,
        const asset::ISpecializedShader::SInfo& specInfo) override
    {
        if (_unspecialized->getAPIType() != EAT_VULKAN)
            return nullptr;

        const CVulkanShader* vulkanShader = IBackendObject::device_compatibility_cast<const CVulkanShader*>(_unspecialized, this);

        return core::make_smart_refctd_ptr<CVulkanSpecializedShader>(
            core::smart_refctd_ptr<CVulkanLogicalDevice>(this),
            core::smart_refctd_ptr<const CVulkanShader>(vulkanShader), specInfo);
    }

    
    core::smart_refctd_ptr<IGPUAccelerationStructure> createAccelerationStructure_impl(IGPUAccelerationStructure::SCreationParams&& params) override;

    // For consistency's sake why not pass IGPUComputePipeline::SCreationParams as
    // only second argument, like in createComputePipelines_impl below? Especially
    // now, since I've added more members to IGPUComputePipeline::SCreationParams
    core::smart_refctd_ptr<IGPUComputePipeline> createComputePipeline_impl(
        IGPUPipelineCache* _pipelineCache, core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
        core::smart_refctd_ptr<IGPUSpecializedShader>&& _shader) override
    {
        core::smart_refctd_ptr<IGPUComputePipeline> result = nullptr;

        IGPUComputePipeline::SCreationParams creationParams = {};
        creationParams.flags = static_cast<video::IGPUComputePipeline::E_PIPELINE_CREATION>(0); // No way to get this now!
        creationParams.layout = std::move(_layout);
        creationParams.shader = std::move(_shader);
        creationParams.basePipeline = nullptr; // No way to get this now!
        creationParams.basePipelineIndex = ~0u; // No way to get this now!

        core::SRange<const IGPUComputePipeline::SCreationParams> creationParamsRange(&creationParams,
            &creationParams + 1);

        if (createComputePipelines_impl(_pipelineCache, creationParamsRange, &result))
        {
            return result;
        }
        else
        {
            return nullptr;
        }
    }

    bool createComputePipelines_impl(IGPUPipelineCache* pipelineCache,
        core::SRange<const IGPUComputePipeline::SCreationParams> createInfos,
        core::smart_refctd_ptr<IGPUComputePipeline>* output) override
    {
        constexpr uint32_t MAX_PIPELINE_COUNT = 100u;
        assert(createInfos.size() <= MAX_PIPELINE_COUNT);

        const IGPUComputePipeline::SCreationParams* creationParams = createInfos.begin();
        for (size_t i = 0ull; i < createInfos.size(); ++i)
        {
            if ((creationParams[i].layout->getAPIType() != EAT_VULKAN) ||
                (creationParams[i].shader->getAPIType() != EAT_VULKAN))
            {
                return false;
            }
        }

        VkPipelineCache vk_pipelineCache = VK_NULL_HANDLE;
        if (pipelineCache && pipelineCache->getAPIType() == EAT_VULKAN)
            vk_pipelineCache = IBackendObject::device_compatibility_cast<const CVulkanPipelineCache*>(pipelineCache, this)->getInternalObject();

        VkPipelineShaderStageCreateInfo vk_shaderStageCreateInfos[MAX_PIPELINE_COUNT];
        VkSpecializationInfo vk_specializationInfos[MAX_PIPELINE_COUNT];
        constexpr uint32_t MAX_SPEC_CONSTANTS_PER_PIPELINE = 100u;
        uint32_t mapEntryCount_total = 0u;
        VkSpecializationMapEntry vk_mapEntries[MAX_PIPELINE_COUNT * MAX_SPEC_CONSTANTS_PER_PIPELINE];

        VkComputePipelineCreateInfo vk_createInfos[MAX_PIPELINE_COUNT];
        for (size_t i = 0ull; i < createInfos.size(); ++i)
        {
            const auto createInfo = createInfos.begin() + i;

            vk_createInfos[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            vk_createInfos[i].pNext = nullptr; // pNext must be either NULL or a pointer to a valid instance of VkPipelineCompilerControlCreateInfoAMD, VkPipelineCreationFeedbackCreateInfoEXT, or VkSubpassShadingPipelineCreateInfoHUAWEI
            vk_createInfos[i].flags = static_cast<VkPipelineCreateFlags>(createInfo->flags);

            if (createInfo->shader->getAPIType() != EAT_VULKAN)
                continue;

            const auto* specShader = IBackendObject::device_compatibility_cast<const CVulkanSpecializedShader*>(createInfo->shader.get(), this);

            vk_shaderStageCreateInfos[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            vk_shaderStageCreateInfos[i].pNext = nullptr; // pNext must be NULL or a pointer to a valid instance of VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT
            vk_shaderStageCreateInfos[i].flags = 0;
            vk_shaderStageCreateInfos[i].stage = static_cast<VkShaderStageFlagBits>(specShader->getStage());
            vk_shaderStageCreateInfos[i].module = specShader->getInternalObject();
            vk_shaderStageCreateInfos[i].pName = "main";
            if (specShader->getSpecInfo().m_entries && specShader->getSpecInfo().m_backingBuffer)
            {
                uint32_t offset = mapEntryCount_total;
                assert(specShader->getSpecInfo().m_entries->size() <= MAX_SPEC_CONSTANTS_PER_PIPELINE);

                for (size_t s = 0ull; s < specShader->getSpecInfo().m_entries->size(); ++s)
                {
                    const auto entry = specShader->getSpecInfo().m_entries->begin() + s;
                    vk_mapEntries[offset + s].constantID = entry->specConstID;
                    vk_mapEntries[offset + s].offset = entry->offset;
                    vk_mapEntries[offset + s].size = entry->size;
                }
                mapEntryCount_total += specShader->getSpecInfo().m_entries->size();

                vk_specializationInfos[i].mapEntryCount = static_cast<uint32_t>(specShader->getSpecInfo().m_entries->size());
                vk_specializationInfos[i].pMapEntries = vk_mapEntries + offset;
                vk_specializationInfos[i].dataSize = specShader->getSpecInfo().m_backingBuffer->getSize();
                vk_specializationInfos[i].pData = specShader->getSpecInfo().m_backingBuffer->getPointer();

                vk_shaderStageCreateInfos[i].pSpecializationInfo = &vk_specializationInfos[i];
            }
            else
            {
                vk_shaderStageCreateInfos[i].pSpecializationInfo = nullptr;
            }

            vk_createInfos[i].stage = vk_shaderStageCreateInfos[i];

            vk_createInfos[i].layout = VK_NULL_HANDLE;
            if (createInfo->layout && (createInfo->layout->getAPIType() == EAT_VULKAN))
                vk_createInfos[i].layout = IBackendObject::device_compatibility_cast<const CVulkanPipelineLayout*>(createInfo->layout.get(), this)->getInternalObject();

            vk_createInfos[i].basePipelineHandle = VK_NULL_HANDLE;
            if (createInfo->basePipeline && (createInfo->basePipeline->getAPIType() == EAT_VULKAN))
                vk_createInfos[i].basePipelineHandle = IBackendObject::device_compatibility_cast<const CVulkanComputePipeline*>(createInfo->basePipeline.get(), this)->getInternalObject();

            vk_createInfos[i].basePipelineIndex = createInfo->basePipelineIndex;
        }
        
        VkPipeline vk_pipelines[MAX_PIPELINE_COUNT];
        if (m_devf.vk.vkCreateComputePipelines(m_vkdev, vk_pipelineCache, static_cast<uint32_t>(createInfos.size()),
            vk_createInfos, nullptr, vk_pipelines) == VK_SUCCESS)
        {
            for (size_t i = 0ull; i < createInfos.size(); ++i)
            {
                const auto createInfo = createInfos.begin() + i;

                output[i] = core::make_smart_refctd_ptr<CVulkanComputePipeline>(
                    core::smart_refctd_ptr<CVulkanLogicalDevice>(this),
                    core::smart_refctd_ptr(createInfo->layout),
                    core::smart_refctd_ptr(createInfo->shader), vk_pipelines[i]);
            }
            return true;
        }
        else
        {
            return false;
        }
    }

    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> createRenderpassIndependentPipeline_impl(
        IGPUPipelineCache* _pipelineCache,
        core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
        IGPUSpecializedShader* const* _shadersBegin, IGPUSpecializedShader* const* _shadersEnd,
        const asset::SVertexInputParams& _vertexInputParams,
        const asset::SBlendParams& _blendParams,
        const asset::SPrimitiveAssemblyParams& _primAsmParams,
        const asset::SRasterizationParams& _rasterParams) override
    {
        IGPURenderpassIndependentPipeline::SCreationParams creationParams = {};
        creationParams.layout = std::move(_layout);
        const uint32_t shaderCount = std::distance(_shadersBegin, _shadersEnd);
        for (uint32_t i = 0u; i < shaderCount; ++i)
            creationParams.shaders[i] = core::smart_refctd_ptr<const IGPUSpecializedShader>(_shadersBegin[i]);
        creationParams.vertexInput = _vertexInputParams;
        creationParams.blend = _blendParams;
        creationParams.primitiveAssembly = _primAsmParams;
        creationParams.rasterization = _rasterParams;

        core::SRange<const IGPURenderpassIndependentPipeline::SCreationParams> creationParamsRange(&creationParams, &creationParams + 1);

        core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> result = nullptr;
        createRenderpassIndependentPipelines_impl(_pipelineCache, creationParamsRange, &result);
        return result;
    }

    bool createRenderpassIndependentPipelines_impl(IGPUPipelineCache* pipelineCache,
        core::SRange<const IGPURenderpassIndependentPipeline::SCreationParams> createInfos,
        core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>* output) override
    {
        if (pipelineCache && pipelineCache->getAPIType() != EAT_VULKAN)
            return false;

        auto creationParams = createInfos.begin();
        for (size_t i = 0ull; i < createInfos.size(); ++i)
        {
            if (creationParams[i].layout->getAPIType() != EAT_VULKAN)
                continue;

            uint32_t shaderCount = 0u;
            for (uint32_t ss = 0u; ss < IGPURenderpassIndependentPipeline::GRAPHICS_SHADER_STAGE_COUNT; ++ss)
            {
                auto shader = creationParams[i].shaders[ss];
                if (shader)
                {
                    if (shader->getAPIType() != EAT_VULKAN)
                        continue;

                    ++shaderCount;
                }
            }
            
            output[i] = core::make_smart_refctd_ptr<CVulkanRenderpassIndependentPipeline>(
                core::smart_refctd_ptr<const CVulkanLogicalDevice>(this),
                core::smart_refctd_ptr(creationParams[i].layout),
                reinterpret_cast<IGPUSpecializedShader* const*>(creationParams[i].shaders),
                reinterpret_cast<IGPUSpecializedShader* const*>(creationParams[i].shaders) + shaderCount,
                creationParams[i].vertexInput,
                creationParams[i].blend,
                creationParams[i].primitiveAssembly,
                creationParams[i].rasterization);
        }

        return true;
    }
    
    template<typename AddressType>
    IGPUAccelerationStructure::BuildSizes getAccelerationStructureBuildSizes_impl(VkAccelerationStructureBuildTypeKHR buildType, const IGPUAccelerationStructure::BuildGeometryInfo<AddressType>& pBuildInfo, const uint32_t* pMaxPrimitiveCounts) 
    {
        VkAccelerationStructureBuildSizesInfoKHR vk_ret = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR, nullptr};

        if(pMaxPrimitiveCounts == nullptr) {
            assert(false);
            return IGPUAccelerationStructure::BuildSizes{};
        }

        static constexpr size_t MaxGeometryPerBuildInfoCount = 64;
                
        VkAccelerationStructureBuildGeometryInfoKHR vk_buildGeomsInfo = {};

        // TODO: Use better container when ready for these stack allocated memories.
        uint32_t geometryArrayOffset = 0u;
        VkAccelerationStructureGeometryKHR vk_geometries[MaxGeometryPerBuildInfoCount] = {};

        {
            uint32_t geomCount = pBuildInfo.geometries.size();

            assert(geomCount > 0);
            assert(geomCount <= MaxGeometryPerBuildInfoCount);

            vk_buildGeomsInfo = CVulkanAccelerationStructure::getVkASBuildGeomInfoFromBuildGeomInfo(m_vkdev, &m_devf, pBuildInfo, vk_geometries);
        }

        m_devf.vk.vkGetAccelerationStructureBuildSizesKHR(m_vkdev, buildType, &vk_buildGeomsInfo, pMaxPrimitiveCounts, &vk_ret);

        IGPUAccelerationStructure::BuildSizes ret = {};
        ret.accelerationStructureSize = vk_ret.accelerationStructureSize;
        ret.updateScratchSize = vk_ret.updateScratchSize;
        ret.buildScratchSize = vk_ret.buildScratchSize;

        return ret;
    }

    core::smart_refctd_ptr<IGPUGraphicsPipeline> createGraphicsPipeline_impl(IGPUPipelineCache* pipelineCache, IGPUGraphicsPipeline::SCreationParams&& params);

    bool createGraphicsPipelines_impl(IGPUPipelineCache* pipelineCache, core::SRange<const IGPUGraphicsPipeline::SCreationParams> params, core::smart_refctd_ptr<IGPUGraphicsPipeline>* output) override;

    private:
        inline ~CVulkanLogicalDevice()
        {
            m_devf.vk.vkDestroyDevice(m_vkdev, nullptr);
        }

        VkDevice m_vkdev;
        CVulkanDeviceFunctionTable m_devf;
    
        constexpr static inline uint32_t NODES_PER_BLOCK_DEFERRED_OP = 4096u;
        constexpr static inline uint32_t MAX_BLOCK_COUNT_DEFERRED_OP = 256u;
        memory_pool_mt_t m_deferred_op_mempool;

        core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> m_dummyDSLayout = nullptr;
};

}

#endif