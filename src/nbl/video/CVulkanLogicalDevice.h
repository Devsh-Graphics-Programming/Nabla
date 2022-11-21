#ifndef _NBL_C_VULKAN_LOGICAL_DEVICE_H_INCLUDED_
#define _NBL_C_VULKAN_LOGICAL_DEVICE_H_INCLUDED_

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
#include "nbl/video/CVulkanFence.h"
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
#include "nbl/core/containers/CMemoryPool.h"

namespace nbl::video
{

class CVulkanCommandBuffer;

class CVulkanLogicalDevice final : public ILogicalDevice
{
public:
    using memory_pool_mt_t = core::CMemoryPool<core::PoolAddressAllocator<uint32_t>, core::default_aligned_allocator, true, uint32_t>;

    CVulkanLogicalDevice(core::smart_refctd_ptr<IAPIConnection>&& api, renderdoc_api_t* rdoc, IPhysicalDevice* physicalDevice, VkDevice vkdev, VkInstance vkinst, const SCreationParams& params)
        : ILogicalDevice(std::move(api)
        , physicalDevice,params)
        , m_vkdev(vkdev)
        , m_devf(vkdev)
        , m_deferred_op_mempool(NODES_PER_BLOCK_DEFERRED_OP * sizeof(CVulkanDeferredOperation), 1u, MAX_BLOCK_COUNT_DEFERRED_OP, static_cast<uint32_t>(sizeof(CVulkanDeferredOperation)))
    {
        // create actual queue objects
        for (uint32_t i = 0u; i < params.queueParamsCount; ++i)
        {
            const auto& qci = params.queueParams[i];
            const uint32_t famIx = qci.familyIndex;
            const uint32_t offset = (*m_offsets)[famIx];
            const auto flags = qci.flags;
                    
            for (uint32_t j = 0u; j < qci.count; ++j)
            {
                const float priority = qci.priorities[j];
                        
                VkQueue q;
                m_devf.vk.vkGetDeviceQueue(m_vkdev, famIx, j, &q);
                        
                const uint32_t ix = offset + j;
                (*m_queues)[ix] = new CThreadSafeGPUQueueAdapter(this, new CVulkanQueue(this, rdoc, vkinst, q, famIx, flags, priority));
            }
        }

        m_dummyDSLayout = createDescriptorSetLayout(nullptr, nullptr);
    }
            
    ~CVulkanLogicalDevice()
    {
        m_devf.vk.vkDestroyDevice(m_vkdev, nullptr);
    }
            
    core::smart_refctd_ptr<IGPUSemaphore> createSemaphore() override
    {
        VkSemaphoreCreateInfo createInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        createInfo.pNext = nullptr; // Each pNext member of any structure (including this one) in the pNext chain must be either NULL or a pointer to a valid instance of VkExportSemaphoreCreateInfo, VkExportSemaphoreWin32HandleInfoKHR, or VkSemaphoreTypeCreateInfo
        createInfo.flags = static_cast<VkSemaphoreCreateFlags>(0); // flags must be 0

        VkSemaphore semaphore;
        if (m_devf.vk.vkCreateSemaphore(m_vkdev, &createInfo, nullptr, &semaphore) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanSemaphore>
                (core::smart_refctd_ptr<CVulkanLogicalDevice>(this), semaphore);
        }
        else
        {
            return nullptr;
        }
    }
            
    core::smart_refctd_ptr<IGPUEvent> createEvent(IGPUEvent::E_CREATE_FLAGS flags) override;
    IGPUEvent::E_STATUS getEventStatus(const IGPUEvent* _event) override;
    IGPUEvent::E_STATUS resetEvent(IGPUEvent* _event) override;
    IGPUEvent::E_STATUS setEvent(IGPUEvent* _event) override;
            
    core::smart_refctd_ptr<IGPUFence> createFence(IGPUFence::E_CREATE_FLAGS flags) override
    {
        VkFenceCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        vk_createInfo.pNext = nullptr; // Each pNext member of any structure (including this one) in the pNext chain must be either NULL or a pointer to a valid instance of VkExportFenceCreateInfo or VkExportFenceWin32HandleInfoKHR
        vk_createInfo.flags = static_cast<VkFenceCreateFlags>(flags);

        VkFence vk_fence;
        if (m_devf.vk.vkCreateFence(m_vkdev, &vk_createInfo, nullptr, &vk_fence) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanFence>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this), flags, vk_fence);
        }
        else
        {
            return nullptr;
        }
    }
            
    IGPUFence::E_STATUS getFenceStatus(IGPUFence* _fence) override
    {
        if (!_fence && (_fence->getAPIType() != EAT_VULKAN))
            return IGPUFence::E_STATUS::ES_ERROR;

        VkResult retval = m_devf.vk.vkGetFenceStatus(m_vkdev, IBackendObject::device_compatibility_cast<const CVulkanFence*>(_fence, this)->getInternalObject());

        switch (retval)
        {
        case VK_SUCCESS:
            return IGPUFence::ES_SUCCESS;
        case VK_NOT_READY:
            return IGPUFence::ES_NOT_READY;
        default:
            return IGPUFence::ES_ERROR;
        }
    }
            
    // API needs to change. vkResetFences can fail.
    bool resetFences(uint32_t _count, IGPUFence*const* _fences) override
    {
        constexpr uint32_t MAX_FENCE_COUNT = 100u;
        assert(_count < MAX_FENCE_COUNT);

        VkFence vk_fences[MAX_FENCE_COUNT];
        for (uint32_t i = 0u; i < _count; ++i)
        {
            if (_fences[i]->getAPIType() != EAT_VULKAN)
            {
                assert(false);
                return false;
            }

            vk_fences[i] = IBackendObject::device_compatibility_cast<CVulkanFence*>(_fences[i], this)->getInternalObject();
        }

        auto vk_res = m_devf.vk.vkResetFences(m_vkdev, _count, vk_fences);
        return (vk_res == VK_SUCCESS);
    }
            
    IGPUFence::E_STATUS waitForFences(uint32_t _count, IGPUFence*const* _fences, bool _waitAll, uint64_t _timeout) override
    {
        constexpr uint32_t MAX_FENCE_COUNT = 100u;

        assert(_count <= MAX_FENCE_COUNT);

        VkFence vk_fences[MAX_FENCE_COUNT];
        for (uint32_t i = 0u; i < _count; ++i)
        {
            if (_fences[i]->getAPIType() != EAT_VULKAN)
                return IGPUFence::E_STATUS::ES_ERROR;

            vk_fences[i] = IBackendObject::device_compatibility_cast<CVulkanFence*>(_fences[i], this)->getInternalObject();
        }

        VkResult result = m_devf.vk.vkWaitForFences(m_vkdev, _count, vk_fences, _waitAll, _timeout);
        switch (result)
        {
        case VK_SUCCESS:
            return IGPUFence::ES_SUCCESS;
        case VK_TIMEOUT:
            return IGPUFence::ES_TIMEOUT;
        default:
            return IGPUFence::ES_ERROR;
        }
    }
              
    core::smart_refctd_ptr<IDeferredOperation> createDeferredOperation() override
    {
        VkDeferredOperationKHR vk_deferredOp = VK_NULL_HANDLE;
        VkResult vk_res = m_devf.vk.vkCreateDeferredOperationKHR(m_vkdev, nullptr, &vk_deferredOp);
        if(vk_res!=VK_SUCCESS)
            return nullptr;

        void* memory = m_deferred_op_mempool.allocate(sizeof(CVulkanDeferredOperation),alignof(CVulkanDeferredOperation));
        if (!memory)
            return nullptr;

        new (memory) CVulkanDeferredOperation(core::smart_refctd_ptr<CVulkanLogicalDevice>(this),vk_deferredOp);
        return core::smart_refctd_ptr<CVulkanDeferredOperation>(reinterpret_cast<CVulkanDeferredOperation*>(memory),core::dont_grab);
    }

    core::smart_refctd_ptr<IGPUCommandPool> createCommandPool(uint32_t familyIndex, core::bitflag<IGPUCommandPool::E_CREATE_FLAGS> flags) override
    {
        VkCommandPoolCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        vk_createInfo.pNext = nullptr; // pNext must be NULL
        vk_createInfo.flags = static_cast<VkCommandPoolCreateFlags>(flags.value);
        vk_createInfo.queueFamilyIndex = familyIndex;

        VkCommandPool vk_commandPool = VK_NULL_HANDLE;
        if (m_devf.vk.vkCreateCommandPool(m_vkdev, &vk_createInfo, nullptr, &vk_commandPool) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanCommandPool>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this), flags, familyIndex, vk_commandPool);
        }
        else
        {
            return nullptr;
        }
    }
            
    core::smart_refctd_ptr<IDescriptorPool> createDescriptorPool(
        IDescriptorPool::E_CREATE_FLAGS flags, uint32_t maxSets, uint32_t poolSizeCount,
        const IDescriptorPool::SDescriptorPoolSize* poolSizes) override
    {
        constexpr uint32_t MAX_DESCRIPTOR_POOL_SIZE_COUNT = 100u;

        assert(poolSizeCount <= MAX_DESCRIPTOR_POOL_SIZE_COUNT);

        // I wonder if I can memcpy the entire array
        VkDescriptorPoolSize vk_descriptorPoolSizes[MAX_DESCRIPTOR_POOL_SIZE_COUNT];
        for (uint32_t i = 0u; i < poolSizeCount; ++i)
        {
            vk_descriptorPoolSizes[i].type = static_cast<VkDescriptorType>(poolSizes[i].type);
            vk_descriptorPoolSizes[i].descriptorCount = poolSizes[i].count;
        }

        VkDescriptorPoolCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        vk_createInfo.pNext = nullptr; // Each pNext member of any structure (including this one) in the pNext chain must be either NULL or a pointer to a valid instance of VkDescriptorPoolInlineUniformBlockCreateInfoEXT or VkMutableDescriptorTypeCreateInfoVALVE
        vk_createInfo.flags = static_cast<VkDescriptorPoolCreateFlags>(flags);
        vk_createInfo.maxSets = maxSets;
        vk_createInfo.poolSizeCount = poolSizeCount;
        vk_createInfo.pPoolSizes = vk_descriptorPoolSizes;

        VkDescriptorPool vk_descriptorPool;
        if (m_devf.vk.vkCreateDescriptorPool(m_vkdev, &vk_createInfo, nullptr, &vk_descriptorPool) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanDescriptorPool>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this), maxSets, vk_descriptorPool);
        }
        else
        {
            return nullptr;
        }
    }
            
    core::smart_refctd_ptr<IGPURenderpass> createRenderpass(const IGPURenderpass::SCreationParams& params) override
    {
        VkRenderPassCreateInfo createInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
        createInfo.pNext = nullptr;
        createInfo.flags = static_cast<VkRenderPassCreateFlags>(0u); // No flags are supported
        createInfo.attachmentCount = params.attachmentCount;

        core::vector<VkAttachmentDescription> attachments(createInfo.attachmentCount); // TODO reduce number of allocations/get rid of vectors
        for (uint32_t i = 0u; i < attachments.size(); ++i)
        {
            const auto& att = params.attachments[i];
            auto& vkatt = attachments[i];
            vkatt.flags = static_cast<VkAttachmentDescriptionFlags>(0u); // No flags are supported
            vkatt.format = getVkFormatFromFormat(att.format);
            vkatt.samples = static_cast<VkSampleCountFlagBits>(att.samples);
            vkatt.loadOp = static_cast<VkAttachmentLoadOp>(att.loadOp);
            vkatt.storeOp = static_cast<VkAttachmentStoreOp>(att.storeOp);

            // Todo(achal): Do we want these??
            vkatt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            vkatt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

            vkatt.initialLayout = static_cast<VkImageLayout>(att.initialLayout);
            vkatt.finalLayout = static_cast<VkImageLayout>(att.finalLayout);
        }
        createInfo.pAttachments = attachments.data();

        createInfo.subpassCount = params.subpassCount;
        core::vector<VkSubpassDescription> vk_subpasses(createInfo.subpassCount);
        
        constexpr uint32_t MemSz = 1u << 12;
        constexpr uint32_t MaxAttachmentRefs = MemSz / sizeof(VkAttachmentReference);
        VkAttachmentReference vk_attRefs[MaxAttachmentRefs];
        uint32_t preserveAttRefs[MaxAttachmentRefs];

        uint32_t totalAttRefCount = 0u;
        uint32_t totalPreserveCount = 0u;

        auto fillUpVkAttachmentRefHandles = [&vk_attRefs, &totalAttRefCount](const uint32_t count, const auto* srcRef, uint32_t& dstCount, auto*& dstRef)
        {
            for (uint32_t j = 0u; j < count; ++j)
            {
                vk_attRefs[totalAttRefCount + j].attachment = srcRef[j].attachment;
                vk_attRefs[totalAttRefCount + j].layout = static_cast<VkImageLayout>(srcRef[j].layout);
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
        core::vector<VkSubpassDependency> deps(createInfo.dependencyCount);
        for (uint32_t i = 0u; i < deps.size(); ++i)
        {
            const auto& dep = params.dependencies[i];
            auto& vkdep = deps[i];

            vkdep.srcSubpass = dep.srcSubpass;
            vkdep.dstSubpass = dep.dstSubpass;
            vkdep.srcStageMask = getVkPipelineStageFlagsFromPipelineStageFlags(dep.srcStageMask);
            vkdep.dstStageMask = getVkPipelineStageFlagsFromPipelineStageFlags(dep.dstStageMask);
            vkdep.srcAccessMask = static_cast<VkAccessFlags>(dep.srcAccessMask);
            vkdep.dstAccessMask = static_cast<VkAccessFlags>(dep.dstAccessMask);
            vkdep.dependencyFlags = static_cast<VkDependencyFlags>(dep.dependencyFlags);
        }
        createInfo.pDependencies = deps.data();

        VkRenderPass vk_renderpass;
        if (m_devf.vk.vkCreateRenderPass(m_vkdev, &createInfo, nullptr, &vk_renderpass) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanRenderpass>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this), params, vk_renderpass);
        }
        else
        {
            return nullptr;
        }
    }
           
    // API needs to change, vkFlushMappedMemoryRanges could fail.
    void flushMappedMemoryRanges(core::SRange<const video::IDeviceMemoryAllocation::MappedMemoryRange> ranges) override
    {
        constexpr uint32_t MAX_MEMORY_RANGE_COUNT = 408u;
        VkMappedMemoryRange vk_memoryRanges[MAX_MEMORY_RANGE_COUNT];

        const uint32_t memoryRangeCount = static_cast<uint32_t>(ranges.size());
        assert(memoryRangeCount <= MAX_MEMORY_RANGE_COUNT);

        getVkMappedMemoryRanges(vk_memoryRanges, ranges.begin(), ranges.end());
        
        if (m_devf.vk.vkFlushMappedMemoryRanges(m_vkdev, memoryRangeCount, vk_memoryRanges) != VK_SUCCESS)
        {
            auto logger = (m_physicalDevice->getDebugCallback()) ? m_physicalDevice->getDebugCallback()->getLogger() : nullptr;
            if (logger)
                logger->log("flushMappedMemoryRanges failed!", system::ILogger::ELL_ERROR);
        }
    }
            
    // API needs to change, this could fail
    void invalidateMappedMemoryRanges(core::SRange<const video::IDeviceMemoryAllocation::MappedMemoryRange> ranges) override
    {
        constexpr uint32_t MAX_MEMORY_RANGE_COUNT = 408u;
        VkMappedMemoryRange vk_memoryRanges[MAX_MEMORY_RANGE_COUNT];

        const uint32_t memoryRangeCount = static_cast<uint32_t>(ranges.size());
        assert(memoryRangeCount <= MAX_MEMORY_RANGE_COUNT);

        getVkMappedMemoryRanges(vk_memoryRanges, ranges.begin(), ranges.end());

        if (m_devf.vk.vkInvalidateMappedMemoryRanges(m_vkdev, memoryRangeCount, vk_memoryRanges) != VK_SUCCESS)
        {
            auto logger = (m_physicalDevice->getDebugCallback()) ? m_physicalDevice->getDebugCallback()->getLogger() : nullptr;
            if (logger)
                logger->log("invalidateMappedMemoryRanges failed!", system::ILogger::ELL_ERROR);

        }
    }

    bool bindBufferMemory(uint32_t bindInfoCount, const SBindBufferMemoryInfo* pBindInfos) override
    {
        bool anyFailed = false;
        for (uint32_t i = 0u; i < bindInfoCount; ++i)
        {
            const auto& bindInfo = pBindInfos[i];
            
            if ((bindInfo.buffer->getAPIType() != EAT_VULKAN) || (bindInfo.memory->getAPIType() != EAT_VULKAN))
                continue;

            if (bindInfo.buffer->getCreationParams().usage.hasFlags(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT))
            {
                if(!bindInfo.memory->getAllocateFlags().hasFlags(IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT))
                {
                    // TODO(erfan): Log-> if buffer was created with EUF_SHADER_DEVICE_ADDRESS_BIT set, memory must have been allocated with the EMAF_DEVICE_ADDRESS_BIT bit.
                    _NBL_DEBUG_BREAK_IF(true);
                    anyFailed = true;
                    continue;
                }
            }

            CVulkanBuffer* vulkanBuffer = IBackendObject::device_compatibility_cast<CVulkanBuffer*>(bindInfo.buffer, this);
            VkBuffer vk_buffer = vulkanBuffer->getInternalObject();
            VkDeviceMemory vk_memory = static_cast<const CVulkanMemoryAllocation*>(pBindInfos[i].memory)->getInternalObject();
            if (m_devf.vk.vkBindBufferMemory(m_vkdev, vk_buffer, vk_memory, static_cast<VkDeviceSize>(pBindInfos[i].offset)) == VK_SUCCESS)
            {
                vulkanBuffer->setMemoryAndOffset(
                    core::smart_refctd_ptr<IDeviceMemoryAllocation>(bindInfo.memory), bindInfo.offset);
            }
            else
            {
                // Todo(achal): Log which one failed
                _NBL_DEBUG_BREAK_IF(true);
                anyFailed = true;
            }
        }

        return !anyFailed;
    }

    core::smart_refctd_ptr<IGPUBuffer> createBuffer(IGPUBuffer::SCreationParams&& creationParams)
    {
        VkBufferCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        // Each pNext member of any structure (including this one) in the pNext chain must be either NULL or a pointer to a valid instance of VkBufferDeviceAddressCreateInfoEXT, VkBufferOpaqueCaptureAddressCreateInfo, VkDedicatedAllocationBufferCreateInfoNV, VkExternalMemoryBufferCreateInfo, VkVideoProfileKHR, or VkVideoProfilesKHR
        vk_createInfo.pNext = nullptr;
        vk_createInfo.flags = static_cast<VkBufferCreateFlags>(0u); // Nabla doesn't support any of these flags
        vk_createInfo.size = static_cast<VkDeviceSize>(creationParams.size);
        vk_createInfo.usage = getVkBufferUsageFlagsFromBufferUsageFlags(creationParams.usage);
        vk_createInfo.sharingMode = creationParams.isConcurrentSharing() ? VK_SHARING_MODE_CONCURRENT:VK_SHARING_MODE_EXCLUSIVE;
        vk_createInfo.queueFamilyIndexCount = creationParams.queueFamilyIndexCount;
        vk_createInfo.pQueueFamilyIndices = creationParams.queueFamilyIndices;

        VkBuffer vk_buffer;
        if (m_devf.vk.vkCreateBuffer(m_vkdev, &vk_createInfo, nullptr, &vk_buffer) == VK_SUCCESS)
        {
            VkBufferMemoryRequirementsInfo2 vk_memoryRequirementsInfo = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2 };
            vk_memoryRequirementsInfo.pNext = nullptr; // pNext must be NULL
            vk_memoryRequirementsInfo.buffer = vk_buffer;

            VkMemoryDedicatedRequirements vk_dedicatedMemoryRequirements = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS };
            VkMemoryRequirements2 vk_memoryRequirements = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
            vk_memoryRequirements.pNext = &vk_dedicatedMemoryRequirements;
            m_devf.vk.vkGetBufferMemoryRequirements2(m_vkdev, &vk_memoryRequirementsInfo, &vk_memoryRequirements);

            IDeviceMemoryBacked::SDeviceMemoryRequirements bufferMemoryReqs = {};
            bufferMemoryReqs.size = vk_memoryRequirements.memoryRequirements.size;
            bufferMemoryReqs.memoryTypeBits = vk_memoryRequirements.memoryRequirements.memoryTypeBits;
            bufferMemoryReqs.alignmentLog2 = std::log2(vk_memoryRequirements.memoryRequirements.alignment);
            bufferMemoryReqs.prefersDedicatedAllocation = vk_dedicatedMemoryRequirements.prefersDedicatedAllocation;
            bufferMemoryReqs.requiresDedicatedAllocation = vk_dedicatedMemoryRequirements.requiresDedicatedAllocation;

            return core::make_smart_refctd_ptr<CVulkanBuffer>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this),
                bufferMemoryReqs,
                std::move(creationParams),
                vk_buffer
            );
        }
        else
        {
            return nullptr;
        }
    }
        
    core::smart_refctd_ptr<IGPUShader> createShader(core::smart_refctd_ptr<asset::ICPUShader>&& cpushader) override
    {
        const char* entryPoint = "main";
        const asset::IShader::E_SHADER_STAGE shaderStage = cpushader->getStage();

        const asset::ICPUBuffer* source = cpushader->getContent();

        // TODO:
        core::smart_refctd_ptr<asset::CCompilerSet> compilerSet;

        const char* begin = static_cast<const char*>(source->getPointer());
        const char* end = begin + source->getSize();
        std::string code(begin, end);

        auto compiler = compilerSet->getShaderCompiler(cpushader->getContentType());
        asset::IShaderCompiler::SOptions commonCompileOptions = {};
        commonCompileOptions.logger = (m_physicalDevice->getDebugCallback()) ? m_physicalDevice->getDebugCallback()->getLogger() : nullptr;
        commonCompileOptions.includeFinder = compiler->getDefaultIncludeFinder(); // to resolve includes before compilation
        commonCompileOptions.stage = shaderStage;
        commonCompileOptions.sourceIdentifier = cpushader->getFilepathHint().c_str();
        commonCompileOptions.entryPoint = entryPoint;
        commonCompileOptions.genDebugInfo = true;
        commonCompileOptions.spirvOptimizer = nullptr; // TODO: create/get spirv optimizer in logical device?
        commonCompileOptions.targetSpirvVersion = m_physicalDevice->getLimits().spirvVersion;

        if (cpushader->getContentType() != asset::ICPUShader::E_CONTENT_TYPE::ECT_SPIRV)
            asset::IShader::insertDefines(code, m_physicalDevice->getExtraGLSLDefines());

        core::smart_refctd_ptr<asset::ICPUBuffer> spirv;

        if (cpushader->getContentType() == asset::ICPUShader::E_CONTENT_TYPE::ECT_HLSL)
        {
            // TODO: actually use code to create a ICPUShader (via non allocating CPUBuffer) or change the function signature for "compilerSet"
            // TODO: add specific HLSLCompiler::SOption params
            spirv = compilerSet->compileToSPIRV(cpushader.get(), commonCompileOptions);
        }
        else if (cpushader->getContentType() == asset::ICPUShader::E_CONTENT_TYPE::ECT_GLSL)
        {
            spirv = compilerSet->compileToSPIRV(cpushader.get(), commonCompileOptions);
        }
        else if (cpushader->getContentType() == asset::ICPUShader::E_CONTENT_TYPE::ECT_SPIRV)
        {
            spirv = core::smart_refctd_ptr<asset::ICPUBuffer>(const_cast<asset::ICPUBuffer*>(cpushader->getContent()));
        }

        if (!spirv)
            return nullptr;

        VkShaderModuleCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        vk_createInfo.pNext = nullptr;
        vk_createInfo.flags = static_cast<VkShaderModuleCreateFlags>(0u); // reserved for future use by Vulkan
        vk_createInfo.codeSize = spirv->getSize();
        vk_createInfo.pCode = static_cast<const uint32_t*>(spirv->getPointer());
        
        VkShaderModule vk_shaderModule;
        if (m_devf.vk.vkCreateShaderModule(m_vkdev, &vk_createInfo, nullptr, &vk_shaderModule) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<video::CVulkanShader>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this), cpushader->getStage(), std::string(cpushader->getFilepathHint()), vk_shaderModule);
        }
        else
        {
            return nullptr;
        }
    }

    core::smart_refctd_ptr<IGPUImage> createImage(IGPUImage::SCreationParams&& params) override;

    bool bindImageMemory(uint32_t bindInfoCount, const SBindImageMemoryInfo* pBindInfos) override
    {
        bool anyFailed = false;
        for (uint32_t i = 0u; i < bindInfoCount; ++i)
        {
            const auto& bindInfo = pBindInfos[i];

            if ((bindInfo.image->getAPIType() != EAT_VULKAN) || (bindInfo.memory->getAPIType() != EAT_VULKAN))
                continue;

            CVulkanImage* vulkanImage = IBackendObject::device_compatibility_cast<CVulkanImage*>(bindInfo.image, this);

            VkImage vk_image = vulkanImage->getInternalObject();
            VkDeviceMemory vk_deviceMemory = static_cast<const CVulkanMemoryAllocation*>(bindInfo.memory)->getInternalObject();
            if (m_devf.vk.vkBindImageMemory(m_vkdev, vk_image, vk_deviceMemory, static_cast<VkDeviceSize>(bindInfo.offset)) == VK_SUCCESS)
            {
                vulkanImage->setMemoryAndOffset(
                    core::smart_refctd_ptr<IDeviceMemoryAllocation>(bindInfo.memory),
                    bindInfo.offset);
            }
            else
            {
                // Todo(achal): Log which one failed
                _NBL_DEBUG_BREAK_IF(true);
                anyFailed = true;
            }
        }

        return !anyFailed;
    }

    void updateDescriptorSets(uint32_t descriptorWriteCount, const IGPUDescriptorSet::SWriteDescriptorSet* pDescriptorWrites,
        uint32_t descriptorCopyCount, const IGPUDescriptorSet::SCopyDescriptorSet* pDescriptorCopies) override
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
            vk_writeDescriptorSets[i].descriptorType = static_cast<VkDescriptorType>(pDescriptorWrites[i].descriptorType);
            vk_writeDescriptorSets[i].descriptorCount = pDescriptorWrites[i].count;

            assert(pDescriptorWrites[i].count <= MAX_DESCRIPTOR_ARRAY_COUNT);
            assert(pDescriptorWrites[i].info[0].desc);

            switch (pDescriptorWrites[i].info->desc->getTypeCategory())
            {
                case asset::IDescriptor::EC_BUFFER:
                {
                    VkDescriptorBufferInfo dummyInfo = {};
                    dummyInfo.buffer = static_cast<const CVulkanBuffer*>(pDescriptorWrites[i].info[0].desc.get())->getInternalObject();
                    dummyInfo.offset = pDescriptorWrites[i].info[0].buffer.offset;
                    dummyInfo.range = pDescriptorWrites[i].info[0].buffer.size;

                    for (uint32_t j = 0u; j < pDescriptorWrites[i].count; ++j)
                    {
                        if (pDescriptorWrites[i].info[j].buffer.size)
                        {
                            vk_bufferInfos[bufferInfoOffset + j].buffer = static_cast<const CVulkanBuffer*>(pDescriptorWrites[i].info[j].desc.get())->getInternalObject();
                            vk_bufferInfos[bufferInfoOffset + j].offset = pDescriptorWrites[i].info[j].buffer.offset;
                            vk_bufferInfos[bufferInfoOffset + j].range = pDescriptorWrites[i].info[j].buffer.size;
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
                    const auto& firstDescWriteImageInfo = pDescriptorWrites[i].info[0].image;

                    VkDescriptorImageInfo dummyInfo = { VK_NULL_HANDLE };
                    if (firstDescWriteImageInfo.sampler && (firstDescWriteImageInfo.sampler->getAPIType() == EAT_VULKAN))
                        dummyInfo.sampler = static_cast<const CVulkanSampler*>(firstDescWriteImageInfo.sampler.get())->getInternalObject();
                    dummyInfo.imageView = static_cast<const CVulkanImageView*>(pDescriptorWrites[i].info[0].desc.get())->getInternalObject();
                    dummyInfo.imageLayout = static_cast<VkImageLayout>(pDescriptorWrites[i].info[0].image.imageLayout);

                    for (uint32_t j = 0u; j < pDescriptorWrites[i].count; ++j)
                    {
                        const auto& descriptorWriteImageInfo = pDescriptorWrites[i].info[j].image;
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
                        if (pDescriptorWrites[i].info[j].buffer.size)
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
                    auto & writeAS = vk_writeDescriptorSetAS[i];
                    writeAS = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR, nullptr};
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

        m_devf.vk.vkUpdateDescriptorSets(
            m_vkdev,
            descriptorWriteCount, vk_writeDescriptorSets.data(),
            descriptorCopyCount, vk_copyDescriptorSets.data());
    }

    SMemoryOffset allocate(const SAllocateInfo& info) override;

    core::smart_refctd_ptr<IGPUSampler> createSampler(const IGPUSampler::SParams& _params) override
    {
        VkSamplerCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        vk_createInfo.pNext = nullptr; // Each pNext member of any structure (including this one) in the pNext chain must be either NULL or a pointer to a valid instance of VkSamplerCustomBorderColorCreateInfoEXT, VkSamplerReductionModeCreateInfo, or VkSamplerYcbcrConversionInfo
        vk_createInfo.flags = static_cast<VkSamplerCreateFlags>(0); // No flags supported yet
        assert(_params.MaxFilter <= asset::ISampler::ETF_LINEAR);
        vk_createInfo.magFilter = static_cast<VkFilter>(_params.MaxFilter);
        assert(_params.MinFilter <= asset::ISampler::ETF_LINEAR);
        vk_createInfo.minFilter = static_cast<VkFilter>(_params.MinFilter);
        vk_createInfo.mipmapMode = static_cast<VkSamplerMipmapMode>(_params.MipmapMode);
        vk_createInfo.addressModeU = getVkAddressModeFromTexClamp(static_cast<asset::ISampler::E_TEXTURE_CLAMP>(_params.TextureWrapU));
        vk_createInfo.addressModeV = getVkAddressModeFromTexClamp(static_cast<asset::ISampler::E_TEXTURE_CLAMP>(_params.TextureWrapV));
        vk_createInfo.addressModeW = getVkAddressModeFromTexClamp(static_cast<asset::ISampler::E_TEXTURE_CLAMP>(_params.TextureWrapW));
        vk_createInfo.mipLodBias = _params.LodBias;
        assert(_params.AnisotropicFilter <= m_physicalDevice->getLimits().maxSamplerAnisotropyLog2);
        vk_createInfo.maxAnisotropy = std::exp2(_params.AnisotropicFilter);
        vk_createInfo.anisotropyEnable = m_physicalDevice->getFeatures().samplerAnisotropy;
        vk_createInfo.compareEnable = _params.CompareEnable;
        vk_createInfo.compareOp = static_cast<VkCompareOp>(_params.CompareFunc);
        vk_createInfo.minLod = _params.MinLod;
        vk_createInfo.maxLod = _params.MaxLod;
        assert(_params.BorderColor < asset::ISampler::ETBC_COUNT);
        vk_createInfo.borderColor = static_cast<VkBorderColor>(_params.BorderColor);
        vk_createInfo.unnormalizedCoordinates = VK_FALSE;

        VkSampler vk_sampler;
        if (m_devf.vk.vkCreateSampler(m_vkdev, &vk_createInfo, nullptr, &vk_sampler) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanSampler>(core::smart_refctd_ptr<ILogicalDevice>(this), _params, vk_sampler);
        }
        else
        {
            return nullptr;
        }
    }

    // API changes needed, this could also fail.
    void waitIdle() override
    {
        VkResult retval = m_devf.vk.vkDeviceWaitIdle(m_vkdev);

        // Todo(achal): Handle errors
        assert(retval == VK_SUCCESS);
    }

    void* mapMemory(const IDeviceMemoryAllocation::MappedMemoryRange& memory, core::bitflag<IDeviceMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAGS> accessHint = IDeviceMemoryAllocation::EMCAF_READ_AND_WRITE) override
    {
        if (memory.memory == nullptr || memory.memory->getAPIType() != EAT_VULKAN)
            return nullptr;
        assert(IDeviceMemoryAllocation::isMappingAccessConsistentWithMemoryType(accessHint, memory.memory->getMemoryPropertyFlags()));

        VkMemoryMapFlags vk_memoryMapFlags = 0; // reserved for future use, by Vulkan
        auto vulkanMemory = static_cast<CVulkanMemoryAllocation*>(memory.memory);
        VkDeviceMemory vk_memory = vulkanMemory->getInternalObject();
        void* mappedPtr;
        if (m_devf.vk.vkMapMemory(m_vkdev, vk_memory, static_cast<VkDeviceSize>(memory.offset),
            static_cast<VkDeviceSize>(memory.length), vk_memoryMapFlags, &mappedPtr) == VK_SUCCESS)
        {
            post_mapMemory(vulkanMemory, mappedPtr, memory.range, accessHint);
            return vulkanMemory->getMappedPointer(); // so pointer is rewound
        }
        else
        {
            return nullptr;
        }
    }

    void unmapMemory(IDeviceMemoryAllocation* memory) override
    {
        if (memory->getAPIType() != EAT_VULKAN)
            return;

        VkDeviceMemory vk_deviceMemory = static_cast<const CVulkanMemoryAllocation*>(memory)->getInternalObject();
        m_devf.vk.vkUnmapMemory(m_vkdev, vk_deviceMemory);
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

    inline memory_pool_mt_t & getMemoryPoolForDeferredOperations()
    {
        return m_deferred_op_mempool;
    }

    // At the moment this NEEDS requiredCount to be zero for the root call. We can fix this
    // by introducing a `offset` param which might make the code a little bit more verbose,
    // since it the function is not used very frequently I think its fine.
    static void getRequiredFeatures(const E_FEATURE feature, uint32_t& requiredCount, E_FEATURE* required)
    {
        switch (feature)
        {
        case EF_SWAPCHAIN:
        {
            required[requiredCount++] = EF_SWAPCHAIN;
        } break;

        case EF_DEFERRED_HOST_OPERATIONS:
        {
            required[requiredCount++] = EF_DEFERRED_HOST_OPERATIONS;
        } break;

        case EF_BUFFER_DEVICE_ADDRESS:
        {
            required[requiredCount++] = EF_BUFFER_DEVICE_ADDRESS;
        } break;

        case EF_DESCRIPTOR_INDEXING:
        {
            required[requiredCount++] = EF_DESCRIPTOR_INDEXING;
        } break;

        case EF_SHADER_FLOAT_CONTROLS:
        {
            required[requiredCount++] = EF_SHADER_FLOAT_CONTROLS;
        } break;

        case EF_SPIRV_1_4:
        {
            required[requiredCount++] = EF_SPIRV_1_4;

            const uint32_t requiredForThisCount = 1u;
            E_FEATURE requiredForThis[requiredForThisCount] = { EF_SHADER_FLOAT_CONTROLS };

            for (uint32_t i = 0u; i < requiredForThisCount; ++i)
                getRequiredFeatures(requiredForThis[i], requiredCount, required);
        } break;

        case EF_ACCELERATION_STRUCTURE:
        {
            required[requiredCount++] = EF_ACCELERATION_STRUCTURE;

            const uint32_t requiredForThisCount = 3u;
            E_FEATURE requiredForThis[requiredForThisCount] = { EF_DESCRIPTOR_INDEXING,
                EF_BUFFER_DEVICE_ADDRESS,
                EF_DEFERRED_HOST_OPERATIONS
            };

            for (uint32_t i = 0u; i < requiredForThisCount; ++i)
                getRequiredFeatures(requiredForThis[i], requiredCount, required);
        } break;

        case EF_RAY_TRACING_PIPELINE:
        {
            required[requiredCount++] = EF_RAY_TRACING_PIPELINE;

            const uint32_t requiredForThisCount = 2u;
            E_FEATURE requiredForThis[requiredForThisCount] = { EF_ACCELERATION_STRUCTURE, EF_SPIRV_1_4 };

            for (uint32_t i = 0u; i < requiredForThisCount; ++i)
                getRequiredFeatures(requiredForThis[i], requiredCount, required);
        } break;

        case EF_RAY_QUERY:
        {
            required[requiredCount++] = EF_RAY_QUERY;

            const uint32_t requiredForThisCount = 2u;
            E_FEATURE requiredForThis[requiredForThisCount] = { EF_ACCELERATION_STRUCTURE, EF_SPIRV_1_4 };

            for (uint32_t i = 0u; i < requiredForThisCount; ++i)
                getRequiredFeatures(requiredForThis[i], requiredCount, required);
        } break;
        
        case EF_FRAGMENT_SHADER_INTERLOCK:
        {
            required[requiredCount++] = EF_FRAGMENT_SHADER_INTERLOCK;
        } break;

        default:
            break;
        }
    };

    static inline const char* getVulkanExtensionName(const E_FEATURE feature)
    {
        switch (feature)
        {
        case EF_SWAPCHAIN:
            return VK_KHR_SWAPCHAIN_EXTENSION_NAME;
        case EF_DEFERRED_HOST_OPERATIONS:
            return VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME;
        case EF_BUFFER_DEVICE_ADDRESS:
            return VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME;
        case EF_DESCRIPTOR_INDEXING:
            return VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME;
        case EF_ACCELERATION_STRUCTURE:
            return VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME;
        case EF_SHADER_FLOAT_CONTROLS:
            return VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME;
        case EF_SPIRV_1_4:
            return VK_KHR_SPIRV_1_4_EXTENSION_NAME;
        case EF_RAY_TRACING_PIPELINE:
            return VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME;
        case EF_RAY_QUERY:
            return VK_KHR_RAY_QUERY_EXTENSION_NAME;
        case EF_FRAGMENT_SHADER_INTERLOCK:
            return VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME;
        default:
            assert(!"Extension unknown");
            return "";
        }
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

    core::smart_refctd_ptr<IGPUBufferView> createBufferView_impl(IGPUBuffer* _underlying, asset::E_FORMAT _fmt, size_t _offset = 0ull, size_t _size = IGPUBufferView::whole_buffer) override
    {
        if (_underlying->getAPIType() != EAT_VULKAN)
            return nullptr;

        VkBuffer vk_buffer = IBackendObject::device_compatibility_cast<const CVulkanBuffer*>(_underlying, this)->getInternalObject();

        VkBufferViewCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO };
        vk_createInfo.pNext = nullptr; // pNext must be NULL
        vk_createInfo.flags = static_cast<VkBufferViewCreateFlags>(0); // flags must be 0
        vk_createInfo.buffer = vk_buffer;
        vk_createInfo.format = getVkFormatFromFormat(_fmt);
        vk_createInfo.offset = _offset;
        vk_createInfo.range = _size;

        VkBufferView vk_bufferView;
        if (m_devf.vk.vkCreateBufferView(m_vkdev, &vk_createInfo, nullptr, &vk_bufferView) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanBufferView>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this),
                core::smart_refctd_ptr<IGPUBuffer>(_underlying), _fmt, _offset,
                _size, vk_bufferView);
        }
        else
        {
            return nullptr;
        }
    }

    core::smart_refctd_ptr<IGPUImageView> createImageView_impl(IGPUImageView::SCreationParams&& params) override
    {
        VkImageViewCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vk_createInfo.pNext = nullptr; // Each pNext member of any structure (including this one) in the pNext chain must be either NULL or a pointer to a valid instance of VkImageViewASTCDecodeModeEXT, VkImageViewUsageCreateInfo, VkSamplerYcbcrConversionInfo, VkVideoProfileKHR, or VkVideoProfilesKHR
        vk_createInfo.flags = static_cast<VkImageViewCreateFlags>(params.flags);

        if (params.image->getAPIType() != EAT_VULKAN)
            return nullptr;

        VkImage vk_image = IBackendObject::device_compatibility_cast<const CVulkanImage*>(params.image.get(), this)->getInternalObject();
        vk_createInfo.image = vk_image;
        vk_createInfo.viewType = static_cast<VkImageViewType>(params.viewType);
        vk_createInfo.format = getVkFormatFromFormat(params.format);
        vk_createInfo.components.r = static_cast<VkComponentSwizzle>(params.components.r);
        vk_createInfo.components.g = static_cast<VkComponentSwizzle>(params.components.g);
        vk_createInfo.components.b = static_cast<VkComponentSwizzle>(params.components.b);
        vk_createInfo.components.a = static_cast<VkComponentSwizzle>(params.components.a);
        vk_createInfo.subresourceRange.aspectMask = static_cast<VkImageAspectFlags>(params.subresourceRange.aspectMask);
        vk_createInfo.subresourceRange.baseMipLevel = params.subresourceRange.baseMipLevel;
        vk_createInfo.subresourceRange.levelCount = params.subresourceRange.levelCount;
        vk_createInfo.subresourceRange.baseArrayLayer = params.subresourceRange.baseArrayLayer;
        vk_createInfo.subresourceRange.layerCount = params.subresourceRange.layerCount;

        VkImageView vk_imageView;
        if (m_devf.vk.vkCreateImageView(m_vkdev, &vk_createInfo, nullptr, &vk_imageView) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanImageView>(core::smart_refctd_ptr<CVulkanLogicalDevice>(this),
                std::move(params), vk_imageView);
        }
        else
        {
            return nullptr;
        }
    }

    core::smart_refctd_ptr<IGPUDescriptorSet> createDescriptorSet_impl(IDescriptorPool* pool, core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout) override
    {
        if (pool->getAPIType() != EAT_VULKAN)
            return nullptr;

        const CVulkanDescriptorPool* vulkanDescriptorPool = IBackendObject::device_compatibility_cast<const CVulkanDescriptorPool*>(pool, this);

        VkDescriptorSetAllocateInfo vk_allocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        vk_allocateInfo.pNext = nullptr; // pNext must be NULL or a pointer to a valid instance of VkDescriptorSetVariableDescriptorCountAllocateInfo

        vk_allocateInfo.descriptorPool = vulkanDescriptorPool->getInternalObject();
        vk_allocateInfo.descriptorSetCount = 1u; // Isn't creating only descriptor set every time wasteful?

        if (layout->getAPIType() != EAT_VULKAN)
            return nullptr;
        VkDescriptorSetLayout vk_dsLayout = IBackendObject::device_compatibility_cast<const CVulkanDescriptorSetLayout*>(layout.get(), this)->getInternalObject();
        vk_allocateInfo.pSetLayouts = &vk_dsLayout;

        VkDescriptorSet vk_descriptorSet;
        if (m_devf.vk.vkAllocateDescriptorSets(m_vkdev, &vk_allocateInfo, &vk_descriptorSet) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanDescriptorSet>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this), std::move(layout),
                core::smart_refctd_ptr<const CVulkanDescriptorPool>(vulkanDescriptorPool),
                vk_descriptorSet);
        }
        else
        {
            return nullptr;
        }
    }

    core::smart_refctd_ptr<IGPUDescriptorSetLayout> createDescriptorSetLayout_impl(const IGPUDescriptorSetLayout::SBinding* _begin, const IGPUDescriptorSetLayout::SBinding* _end) override
    {
        uint32_t bindingCount = std::distance(_begin, _end);
        uint32_t maxSamplersCount = 0u;
        for (uint32_t b = 0u; b < bindingCount; ++b)
        {
            auto binding = _begin + b;
            if (binding->samplers && binding->count > 0u)
                maxSamplersCount += binding->count;
        }

        std::vector<VkSampler> vk_samplers;
        std::vector<VkDescriptorSetLayoutBinding> vk_dsLayoutBindings;
        vk_samplers.reserve(maxSamplersCount); // Reserve to avoid resizing and pointer change while iterating 
        vk_dsLayoutBindings.reserve(bindingCount);

        for (uint32_t b = 0u; b < bindingCount; ++b)
        {
            auto binding = _begin + b;

            VkDescriptorSetLayoutBinding vkDescSetLayoutBinding = {};
            vkDescSetLayoutBinding.binding = binding->binding;
            vkDescSetLayoutBinding.descriptorType = static_cast<VkDescriptorType>(binding->type);
            vkDescSetLayoutBinding.descriptorCount = binding->count;
            vkDescSetLayoutBinding.stageFlags = static_cast<VkShaderStageFlags>(binding->stageFlags);
            vkDescSetLayoutBinding.pImmutableSamplers = nullptr;

            if (binding->type==asset::ESRT_SAMPLED_IMAGE && binding->samplers && binding->count > 0u)
            {
                // If descriptorType is VK_DESCRIPTOR_TYPE_SAMPLER or VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, and descriptorCount is not 0 and pImmutableSamplers is not NULL:
                // pImmutableSamplers must be a valid pointer to an array of descriptorCount valid VkSampler handles.
                const uint32_t samplerOffset = vk_samplers.size();

                for (uint32_t i = 0u; i < binding->count; ++i)
                {
                    if (binding->samplers[i]->getAPIType() != EAT_VULKAN) {
                        assert(false);
                        vk_samplers.push_back(VK_NULL_HANDLE); // To get validation errors on Release Builds
                        continue;
                    }

                    VkSampler vkSampler = IBackendObject::device_compatibility_cast<const CVulkanSampler*>(binding->samplers[i].get(), this)->getInternalObject();
                    vk_samplers.push_back(vkSampler);
                }

                vkDescSetLayoutBinding.pImmutableSamplers = vk_samplers.data() + samplerOffset;
            }

            vk_dsLayoutBindings.push_back(vkDescSetLayoutBinding);
        }

        VkDescriptorSetLayoutCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        vk_createInfo.pNext = nullptr; // Each pNext member of any structure (including this one) in the pNext chain must be either NULL or a pointer to a valid instance of VkDescriptorSetLayoutBindingFlagsCreateInfo or VkMutableDescriptorTypeCreateInfoVALVE
        vk_createInfo.flags = 0; // Todo(achal): I would need to create a IDescriptorSetLayout::SCreationParams for this
        vk_createInfo.bindingCount = vk_dsLayoutBindings.size();
        vk_createInfo.pBindings = vk_dsLayoutBindings.data();

        VkDescriptorSetLayout vk_dsLayout;
        if (m_devf.vk.vkCreateDescriptorSetLayout(m_vkdev, &vk_createInfo, nullptr, &vk_dsLayout) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanDescriptorSetLayout>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this), _begin, _end, vk_dsLayout);
        }
        else
        {
            return nullptr;
        }
    }
    
    core::smart_refctd_ptr<IGPUAccelerationStructure> createAccelerationStructure_impl(IGPUAccelerationStructure::SCreationParams&& params) override;

    core::smart_refctd_ptr<IGPUPipelineLayout> createPipelineLayout_impl(
        const asset::SPushConstantRange* const _pcRangesBegin = nullptr,
        const asset::SPushConstantRange* const _pcRangesEnd = nullptr,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& layout0 = nullptr,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& layout1 = nullptr,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& layout2 = nullptr,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& layout3 = nullptr) override
    {
        constexpr uint32_t MAX_PC_RANGE_COUNT = 100u;

        const core::smart_refctd_ptr<IGPUDescriptorSetLayout> tmp[] = { layout0, layout1, layout2,
            layout3 };

        VkDescriptorSetLayout vk_dsLayouts[asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT];
        uint32_t setLayoutCount = 0u;
        for (uint32_t i = 0u; i < asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
        {
            if (tmp[i] && (tmp[i]->getAPIType() == EAT_VULKAN))
            {
                vk_dsLayouts[i] = IBackendObject::device_compatibility_cast<const CVulkanDescriptorSetLayout*>(tmp[i].get(), this)->getInternalObject();
                setLayoutCount = i + 1;
            }
            else
                vk_dsLayouts[i] = IBackendObject::device_compatibility_cast<const CVulkanDescriptorSetLayout*>(m_dummyDSLayout.get(), this)->getInternalObject();
        }

        const auto pcRangeCount = std::distance(_pcRangesBegin, _pcRangesEnd);
        assert(pcRangeCount <= MAX_PC_RANGE_COUNT);
        VkPushConstantRange vk_pushConstantRanges[MAX_PC_RANGE_COUNT];
        for (uint32_t i = 0u; i < pcRangeCount; ++i)
        {
            const auto pcRange = _pcRangesBegin + i;

            vk_pushConstantRanges[i].stageFlags = static_cast<VkShaderStageFlags>(pcRange->stageFlags);
            vk_pushConstantRanges[i].offset = pcRange->offset;
            vk_pushConstantRanges[i].size = pcRange->size;
        }

        VkPipelineLayoutCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        vk_createInfo.pNext = nullptr; // pNext must be NULL
        vk_createInfo.flags = static_cast<VkPipelineLayoutCreateFlags>(0); // flags must be 0
        vk_createInfo.setLayoutCount = setLayoutCount;
        vk_createInfo.pSetLayouts = vk_dsLayouts;
        vk_createInfo.pushConstantRangeCount = pcRangeCount;
        vk_createInfo.pPushConstantRanges = vk_pushConstantRanges;
                
        VkPipelineLayout vk_pipelineLayout;
        if (m_devf.vk.vkCreatePipelineLayout(m_vkdev, &vk_createInfo, nullptr, &vk_pipelineLayout) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanPipelineLayout>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this), _pcRangesBegin, _pcRangesEnd,
                std::move(layout0), std::move(layout1), std::move(layout2), std::move(layout3),
                vk_pipelineLayout);
        }
        else
        {
            return nullptr;
        }
    }

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
            for (uint32_t ss = 0u; ss < IGPURenderpassIndependentPipeline::SHADER_STAGE_COUNT; ++ss)
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
    inline void getVkMappedMemoryRanges(VkMappedMemoryRange* outRanges, const IDeviceMemoryAllocation::MappedMemoryRange* inRangeBegin, const IDeviceMemoryAllocation::MappedMemoryRange* inRangeEnd)
    {
        uint32_t k = 0u;
        for (auto currentRange = inRangeBegin; currentRange != inRangeEnd; ++currentRange)
        {
            VkMappedMemoryRange& vk_memoryRange = outRanges[k++];
            vk_memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            vk_memoryRange.pNext = nullptr; // pNext must be NULL

            if (currentRange->memory->getAPIType() != EAT_VULKAN)
                continue;

            vk_memoryRange.memory = static_cast<const CVulkanMemoryAllocation*>(currentRange->memory)->getInternalObject();
            vk_memoryRange.offset = static_cast<VkDeviceSize>(currentRange->range.offset);
            vk_memoryRange.size = static_cast<VkDeviceSize>(currentRange->range.length);
        }
    }

    VkDevice m_vkdev;
    CVulkanDeviceFunctionTable m_devf; // Todo(achal): I don't have a function table yet
    
    constexpr static inline uint32_t NODES_PER_BLOCK_DEFERRED_OP = 4096u;
    constexpr static inline uint32_t MAX_BLOCK_COUNT_DEFERRED_OP = 256u;
    memory_pool_mt_t m_deferred_op_mempool;

    core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> m_dummyDSLayout = nullptr;
};

}

#endif