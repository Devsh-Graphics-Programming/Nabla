#ifndef __NBL_C_VULKAN_LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_C_VULKAN_LOGICAL_DEVICE_H_INCLUDED__

#include <algorithm>

#include "nbl/video/ILogicalDevice.h"
// Todo(achal): I should probably consider putting some defintions in CVulkanLogicalDevice.cpp
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
#include "nbl/video/CVulkanCommandBuffer.h"
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
#include "nbl/video/CVulkanForeignImage.h"
#include "nbl/video/surface/CSurfaceVulkan.h"

namespace nbl::video
{

class CVulkanLogicalDevice final : public ILogicalDevice
{
public:
    CVulkanLogicalDevice(IPhysicalDevice* physicalDevice, VkDevice vkdev,
        const SCreationParams& params, core::smart_refctd_ptr<system::ISystem>&& sys)
        : ILogicalDevice(physicalDevice, params), m_vkdev(vkdev), m_devf(vkdev)
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
                // m_devf.vk.vkGetDeviceQueue(m_vkdev, famIx, j, &q);
                vkGetDeviceQueue(m_vkdev, famIx, j, &q);
                        
                const uint32_t ix = offset + j;
                (*m_queues)[ix] = new CThreadSafeGPUQueueAdapter(this, new CVulkanQueue(this, q, famIx, flags, priority));
            }
        }
    }
            
    ~CVulkanLogicalDevice()
    {
        // m_devf.vk.vkDestroyDevice(m_vkdev, nullptr);
        vkDestroyDevice(m_vkdev, nullptr);
    }
            
    core::smart_refctd_ptr<ISwapchain> createSwapchain(ISwapchain::SCreationParams&& params) override
    {
        constexpr uint32_t MAX_SWAPCHAIN_IMAGE_COUNT = 100u;

        if (params.surface->getAPIType() != EAT_VULKAN)
            return nullptr;

        VkSurfaceKHR vk_surface = static_cast<const CSurfaceVulkanWin32*>(params.surface.get())->getInternalObject();

        VkPresentModeKHR vkPresentMode;
        if((params.presentMode & ISurface::E_PRESENT_MODE::EPM_IMMEDIATE) == ISurface::E_PRESENT_MODE::EPM_IMMEDIATE)
            vkPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        else if((params.presentMode & ISurface::E_PRESENT_MODE::EPM_MAILBOX) == ISurface::E_PRESENT_MODE::EPM_MAILBOX)
            vkPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
        else if((params.presentMode & ISurface::E_PRESENT_MODE::EPM_FIFO) == ISurface::E_PRESENT_MODE::EPM_FIFO)
            vkPresentMode = VK_PRESENT_MODE_FIFO_KHR;
        else if((params.presentMode & ISurface::E_PRESENT_MODE::EPM_FIFO_RELAXED) == ISurface::E_PRESENT_MODE::EPM_FIFO_RELAXED)
            vkPresentMode = VK_PRESENT_MODE_FIFO_RELAXED_KHR;

        VkSwapchainCreateInfoKHR vk_createInfo = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
        vk_createInfo.surface = vk_surface;
        vk_createInfo.minImageCount = params.minImageCount;
        vk_createInfo.imageFormat = getVkFormatFromFormat(params.surfaceFormat.format);
        vk_createInfo.imageColorSpace = getVkColorSpaceKHRFromColorSpace(params.surfaceFormat.colorSpace);
        vk_createInfo.imageExtent = { params.width, params.height };
        vk_createInfo.imageArrayLayers = params.arrayLayers;
        vk_createInfo.imageUsage = static_cast<VkImageUsageFlags>(params.imageUsage);
        vk_createInfo.imageSharingMode = static_cast<VkSharingMode>(params.imageSharingMode);
        vk_createInfo.queueFamilyIndexCount = params.queueFamilyIndexCount;
        vk_createInfo.pQueueFamilyIndices = params.queueFamilyIndices;
        vk_createInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR; // Todo(achal)     
        vk_createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // Todo(achal)
        vk_createInfo.presentMode = vkPresentMode;
        vk_createInfo.clipped = VK_TRUE;
        vk_createInfo.oldSwapchain = VK_NULL_HANDLE; // Todo(achal)

        VkSwapchainKHR vk_swapchain;
        if (vkCreateSwapchainKHR(m_vkdev, &vk_createInfo, nullptr, &vk_swapchain) != VK_SUCCESS)
            return nullptr;

        uint32_t imageCount;
        VkResult retval = vkGetSwapchainImagesKHR(m_vkdev, vk_swapchain, &imageCount, nullptr);
        if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE)) // Todo(achal): Would there be a need to handle VK_INCOMPLETE separately?
            return nullptr;

        assert(imageCount <= MAX_SWAPCHAIN_IMAGE_COUNT);

        VkImage vk_images[MAX_SWAPCHAIN_IMAGE_COUNT];
        retval = vkGetSwapchainImagesKHR(m_vkdev, vk_swapchain, &imageCount, vk_images);
        if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE)) // Todo(achal): Would there be a need to handle VK_INCOMPLETE separately?
            return nullptr;

        ISwapchain::images_array_t images = core::make_refctd_dynamic_array<ISwapchain::images_array_t>(imageCount);

        uint32_t i = 0u;
        for (auto& image : (*images))
        {
            CVulkanForeignImage::SCreationParams creationParams;
            creationParams.arrayLayers = params.arrayLayers;
            creationParams.extent = { params.width, params.height, 1u };
            creationParams.flags = static_cast<CVulkanForeignImage::E_CREATE_FLAGS>(0); // Todo(achal)
            creationParams.format = params.surfaceFormat.format;
            creationParams.mipLevels = 1u;
            creationParams.samples = CVulkanImage::ESCF_1_BIT; // Todo(achal)
            creationParams.type = CVulkanImage::ET_2D;

            image = core::make_smart_refctd_ptr<CVulkanForeignImage>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this), std::move(creationParams),
                vk_images[i++]);
        }

        return core::make_smart_refctd_ptr<CVulkanSwapchain>(
            core::smart_refctd_ptr<CVulkanLogicalDevice>(this), std::move(params),
            std::move(images), vk_swapchain);
    }
    
    core::smart_refctd_ptr<IGPUSemaphore> createSemaphore() override
    {
        VkSemaphoreCreateInfo createInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        createInfo.pNext = nullptr; // Each pNext member of any structure (including this one) in the pNext chain must be either NULL or a pointer to a valid instance of VkExportSemaphoreCreateInfo, VkExportSemaphoreWin32HandleInfoKHR, or VkSemaphoreTypeCreateInfo
        createInfo.flags = static_cast<VkSemaphoreCreateFlags>(0); // flags must be 0

        VkSemaphore semaphore;
        if (vkCreateSemaphore(m_vkdev, &createInfo, nullptr, &semaphore) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanSemaphore>
                (core::smart_refctd_ptr<CVulkanLogicalDevice>(this), semaphore);
        }
        else
        {
            return nullptr;
        }
    }
            
    core::smart_refctd_ptr<IGPUEvent> createEvent(IGPUEvent::E_CREATE_FLAGS flags) override
    {
        return nullptr;
    };
            
    IGPUEvent::E_STATUS getEventStatus(const IGPUEvent* _event) override
    {
        return IGPUEvent::E_STATUS::ES_FAILURE;
    }
            
    IGPUEvent::E_STATUS resetEvent(IGPUEvent* _event) override
    {
        return IGPUEvent::E_STATUS::ES_FAILURE;
    }
            
    IGPUEvent::E_STATUS setEvent(IGPUEvent* _event) override
    {
        return IGPUEvent::E_STATUS::ES_FAILURE;
    }
            
    core::smart_refctd_ptr<IGPUFence> createFence(IGPUFence::E_CREATE_FLAGS flags) override
    {
        VkFenceCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        vk_createInfo.pNext = nullptr; // Each pNext member of any structure (including this one) in the pNext chain must be either NULL or a pointer to a valid instance of VkExportFenceCreateInfo or VkExportFenceWin32HandleInfoKHR
        vk_createInfo.flags = static_cast<VkFenceCreateFlags>(flags);

        VkFence vk_fence;
        if (vkCreateFence(m_vkdev, &vk_createInfo, nullptr, &vk_fence) == VK_SUCCESS)
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

        VkResult retval = vkGetFenceStatus(m_vkdev, static_cast<const CVulkanFence*>(_fence)->getInternalObject());

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
    void resetFences(uint32_t _count, IGPUFence*const* _fences) override
    {
        constexpr uint32_t MAX_FENCE_COUNT = 100u;
        assert(_count < MAX_FENCE_COUNT);

        VkFence vk_fences[MAX_FENCE_COUNT];
        for (uint32_t i = 0u; i < _count; ++i)
        {
            if (_fences[i]->getAPIType() != EAT_VULKAN)
                return;

            vk_fences[i] = static_cast<CVulkanFence*>(_fences[i])->getInternalObject();
        }

        vkResetFences(m_vkdev, _count, vk_fences);
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

            vk_fences[i] = static_cast<CVulkanFence*>(_fences[i])->getInternalObject();
        }

        VkResult result = vkWaitForFences(m_vkdev, _count, vk_fences, _waitAll, _timeout);
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
            
    const core::smart_refctd_dynamic_array<std::string> getSupportedGLSLExtensions() const override
    {
        return nullptr;
    }
            
    core::smart_refctd_ptr<IGPUCommandPool> createCommandPool(uint32_t familyIndex, core::bitflag<IGPUCommandPool::E_CREATE_FLAGS> flags) override
    {
        VkCommandPoolCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        vk_createInfo.pNext = nullptr; // pNext must be NULL
        vk_createInfo.flags = static_cast<VkCommandPoolCreateFlags>(flags.value);
        vk_createInfo.queueFamilyIndex = familyIndex;

        VkCommandPool vk_commandPool = VK_NULL_HANDLE;
        if (vkCreateCommandPool(m_vkdev, &vk_createInfo, nullptr, &vk_commandPool) == VK_SUCCESS)
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
        if (vkCreateDescriptorPool(m_vkdev, &vk_createInfo, nullptr, &vk_descriptorPool) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanDescriptorPool>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this), maxSets, vk_descriptorPool);
        }
        else
        {
            return nullptr;
        }
    }
            
    core::smart_refctd_ptr<IGPURenderpass> createGPURenderpass(const IGPURenderpass::SCreationParams& params) override
    {
        // auto* vk = m_vkdev->getFunctionTable();

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
            vkdep.srcStageMask = static_cast<VkPipelineStageFlags>(dep.srcStageMask);
            vkdep.dstStageMask = static_cast<VkPipelineStageFlags>(dep.dstStageMask);
            vkdep.srcAccessMask = static_cast<VkAccessFlags>(dep.srcAccessMask);
            vkdep.dstAccessMask = static_cast<VkAccessFlags>(dep.dstAccessMask);
            vkdep.dependencyFlags = static_cast<VkDependencyFlags>(dep.dependencyFlags);
        }
        createInfo.pDependencies = deps.data();

        // vk->vk.vkCreateRenderPass(vkdev, &ci, nullptr, &m_renderpass);
        VkRenderPass vk_renderpass;
        if (vkCreateRenderPass(m_vkdev, &createInfo, nullptr, &vk_renderpass) == VK_SUCCESS)
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
    void flushMappedMemoryRanges(core::SRange<const video::IDriverMemoryAllocation::MappedMemoryRange> ranges) override
    {
        constexpr uint32_t MAX_MEMORY_RANGE_COUNT = 408u;
        VkMappedMemoryRange vk_memoryRanges[MAX_MEMORY_RANGE_COUNT];

        const uint32_t memoryRangeCount = static_cast<uint32_t>(ranges.size());
        assert(memoryRangeCount <= MAX_MEMORY_RANGE_COUNT);

        getVkMappedMemoryRanges(vk_memoryRanges, ranges.begin(), ranges.end());
        
        if (vkFlushMappedMemoryRanges(m_vkdev, memoryRangeCount, vk_memoryRanges) != VK_SUCCESS)
            printf("flushMappedMemoryRanges failed\n");
    }
            
    // API needs to change, this could fail
    void invalidateMappedMemoryRanges(core::SRange<const video::IDriverMemoryAllocation::MappedMemoryRange> ranges) override
    {
        constexpr uint32_t MAX_MEMORY_RANGE_COUNT = 408u;
        VkMappedMemoryRange vk_memoryRanges[MAX_MEMORY_RANGE_COUNT];

        const uint32_t memoryRangeCount = static_cast<uint32_t>(ranges.size());
        assert(memoryRangeCount <= MAX_MEMORY_RANGE_COUNT);

        getVkMappedMemoryRanges(vk_memoryRanges, ranges.begin(), ranges.end());

        if (vkInvalidateMappedMemoryRanges(m_vkdev, memoryRangeCount, vk_memoryRanges) != VK_SUCCESS)
            printf("invalidateMappedMemoryRanges failed!\n");
    }

    bool bindBufferMemory(uint32_t bindInfoCount, const SBindBufferMemoryInfo* pBindInfos) override
    {
        bool anyFailed = false;
        for (uint32_t i = 0u; i < bindInfoCount; ++i)
        {
            const auto& bindInfo = pBindInfos[i];
            
            if ((bindInfo.buffer->getAPIType() != EAT_VULKAN) || (bindInfo.memory->getAPIType() != EAT_VULKAN))
                continue;

            CVulkanBuffer* vulkanBuffer = static_cast<CVulkanBuffer*>(bindInfo.buffer);
            vulkanBuffer->setMemoryAndOffset(
                core::smart_refctd_ptr<IDriverMemoryAllocation>(bindInfo.memory), bindInfo.offset);

            VkBuffer vk_buffer = vulkanBuffer->getInternalObject();
            VkDeviceMemory vk_memory = static_cast<const CVulkanMemoryAllocation*>(pBindInfos[i].memory)->getInternalObject();
            if (vkBindBufferMemory(m_vkdev, vk_buffer, vk_memory, static_cast<VkDeviceSize>(pBindInfos[i].offset)) != VK_SUCCESS)
            {
                // Todo(achal): Log which one failed
                anyFailed = true;
            }
        }   

        return !anyFailed;
    }

    core::smart_refctd_ptr<IGPUBuffer> createGPUBuffer(const IGPUBuffer::SCreationParams& creationParams, const bool canModifySubData = false) override
    {
        VkBufferCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        vk_createInfo.pNext = nullptr; // Each pNext member of any structure (including this one) in the pNext chain must be either NULL or a pointer to a valid instance of VkBufferDeviceAddressCreateInfoEXT, VkBufferOpaqueCaptureAddressCreateInfo, VkDedicatedAllocationBufferCreateInfoNV, VkExternalMemoryBufferCreateInfo, VkVideoProfileKHR, or VkVideoProfilesKHR
        vk_createInfo.flags = static_cast<VkBufferCreateFlags>(0); // Nabla doesn't support any of these flags
        vk_createInfo.size = static_cast<VkDeviceSize>(creationParams.size);
        vk_createInfo.usage = static_cast<VkBufferUsageFlags>(creationParams.usage);
        vk_createInfo.sharingMode = static_cast<VkSharingMode>(creationParams.sharingMode); 
        vk_createInfo.queueFamilyIndexCount = creationParams.queueFamilyIndexCount;
        vk_createInfo.pQueueFamilyIndices = creationParams.queueFamilyIndices;

        VkBuffer vk_buffer;
        if (vkCreateBuffer(m_vkdev, &vk_createInfo, nullptr, &vk_buffer) == VK_SUCCESS)
        {
            VkBufferMemoryRequirementsInfo2 vk_memoryRequirementsInfo = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2 };
            vk_memoryRequirementsInfo.pNext = nullptr; // pNext must be NULL
            vk_memoryRequirementsInfo.buffer = vk_buffer;

            VkMemoryDedicatedRequirements vk_dedicatedMemoryRequirements = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS };
            VkMemoryRequirements2 vk_memoryRequirements = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
            vk_memoryRequirements.pNext = &vk_dedicatedMemoryRequirements;
            vkGetBufferMemoryRequirements2(m_vkdev, &vk_memoryRequirementsInfo, &vk_memoryRequirements);

            IDriverMemoryBacked::SDriverMemoryRequirements bufferMemoryReqs = {};
            bufferMemoryReqs.vulkanReqs.alignment = vk_memoryRequirements.memoryRequirements.alignment;
            bufferMemoryReqs.vulkanReqs.size = vk_memoryRequirements.memoryRequirements.size;
            bufferMemoryReqs.vulkanReqs.memoryTypeBits = vk_memoryRequirements.memoryRequirements.memoryTypeBits;
            bufferMemoryReqs.memoryHeapLocation = 0u; // doesn't matter, would get overwritten during memory allocation for this resource anyway
            bufferMemoryReqs.mappingCapability = 0u; // doesn't matter, would get overwritten during memory allocation for this resource anyway
            bufferMemoryReqs.prefersDedicatedAllocation = vk_dedicatedMemoryRequirements.prefersDedicatedAllocation;
            bufferMemoryReqs.requiresDedicatedAllocation = vk_dedicatedMemoryRequirements.requiresDedicatedAllocation;

            return core::make_smart_refctd_ptr<CVulkanBuffer>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this), bufferMemoryReqs, canModifySubData, vk_buffer);
        }
        else
        {
            return nullptr;
        }
    }

    core::smart_refctd_ptr<IGPUBuffer> createGPUBufferOnDedMem(const IGPUBuffer::SCreationParams& creationParams, const IDriverMemoryBacked::SDriverMemoryRequirements& additionalMemoryReqs, const bool canModifySubData = false) override
    {
        core::smart_refctd_ptr<IGPUBuffer> gpuBuffer = createGPUBuffer(creationParams);

        if (!gpuBuffer)
            return nullptr;

        IDriverMemoryBacked::SDriverMemoryRequirements memoryReqs = gpuBuffer->getMemoryReqs();
        memoryReqs.vulkanReqs.size = core::max(memoryReqs.vulkanReqs.size, additionalMemoryReqs.vulkanReqs.size);
        memoryReqs.vulkanReqs.alignment = core::max(memoryReqs.vulkanReqs.alignment, additionalMemoryReqs.vulkanReqs.alignment);
        memoryReqs.vulkanReqs.memoryTypeBits &= additionalMemoryReqs.vulkanReqs.memoryTypeBits;
        memoryReqs.memoryHeapLocation = additionalMemoryReqs.memoryHeapLocation;
        memoryReqs.mappingCapability = additionalMemoryReqs.mappingCapability;

        core::smart_refctd_ptr<video::IDriverMemoryAllocation> bufferMemory =
            allocateGPUMemory(memoryReqs);

        if (!bufferMemory)
            return nullptr;

        ILogicalDevice::SBindBufferMemoryInfo bindBufferInfo = {};
        bindBufferInfo.buffer = gpuBuffer.get();
        bindBufferInfo.memory = bufferMemory.get();
        bindBufferInfo.offset = 0ull;

        if (!bindBufferMemory(1u, &bindBufferInfo))
            return nullptr;

        return gpuBuffer;
    }
        
    core::smart_refctd_ptr<IGPUShader> createGPUShader(core::smart_refctd_ptr<asset::ICPUShader>&& cpushader) override
    {
        const asset::ICPUBuffer* source = cpushader->getSPVorGLSL();
        core::smart_refctd_ptr<asset::ICPUBuffer> clone =
            core::smart_refctd_ptr_static_cast<asset::ICPUBuffer>(source->clone(1u));
        if (cpushader->containsGLSL())
        {
            return core::make_smart_refctd_ptr<CVulkanShader>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this), std::move(clone),
                IGPUShader::buffer_contains_glsl);
        }
        else
        {
            return core::make_smart_refctd_ptr<CVulkanShader>(core::smart_refctd_ptr<CVulkanLogicalDevice>(this),
                std::move(clone));
        }
    }

    core::smart_refctd_ptr<IGPUImage> createGPUImage(asset::IImage::SCreationParams&& params) override
    {
        VkImageCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        vk_createInfo.pNext = nullptr; // there are a lot of extensions
        vk_createInfo.flags = static_cast<VkImageCreateFlags>(params.flags);
        vk_createInfo.imageType = static_cast<VkImageType>(params.type);
        vk_createInfo.format = getVkFormatFromFormat(params.format);
        vk_createInfo.extent = { params.extent.width, params.extent.height, params.extent.depth };
        vk_createInfo.mipLevels = params.mipLevels;
        vk_createInfo.arrayLayers = params.arrayLayers;
        vk_createInfo.samples = static_cast<VkSampleCountFlagBits>(params.samples);
        vk_createInfo.tiling = static_cast<VkImageTiling>(params.tiling);
        vk_createInfo.usage = static_cast<VkImageUsageFlags>(params.usage.value);
        vk_createInfo.sharingMode = static_cast<VkSharingMode>(params.sharingMode);
        vk_createInfo.queueFamilyIndexCount = params.queueFamilyIndexCount;
        vk_createInfo.pQueueFamilyIndices = params.queueFamilyIndices;
        vk_createInfo.initialLayout = static_cast<VkImageLayout>(params.initialLayout);

        VkImage vk_image;
        if (vkCreateImage(m_vkdev, &vk_createInfo, nullptr, &vk_image) == VK_SUCCESS)
        {
            VkImageMemoryRequirementsInfo2 vk_memReqsInfo = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2 };
            vk_memReqsInfo.pNext = nullptr;
            vk_memReqsInfo.image = vk_image;

            VkMemoryDedicatedRequirements vk_memDedReqs = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS };
            VkMemoryRequirements2 vk_memReqs = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
            vk_memReqs.pNext = &vk_memDedReqs;

            vkGetImageMemoryRequirements2(m_vkdev, &vk_memReqsInfo, &vk_memReqs);

            IDriverMemoryBacked::SDriverMemoryRequirements imageMemReqs = {};
            imageMemReqs.vulkanReqs.alignment = vk_memReqs.memoryRequirements.alignment;
            imageMemReqs.vulkanReqs.size = vk_memReqs.memoryRequirements.size;
            imageMemReqs.vulkanReqs.memoryTypeBits = vk_memReqs.memoryRequirements.memoryTypeBits;
            imageMemReqs.memoryHeapLocation = 0u; // doesn't matter, would get overwritten during memory allocation for this resource anyway
            imageMemReqs.mappingCapability = 0u; // doesn't matter, would get overwritten during memory allocation for this resource anyway
            imageMemReqs.prefersDedicatedAllocation = vk_memDedReqs.prefersDedicatedAllocation;
            imageMemReqs.requiresDedicatedAllocation = vk_memDedReqs.requiresDedicatedAllocation;

            return core::make_smart_refctd_ptr<CVulkanImage>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this), std::move(params),
                vk_image, imageMemReqs);
        }
        else
        {
            return nullptr;
        }
    }

    bool bindImageMemory(uint32_t bindInfoCount, const SBindImageMemoryInfo* pBindInfos) override
    {
        bool anyFailed = false;
        for (uint32_t i = 0u; i < bindInfoCount; ++i)
        {
            const auto& bindInfo = pBindInfos[i];

            if ((bindInfo.image->getAPIType() != EAT_VULKAN) || (bindInfo.memory->getAPIType() != EAT_VULKAN))
                continue;

            CVulkanImage* vulkanImage = static_cast<CVulkanImage*>(bindInfo.image);
            vulkanImage->setMemoryAndOffset(
                core::smart_refctd_ptr<IDriverMemoryAllocation>(bindInfo.memory),
                bindInfo.offset);

            VkImage vk_image = vulkanImage->getInternalObject();
            VkDeviceMemory vk_deviceMemory = static_cast<const CVulkanMemoryAllocation*>(bindInfo.memory)->getInternalObject();
            if (vkBindImageMemory(m_vkdev, vk_image, vk_deviceMemory, static_cast<VkDeviceSize>(bindInfo.offset)) != VK_SUCCESS)
            {
                // Todo(achal): Log which one failed
                anyFailed = true;
            }
        }

        return !anyFailed;
    }
            
    core::smart_refctd_ptr<IGPUImage> createGPUImageOnDedMem(IGPUImage::SCreationParams&& params, const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs) override
    {
        core::smart_refctd_ptr<IGPUImage> gpuImage = createGPUImage(std::move(params));

        if (!gpuImage)
            return nullptr;

        IDriverMemoryBacked::SDriverMemoryRequirements memReqs = gpuImage->getMemoryReqs();
        memReqs.vulkanReqs.size = core::max(memReqs.vulkanReqs.size, initialMreqs.vulkanReqs.size);
        memReqs.vulkanReqs.alignment = core::max(memReqs.vulkanReqs.alignment, initialMreqs.vulkanReqs.alignment);
        memReqs.vulkanReqs.memoryTypeBits &= initialMreqs.vulkanReqs.memoryTypeBits;
        memReqs.memoryHeapLocation = initialMreqs.memoryHeapLocation;
        memReqs.mappingCapability = initialMreqs.mappingCapability;

        core::smart_refctd_ptr<video::IDriverMemoryAllocation> imageMemory =
            allocateGPUMemory(memReqs);

        if (!imageMemory)
            return nullptr;

        ILogicalDevice::SBindImageMemoryInfo bindImageInfo = {};
        bindImageInfo.image = gpuImage.get();
        bindImageInfo.memory = imageMemory.get();
        bindImageInfo.offset = 0ull;

        if (!bindImageMemory(1u, &bindImageInfo))
            return nullptr;

        return gpuImage;
    }

    void updateDescriptorSets(uint32_t descriptorWriteCount, const IGPUDescriptorSet::SWriteDescriptorSet* pDescriptorWrites,
        uint32_t descriptorCopyCount, const IGPUDescriptorSet::SCopyDescriptorSet* pDescriptorCopies) override
    {
        constexpr uint32_t MAX_DESCRIPTOR_WRITE_COUNT = 25u;
        constexpr uint32_t MAX_DESCRIPTOR_COPY_COUNT = 25u;
        constexpr uint32_t MAX_DESCRIPTOR_ARRAY_COUNT = MAX_DESCRIPTOR_WRITE_COUNT;

        // Todo(achal): This exceeds 16384 bytes on stack, move to heap

        assert(descriptorWriteCount <= MAX_DESCRIPTOR_WRITE_COUNT);
        VkWriteDescriptorSet vk_writeDescriptorSets[MAX_DESCRIPTOR_WRITE_COUNT];

        uint32_t bufferInfoOffset = 0u;
        VkDescriptorBufferInfo vk_bufferInfos[MAX_DESCRIPTOR_WRITE_COUNT * MAX_DESCRIPTOR_ARRAY_COUNT];

        uint32_t imageInfoOffset = 0u;
        VkDescriptorImageInfo vk_imageInfos[MAX_DESCRIPTOR_WRITE_COUNT * MAX_DESCRIPTOR_ARRAY_COUNT];

        uint32_t bufferViewOffset = 0u;
        VkBufferView vk_bufferViews[MAX_DESCRIPTOR_WRITE_COUNT * MAX_DESCRIPTOR_ARRAY_COUNT];

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
            vk_writeDescriptorSets[i].descriptorCount = pDescriptorWrites[i].count;
            vk_writeDescriptorSets[i].descriptorType = static_cast<VkDescriptorType>(pDescriptorWrites[i].descriptorType);

            assert(pDescriptorWrites[i].count <= MAX_DESCRIPTOR_ARRAY_COUNT);

            switch (pDescriptorWrites[i].info->desc->getTypeCategory())
            {
                case asset::IDescriptor::EC_BUFFER:
                {
                    for (uint32_t j = 0u; j < pDescriptorWrites[i].count; ++j)
                    {
                        // if (pDescriptorWrites[i].info[j].desc->getAPIType() != EAT_VULKAN)
                        //     continue;

                        VkBuffer vk_buffer = static_cast<const CVulkanBuffer*>(pDescriptorWrites[i].info[j].desc.get())->getInternalObject();

                        vk_bufferInfos[bufferInfoOffset + j].buffer = vk_buffer;
                        vk_bufferInfos[bufferInfoOffset + j].offset = pDescriptorWrites[i].info[j].buffer.offset;
                        vk_bufferInfos[bufferInfoOffset + j].range = pDescriptorWrites[i].info[j].buffer.size;
                    }

                    vk_writeDescriptorSets[i].pBufferInfo = vk_bufferInfos + bufferInfoOffset;
                    bufferInfoOffset += pDescriptorWrites[i].count;
                } break;

                case asset::IDescriptor::EC_IMAGE:
                {
                    for (uint32_t j = 0u; j < pDescriptorWrites[i].count; ++j)
                    {
                        auto descriptorWriteImageInfo = pDescriptorWrites[i].info[j].image;

                        VkSampler vk_sampler = VK_NULL_HANDLE;
                        if (descriptorWriteImageInfo.sampler && (descriptorWriteImageInfo.sampler->getAPIType() == EAT_VULKAN))
                            vk_sampler = static_cast<const CVulkanSampler*>(descriptorWriteImageInfo.sampler.get())->getInternalObject();

                        // if (pDescriptorWrites[i].info[j].desc->getAPIType() != EAT_VULKAN)
                        //     continue;
                        VkImageView vk_imageView = static_cast<const CVulkanImageView*>(pDescriptorWrites[i].info[j].desc.get())->getInternalObject();

                        vk_imageInfos[imageInfoOffset + j].sampler = vk_sampler;
                        vk_imageInfos[imageInfoOffset + j].imageView = vk_imageView;
                        vk_imageInfos[imageInfoOffset + j].imageLayout = static_cast<VkImageLayout>(descriptorWriteImageInfo.imageLayout);
                    }

                    vk_writeDescriptorSets[i].pImageInfo = vk_imageInfos + imageInfoOffset;
                    imageInfoOffset += pDescriptorWrites[i].count;
                } break;

                case asset::IDescriptor::EC_BUFFER_VIEW:
                {
                    for (uint32_t j = 0u; j < pDescriptorWrites[i].count; ++j)
                    {
                        // if (pDescriptorWrites[i].info[j].desc->getAPIType() != EAT_VULKAN)
                        //     continue;

                        VkBufferView vk_bufferView = static_cast<const CVulkanBufferView*>(pDescriptorWrites[i].info[j].desc.get())->getInternalObject();
                        vk_bufferViews[bufferViewOffset + j] = vk_bufferView;
                    }

                    vk_writeDescriptorSets[i].pTexelBufferView = vk_bufferViews + bufferViewOffset;
                    bufferViewOffset += pDescriptorWrites[i].count;
                } break;
            }
        }

        assert(descriptorCopyCount <= MAX_DESCRIPTOR_COPY_COUNT);
        VkCopyDescriptorSet vk_copyDescriptorSets[MAX_DESCRIPTOR_COPY_COUNT];

        for (uint32_t i = 0u; i < descriptorCopyCount; ++i)
        {
            vk_copyDescriptorSets[i].sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET;
            vk_copyDescriptorSets[i].pNext = nullptr; // pNext must be NULL

            // if (pDescriptorCopies[i].srcSet->getAPIType() != EAT_VULKAN)
            //     continue;
            vk_copyDescriptorSets[i].srcSet = static_cast<const CVulkanDescriptorSet*>(pDescriptorCopies[i].srcSet)->getInternalObject();

            vk_copyDescriptorSets[i].srcBinding = pDescriptorCopies[i].srcBinding;
            vk_copyDescriptorSets[i].srcArrayElement = pDescriptorCopies[i].srcArrayElement;

            // if (pDescriptorCopies[i].dstSet->getAPIType() != EAT_VULKAN)
            //     continue;
            vk_copyDescriptorSets[i].dstSet = static_cast<const CVulkanDescriptorSet*>(pDescriptorCopies[i].dstSet)->getInternalObject();

            vk_copyDescriptorSets[i].dstBinding = pDescriptorCopies[i].dstBinding;
            vk_copyDescriptorSets[i].dstArrayElement = pDescriptorCopies[i].dstArrayElement;
            vk_copyDescriptorSets[i].descriptorCount = pDescriptorCopies[i].count;
        }

        vkUpdateDescriptorSets(m_vkdev, descriptorWriteCount, vk_writeDescriptorSets, descriptorCopyCount, vk_copyDescriptorSets);
    }

    core::smart_refctd_ptr<IDriverMemoryAllocation> allocateDeviceLocalMemory(
        const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs) override;

    core::smart_refctd_ptr<IDriverMemoryAllocation> allocateSpilloverMemory(
        const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs) override;

    core::smart_refctd_ptr<IDriverMemoryAllocation> allocateUpStreamingMemory(
        const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs) override;

    core::smart_refctd_ptr<IDriverMemoryAllocation> allocateDownStreamingMemory(
        const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs) override;

    core::smart_refctd_ptr<IDriverMemoryAllocation> allocateCPUSideGPUVisibleMemory(
        const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs) override;

    core::smart_refctd_ptr<IDriverMemoryAllocation> allocateGPUMemory(
        const IDriverMemoryBacked::SDriverMemoryRequirements& reqs) override;

    core::smart_refctd_ptr<IGPUSampler> createGPUSampler(const IGPUSampler::SParams& _params) override
    {
        return nullptr;
    }

    // API changes needed, this could also fail.
    void waitIdle() override
    {
        VkResult retval = vkDeviceWaitIdle(m_vkdev);

        // Todo(achal): Handle errors
        assert(retval == VK_SUCCESS);
    }

    void* mapMemory(const IDriverMemoryAllocation::MappedMemoryRange& memory, IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAG accessHint = IDriverMemoryAllocation::EMCAF_READ_AND_WRITE) override
    {
        if (memory.memory->getAPIType() != EAT_VULKAN)
            return nullptr;

        VkMemoryMapFlags vk_memoryMapFlags = 0; // reserved for future use, by Vulkan
        VkDeviceMemory vk_memory = static_cast<const CVulkanMemoryAllocation*>(memory.memory)->getInternalObject();
        void* mappedPtr;
        if (vkMapMemory(m_vkdev, vk_memory, static_cast<VkDeviceSize>(memory.offset),
            static_cast<VkDeviceSize>(memory.length), vk_memoryMapFlags, &mappedPtr) == VK_SUCCESS)
        {
            return mappedPtr;
        }
        else
        {
            return nullptr;
        }
    }

    void unmapMemory(IDriverMemoryAllocation* memory) override
    {
        if (memory->getAPIType() != EAT_VULKAN)
            return;

        VkDeviceMemory vk_deviceMemory = static_cast<const CVulkanMemoryAllocation*>(memory)->getInternalObject();
        vkUnmapMemory(m_vkdev, vk_deviceMemory);
    }

    CVulkanDeviceFunctionTable* getFunctionTable() { return &m_devf; }

    VkDevice getInternalObject() const { return m_vkdev; }

protected:
    bool createCommandBuffers_impl(IGPUCommandPool* cmdPool, IGPUCommandBuffer::E_LEVEL level,
        uint32_t count, core::smart_refctd_ptr<IGPUCommandBuffer>* outCmdBufs) override
    {
        constexpr uint32_t MAX_COMMAND_BUFFER_COUNT = 1000u;

        if (cmdPool->getAPIType() != EAT_VULKAN)
            return false;

        auto vulkanCommandPool = static_cast<CVulkanCommandPool*>(cmdPool)->getInternalObject();

        assert(count <= MAX_COMMAND_BUFFER_COUNT);
        VkCommandBuffer vk_commandBuffers[MAX_COMMAND_BUFFER_COUNT];

        VkCommandBufferAllocateInfo vk_allocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        vk_allocateInfo.pNext = nullptr; // this must be NULL
        vk_allocateInfo.commandPool = vulkanCommandPool;
        vk_allocateInfo.level = static_cast<VkCommandBufferLevel>(level);
        vk_allocateInfo.commandBufferCount = count;

        if (vkAllocateCommandBuffers(m_vkdev, &vk_allocateInfo, vk_commandBuffers) == VK_SUCCESS)
        {
            for (uint32_t i = 0u; i < count; ++i)
            {
                outCmdBufs[i] = core::make_smart_refctd_ptr<CVulkanCommandBuffer>(
                    core::smart_refctd_ptr<ILogicalDevice>(this), level, vk_commandBuffers[i],
                    cmdPool);
            }

            return true;
        }
        else
        {
            return false;
        }
    }

    bool freeCommandBuffers_impl(IGPUCommandBuffer** _cmdbufs, uint32_t _count) override
    {
        return false;
    }

    core::smart_refctd_ptr<IGPUFramebuffer> createGPUFramebuffer_impl(IGPUFramebuffer::SCreationParams&& params) override
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
                vk_attachments[i] = static_cast<const CVulkanImageView*>(params.attachments[i].get())->getInternalObject();
                ++attachmentCount;
            }
        }
        assert(attachmentCount <= MaxAttachments);

        VkFramebufferCreateInfo createInfo = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        createInfo.pNext = nullptr;
        createInfo.flags = static_cast<VkFramebufferCreateFlags>(params.flags);

        if (params.renderpass->getAPIType() != EAT_VULKAN)
            return nullptr;

        createInfo.renderPass = static_cast<const CVulkanRenderpass*>(params.renderpass.get())->getInternalObject();
        createInfo.attachmentCount = attachmentCount;
        createInfo.pAttachments = vk_attachments;
        createInfo.width = params.width;
        createInfo.height = params.height;
        createInfo.layers = params.layers;

        // vk->vk.vkCreateFramebuffer(vkdev, &createInfo, nullptr, &m_vkfbo);
        VkFramebuffer vk_framebuffer;
        if (vkCreateFramebuffer(m_vkdev, &createInfo, nullptr, &vk_framebuffer) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanFramebuffer>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this), std::move(params), vk_framebuffer);
        }
        else
        {
            return nullptr;
        }
    }

    // Todo(achal): For some reason this is not printing shader errors to console
    core::smart_refctd_ptr<IGPUSpecializedShader> createGPUSpecializedShader_impl(const IGPUShader* _unspecialized, const asset::ISpecializedShader::SInfo& specInfo, const asset::ISPIRVOptimizer* spvopt) override
    {
        if (_unspecialized->getAPIType() != EAT_VULKAN)
            return nullptr;

        const CVulkanShader* unspecializedShader = static_cast<const CVulkanShader*>(_unspecialized);

        const std::string& entryPoint = specInfo.entryPoint;
        const asset::ISpecializedShader::E_SHADER_STAGE shaderStage = specInfo.shaderStage;

        core::smart_refctd_ptr<asset::ICPUBuffer> spirv = nullptr;
        if (unspecializedShader->containsGLSL())
        {
            const char* begin = reinterpret_cast<const char*>(unspecializedShader->getSPVorGLSL()->getPointer());
            const char* end = begin + unspecializedShader->getSPVorGLSL()->getSize();
            std::string glsl(begin, end);
            core::smart_refctd_ptr<asset::ICPUShader> glslShader_woIncludes =
                m_physicalDevice->getGLSLCompiler()->resolveIncludeDirectives(glsl.c_str(),
                    shaderStage, specInfo.m_filePathHint.string().c_str());

            spirv = m_physicalDevice->getGLSLCompiler()->compileSPIRVFromGLSL(
                reinterpret_cast<const char*>(glslShader_woIncludes->getSPVorGLSL()->getPointer()),
                shaderStage, entryPoint.c_str(), specInfo.m_filePathHint.string().c_str());
        }
        else
        {
            spirv = unspecializedShader->getSPVorGLSL_refctd();
        }

        // Should just do this check in ISPIRVOptimizer::optimize
        if (!spirv)
            return nullptr;

        if (spvopt)
            spirv = spvopt->optimize(spirv.get(), m_physicalDevice->getDebugCallback()->getLogger());

        if (!spirv)
            return nullptr;

        VkShaderModuleCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        vk_createInfo.pNext = nullptr;
        vk_createInfo.flags = static_cast<VkShaderModuleCreateFlags>(0); // reserved for future use by Vulkan
        vk_createInfo.codeSize = spirv->getSize();
        vk_createInfo.pCode = reinterpret_cast<const uint32_t*>(spirv->getPointer());

        VkShaderModule vk_shaderModule;
        if (vkCreateShaderModule(m_vkdev, &vk_createInfo, nullptr, &vk_shaderModule) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<video::CVulkanSpecializedShader>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this), vk_shaderModule, shaderStage);
        }
        else
        {
            return nullptr;
        }
    }

    core::smart_refctd_ptr<IGPUBufferView> createGPUBufferView_impl(IGPUBuffer* _underlying, asset::E_FORMAT _fmt, size_t _offset = 0ull, size_t _size = IGPUBufferView::whole_buffer) override
    {
        if (_underlying->getAPIType() != EAT_VULKAN)
            return nullptr;

        VkBuffer vk_buffer = static_cast<const CVulkanBuffer*>(_underlying)->getInternalObject();

        VkBufferViewCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO };
        vk_createInfo.pNext = nullptr; // pNext must be NULL
        vk_createInfo.flags = static_cast<VkBufferViewCreateFlags>(0); // flags must be 0
        vk_createInfo.buffer = vk_buffer;
        vk_createInfo.format = getVkFormatFromFormat(_fmt);
        vk_createInfo.offset = _offset;
        vk_createInfo.range = _size;

        VkBufferView vk_bufferView;
        if (vkCreateBufferView(m_vkdev, &vk_createInfo, nullptr, &vk_bufferView) == VK_SUCCESS)
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

    core::smart_refctd_ptr<IGPUImageView> createGPUImageView_impl(IGPUImageView::SCreationParams&& params) override
    {
        VkImageViewCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vk_createInfo.pNext = nullptr; // Each pNext member of any structure (including this one) in the pNext chain must be either NULL or a pointer to a valid instance of VkImageViewASTCDecodeModeEXT, VkImageViewUsageCreateInfo, VkSamplerYcbcrConversionInfo, VkVideoProfileKHR, or VkVideoProfilesKHR
        vk_createInfo.flags = static_cast<VkImageViewCreateFlags>(params.flags);

        if (params.image->getAPIType() != EAT_VULKAN)
            return nullptr;

        VkImage vk_image = static_cast<const CVulkanImage*>(params.image.get())->getInternalObject();
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
        if (vkCreateImageView(m_vkdev, &vk_createInfo, nullptr, &vk_imageView) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanImageView>(core::smart_refctd_ptr<CVulkanLogicalDevice>(this),
                std::move(params), vk_imageView);
        }
        else
        {
            return nullptr;
        }
    }

    core::smart_refctd_ptr<IGPUDescriptorSet> createGPUDescriptorSet_impl(IDescriptorPool* pool, core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout) override
    {
        if (pool->getAPIType() != EAT_VULKAN)
            return nullptr;

        const CVulkanDescriptorPool* vulkanDescriptorPool = static_cast<const CVulkanDescriptorPool*>(pool);

        VkDescriptorSetAllocateInfo vk_allocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        vk_allocateInfo.pNext = nullptr; // pNext must be NULL or a pointer to a valid instance of VkDescriptorSetVariableDescriptorCountAllocateInfo

        vk_allocateInfo.descriptorPool = vulkanDescriptorPool->getInternalObject();
        vk_allocateInfo.descriptorSetCount = 1u; // Isn't creating only descriptor set every time wasteful?

        if (layout->getAPIType() != EAT_VULKAN)
            return nullptr;
        VkDescriptorSetLayout vk_dsLayout = static_cast<const CVulkanDescriptorSetLayout*>(layout.get())->getInternalObject();
        vk_allocateInfo.pSetLayouts = &vk_dsLayout;

        VkDescriptorSet vk_descriptorSet;
        if (vkAllocateDescriptorSets(m_vkdev, &vk_allocateInfo, &vk_descriptorSet) == VK_SUCCESS)
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

    core::smart_refctd_ptr<IGPUDescriptorSetLayout> createGPUDescriptorSetLayout_impl(const IGPUDescriptorSetLayout::SBinding* _begin, const IGPUDescriptorSetLayout::SBinding* _end) override
    {
        constexpr uint32_t MAX_BINDING_COUNT = 25u;
        constexpr uint32_t MAX_SAMPLER_COUNT_PER_BINDING = 25u;

        uint32_t bindingCount = std::distance(_begin, _end);
        assert(bindingCount <= MAX_BINDING_COUNT);

        uint32_t samplerOffset = 0u;
        VkSampler vk_samplers[MAX_SAMPLER_COUNT_PER_BINDING * MAX_BINDING_COUNT];
        VkDescriptorSetLayoutBinding vk_dsLayoutBindings[MAX_BINDING_COUNT];

        for (uint32_t b = 0u; b < bindingCount; ++b)
        {
            auto binding = _begin + b;

            vk_dsLayoutBindings[b].binding = binding->binding;
            vk_dsLayoutBindings[b].descriptorType = static_cast<VkDescriptorType>(binding->type);
            vk_dsLayoutBindings[b].descriptorCount = binding->count;
            vk_dsLayoutBindings[b].stageFlags = static_cast<VkShaderStageFlags>(binding->stageFlags);
            vk_dsLayoutBindings[b].pImmutableSamplers = nullptr;

            if (binding->samplers)
            {
                assert(binding->count <= MAX_SAMPLER_COUNT_PER_BINDING);

                for (uint32_t i = 0u; i < binding->count; ++i)
                {
                    if (binding->samplers[i]->getAPIType() != EAT_VULKAN)
                        continue;

                    vk_samplers[samplerOffset + i] = static_cast<const CVulkanSampler*>(binding->samplers[i].get())->getInternalObject();
                }

                vk_dsLayoutBindings[b].pImmutableSamplers = vk_samplers + samplerOffset;
                samplerOffset += binding->count;
            }
        }

        VkDescriptorSetLayoutCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        vk_createInfo.pNext = nullptr; // Each pNext member of any structure (including this one) in the pNext chain must be either NULL or a pointer to a valid instance of VkDescriptorSetLayoutBindingFlagsCreateInfo or VkMutableDescriptorTypeCreateInfoVALVE
        vk_createInfo.flags = 0; // Todo(achal): I would need to create a IDescriptorSetLayout::SCreationParams for this
        vk_createInfo.bindingCount = bindingCount;
        vk_createInfo.pBindings = vk_dsLayoutBindings;

        VkDescriptorSetLayout vk_dsLayout;
        if (vkCreateDescriptorSetLayout(m_vkdev, &vk_createInfo, nullptr, &vk_dsLayout) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanDescriptorSetLayout>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this), _begin, _end, vk_dsLayout);
        }
        else
        {
            return nullptr;
        }
    }

    core::smart_refctd_ptr<IGPUPipelineLayout> createGPUPipelineLayout_impl(const asset::SPushConstantRange* const _pcRangesBegin = nullptr,
        const asset::SPushConstantRange* const _pcRangesEnd = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& layout0 = nullptr,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& layout1 = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& layout2 = nullptr,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& layout3 = nullptr) override
    {
        constexpr uint32_t MAX_PC_RANGE_COUNT = 100u;
        constexpr uint32_t MAX_DESCRIPTOR_SET_LAYOUT_COUNT = 4u;

        const core::smart_refctd_ptr<IGPUDescriptorSetLayout> tmp[] = { layout0, layout1, layout2,
            layout3 };

        VkDescriptorSetLayout vk_dsLayouts[MAX_DESCRIPTOR_SET_LAYOUT_COUNT];
        uint32_t dsLayoutCount = 0u;
        for (uint32_t i = 0u; i < MAX_DESCRIPTOR_SET_LAYOUT_COUNT; ++i)
        {
            if (tmp[i] && (tmp[i]->getAPIType() == EAT_VULKAN))
                vk_dsLayouts[dsLayoutCount++] = static_cast<const CVulkanDescriptorSetLayout*>(tmp[i].get())->getInternalObject();
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
        vk_createInfo.setLayoutCount = dsLayoutCount;
        vk_createInfo.pSetLayouts = vk_dsLayouts;
        vk_createInfo.pushConstantRangeCount = pcRangeCount;
        vk_createInfo.pPushConstantRanges = vk_pushConstantRanges;
                
        VkPipelineLayout vk_pipelineLayout;
        if (vkCreatePipelineLayout(m_vkdev, &vk_createInfo, nullptr, &vk_pipelineLayout) == VK_SUCCESS)
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
    // only second argument, like in createGPUComputePipelines_impl below? Especially
    // now, since I've added more members to IGPUComputePipeline::SCreationParams
    core::smart_refctd_ptr<IGPUComputePipeline> createGPUComputePipeline_impl(
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

        if (createGPUComputePipelines_impl(_pipelineCache, creationParamsRange, &result))
        {
            return result;
        }
        else
        {
            return nullptr;
        }
    }

    bool createGPUComputePipelines_impl(IGPUPipelineCache* pipelineCache,
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
            vk_pipelineCache = static_cast<const CVulkanPipelineCache*>(pipelineCache)->getInternalObject();

        VkPipelineShaderStageCreateInfo vk_shaderStageCreateInfos[MAX_PIPELINE_COUNT];

        VkComputePipelineCreateInfo vk_createInfos[MAX_PIPELINE_COUNT];
        for (size_t i = 0ull; i < createInfos.size(); ++i)
        {
            const auto createInfo = createInfos.begin() + i;

            vk_createInfos[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            vk_createInfos[i].pNext = nullptr; // pNext must be either NULL or a pointer to a valid instance of VkPipelineCompilerControlCreateInfoAMD, VkPipelineCreationFeedbackCreateInfoEXT, or VkSubpassShadingPipelineCreateInfoHUAWEI
            vk_createInfos[i].flags = static_cast<VkPipelineCreateFlags>(createInfo->flags);

            if (createInfo->shader->getAPIType() != EAT_VULKAN)
                continue;

            const CVulkanSpecializedShader* specShader
                = static_cast<const CVulkanSpecializedShader*>(createInfo->shader.get());

            vk_shaderStageCreateInfos[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            vk_shaderStageCreateInfos[i].pNext = nullptr; // pNext must be NULL or a pointer to a valid instance of VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT
            vk_shaderStageCreateInfos[i].flags = 0; // currently there is no way to get this in the API https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPipelineShaderStageCreateFlagBits.html
            vk_shaderStageCreateInfos[i].stage = static_cast<VkShaderStageFlagBits>(specShader->getStage());
            vk_shaderStageCreateInfos[i].module = specShader->getInternalObject();
            vk_shaderStageCreateInfos[i].pName = "main"; // Probably want to change the API of IGPUSpecializedShader to have something like getEntryPointName like theres getStage
            vk_shaderStageCreateInfos[i].pSpecializationInfo = nullptr; // Todo(achal): Should we have a asset::ISpecializedShader::SInfo member in CVulkanSpecializedShader, otherwise I don't know how I'm gonna get the values required for VkSpecializationInfo https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkSpecializationInfo.html

            vk_createInfos[i].stage = vk_shaderStageCreateInfos[i];

            vk_createInfos[i].layout = VK_NULL_HANDLE;
            if (createInfo->layout && (createInfo->layout->getAPIType() == EAT_VULKAN))
                vk_createInfos[i].layout = static_cast<const CVulkanPipelineLayout*>(createInfo->layout.get())->getInternalObject();

            vk_createInfos[i].basePipelineHandle = VK_NULL_HANDLE;
            if (createInfo->basePipeline && (createInfo->basePipeline->getAPIType() == EAT_VULKAN))
                vk_createInfos[i].basePipelineHandle = static_cast<const CVulkanComputePipeline*>(createInfo->basePipeline.get())->getInternalObject();

            vk_createInfos[i].basePipelineIndex = createInfo->basePipelineIndex;
        }
        
        VkPipeline vk_pipelines[MAX_PIPELINE_COUNT];
        if (vkCreateComputePipelines(m_vkdev, vk_pipelineCache, static_cast<uint32_t>(createInfos.size()),
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

    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> createGPURenderpassIndependentPipeline_impl(IGPUPipelineCache* _pipelineCache,
        core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout, IGPUSpecializedShader* const* _shaders, IGPUSpecializedShader* const* _shadersEnd,
        const asset::SVertexInputParams& _vertexInputParams, const asset::SBlendParams& _blendParams, const asset::SPrimitiveAssemblyParams& _primAsmParams,
        const asset::SRasterizationParams& _rasterParams) override
    {
        return nullptr;
    }

    bool createGPURenderpassIndependentPipelines_impl(IGPUPipelineCache* pipelineCache, core::SRange<const IGPURenderpassIndependentPipeline::SCreationParams> createInfos,
        core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>* output) override
    {
        return false;
    }

    core::smart_refctd_ptr<IGPUGraphicsPipeline> createGPUGraphicsPipeline_impl(IGPUPipelineCache* pipelineCache, IGPUGraphicsPipeline::SCreationParams&& params) override
    {
        return nullptr;
    }

    bool createGPUGraphicsPipelines_impl(IGPUPipelineCache* pipelineCache, core::SRange<const IGPUGraphicsPipeline::SCreationParams> params, core::smart_refctd_ptr<IGPUGraphicsPipeline>* output) override
    {
        return false;
    }

private:

    inline void getVkMappedMemoryRanges(VkMappedMemoryRange* outRanges, const IDriverMemoryAllocation::MappedMemoryRange* inRangeBegin, const IDriverMemoryAllocation::MappedMemoryRange* inRangeEnd)
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
};

}

#endif