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
#include "nbl/video/CVulkanDeferredOperation.h"
#include "nbl/video/CVulkanAccelerationStructure.h"
#include "nbl/video/surface/CSurfaceVulkan.h"
#include "nbl/core/containers/CMemoryPool.h"

namespace nbl::video
{

class CVulkanLogicalDevice final : public ILogicalDevice
{
public:
    using memory_pool_mt_t = core::CMemoryPool<core::PoolAddressAllocator<uint32_t>, core::default_aligned_allocator, true, uint32_t>;

public:
    CVulkanLogicalDevice(IPhysicalDevice* physicalDevice, VkDevice vkdev,
        const SCreationParams& params, core::smart_refctd_ptr<system::ISystem>&& sys)
        : ILogicalDevice(physicalDevice, params), m_vkdev(vkdev), m_devf(vkdev),
          m_deferred_op_mempool(NODES_PER_BLOCK_DEFERRED_OP * sizeof(CVulkanDeferredOperation), 1u, MAX_BLOCK_COUNT_DEFERRED_OP, static_cast<uint32_t>(sizeof(CVulkanDeferredOperation)))
    {
        // create actual queue objects
        for (uint32_t i = 0u; i < params.queueParamsCount; ++i)
        {
            const auto& qci = params.queueCreateInfos[i];
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

        // Todo(achal): not sure yet, how would I handle multiple platforms without making
        // this function templated
        VkSurfaceKHR vk_surface = static_cast<const CSurfaceVulkanWin32*>(params.surface.get())->getInternalObject();

        VkSwapchainCreateInfoKHR vk_createInfo = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
        vk_createInfo.surface = vk_surface;
        vk_createInfo.minImageCount = params.minImageCount;
        vk_createInfo.imageFormat = getVkFormatFromFormat(params.surfaceFormat.format);
        vk_createInfo.imageColorSpace = getVkColorSpaceKHRFromColorSpace(params.surfaceFormat.colorSpace);
        vk_createInfo.imageExtent = { params.width, params.height };
        vk_createInfo.imageArrayLayers = params.arrayLayers;
        vk_createInfo.imageUsage = static_cast<VkImageUsageFlags>(params.imageUsage);
        vk_createInfo.imageSharingMode = static_cast<VkSharingMode>(params.imageSharingMode);
        vk_createInfo.queueFamilyIndexCount = static_cast<uint32_t>(params.queueFamilyIndices->size());
        vk_createInfo.pQueueFamilyIndices = params.queueFamilyIndices->data();
        vk_createInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR; // Todo(achal)     
        vk_createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // Todo(achal)
        vk_createInfo.presentMode = static_cast<VkPresentModeKHR>(params.presentMode);
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
        assert(!"Not implemented!\n");
        return IGPUFence::E_STATUS::ES_ERROR;
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

            vk_fences[i] = reinterpret_cast<CVulkanFence*>(_fences[i])->getInternalObject();
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

            vk_fences[i] = reinterpret_cast<CVulkanFence*>(_fences[i])->getInternalObject();
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
            
    core::smart_refctd_ptr<IDeferredOperation> createDeferredOperation() override
    {
        VkDeferredOperationKHR vk_deferredOp = VK_NULL_HANDLE;
        VkResult vk_res = vkCreateDeferredOperationKHR(m_vkdev, nullptr, &vk_deferredOp);
        if(vk_res!=VK_SUCCESS)
            return nullptr;

        void* memory = m_deferred_op_mempool.allocate(sizeof(CVulkanDeferredOperation),alignof(CVulkanDeferredOperation));
        if (!memory)
            return nullptr;

        new (memory) CVulkanDeferredOperation(core::smart_refctd_ptr<CVulkanLogicalDevice>(this),vk_deferredOp);
        return core::smart_refctd_ptr<CVulkanDeferredOperation>(reinterpret_cast<CVulkanDeferredOperation*>(memory),core::dont_grab);
    }

    core::smart_refctd_ptr<IGPUCommandPool> createCommandPool(uint32_t familyIndex, std::underlying_type_t<IGPUCommandPool::E_CREATE_FLAGS> flags) override
    {
        VkCommandPoolCreateInfo vk_createInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        vk_createInfo.pNext = nullptr; // pNext must be NULL
        vk_createInfo.flags = static_cast<VkCommandPoolCreateFlags>(flags);
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
        // Todo(achal): Hoist creation out of constructor
        return nullptr; // return core::make_smart_refctd_ptr<CVulkanRenderpass>(this, params);
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
        vk_createInfo.pQueueFamilyIndices = creationParams.queuueFamilyIndices;

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

    core::smart_refctd_ptr<IGPUBuffer> createGPUBufferOnDedMem(const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs, const bool canModifySubData = false) override
    {
#if 0
        core::smart_refctd_ptr<IGPUBuffer> gpuBuffer = createGPUBuffer(initialMreqs, false);

        if (!gpuBuffer)
            return nullptr;

        // Todo(achal): Probably do not call getMemoryReqs at all but
        // vkGetBufferMemoryRequirements or equivalent
        core::smart_refctd_ptr<video::IDriverMemoryAllocation> bufferMemory =
            allocateDeviceLocalMemory(gpuBuffer->getMemoryReqs());

        if (!bufferMemory)
            return nullptr;

        ILogicalDevice::SBindBufferMemoryInfo bindBufferInfo = {};
        bindBufferInfo.buffer = gpuBuffer.get();
        bindBufferInfo.memory = bufferMemory.get();
        bindBufferInfo.offset = 0ull;
        bindBufferMemory(1u, &bindBufferInfo);

        return gpuBuffer;
#endif
        return nullptr;
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
        vk_createInfo.usage = static_cast<VkImageUsageFlags>(params.usage);
        vk_createInfo.sharingMode = static_cast<VkSharingMode>(params.sharingMode);
        vk_createInfo.queueFamilyIndexCount = params.queueFamilyIndices->size();
        vk_createInfo.pQueueFamilyIndices = params.queueFamilyIndices->data();
        vk_createInfo.initialLayout = static_cast<VkImageLayout>(params.initialLayout);

        VkImage vk_image;
        if (vkCreateImage(m_vkdev, &vk_createInfo, nullptr, &vk_image) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanImage>(core::smart_refctd_ptr<CVulkanLogicalDevice>(this),
                std::move(params), vk_image);
        }
        else
        {
            return nullptr;
        }
    }

    bool bindImageMemory(uint32_t bindInfoCount, const SBindImageMemoryInfo* pBindInfos) override
    {
        for (uint32_t i = 0u; i < bindInfoCount; ++i)
        {
            if (pBindInfos[i].image->getAPIType() != EAT_VULKAN)
                continue;
            VkImage vk_image = static_cast<const CVulkanImage*>(pBindInfos[i].image)->getInternalObject();

            // if (pBindInfos[i].memory->getAPIType() != EAT_VULKAN)
            //     continue;
            VkDeviceMemory vk_deviceMemory = static_cast<const CVulkanMemoryAllocation*>(pBindInfos[i].memory)->getInternalObject();

            if (vkBindImageMemory(m_vkdev, vk_image, vk_deviceMemory,
                static_cast<VkDeviceSize>(pBindInfos[i].offset)) != VK_SUCCESS)
            {
                return false;
            }
        }

        return true;
    }
            
    core::smart_refctd_ptr<IGPUImage> createGPUImageOnDedMem(IGPUImage::SCreationParams&& params, const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs) override
    {
        // Todo(achal)
#if 0
        core::smart_refctd_ptr<IGPUImage> gpuImage = createGPUImage(core::smart_refctd_ptr(params));
        if (!gpuImage)
            return nullptr;
        
        core::smart_refctd_ptr<video::IDriverMemoryAllocation> imageMemory =
            allocateDeviceLocalMemory(gpuImage->getMemoryReqs());
#endif

        return nullptr;
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

        VkWriteDescriptorSetAccelerationStructureKHR vk_writeDescriptorSetAS[MAX_DESCRIPTOR_WRITE_COUNT];
        
        uint32_t accelerationStructuresOffset = 0u;
        VkAccelerationStructureKHR vk_accelerationStructures[MAX_DESCRIPTOR_WRITE_COUNT * MAX_DESCRIPTOR_ARRAY_COUNT];

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

                        vk_bufferInfos[j].buffer = vk_buffer;
                        vk_bufferInfos[j].offset = pDescriptorWrites[i].info[j].buffer.offset;
                        vk_bufferInfos[j].range = pDescriptorWrites[i].info[j].buffer.size;
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

                        vk_imageInfos[j].sampler = vk_sampler;
                        vk_imageInfos[j].imageView = vk_imageView;
                        vk_imageInfos[j].imageLayout = static_cast<VkImageLayout>(descriptorWriteImageInfo.imageLayout);
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
                        vk_bufferViews[j] = vk_bufferView;
                    }

                    vk_writeDescriptorSets[i].pTexelBufferView = vk_bufferViews + bufferViewOffset;
                    bufferViewOffset += pDescriptorWrites[i].count;
                } break;
                
                case asset::IDescriptor::EC_ACCELERATION_STRUCTURE:
                {
                    // Get WriteAS
                    auto & writeAS = vk_writeDescriptorSetAS[i];
                    
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
            
    core::smart_refctd_ptr<IQueryPool> createQueryPool(IQueryPool::SCreationParams&& params) override;
    
    bool getQueryPoolResults(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, size_t dataSize, void * pData, uint64_t stride, IQueryPool::E_QUERY_RESULTS_FLAGS flags) override;

    bool buildAccelerationStructures(
        core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation,
        const core::SRange<IGPUAccelerationStructure::HostBuildGeometryInfo>& pInfos,
        IGPUAccelerationStructure::BuildRangeInfo* const* ppBuildRangeInfos) override;

    bool copyAccelerationStructure(core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation, const IGPUAccelerationStructure::CopyInfo& copyInfo) override;
    
    bool copyAccelerationStructureToMemory(core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation, const IGPUAccelerationStructure::HostCopyToMemoryInfo& copyInfo) override;

    bool copyAccelerationStructureFromMemory(core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation, const IGPUAccelerationStructure::HostCopyFromMemoryInfo& copyInfo) override;

    IGPUAccelerationStructure::BuildSizes getAccelerationStructureBuildSizes(const IGPUAccelerationStructure::HostBuildGeometryInfo& pPartialInfos, const uint32_t* pMaxPrimitiveCounts) override;

    IGPUAccelerationStructure::BuildSizes getAccelerationStructureBuildSizes(const IGPUAccelerationStructure::DeviceBuildGeometryInfo& pPartialInfos, const uint32_t* pMaxPrimitiveCounts) override;

    CVulkanDeviceFunctionTable* getFunctionTable() { return &m_devf; }

    VkDevice getInternalObject() const { return m_vkdev; }
        
    inline memory_pool_mt_t & getMemoryPoolForDeferredOperations() {
        return m_deferred_op_mempool;
    }

protected:
    bool createCommandBuffers_impl(IGPUCommandPool* cmdPool, IGPUCommandBuffer::E_LEVEL level,
        uint32_t count, core::smart_refctd_ptr<IGPUCommandBuffer>* outCmdBufs) override
    {
        constexpr uint32_t MAX_COMMAND_BUFFER_COUNT = 1000u;

        if (cmdPool->getAPIType() != EAT_VULKAN)
            return false;

        auto vulkanCommandPool = reinterpret_cast<CVulkanCommandPool*>(cmdPool)->getInternalObject();

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
        // Todo(achal): Hoist creation out of constructor
        return nullptr; // return core::make_smart_refctd_ptr<CVulkanFramebuffer>(this, std::move(params));
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
    
    core::smart_refctd_ptr<IGPUAccelerationStructure> createGPUAccelerationStructure_impl(IGPUAccelerationStructure::SCreationParams&& params) override;

    core::smart_refctd_ptr<IGPUPipelineLayout> createGPUPipelineLayout_impl(const asset::SPushConstantRange* const _pcRangesBegin = nullptr,
        const asset::SPushConstantRange* const _pcRangesEnd = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& layout0 = nullptr,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& layout1 = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& layout2 = nullptr,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& layout3 = nullptr) override
    {
        constexpr uint32_t MAX_PC_RANGE_COUNT = 100u;
        constexpr uint32_t MAX_DESCRIPTOR_SET_LAYOUT_COUNT = 4u; // temporary max, I believe

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
        core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout, IGPUSpecializedShader** _shaders, IGPUSpecializedShader** _shadersEnd,
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
    
    template<typename AddressType>
    IGPUAccelerationStructure::BuildSizes getAccelerationStructureBuildSizes_impl(VkAccelerationStructureBuildTypeKHR buildType, const IGPUAccelerationStructure::BuildGeometryInfo<AddressType>& pPartialInfos, const uint32_t* pMaxPrimitiveCounts) 
    {
        VkAccelerationStructureBuildSizesInfoKHR vk_ret = {};

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
            uint32_t geomCount = pPartialInfos.geometries.size();

            assert(geomCount > 0);
            assert(geomCount <= MaxGeometryPerBuildInfoCount);

            vk_buildGeomsInfo = CVulkanAccelerationStructure::getVkASBuildGeomInfoFromBuildGeomInfo(m_vkdev, pPartialInfos, &vk_geometries[geometryArrayOffset]);
            geometryArrayOffset += geomCount;
        }

        vkGetAccelerationStructureBuildSizesKHR(m_vkdev, buildType, &vk_buildGeomsInfo, pMaxPrimitiveCounts, &vk_ret);

        IGPUAccelerationStructure::BuildSizes ret = {};
        ret.accelerationStructureSize = vk_ret.accelerationStructureSize;
        ret.updateScratchSize = vk_ret.updateScratchSize;
        ret.buildScratchSize = vk_ret.buildScratchSize;
    }

private:
    core::smart_refctd_ptr<IDriverMemoryAllocation> allocateGPUMemory(const IDriverMemoryBacked::SDriverMemoryRequirements& reqs);

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
    
    constexpr static inline uint32_t NODES_PER_BLOCK_DEFERRED_OP = 4096u;
    constexpr static inline uint32_t MAX_BLOCK_COUNT_DEFERRED_OP = 256u;
    memory_pool_mt_t m_deferred_op_mempool;
};

}

#endif