#ifndef __NBL_C_VK_LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_C_VK_LOGICAL_DEVICE_H_INCLUDED__

#include <algorithm>

#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/CVulkanDeviceFunctionTable.h"
#include "nbl/video/CVKSwapchain.h"
#include "nbl/video/CVulkanQueue.h"
#include "nbl/video/CVulkanRenderpass.h"
#include "nbl/video/CVulkanImageView.h"
#include "nbl/video/CVulkanFramebuffer.h"
#include "nbl/video/CVulkanSemaphore.h"
// #include "nbl/video/surface/ISurfaceVK.h"

namespace nbl::video
{

class CVKLogicalDevice final : public ILogicalDevice
{
public:
    CVKLogicalDevice(VkDevice vkdev, const SCreationParams& params, core::smart_refctd_ptr<system::ISystem>&& sys, core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc) :
    ILogicalDevice(EAT_VULKAN, params, std::move(sys), std::move(glslc)),
    m_vkdev(vkdev),
    m_devf(vkdev)
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
                m_devf.vk.vkGetDeviceQueue(m_vkdev, famIx, j, &q);
                        
                const uint32_t ix = offset + j;
                (*m_queues)[ix] = core::make_smart_refctd_ptr<CThreadSafeGPUQueueAdapter>(core::make_smart_refctd_ptr<CVulkanQueue>(this, q, famIx, flags, priority), this);
            }
        }
    }
            
    ~CVKLogicalDevice()
    {
        m_devf.vk.vkDestroyDevice(m_vkdev, nullptr);
    }
            
    core::smart_refctd_ptr<ISwapchain> createSwapchain(ISwapchain::SCreationParams&& params) override
    {
        return core::make_smart_refctd_ptr<CVKSwapchain>(std::move(params), this);
    }
    
    core::smart_refctd_ptr<IGPUSemaphore> createSemaphore() override
    {
        VkSemaphoreCreateInfo createInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        // createInfo.pNext = nullptr;

        VkSemaphore semaphore;
        if (vkCreateSemaphore(m_vkdev, &createInfo, nullptr, &semaphore) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanSemaphore>(this, semaphore);
        }
        
        // Probably log an error/warning here?
        return nullptr;
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
            
    core::smart_refctd_ptr<IGPUFence> createFence(IGPUFence::E_CREATE_FLAGS _flags) override
    {
        return nullptr;
    }
            
    IGPUFence::E_STATUS getFenceStatus(IGPUFence* _fence) override
    {
        return IGPUFence::E_STATUS::ES_ERROR;
    }
            
    void resetFences(uint32_t _count, IGPUFence** _fences) override
    {
        return;
    }
            
    IGPUFence::E_STATUS waitForFences(uint32_t _count, IGPUFence** _fences, bool _waitAll, uint64_t _timeout) override
    {
        return IGPUFence::E_STATUS::ES_ERROR;
    }
            
    const core::smart_refctd_dynamic_array<std::string> getSupportedGLSLExtensions() const override
    {
        return nullptr;
    }
            
    core::smart_refctd_ptr<IGPUCommandPool> createCommandPool(uint32_t _familyIx, IGPUCommandPool::E_CREATE_FLAGS flags) override
    {
        return nullptr; // return core::smart_refctd_ptr<CVulkanCommandPool>();
    }
            
    core::smart_refctd_ptr<IDescriptorPool> createDescriptorPool(IDescriptorPool::E_CREATE_FLAGS flags, uint32_t maxSets, uint32_t poolSizeCount, const IDescriptorPool::SDescriptorPoolSize* poolSizes) override
    {
        return nullptr;
    }
            
    core::smart_refctd_ptr<IGPURenderpass> createGPURenderpass(const IGPURenderpass::SCreationParams& params) override
    {
        return core::make_smart_refctd_ptr<CVulkanRenderpass>(this, params);
    }
            
    void flushMappedMemoryRanges(core::SRange<const video::IDriverMemoryAllocation::MappedMemoryRange> ranges) override
    {
        return;
    }
            
    void invalidateMappedMemoryRanges(core::SRange<const video::IDriverMemoryAllocation::MappedMemoryRange> ranges) override
    {
        return;
    }
            
    core::smart_refctd_ptr<IGPUShader> createGPUShader(core::smart_refctd_ptr<asset::ICPUShader>&& cpushader) override
    {
        return nullptr;
    }

    core::smart_refctd_ptr<IGPUImage> createGPUImage(IGPUImage::SCreationParams&& params)
    {
#if 0
        VkImageCreateInfo createInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        createInfo.flags = static_cast<VkImageCreateFlags>(params.flags);
        createInfo.imageType = createInfo.imageType = static_cast<VkImageType>(params.type);
        createInfo.format = ISurfaceVK::getVkFormat(params.format);
        createInfo.extent = { params.extent.width, params.extent.height, params.extent.depth };
        createInfo.mipLevels = params.mipLevels;
        createInfo.arrayLayers = params.arrayLayers;
        createInfo.samples = static_cast<VkSampleCountFlagBits>(params.samples);
        createInfo.tiling = static_cast<VkImageTiling>(params.tiling);
        createInfo.usage = static_cast<VkImageUsageFlags>(params.usage);
        createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // Todo(achal): enumize this
        createInfo.queueFamilyIndexCount = params.queueFamilyIndices->size();
        createInfo.pQueueFamilyIndices = params.queueFamilyIndices->data();
        createInfo.initialLayout = static_cast<VkImageLayout>(params.initialLayout);

        VkImage vk_image;
        assert(vkCreateImage(m_vkdev, &createInfo, nullptr, &vk_image) == VK_SUCCESS); // Todo(achal): error handling

        return core::make_smart_refctd_ptr<CVulkanImage>(this, std::move(params));
#endif
        return nullptr;
    }
            
    core::smart_refctd_ptr<IGPUImage> createGPUImageOnDedMem(IGPUImage::SCreationParams&& params, const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs) override
    {
#if 0
        VkImageCreateInfo createInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        createInfo.flags = static_cast<VkImageCreateFlags>(params.flags);
        createInfo.imageType = createInfo.imageType = static_cast<VkImageType>(params.type);
        createInfo.format = ISurfaceVK::getVkFormat(params.format);
        createInfo.extent = { params.extent.width, params.extent.height, params.extent.depth };
        createInfo.mipLevels = params.mipLevels;
        createInfo.arrayLayers = params.arrayLayers;
        createInfo.samples = static_cast<VkSampleCountFlagBits>(params.samples);
        createInfo.tiling = static_cast<VkImageTiling>(params.tiling);
        createInfo.usage = static_cast<VkImageUsageFlags>(params.usage);
        createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // Todo(achal): enumize this
        createInfo.queueFamilyIndexCount = params.queueFamilyIndices->size();
        createInfo.pQueueFamilyIndices = params.queueFamilyIndices->data();
        createInfo.initialLayout = static_cast<VkImageLayout>(params.initialLayout);

        VkImage vk_image;
        assert(vkCreateImage(m_vkdev, &createInfo, nullptr, &vk_image) == VK_SUCCESS); // Todo(achal): error handling

        return core::make_smart_refctd_ptr<CVulkanImage>(this, std::move(params));
#endif
        return nullptr;
    }

    void updateDescriptorSets(uint32_t descriptorWriteCount, const IGPUDescriptorSet::SWriteDescriptorSet* pDescriptorWrites,
        uint32_t descriptorCopyCount, const IGPUDescriptorSet::SCopyDescriptorSet* pDescriptorCopies) override
    {
        return;
    }

    core::smart_refctd_ptr<IGPUSampler> createGPUSampler(const IGPUSampler::SParams& _params) override
    {
        return nullptr;
    }

    void waitIdle() override
    {

    }

    void* mapMemory(const IDriverMemoryAllocation::MappedMemoryRange& memory, IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAG accessHint = IDriverMemoryAllocation::EMCAF_READ_AND_WRITE) override
    {
        return nullptr;
    }

    void unmapMemory(IDriverMemoryAllocation* memory) override
    {

    }

    CVulkanDeviceFunctionTable* getFunctionTable() { return &m_devf; }

    VkDevice getInternalObject() const { return m_vkdev; }

protected:
    bool createCommandBuffers_impl(IGPUCommandPool* _cmdPool, IGPUCommandBuffer::E_LEVEL _level, uint32_t _count, core::smart_refctd_ptr<IGPUCommandBuffer>* _outCmdBufs) override
    {
        return false;
    }

    bool freeCommandBuffers_impl(IGPUCommandBuffer** _cmdbufs, uint32_t _count) override
    {
        return false;
    }

    core::smart_refctd_ptr<IGPUFramebuffer> createGPUFramebuffer_impl(IGPUFramebuffer::SCreationParams&& params) override
    {
        return core::make_smart_refctd_ptr<CVulkanFramebuffer>(this, std::move(params));
    }

    core::smart_refctd_ptr<IGPUSpecializedShader> createGPUSpecializedShader_impl(const IGPUShader* _unspecialized, const asset::ISpecializedShader::SInfo& _specInfo, const asset::ISPIRVOptimizer* _spvopt) override
    {
        return nullptr;
    }

    core::smart_refctd_ptr<IGPUBufferView> createGPUBufferView_impl(IGPUBuffer* _underlying, asset::E_FORMAT _fmt, size_t _offset = 0ull, size_t _size = IGPUBufferView::whole_buffer) override
    {
        return nullptr;
    }

    core::smart_refctd_ptr<IGPUImageView> createGPUImageView_impl(IGPUImageView::SCreationParams&& params) override
    {
        return core::make_smart_refctd_ptr<CVulkanImageView>(this, std::move(params));
    }

    core::smart_refctd_ptr<IGPUDescriptorSet> createGPUDescriptorSet_impl(IDescriptorPool* pool, core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout) override
    {
        return nullptr;
    }

    core::smart_refctd_ptr<IGPUDescriptorSetLayout> createGPUDescriptorSetLayout_impl(const IGPUDescriptorSetLayout::SBinding* _begin, const IGPUDescriptorSetLayout::SBinding* _end) override
    {
        return nullptr;
    }

    core::smart_refctd_ptr<IGPUPipelineLayout> createGPUPipelineLayout_impl(const asset::SPushConstantRange* const _pcRangesBegin = nullptr,
        const asset::SPushConstantRange* const _pcRangesEnd = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0 = nullptr,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1 = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2 = nullptr,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3 = nullptr) override
    {
        return nullptr;
    }

    core::smart_refctd_ptr<IGPUComputePipeline> createGPUComputePipeline_impl(IGPUPipelineCache* _pipelineCache,
        core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout, core::smart_refctd_ptr<IGPUSpecializedShader>&& _shader) override
    {
        return nullptr;
    }

    bool createGPUComputePipelines_impl(IGPUPipelineCache* pipelineCache, core::SRange<const IGPUComputePipeline::SCreationParams> createInfos,
        core::smart_refctd_ptr<IGPUComputePipeline>* output) override
    {
        return false;
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
            
private:
    VkDevice m_vkdev;
    CVulkanDeviceFunctionTable m_devf; // Todo(achal): I don't have a function table yet
};

}

#endif