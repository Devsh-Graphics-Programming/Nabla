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
#include "nbl/video/CVulkanFence.h"
#include "nbl/video/CVulkanShader.h"
#include "nbl/video/CVulkanSpecializedShader.h"
#include "nbl/video/CVulkanCommandPool.h"
#include "nbl/video/CVulkanCommandBuffer.h"
// #include "nbl/video/surface/ISurfaceVK.h"

namespace nbl::video
{

// Todo(achal): There are methods in this class which aren't pure virtual in ILogicalDevice,
// need to implement those as well
class CVKLogicalDevice final : public ILogicalDevice
{
public:
    CVKLogicalDevice(IPhysicalDevice* physicalDevice, VkDevice vkdev, const SCreationParams& params,
        core::smart_refctd_ptr<system::ISystem>&& sys,
        core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc,
        system::logger_opt_smart_ptr&& logger)
        : ILogicalDevice(physicalDevice, params, std::move(sys), std::move(glslc)),
    m_vkdev(vkdev), m_devf(vkdev), m_logger(std::move(logger))
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
                (*m_queues)[ix] = core::make_smart_refctd_ptr<CThreadSafeGPUQueueAdapter>(
                    core::make_smart_refctd_ptr<CVulkanQueue>(this, q, famIx, flags, priority),
                    this);
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
        // createInfo.pNext = extensions are not supported yet;

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
        VkFenceCreateInfo createInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        // createInfo.pNext = API doesnt support extensions yet;
        createInfo.flags = static_cast<VkFenceCreateFlags>(_flags);

        VkFence fence;
        if (vkCreateFence(m_vkdev, &createInfo, nullptr, &fence) == VK_SUCCESS)
            return core::make_smart_refctd_ptr<CVulkanFence>(this, _flags, fence);

        // Probably log error/warning?
        return nullptr;
    }
            
    IGPUFence::E_STATUS getFenceStatus(IGPUFence* _fence) override
    {
        return IGPUFence::E_STATUS::ES_ERROR;
    }
            
    // API needs to change. vkResetFences can fail.
    void resetFences(uint32_t _count, IGPUFence*const* _fences) override
    {
        assert(_count < 100);

        VkFence vk_fences[100];
        for (uint32_t i = 0u; i < _count; ++i)
        {
            if (_fences[i]->getAPIType() != EAT_VULKAN)
            {
                // Probably log warning?
                assert(false);
            }

            vk_fences[i] = reinterpret_cast<CVulkanFence*>(_fences[i])->getInternalObject();
        }

        vkResetFences(m_vkdev, _count, vk_fences);
    }
            
    IGPUFence::E_STATUS waitForFences(uint32_t _count, IGPUFence*const* _fences, bool _waitAll, uint64_t _timeout) override
    {
        assert(_count < 100);

        VkFence vk_fences[100];
        for (uint32_t i = 0u; i < _count; ++i)
        {
            if (_fences[i]->getAPIType() != EAT_VULKAN)
            {
                // Probably log warning?
                return IGPUFence::E_STATUS::ES_ERROR;
            }

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
            
    core::smart_refctd_ptr<IGPUCommandPool> createCommandPool(uint32_t _familyIx, std::underlying_type_t<IGPUCommandPool::E_CREATE_FLAGS> flags) override
    {
        VkCommandPool vk_commandPool = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo createInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        createInfo.pNext = nullptr; // Todo(achal)
        createInfo.flags = static_cast<VkCommandPoolCreateFlags>(flags);
        createInfo.queueFamilyIndex = _familyIx;

        if (vkCreateCommandPool(m_vkdev, &createInfo, nullptr, &vk_commandPool) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<CVulkanCommandPool>(this, flags, _familyIx,
                vk_commandPool);
        }
        else
        {
            // Probably log a warning
            return nullptr;
        }
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
        const asset::ICPUBuffer* source = cpushader->getSPVorGLSL();
        core::smart_refctd_ptr<asset::ICPUBuffer> clone =
            core::smart_refctd_ptr_static_cast<asset::ICPUBuffer>(source->clone(1u));
        if (cpushader->containsGLSL())
            return core::make_smart_refctd_ptr<CVulkanShader>(this, std::move(clone), IGPUShader::buffer_contains_glsl);
        else
            return core::make_smart_refctd_ptr<CVulkanShader>(this, std::move(clone));
        
    }

    // Todo(achal): There's already a createGPUImage method in ILogicalDevice
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

    // API changes needed, this could also fail.
    void waitIdle() override
    {
        // Todo(achal): Handle errors
        assert(vkDeviceWaitIdle(m_vkdev) == VK_SUCCESS);
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
        if (_cmdPool->getAPIType() != EAT_VULKAN)
            return false;

        auto vk_commandPool = reinterpret_cast<CVulkanCommandPool*>(_cmdPool)->getInternalObject();

        assert(_count <= 100);
        VkCommandBuffer vk_commandBuffers[100];

        VkCommandBufferAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        // allocateInfo.pNext = nullptr; (this must be NULL)
        allocateInfo.commandPool = vk_commandPool;
        allocateInfo.level = static_cast<VkCommandBufferLevel>(_level);
        allocateInfo.commandBufferCount = _count;

        if (vkAllocateCommandBuffers(m_vkdev, &allocateInfo, vk_commandBuffers) == VK_SUCCESS)
        {
            for (uint32_t i = 0u; i < _count; ++i)
                _outCmdBufs[i] = core::make_smart_refctd_ptr<CVulkanCommandBuffer>(this,
                    _level, vk_commandBuffers[i], _cmdPool);

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
        return core::make_smart_refctd_ptr<CVulkanFramebuffer>(this, std::move(params));
    }

    // Todo(achal): For some reason this is not printing shader compilation errors to console
    core::smart_refctd_ptr<IGPUSpecializedShader> createGPUSpecializedShader_impl(const IGPUShader* _unspecialized, const asset::ISpecializedShader::SInfo& _specInfo, const asset::ISPIRVOptimizer* _spvopt) override
    {
        if (_unspecialized->getAPIType() != EAT_VULKAN)
        {
            // Log a warning
            return nullptr;
        }
        const CVulkanShader* unspecializedShader = static_cast<const CVulkanShader*>(_unspecialized);

        const std::string& entryPoint = _specInfo.entryPoint;
        const asset::ISpecializedShader::E_SHADER_STAGE shaderStage = _specInfo.shaderStage;

        core::smart_refctd_ptr<asset::ICPUBuffer> spirv = nullptr;
        if (unspecializedShader->containsGLSL())
        {
            const char* begin = reinterpret_cast<const char*>(unspecializedShader->getSPVorGLSL()->getPointer());
            const char* end = begin + unspecializedShader->getSPVorGLSL()->getSize();
            std::string glsl(begin, end);
            core::smart_refctd_ptr<asset::ICPUShader> glslShader_woIncludes =
                m_GLSLCompiler->resolveIncludeDirectives(glsl.c_str(), shaderStage,
                    _specInfo.m_filePathHint.string().c_str());

            spirv = m_GLSLCompiler->compileSPIRVFromGLSL(
                reinterpret_cast<const char*>(glslShader_woIncludes->getSPVorGLSL()->getPointer()),
                shaderStage, entryPoint.c_str(), _specInfo.m_filePathHint.string().c_str());
        }
        else
        {
            spirv = unspecializedShader->getSPVorGLSL_refctd();
        }

        // Should just do this check in ISPIRVOptimizer::optimize
        if (!spirv)
            return nullptr;

        if (_spvopt)
            spirv = _spvopt->optimize(spirv.get(), m_logger.getOptRawPtr());

        if (!spirv)
            return nullptr;

        VkShaderModuleCreateInfo createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        createInfo.pNext = nullptr;
        // createInfo.flags = 0; (reserved for future use by Vulkan)
        createInfo.codeSize = spirv->getSize();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(spirv->getPointer());
        
        VkShaderModule vk_shaderModule;
        if (vkCreateShaderModule(m_vkdev, &createInfo, nullptr, &vk_shaderModule) == VK_SUCCESS)
        {
            return core::make_smart_refctd_ptr<video::CVulkanSpecializedShader>(this, vk_shaderModule, shaderStage);
        }
        else
        {
            // Probably log a warning
            return nullptr;
        }
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
    system::logger_opt_smart_ptr m_logger;
    VkDevice m_vkdev;
    CVulkanDeviceFunctionTable m_devf; // Todo(achal): I don't have a function table yet
};

}

#endif