#include "CVulkanLogicalDevice.h"

#include "nbl/video/CVulkanPhysicalDevice.h"
#include "nbl/video/CVulkanCommandBuffer.h"

namespace nbl::video
{

core::smart_refctd_ptr<IDriverMemoryAllocation> CVulkanLogicalDevice::allocateDeviceLocalMemory(
    const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs)
{
    IDriverMemoryBacked::SDriverMemoryRequirements memoryReqs = getDeviceLocalGPUMemoryReqs();
    memoryReqs.vulkanReqs.alignment = core::max(memoryReqs.vulkanReqs.alignment, additionalReqs.vulkanReqs.alignment);
    memoryReqs.vulkanReqs.size = core::max(memoryReqs.vulkanReqs.size, additionalReqs.vulkanReqs.size);
    memoryReqs.vulkanReqs.memoryTypeBits &= additionalReqs.vulkanReqs.memoryTypeBits;
    memoryReqs.mappingCapability = additionalReqs.mappingCapability;
    memoryReqs.memoryHeapLocation = IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
    memoryReqs.prefersDedicatedAllocation = additionalReqs.prefersDedicatedAllocation;
    memoryReqs.requiresDedicatedAllocation = additionalReqs.requiresDedicatedAllocation;

    return allocateGPUMemory(additionalReqs);
}

core::smart_refctd_ptr<IDriverMemoryAllocation> CVulkanLogicalDevice::allocateSpilloverMemory(
    const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs)
{
    if (additionalReqs.memoryHeapLocation == IDriverMemoryAllocation::ESMT_DEVICE_LOCAL)
        return nullptr;

    IDriverMemoryBacked::SDriverMemoryRequirements memoryReqs = getSpilloverGPUMemoryReqs();
    memoryReqs.vulkanReqs.alignment = core::max(memoryReqs.vulkanReqs.alignment, additionalReqs.vulkanReqs.alignment);
    memoryReqs.vulkanReqs.size = core::max(memoryReqs.vulkanReqs.size, additionalReqs.vulkanReqs.size);
    memoryReqs.vulkanReqs.memoryTypeBits &= additionalReqs.vulkanReqs.memoryTypeBits;
    memoryReqs.mappingCapability = additionalReqs.mappingCapability;
    memoryReqs.memoryHeapLocation = additionalReqs.memoryHeapLocation;
    memoryReqs.prefersDedicatedAllocation = additionalReqs.prefersDedicatedAllocation;
    memoryReqs.requiresDedicatedAllocation = additionalReqs.requiresDedicatedAllocation;

    return allocateGPUMemory(memoryReqs);
}

core::smart_refctd_ptr<IDriverMemoryAllocation> CVulkanLogicalDevice::allocateUpStreamingMemory(
    const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs)
{
    if (getUpStreamingMemoryReqs().memoryHeapLocation != additionalReqs.memoryHeapLocation)
        return nullptr;

    IDriverMemoryBacked::SDriverMemoryRequirements memoryReqs = getUpStreamingMemoryReqs();
    memoryReqs.vulkanReqs.alignment = core::max(memoryReqs.vulkanReqs.alignment, additionalReqs.vulkanReqs.alignment);
    memoryReqs.vulkanReqs.size = core::max(memoryReqs.vulkanReqs.size, additionalReqs.vulkanReqs.size);
    memoryReqs.vulkanReqs.memoryTypeBits &= additionalReqs.vulkanReqs.memoryTypeBits;
    memoryReqs.mappingCapability = additionalReqs.mappingCapability;
    memoryReqs.memoryHeapLocation = additionalReqs.memoryHeapLocation;
    memoryReqs.prefersDedicatedAllocation = additionalReqs.prefersDedicatedAllocation;
    memoryReqs.requiresDedicatedAllocation = additionalReqs.requiresDedicatedAllocation;

    return allocateGPUMemory(memoryReqs);
}

core::smart_refctd_ptr<IDriverMemoryAllocation> CVulkanLogicalDevice::allocateDownStreamingMemory(
    const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs)
{
    if (getDownStreamingMemoryReqs().memoryHeapLocation != additionalReqs.memoryHeapLocation)
        return nullptr;

    IDriverMemoryBacked::SDriverMemoryRequirements memoryReqs = getDownStreamingMemoryReqs();
    memoryReqs.vulkanReqs.alignment = core::max(memoryReqs.vulkanReqs.alignment, additionalReqs.vulkanReqs.alignment);
    memoryReqs.vulkanReqs.size = core::max(memoryReqs.vulkanReqs.size, additionalReqs.vulkanReqs.size);
    memoryReqs.vulkanReqs.memoryTypeBits &= additionalReqs.vulkanReqs.memoryTypeBits;
    memoryReqs.mappingCapability = additionalReqs.mappingCapability;
    memoryReqs.memoryHeapLocation = additionalReqs.memoryHeapLocation;
    memoryReqs.prefersDedicatedAllocation = additionalReqs.prefersDedicatedAllocation;
    memoryReqs.requiresDedicatedAllocation = additionalReqs.requiresDedicatedAllocation;

    return allocateGPUMemory(memoryReqs);
}

core::smart_refctd_ptr<IDriverMemoryAllocation> CVulkanLogicalDevice::allocateCPUSideGPUVisibleMemory(
    const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs)
{
    if (additionalReqs.memoryHeapLocation != IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL)
        return nullptr;

    IDriverMemoryBacked::SDriverMemoryRequirements memoryReqs = getCPUSideGPUVisibleGPUMemoryReqs();
    memoryReqs.vulkanReqs.alignment = core::max(memoryReqs.vulkanReqs.alignment, additionalReqs.vulkanReqs.alignment);
    memoryReqs.vulkanReqs.size = core::max(memoryReqs.vulkanReqs.size, additionalReqs.vulkanReqs.size);
    memoryReqs.vulkanReqs.memoryTypeBits &= additionalReqs.vulkanReqs.memoryTypeBits;
    memoryReqs.mappingCapability = additionalReqs.mappingCapability;
    memoryReqs.memoryHeapLocation = additionalReqs.memoryHeapLocation;
    memoryReqs.prefersDedicatedAllocation = additionalReqs.prefersDedicatedAllocation;
    memoryReqs.requiresDedicatedAllocation = additionalReqs.requiresDedicatedAllocation;

    return allocateGPUMemory(memoryReqs);
}

core::smart_refctd_ptr<IDriverMemoryAllocation> CVulkanLogicalDevice::allocateGPUMemory(
    const IDriverMemoryBacked::SDriverMemoryRequirements& reqs)
{
    VkMemoryPropertyFlags desiredMemoryProperties = static_cast<VkMemoryPropertyFlags>(0u);

    if (reqs.memoryHeapLocation == IDriverMemoryAllocation::ESMT_DEVICE_LOCAL)
        desiredMemoryProperties |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    if ((reqs.mappingCapability & IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ) ||
        (reqs.mappingCapability & IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_WRITE))
        desiredMemoryProperties |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

    if (reqs.mappingCapability & IDriverMemoryAllocation::EMCF_COHERENT)
        desiredMemoryProperties |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    if (reqs.mappingCapability & IDriverMemoryAllocation::EMCF_CACHED)
        desiredMemoryProperties |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    const IPhysicalDevice::SMemoryProperties& memoryProperties = m_physicalDevice->getMemoryProperties();

    uint32_t compatibleMemoryTypeCount = 0u;
    uint32_t compatibleMemoryTypeIndices[VK_MAX_MEMORY_TYPES];

    for (uint32_t i = 0u; i < memoryProperties.memoryTypeCount; ++i)
    {
        const bool memoryTypeSupportedForResource = (reqs.vulkanReqs.memoryTypeBits & (1 << i));

        const bool memoryHasDesirableProperties = (memoryProperties.memoryTypes[i].propertyFlags
            & desiredMemoryProperties) == desiredMemoryProperties;

        if (memoryTypeSupportedForResource && memoryHasDesirableProperties)
            compatibleMemoryTypeIndices[compatibleMemoryTypeCount++] = i;
    }

    for (uint32_t i = 0u; i < compatibleMemoryTypeCount; ++i)
    {
        // Todo(achal): Make use of requiresDedicatedAllocation and prefersDedicatedAllocation

        VkMemoryAllocateInfo vk_allocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        vk_allocateInfo.pNext = nullptr; // No extensions for now
        vk_allocateInfo.allocationSize = reqs.vulkanReqs.size;
        vk_allocateInfo.memoryTypeIndex = compatibleMemoryTypeIndices[i];

        VkDeviceMemory vk_deviceMemory;
        if (m_devf.vk.vkAllocateMemory(m_vkdev, &vk_allocateInfo, nullptr, &vk_deviceMemory) == VK_SUCCESS)
        {
            // Todo(achal): Change dedicate to not always be false
            return core::make_smart_refctd_ptr<CVulkanMemoryAllocation>(this, reqs.vulkanReqs.size, false, vk_deviceMemory);
        }
    }

    return nullptr;
}

bool CVulkanLogicalDevice::createCommandBuffers_impl(IGPUCommandPool* cmdPool, IGPUCommandBuffer::E_LEVEL level,
    uint32_t count, core::smart_refctd_ptr<IGPUCommandBuffer>* outCmdBufs)
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

    if (m_devf.vk.vkAllocateCommandBuffers(m_vkdev, &vk_allocateInfo, vk_commandBuffers) == VK_SUCCESS)
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

core::smart_refctd_ptr<IGPUImage> CVulkanLogicalDevice::createGPUImage(asset::IImage::SCreationParams&& params)
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
    if (m_devf.vk.vkCreateImage(m_vkdev, &vk_createInfo, nullptr, &vk_image) == VK_SUCCESS)
    {
        VkImageMemoryRequirementsInfo2 vk_memReqsInfo = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2 };
        vk_memReqsInfo.pNext = nullptr;
        vk_memReqsInfo.image = vk_image;

        VkMemoryDedicatedRequirements vk_memDedReqs = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS };
        VkMemoryRequirements2 vk_memReqs = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
        vk_memReqs.pNext = &vk_memDedReqs;

        m_devf.vk.vkGetImageMemoryRequirements2(m_vkdev, &vk_memReqsInfo, &vk_memReqs);

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

core::smart_refctd_ptr<IGPUGraphicsPipeline> CVulkanLogicalDevice::createGPUGraphicsPipeline_impl(
    IGPUPipelineCache* pipelineCache,
    IGPUGraphicsPipeline::SCreationParams&& params)
{
    core::smart_refctd_ptr<IGPUGraphicsPipeline> result;
    if (createGPUGraphicsPipelines_impl(pipelineCache, { &params, &params + 1 }, &result))
        return result;
    else
        return nullptr;
}

bool CVulkanLogicalDevice::createGPUGraphicsPipelines_impl(
    IGPUPipelineCache* pipelineCache,
    core::SRange<const IGPUGraphicsPipeline::SCreationParams> params,
    core::smart_refctd_ptr<IGPUGraphicsPipeline>* output)
{
    IGPUGraphicsPipeline::SCreationParams* creationParams = const_cast<IGPUGraphicsPipeline::SCreationParams*>(params.begin());

    // core::smart_refctd_ptr<const renderpass_independent_t> renderpassIndependent;
    // IImage::E_SAMPLE_COUNT_FLAGS rasterizationSamplesHint = IImage::ESCF_1_BIT;
    // core::smart_refctd_ptr<RenderpassType> renderpass;
    // uint32_t subpassIx = 0u;

    // for (size_t i = 0ull; i < params.size(); ++i)
    // {
    //     if ((creationParams[i].layout->getAPIType() != EAT_VULKAN) ||
    //         (creationParams[i].shader->getAPIType() != EAT_VULKAN))
    //     {
    //         return false;
    //     }
    // }

    VkPipelineCache vk_pipelineCache = VK_NULL_HANDLE;
    if (pipelineCache && pipelineCache->getAPIType() == EAT_VULKAN)
        vk_pipelineCache = static_cast<const CVulkanPipelineCache*>(pipelineCache)->getInternalObject();

    uint32_t totalShaderStageCount = 0u;
    core::vector<VkPipelineShaderStageCreateInfo> vk_shaderStages(params.size() * IGPURenderpassIndependentPipeline::SHADER_STAGE_COUNT);
    core::vector<VkPipelineVertexInputStateCreateInfo> vk_vertexInputStates(params.size());
    core::vector<VkPipelineInputAssemblyStateCreateInfo> vk_inputAssemblyStates(params.size());

    VkViewport vk_viewport = {};
    vk_viewport.x = 0.0f;
    vk_viewport.y = 0.0f;
    vk_viewport.width = (float)1280; // Todo(achal)
    vk_viewport.height = (float)720; // Todo(achal)
    vk_viewport.minDepth = 0.0f;
    vk_viewport.maxDepth = 1.0f;

    VkRect2D vk_scissor = {};
    vk_scissor.offset = { 0, 0 };
    vk_scissor.extent = { 1280u,720u }; // Todo(achal)
    // Todo(achal): Each of these could have multiple viewports and multiple scissors
    core::vector<VkPipelineViewportStateCreateInfo> vk_viewportStates(params.size());

    core::vector<VkPipelineRasterizationStateCreateInfo> vk_rasterizationStates(params.size());
    core::vector<VkPipelineMultisampleStateCreateInfo> vk_multisampleStates(params.size());

    VkPipelineColorBlendAttachmentState vk_colorBlendAttachmentState = {};
    vk_colorBlendAttachmentState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    vk_colorBlendAttachmentState.blendEnable = VK_FALSE;
    // Each of these could have multiple VkPipelineColorBlendAttachmentStates
    core::vector<VkPipelineColorBlendStateCreateInfo> vk_colorBlendStates(params.size());

    core::vector<VkGraphicsPipelineCreateInfo> vk_createInfos(params.size());
    for (size_t i = 0ull; i < params.size(); ++i)
    {
        const auto& rpIndie = creationParams[i].renderpassIndependent;

        vk_createInfos[i].sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        vk_createInfos[i].pNext = nullptr;
        vk_createInfos[i].flags = static_cast<VkPipelineCreateFlags>(creationParams[i].createFlags.value);

        uint32_t shaderStageCount = 0u;
        for (uint32_t ss = 0u; ss < IGPURenderpassIndependentPipeline::SHADER_STAGE_COUNT; ++ss)
        {
            const IGPUSpecializedShader* shader = rpIndie->getShaderAtIndex(ss);
            if (shader)
            {
                if (shader->getAPIType() == EAT_VULKAN)
                {
                    auto& vk_shaderStage = vk_shaderStages[totalShaderStageCount + shaderStageCount];
                    vk_shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                    vk_shaderStage.pNext = nullptr;
                    vk_shaderStage.flags = 0u;
                    vk_shaderStage.stage = static_cast<VkShaderStageFlagBits>(shader->getStage());
                    vk_shaderStage.module = static_cast<const CVulkanSpecializedShader*>(shader)->getInternalObject();
                    vk_shaderStage.pName = "main";
                    vk_shaderStage.pSpecializationInfo = nullptr; // Todo(achal): Seems like there is no way to specify these right now

                    ++shaderStageCount;
                }
            }
        }
        vk_createInfos[i].stageCount = shaderStageCount;
        vk_createInfos[i].pStages = vk_shaderStages.data() + totalShaderStageCount;
        totalShaderStageCount += shaderStageCount;

        // Vertex Input
        vk_vertexInputStates[i].sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vk_vertexInputStates[i].pNext = nullptr;
        vk_vertexInputStates[i].flags = 0u;
        // Todo(achal):
        vk_vertexInputStates[i].vertexBindingDescriptionCount = 0u;
        vk_vertexInputStates[i].vertexAttributeDescriptionCount = 0u;

        vk_createInfos[i].pVertexInputState = &vk_vertexInputStates[i];

        // Input Assembly
        vk_inputAssemblyStates[i].sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        vk_inputAssemblyStates[i].topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        vk_inputAssemblyStates[i].primitiveRestartEnable = VK_FALSE;
        vk_createInfos[i].pInputAssemblyState = &vk_inputAssemblyStates[i];

        // Tesselation
        vk_createInfos[i].pTessellationState = nullptr;

        // Viewport
        vk_viewportStates[i].sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vk_viewportStates[i].viewportCount = 1;
        vk_viewportStates[i].pViewports = &vk_viewport;
        vk_viewportStates[i].scissorCount = 1;
        vk_viewportStates[i].pScissors = &vk_scissor;
        vk_createInfos[i].pViewportState = &vk_viewportStates[i];

        // Rasterization
        vk_rasterizationStates[i].sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        vk_rasterizationStates[i].depthClampEnable = VK_FALSE;
        vk_rasterizationStates[i].rasterizerDiscardEnable = VK_FALSE;
        vk_rasterizationStates[i].polygonMode = VK_POLYGON_MODE_FILL;
        vk_rasterizationStates[i].lineWidth = 1.0f;
        vk_rasterizationStates[i].cullMode = VK_CULL_MODE_BACK_BIT;
        vk_rasterizationStates[i].frontFace = VK_FRONT_FACE_CLOCKWISE;
        vk_rasterizationStates[i].depthBiasEnable = VK_FALSE;
        vk_createInfos[i].pRasterizationState = &vk_rasterizationStates[i];

        // Multisampling
        vk_multisampleStates[i].sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        vk_multisampleStates[i].sampleShadingEnable = VK_FALSE;
        vk_multisampleStates[i].rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        vk_createInfos[i].pMultisampleState = &vk_multisampleStates[i];

        // Depth-stencil
        vk_createInfos[i].pDepthStencilState = nullptr;

        // Color blend
        vk_colorBlendStates[i].sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        vk_colorBlendStates[i].logicOpEnable = VK_FALSE;
        vk_colorBlendStates[i].logicOp = VK_LOGIC_OP_COPY;
        vk_colorBlendStates[i].attachmentCount = 1;
        vk_colorBlendStates[i].pAttachments = &vk_colorBlendAttachmentState;
        vk_colorBlendStates[i].blendConstants[0] = 0.0f;
        vk_colorBlendStates[i].blendConstants[1] = 0.0f;
        vk_colorBlendStates[i].blendConstants[2] = 0.0f;
        vk_colorBlendStates[i].blendConstants[3] = 0.0f;
        vk_createInfos[i].pColorBlendState = &vk_colorBlendStates[i];

        // Dynamic state
        vk_createInfos[i].pDynamicState = nullptr;

        vk_createInfos[i].layout = static_cast<const CVulkanPipelineLayout*>(rpIndie->getLayout())->getInternalObject();
        vk_createInfos[i].renderPass = static_cast<const CVulkanRenderpass*>(creationParams[i].renderpass.get())->getInternalObject();
        vk_createInfos[i].subpass = 0u; // Todo(achal)
        vk_createInfos[i].basePipelineHandle = VK_NULL_HANDLE; // Todo(achal): I think there isn't even any way to get this right now
        vk_createInfos[i].basePipelineIndex = 0u; // Todo(achal) : I think there isn't even any way to get this right now
    }

    core::vector<VkPipeline> vk_pipelines(params.size());
    if (m_devf.vk.vkCreateGraphicsPipelines(m_vkdev, vk_pipelineCache,
        static_cast<uint32_t>(params.size()), vk_createInfos.data(), nullptr, vk_pipelines.data()) == VK_SUCCESS)
    {
        for (size_t i = 0ull; i < params.size(); ++i)
        {
            output[i] = core::make_smart_refctd_ptr<CVulkanGraphicsPipeline>(
                core::smart_refctd_ptr<CVulkanLogicalDevice>(this),
                std::move(creationParams[i]),
                vk_pipelines[i]);
        }
        return true;
    }
    else
    {
        return false;
    }
}

}