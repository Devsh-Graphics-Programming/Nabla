#include "nbl/video/IPhysicalDevice.h"

using namespace nbl;
using namespace nbl::video;


E_API_TYPE ILogicalDevice::getAPIType() const { return m_physicalDevice->getAPIType(); }

core::smart_refctd_ptr<IGPUDescriptorSetLayout> ILogicalDevice::createDescriptorSetLayout(const IGPUDescriptorSetLayout::SBinding* _begin, const IGPUDescriptorSetLayout::SBinding* _end)
{
    uint32_t dynamicSSBOCount=0u,dynamicUBOCount=0u;
    for (auto b=_begin; b!=_end; ++b)
    {
        if (b->type == asset::EDT_STORAGE_BUFFER_DYNAMIC)
            dynamicSSBOCount++;
        else if (b->type == asset::EDT_UNIFORM_BUFFER_DYNAMIC)
            dynamicUBOCount++;
        else if (b->type == asset::EDT_COMBINED_IMAGE_SAMPLER && b->samplers)
        {
            auto* samplers = b->samplers;
            for (uint32_t i = 0u; i < b->count; ++i)
                if (!samplers[i]->wasCreatedBy(this))
                    return nullptr;
        }
    }
    const auto& limits = m_physicalDevice->getLimits();
    if (dynamicSSBOCount>limits.maxDescriptorSetDynamicOffsetSSBOs || dynamicUBOCount>limits.maxDescriptorSetDynamicOffsetUBOs)
        return nullptr;
    return createDescriptorSetLayout_impl(_begin,_end);
}

bool ILogicalDevice::updateDescriptorSets(uint32_t descriptorWriteCount, const IGPUDescriptorSet::SWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const IGPUDescriptorSet::SCopyDescriptorSet* pDescriptorCopies)
{
    for (auto i = 0; i < descriptorWriteCount; ++i)
    {
        auto* ds = static_cast<IGPUDescriptorSet*>(pDescriptorWrites[i].dstSet);
        ds->incrementVersion();

        auto* descriptors = ds->getDescriptors(pDescriptorWrites[i].descriptorType, pDescriptorWrites[i].binding);
        auto* samplers = ds->getMutableSamplers(pDescriptorWrites[i].binding);
        for (auto j = 0; j < pDescriptorWrites[i].count; ++j)
        {
            descriptors[j] = pDescriptorWrites[i].info[j].desc;

            if (samplers)
                samplers[j] = pDescriptorWrites[i].info[j].info.image.sampler;
        }
    }

    for (auto i = 0; i < descriptorCopyCount; ++i)
    {
        const auto* srcDS = static_cast<const IGPUDescriptorSet*>(pDescriptorCopies[i].srcSet);
        auto* dstDS = static_cast<IGPUDescriptorSet*>(pDescriptorCopies[i].dstSet);

        auto foundBindingInfo = std::lower_bound(srcDS->getLayout()->getBindings().begin(), srcDS->getLayout()->getBindings().end(), pDescriptorCopies[i].srcBinding,
            [](const IGPUDescriptorSetLayout::SBinding& a, const uint32_t b) -> bool
            {
                return a.binding < b;
            });

        if (foundBindingInfo->binding != pDescriptorCopies[i].srcBinding)
            return false;

        const asset::E_DESCRIPTOR_TYPE descriptorType = foundBindingInfo->type;

        auto* srcDescriptors = srcDS->getDescriptors(descriptorType, pDescriptorCopies[i].srcBinding);
        auto* srcSamplers = srcDS->getMutableSamplers(pDescriptorCopies[i].srcBinding);
        if (!srcDescriptors)
            return false;

        auto* dstDescriptors = dstDS->getDescriptors(descriptorType, pDescriptorCopies[i].dstBinding);
        auto* dstSamplers = dstDS->getMutableSamplers(pDescriptorCopies[i].dstBinding);
        if (!dstDescriptors)
            return false;

        // TODO(achal): Use copy_n.
        memcpy(dstDescriptors, srcDescriptors, pDescriptorCopies[i].count * sizeof(core::smart_refctd_ptr<const asset::IDescriptor>));

        if (srcSamplers && dstSamplers)
            memcpy(dstSamplers, srcSamplers, pDescriptorCopies[i].count * sizeof(core::smart_refctd_ptr<const IGPUSampler>));
    }

    updateDescriptorSets_impl(descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);

    return true;
}

void ILogicalDevice::addCommonGLSLDefines(std::ostringstream& pool, const bool runningInRenderdoc)
{
    const auto& limits = m_physicalDevice->getProperties().limits;
    const auto& features = getEnabledFeatures();

    // SPhysicalDeviceLimits
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_IMAGE_DIMENSION_1D", limits.maxImageDimension1D);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_IMAGE_DIMENSION_2D", limits.maxImageDimension2D);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_IMAGE_DIMENSION_3D",limits.maxImageDimension3D);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_IMAGE_DIMENSION_CUBE",limits.maxImageDimensionCube);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_IMAGE_ARRAY_LAYERS", limits.maxImageArrayLayers);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_BUFFER_VIEW_TEXELS", limits.maxBufferViewTexels);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_UBO_SIZE",limits.maxUBOSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SSBO_SIZE",limits.maxSSBOSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PUSH_CONSTANTS_SIZE", limits.maxPushConstantsSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_MEMORY_ALLOCATION_COUNT", limits.maxMemoryAllocationCount);
    // addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLER_ALLOCATION_COUNT",limits.maxSamplerAllocationCount); // shader doesn't need to know about that
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_BUFFER_IMAGE_GRANULARITY",core::min(limits.bufferImageGranularity, std::numeric_limits<int32_t>::max()));

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_SAMPLERS", limits.maxPerStageDescriptorSamplers);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UBOS", limits.maxPerStageDescriptorUBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_SSBOS",limits.maxPerStageDescriptorSSBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_IMAGES", limits.maxPerStageDescriptorImages);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_STORAGE_IMAGES", limits.maxPerStageDescriptorStorageImages);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_INPUT_ATTACHMENTS",limits.maxPerStageDescriptorInputAttachments);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_RESOURCES",limits.maxPerStageResources);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_SAMPLERS",limits.maxDescriptorSetSamplers);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UBOS",limits.maxDescriptorSetUBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_DYNAMIC_OFFSET_UBOS",limits.maxDescriptorSetDynamicOffsetUBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_SSBOS",limits.maxDescriptorSetSSBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_DYNAMIC_OFFSET_SSBOS",limits.maxDescriptorSetDynamicOffsetSSBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_IMAGES",limits.maxDescriptorSetImages);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_STORAGE_IMAGES",limits.maxDescriptorSetStorageImages);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_INPUT_ATTACHMENTS",limits.maxDescriptorSetInputAttachments);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_GENERATION_LEVEL",limits.maxTessellationGenerationLevel);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_PATCH_SIZE",limits.maxTessellationPatchSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_CONTROL_PER_VERTEX_INPUT_COMPONENTS",limits.maxTessellationControlPerVertexInputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_CONTROL_PER_VERTEX_OUTPUT_COMPONENTS",limits.maxTessellationControlPerVertexOutputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_CONTROL_PER_PATCH_OUTPUT_COMPONENTS",limits.maxTessellationControlPerPatchOutputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_CONTROL_TOTAL_OUTPUT_COMPONENTS",limits.maxTessellationControlTotalOutputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_EVALUATION_INPUT_COMPONENTS",limits.maxTessellationEvaluationInputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_EVALUATION_OUTPUT_COMPONENTS",limits.maxTessellationEvaluationOutputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_GEOMETRY_SHADER_INVOCATIONS",limits.maxGeometryShaderInvocations);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_GEOMETRY_INPUT_COMPONENTS",limits.maxGeometryInputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_GEOMETRY_OUTPUT_COMPONENTS",limits.maxGeometryOutputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_GEOMETRY_OUTPUT_VERTICES",limits.maxGeometryOutputVertices);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS",limits.maxGeometryTotalOutputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAGMENT_INPUT_COMPONENTS",limits.maxFragmentInputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAGMENT_OUTPUT_ATTACHMENTS",limits.maxFragmentOutputAttachments);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAGMENT_COMBINED_OUTPUT_RESOURCES",limits.maxFragmentCombinedOutputResources);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAGMENT_DUAL_SRC_ATTACHMENTS",limits.maxFragmentDualSrcAttachments);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_SHARED_MEMORY_SIZE",limits.maxComputeSharedMemorySize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_X",limits.maxComputeWorkGroupCount[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Y",limits.maxComputeWorkGroupCount[1]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Z",limits.maxComputeWorkGroupCount[2]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_WORKGROUP_INVOCATIONS",limits.maxComputeWorkGroupInvocations);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_WORKGROUP_SIZE_X",limits.maxWorkgroupSize[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_WORKGROUP_SIZE_Y",limits.maxWorkgroupSize[1]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_WORKGROUP_SIZE_Z",limits.maxWorkgroupSize[2]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SUB_PIXEL_PRECISION_BITS",limits.subPixelPrecisionBits);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DRAW_INDIRECT_COUNT",limits.maxDrawIndirectCount);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLER_LOD_BIAS",limits.maxSamplerLodBias);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLER_ANISOTROPY_LOG2",limits.maxSamplerAnisotropyLog2);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VIEWPORTS",limits.maxViewports);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VIEWPORT_DIMS_X",limits.maxViewportDims[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VIEWPORT_DIMS_Y",limits.maxViewportDims[1]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_VIEWPORT_BOUNDS_RANGE_BEGIN",limits.viewportBoundsRange[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_VIEWPORT_BOUNDS_RANGE_END",limits.viewportBoundsRange[1]);
    
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_VIEWPORT_SUB_PIXEL_BITS",limits.viewportSubPixelBits);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_MEMORY_MAP_ALIGNMENT",core::min(limits.minMemoryMapAlignment, 1u << 30));
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_BUFFER_VIEW_ALIGNMENT",limits.bufferViewAlignment);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_UBO_ALIGNMENT",limits.minUBOAlignment);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_SSBO_ALIGNMENT",limits.minSSBOAlignment);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_TEXEL_OFFSET",limits.minTexelOffset);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TEXEL_OFFSET",limits.maxTexelOffset);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_TEXEL_GATHER_OFFSET",limits.minTexelGatherOffset);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TEXEL_GATHER_OFFSET",limits.maxTexelGatherOffset);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_INTERPOLATION_OFFSET",limits.minInterpolationOffset);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_INTERPOLATION_OFFSET",limits.maxInterpolationOffset);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAMEBUFFER_WIDTH",limits.maxFramebufferWidth);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAMEBUFFER_HEIGHT",limits.maxFramebufferHeight);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAMEBUFFER_LAYERS",limits.maxFramebufferLayers);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FRAMEBUFFER_COLOR_SAMPLE_COUNTS",limits.framebufferColorSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FRAMEBUFFER_DEPTH_SAMPLE_COUNTS",limits.framebufferDepthSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FRAMEBUFFER_STENCIL_SAMPLE_COUNTS",limits.framebufferStencilSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FRAMEBUFFER_NO_ATTACHMENTS_SAMPLE_COUNTS",limits.framebufferNoAttachmentsSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COLOR_ATTACHMENTS",limits.maxColorAttachments);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLED_IMAGE_COLOR_SAMPLE_COUNTS",limits.sampledImageColorSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLED_IMAGE_INTEGER_SAMPLE_COUNTS",limits.sampledImageIntegerSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLED_IMAGE_DEPTH_SAMPLE_COUNTS",limits.sampledImageDepthSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLED_IMAGE_STENCIL_SAMPLE_COUNTS",limits.sampledImageStencilSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_STORAGE_IMAGE_SAMPLE_COUNTS",limits.storageImageSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLE_MASK_WORDS",limits.maxSampleMaskWords);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_TIMESTAMP_COMPUTE_AND_GRAPHICS",limits.timestampComputeAndGraphics);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_TIMESTAMP_PERIOD_IN_NANO_SECONDS",limits.timestampPeriodInNanoSeconds);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_CLIP_DISTANCES",limits.maxClipDistances);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_CULL_DISTANCES",limits.maxCullDistances);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMBINED_CLIP_AND_CULL_DISTANCES",limits.maxCombinedClipAndCullDistances);
    // addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_DISCRETE_QUEUE_PRIORITIES",limits.discreteQueuePriorities); // shader doesn't need to know about that

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_POINT_SIZE",limits.pointSizeRange[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_POINT_SIZE",limits.pointSizeRange[1]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_LINE_WIDTH",limits.lineWidthRange[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_LINE_WIDTH",limits.lineWidthRange[1]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_POINT_SIZE_GRANULARITY",limits.pointSizeGranularity);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_LINE_WIDTH_GRANULARITY",limits.lineWidthGranularity);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_STRICT_LINES",limits.strictLines);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_STANDARD_SAMPLE_LOCATIONS",limits.standardSampleLocations);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_OPTIMAL_BUFFER_COPY_OFFSET_ALIGNMENT",core::min(limits.optimalBufferCopyOffsetAlignment, 1u << 30));
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_OPTIMAL_BUFFER_COPY_ROW_PITCH_ALIGNMENT",core::min(limits.optimalBufferCopyRowPitchAlignment, 1u << 30));
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_NON_COHERENT_ATOM_SIZE",core::min(limits.nonCoherentAtomSize, std::numeric_limits<int32_t>::max()));

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VERTEX_OUTPUT_COMPONENTS",limits.maxVertexOutputComponents);
    
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SUBGROUP_SIZE",limits.subgroupSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SUBGROUP_OPS_SHADER_STAGES",limits.subgroupOpsShaderStages.value);
    if (limits.shaderSubgroupBasic) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_BASIC");
    if (limits.shaderSubgroupVote) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_VOTE");
    if (limits.shaderSubgroupArithmetic) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_ARITHMETIC");
    if (limits.shaderSubgroupBallot) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_BALLOT");
    if (limits.shaderSubgroupShuffle) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_SHUFFLE");
    if (limits.shaderSubgroupShuffleRelative) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_SHUFFLE_RELATIVE");
    if (limits.shaderSubgroupClustered) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_CLUSTERED");
    if (limits.shaderSubgroupQuad) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_QUAD");
    if (limits.shaderSubgroupQuadAllStages) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_QUAD_ALL_STAGES");

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_POINT_CLIPPING_BEHAVIOR",(uint32_t)limits.pointClippingBehavior);
    
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_SET_DESCRIPTORS",limits.maxPerSetDescriptors);
    // addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_MEMORY_ALLOCATION_SIZE",limits.maxMemoryAllocationSize); // shader doesn't need to know about that

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SIGNED_ZERO_INF_NAN_PRESERVE_FLOAT16",(uint32_t)limits.shaderSignedZeroInfNanPreserveFloat16);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SIGNED_ZERO_INF_NAN_PRESERVE_FLOAT32",(uint32_t)limits.shaderSignedZeroInfNanPreserveFloat32);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SIGNED_ZERO_INF_NAN_PRESERVE_FLOAT64",(uint32_t)limits.shaderSignedZeroInfNanPreserveFloat64);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_DENORM_PRESERVE_FLOAT16",(uint32_t)limits.shaderDenormPreserveFloat16);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_DENORM_PRESERVE_FLOAT32",(uint32_t)limits.shaderDenormPreserveFloat32);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_DENORM_PRESERVE_FLOAT64",(uint32_t)limits.shaderDenormPreserveFloat64);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_DENORM_FLUSH_TO_ZERO_FLOAT16",(uint32_t)limits.shaderDenormFlushToZeroFloat16);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_DENORM_FLUSH_TO_ZERO_FLOAT32",(uint32_t)limits.shaderDenormFlushToZeroFloat32);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_DENORM_FLUSH_TO_ZERO_FLOAT64",(uint32_t)limits.shaderDenormFlushToZeroFloat64);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_ROUNDING_MODE_RTE_FLOAT16",(uint32_t)limits.shaderRoundingModeRTEFloat16);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_ROUNDING_MODE_RTE_FLOAT32",(uint32_t)limits.shaderRoundingModeRTEFloat32);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_ROUNDING_MODE_RTE_FLOAT64",(uint32_t)limits.shaderRoundingModeRTEFloat64);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_ROUNDING_MODE_RTZ_FLOAT16",(uint32_t)limits.shaderRoundingModeRTZFloat16);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_ROUNDING_MODE_RTZ_FLOAT32",(uint32_t)limits.shaderRoundingModeRTZFloat32);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_ROUNDING_MODE_RTZ_FLOAT64",(uint32_t)limits.shaderRoundingModeRTZFloat64);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_SAMPLERS",limits.maxPerStageDescriptorUpdateAfterBindSamplers);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_UPDATE_AFTER_BIND_DESCRIPTORS_IN_ALL_POOLS",limits.maxUpdateAfterBindDescriptorsInAllPools);
    if (limits.shaderUniformBufferArrayNonUniformIndexingNative) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_UNIFORM_BUFFER_ARRAY_NON_UNIFORM_INDEXING_NATIVE");
    if (limits.shaderSampledImageArrayNonUniformIndexingNative) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING_NATIVE");
    if (limits.shaderStorageBufferArrayNonUniformIndexingNative) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING_NATIVE");
    if (limits.shaderStorageImageArrayNonUniformIndexingNative) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_STORAGE_IMAGE_ARRAY_NON_UNIFORM_INDEXING_NATIVE");
    if (limits.shaderInputAttachmentArrayNonUniformIndexingNative) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_INPUT_ATTACHMENT_ARRAY_NON_UNIFORM_INDEXING_NATIVE");
    if (limits.robustBufferAccessUpdateAfterBind) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_ROBUST_BUFFER_ACCESS_UPDATE_AFTER_BIND");
    if (limits.quadDivergentImplicitLod) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_QUAD_DIVERGENT_IMPLICIT_LOD");
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_SAMPLERS",limits.maxPerStageDescriptorUpdateAfterBindSamplers);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_UBOS",limits.maxPerStageDescriptorUpdateAfterBindUBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_SSBOS",limits.maxPerStageDescriptorUpdateAfterBindSSBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_IMAGES",limits.maxPerStageDescriptorUpdateAfterBindImages);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_STORAGE_IMAGES",limits.maxPerStageDescriptorUpdateAfterBindStorageImages);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_INPUT_ATTACHMENTS",limits.maxPerStageDescriptorUpdateAfterBindInputAttachments);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_UPDATE_AFTER_BIND_RESOURCES",limits.maxPerStageUpdateAfterBindResources);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_SAMPLERS",limits.maxDescriptorSetUpdateAfterBindSamplers);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_UBOS",limits.maxDescriptorSetUpdateAfterBindUBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_DYNAMIC_OFFSET_UBOS",limits.maxDescriptorSetUpdateAfterBindDynamicOffsetUBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_SSBOS",limits.maxDescriptorSetUpdateAfterBindSSBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_DYNAMIC_OFFSET_SSBOS",limits.maxDescriptorSetUpdateAfterBindDynamicOffsetSSBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_IMAGES",limits.maxDescriptorSetUpdateAfterBindImages);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_STORAGE_IMAGES",limits.maxDescriptorSetUpdateAfterBindStorageImages);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_INPUT_ATTACHMENTS",limits.maxDescriptorSetUpdateAfterBindInputAttachments);

    if (limits.filterMinmaxSingleComponentFormats) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FILTER_MINMAX_SINGLE_COMPONENT_FORMATS");
    if (limits.filterMinmaxImageComponentMapping) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FILTER_MINMAX_IMAGE_COMPONENT_MAPPING");
    
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_SUBGROUP_SIZE",limits.minSubgroupSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SUBGROUP_SIZE",limits.maxSubgroupSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_WORKGROUP_SUBGROUPS",limits.maxComputeWorkgroupSubgroups);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_REQUIRED_SUBGROUP_SIZE_STAGES",limits.requiredSubgroupSizeStages.value);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_STORAGE_TEXEL_BUFFER_OFFSET_ALIGNMENT_BYTES",limits.minSubgroupSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_UNIFORM_TEXEL_BUFFER_OFFSET_ALIGNMENT_BYTES",limits.maxSubgroupSize);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_BUFFER_SIZE",core::min(limits.maxBufferSize, std::numeric_limits<int32_t>::max()));

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_PRIMITIVE_OVERESTIMATION_SIZE", limits.primitiveOverestimationSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_EXTRA_PRIMITIVE_OVERESTIMATION_SIZE", limits.maxExtraPrimitiveOverestimationSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_EXTRA_PRIMITIVE_OVERESTIMATION_SIZE_GRANULARITY", limits.extraPrimitiveOverestimationSizeGranularity);
    if (limits.primitiveUnderestimation) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_PRIMITIVE_UNDERESTIMATION");
    if (limits.conservativePointAndLineRasterization) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_CONSERVATIVE_POINT_AND_LINE_RASTERIZATION");
    if (limits.degenerateTrianglesRasterized) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_DEGENERATE_TRIANGLES_RASTERIZED");
    if (limits.degenerateLinesRasterized) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_DEGENERATE_LINES_RASTERIZED");
    if (limits.fullyCoveredFragmentShaderInputVariable) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FULLY_COVERED_FRAGMENT_SHADER_INPUT_VARIABLE");
    if (limits.conservativeRasterizationPostDepthCoverage) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_CONSERVATIVE_RASTERIZATION_POST_DEPTH_COVERAGE");

    // addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DISCARD_RECTANGLES",limits.maxDiscardRectangles); // shader doesn't need to know about
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_LINE_SUB_PIXEL_PRECISION_BITS",limits.lineSubPixelPrecisionBits);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VERTEX_ATTRIB_DIVISOR",limits.maxVertexAttribDivisor);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SUBPASS_SHADING_WORKGROUP_SIZE_ASPECT_RATIO",limits.maxSubpassShadingWorkgroupSizeAspectRatio);

    if (limits.integerDotProduct8BitUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_8BIT_UNSIGNED_ACCELERATED");
    if (limits.integerDotProduct8BitSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_8BIT_SIGNED_ACCELERATED");
    if (limits.integerDotProduct8BitMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_8BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (limits.integerDotProduct4x8BitPackedUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_4X_8BIT_PACKED_UNSIGNED_ACCELERATED");
    if (limits.integerDotProduct4x8BitPackedSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_4X_8BIT_PACKED_SIGNED_ACCELERATED");
    if (limits.integerDotProduct4x8BitPackedMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_4X_8BIT_PACKED_MIXED_SIGNEDNESS_ACCELERATED");
    if (limits.integerDotProduct16BitUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_16BIT_UNSIGNED_ACCELERATED");
    if (limits.integerDotProduct16BitSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_16BIT_SIGNED_ACCELERATED");
    if (limits.integerDotProduct16BitMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_16BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (limits.integerDotProduct32BitUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_32BIT_UNSIGNED_ACCELERATED");
    if (limits.integerDotProduct32BitSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_32BIT_SIGNED_ACCELERATED");
    if (limits.integerDotProduct32BitMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_32BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (limits.integerDotProduct64BitUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_64BIT_UNSIGNED_ACCELERATED");
    if (limits.integerDotProduct64BitSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_64BIT_SIGNED_ACCELERATED");
    if (limits.integerDotProduct64BitMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_64BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating8BitUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_8BIT_UNSIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating8BitSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_8BIT_SIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_8BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_4X_8BIT_PACKED_UNSIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_4X_8BIT_PACKED_SIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_4X_8BIT_PACKED_MIXED_SIGNEDNESS_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating16BitUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_16BIT_UNSIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating16BitSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_16BIT_SIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_16BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating32BitUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_32BIT_UNSIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating32BitSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_32BIT_SIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_32BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating64BitUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_64BIT_UNSIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating64BitSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_64BIT_SIGNED_ACCELERATED");
    if (limits.integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_64BIT_MIXED_SIGNEDNESS_ACCELERATED");

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_GEOMETRY_COUNT",core::min(limits.maxGeometryCount, std::numeric_limits<int32_t>::max()));
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_INSTANCE_COUNT",core::min(limits.maxInstanceCount, std::numeric_limits<int32_t>::max()));
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PRIMITIVE_COUNT",core::min(limits.maxPrimitiveCount, std::numeric_limits<int32_t>::max()));
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_ACCELERATION_STRUCTURES",limits.maxPerStageDescriptorAccelerationStructures);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_ACCELERATION_STRUCTURES",limits.maxPerStageDescriptorUpdateAfterBindAccelerationStructures);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_ACCELERATION_STRUCTURES",limits.maxDescriptorSetAccelerationStructures);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_ACCELERATION_STRUCTURES",limits.maxDescriptorSetUpdateAfterBindAccelerationStructures);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_ACCELERATION_STRUCTURE_SCRATCH_OFFSET_ALIGNMENT",limits.minAccelerationStructureScratchOffsetAlignment);

    if (limits.variableSampleLocations) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_VARIABLE_SAMPLE_LOCATIONS");
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLE_LOCATION_SUBPIXEL_BITS",limits.sampleLocationSubPixelBits);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLE_LOCATION_SAMPLE_COUNTS",limits.sampleLocationSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLE_LOCATION_GRID_SIZE_X",limits.maxSampleLocationGridSize.width);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLE_LOCATION_GRID_SIZE_Y",limits.maxSampleLocationGridSize.height);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLE_LOCATION_COORDINATE_RANGE_X",limits.sampleLocationCoordinateRange[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLE_LOCATION_COORDINATE_RANGE_Y",limits.sampleLocationCoordinateRange[1]);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_IMPORTED_HOST_POINTER_ALIGNMENT",core::min(limits.minImportedHostPointerAlignment, 1u << 30));

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_FRAGMENT_DENSITY_TEXEL_SIZE_X",limits.minFragmentDensityTexelSize.width);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_FRAGMENT_DENSITY_TEXEL_SIZE_Y",limits.minFragmentDensityTexelSize.height);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAGMENT_DENSITY_TEXEL_SIZE_X",limits.maxFragmentDensityTexelSize.width);
    addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_MAX_FRAGMENT_DENSITY_TEXEL_SIZE_Y",limits.maxFragmentDensityTexelSize.height);
    if (limits.fragmentDensityInvocations) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FRAGMENT_DENSITY_INVOCATIONS");

    if (limits.subsampledLoads) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SUBSAMPLED_LOADS");
    if (limits.subsampledCoarseReconstructionEarlyAccess) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SUBSAMPLED_COARSE_RECONSTRUCTION_EARLY_ACCESS");
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SUBSAMPLED_ARRAY_LAYERS",limits.maxSubsampledArrayLayers);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_SUBSAMPLED_SAMPLERS",limits.maxDescriptorSetSubsampledSamplers);

    // no need to know
    // addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_PCI_DOMAN",limits.pciDomain);
    // addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_PCI_BUS",limits.pciBus);
    // addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_PCI_DEVICE",limits.pciDevice);
    // addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_PCI_FUNCTION",limits.pciFunction);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_GROUP_HANDLE_SIZE",limits.shaderGroupHandleSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_RAY_RECURSION_DEPTH",limits.maxRayRecursionDepth);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SHADER_GROUP_STRIDE",limits.maxShaderGroupStride);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_GROUP_BASE_ALIGNMENT",limits.shaderGroupBaseAlignment);
    // addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_SIZE",limits.shaderGroupHandleCaptureReplaySize); // [DO NOT EXPOSE] for capture tools 
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_RAY_DISPATCH_INVOCATION_COUNT",limits.maxRayDispatchInvocationCount);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_GROUP_HANDLE_ALIGNMENT",limits.shaderGroupHandleAlignment);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_RAY_HIT_ATTRIBUTE_SIZE",limits.maxRayHitAttributeSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_COOPERATIVE_MATRIX_SUPPORTED_STAGES",limits.cooperativeMatrixSupportedStages.value);
  
    if (limits.shaderOutputViewportIndex) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_OUTPUT_VIEWPORT_INDEX");
    if (limits.shaderOutputLayer) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_OUTPUT_LAYER");
    if (limits.shaderIntegerFunctions2) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_INTEGER_FUNCTIONS_2");
    if (limits.shaderSubgroupClock) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_CLOCK");
    if (limits.imageFootprint) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_IMAGE_FOOTPRINT");
    // if (limits.texelBufferAlignment) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_TEXEL_BUFFER_ALIGNMENT"); // shader doesn't need to know about that
    if (limits.shaderSMBuiltins) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SM_BUILTINS");
    if (limits.shaderSubgroupPartitioned) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_PARTITIONED");
    if (limits.gcnShader) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_GCN_SHADER");
    if (limits.gpuShaderHalfFloat) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_GPU_SHADER_HALF_FLOAT");
    if (limits.shaderBallot) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_BALLOT");
    if (limits.shaderImageLoadStoreLod) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_IMAGE_LOAD_STORE_LOD");
    if (limits.shaderTrinaryMinmax) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_TRINARY_MINMAX");
    if (limits.postDepthCoverage) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_POST_DEPTH_COVERAGE");
    if (limits.shaderStencilExport) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_STENCIL_EXPORT");
    if (limits.decorateString) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_DECORATE_STRING");
    // if (limits.externalFence) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_EXTERNAL_FENCE"); // [TODO] requires instance extensions, add them
    // if (limits.externalMemory) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_EXTERNAL_MEMORY"); // [TODO] requires instance extensions, add them
    // if (limits.externalSemaphore) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_EXTERNAL_SEMAPHORE"); // [TODO] requires instance extensions, add them
    if (limits.shaderNonSemanticInfo) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_NON_SEMANTIC_INFO");
    if (limits.fragmentShaderBarycentric) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FRAGMENT_SHADER_BARYCENTRIC");
    if (limits.geometryShaderPassthrough) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_GEOMETRY_SHADER_PASSTHROUGH");
    if (limits.viewportSwizzle) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_VIEWPORT_SWIZZLE");

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_COMPUTE_UNITS",limits.computeUnits);
    if (limits.dispatchBase) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_DISPATCH_BASE");
    if (limits.allowCommandBufferQueryCopies) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_ALLOW_COMMAND_BUFFER_QUERY_COPIES");
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_OPTIMALLY_RESIDENT_WORKGROUP_INVOCATIONS",limits.maxOptimallyResidentWorkgroupInvocations);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_RESIDENT_INVOCATIONS",limits.maxResidentInvocations);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SPIRV_VERSION",(uint32_t)limits.spirvVersion);
    if (limits.vertexPipelineStoresAndAtomics) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_VERTEX_PIPELINE_STORES_AND_ATOMICS");
    if (limits.fragmentStoresAndAtomics) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_FRAGMENT_STORES_AND_ATOMICS");
    if (limits.shaderTessellationAndGeometryPointSize) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_TESSELLATION_AND_GEOMETRY_POINT_SIZE");
    if (limits.shaderImageGatherExtended) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_IMAGE_GATHER_EXTENDED");
    if (limits.shaderInt64) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_INT64");
    if (limits.shaderInt16) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_INT16");
    if (limits.samplerAnisotropy) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_SAMPLER_ANISOTROPY");
    if (limits.storageBuffer16BitAccess) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_STORAGE_BUFFER_16BIT_ACCESS");
    if (limits.uniformAndStorageBuffer16BitAccess) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_UNIFORM_AND_STORAGE_BUFFER_16BIT_ACCESS");
    if (limits.storagePushConstant16) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_STORAGE_PUSH_CONSTANT_16");
    if (limits.storageInputOutput16) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_STORAGE_INPUT_OUTPUT_16");
    if (limits.storageBuffer8BitAccess) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_STORAGE_BUFFER_8BIT_ACCESS");
    if (limits.uniformAndStorageBuffer8BitAccess) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_UNIFORM_AND_STORAGE_BUFFER_8BIT_ACCESS");
    if (limits.storagePushConstant8) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_STORAGE_PUSH_CONSTANT_8");
    if (limits.shaderBufferInt64Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_BUFFER_INT64_ATOMICS");
    if (limits.shaderSharedInt64Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_SHARED_INT64_ATOMICS");
    if (limits.shaderFloat16) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_FLOAT16");
    if (limits.shaderInt8) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_INT8");

    // SPhysicalDeviceFeatures
    if (features.robustBufferAccess) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_ROBUST_BUFFER_ACCESS");
    if (features.fullDrawIndexUint32) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_FULL_DRAW_INDEX_UINT32");
    if (features.imageCubeArray) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_IMAGE_CUBE_ARRAY");
    if (features.independentBlend) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_INDEPENDENT_BLEND");
    if (features.geometryShader) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_GEOMETRY_SHADER");
    if (features.tessellationShader) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_TESSELLATION_SHADER");
    if (features.sampleRateShading) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SAMPLE_RATE_SHADING");
    if (features.dualSrcBlend) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DUAL_SRC_BLEND");
    if (features.logicOp) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_LOGIC_OP");
    if (features.multiDrawIndirect) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_MULTI_DRAW_INDIRECT");
    if (features.drawIndirectFirstInstance) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DRAW_INDIRECT_FIRST_INSTANCE");
    if (features.depthClamp) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DEPTH_CLAMP");
    if (features.depthBiasClamp) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DEPTH_BIAS_CLAMP");
    if (features.fillModeNonSolid) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_FILL_MODE_NON_SOLID");
    if (features.depthBounds) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DEPTH_BOUNDS");
    if (features.wideLines) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_WIDE_LINES");
    if (features.largePoints) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_LARGE_POINTS");
    if (features.alphaToOne) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_ALPHA_TO_ONE");
    if (features.multiViewport) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_MULTI_VIEWPORT");
    if (features.occlusionQueryPrecise) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_OCCLUSION_QUERY_PRECISE");
    // if (features.pipelineStatisticsQuery) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_PIPELINE_STATISTICS_QUERY"); // shader doesn't need to know about
    if (features.shaderStorageImageExtendedFormats) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_IMAGE_EXTENDED_FORMATS");
    if (features.shaderStorageImageMultisample) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_IMAGE_MULTISAMPLE");
    if (features.shaderStorageImageReadWithoutFormat) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_IMAGE_READ_WITHOUT_FORMAT");
    if (features.shaderStorageImageWriteWithoutFormat) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT");
    if (features.shaderUniformBufferArrayDynamicIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_UNIFORM_BUFFER_ARRAY_DYNAMIC_INDEXING");
    if (features.shaderSampledImageArrayDynamicIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SAMPLED_IMAGE_ARRAY_DYNAMIC_INDEXING");
    if (features.shaderStorageBufferArrayDynamicIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_BUFFER_ARRAY_DYNAMIC_INDEXING");
    if (features.shaderStorageImageArrayDynamicIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_IMAGE_ARRAY_DYNAMIC_INDEXING");
    if (features.shaderClipDistance) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_CLIP_DISTANCE");
    if (features.shaderCullDistance) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_CULL_DISTANCE");
    if (features.vertexAttributeDouble) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_VERTEX_ATTRIBUTE_DOUBLE");
    if (features.shaderResourceResidency) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_RESOURCE_RESIDENCY");
    if (features.shaderResourceMinLod) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_RESOURCE_MIN_LOD");
    if (features.variableMultisampleRate) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_VARIABLE_MULTISAMPLE_RATE");
    // if (features.inheritedQueries) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_INHERITED_QUERIES"); // shader doesn't need to know about
    if (features.shaderDrawParameters) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_DRAW_PARAMETERS");
    if (features.samplerMirrorClampToEdge) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SAMPLER_MIRROR_CLAMP_TO_EDGE");
    if (features.drawIndirectCount) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DRAW_INDIRECT_COUNT");
    if (features.descriptorIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_INDEXING");
    if (features.shaderInputAttachmentArrayDynamicIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_INPUT_ATTACHMENT_ARRAY_DYNAMIC_INDEXING");
    if (features.shaderUniformTexelBufferArrayDynamicIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_UNIFORM_TEXEL_BUFFER_ARRAY_DYNAMIC_INDEXING");
    if (features.shaderStorageTexelBufferArrayDynamicIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_TEXEL_BUFFER_ARRAY_DYNAMIC_INDEXING");
    if (features.shaderUniformBufferArrayNonUniformIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_UNIFORM_BUFFER_ARRAY_NON_UNIFORM_INDEXING");
    if (features.shaderSampledImageArrayNonUniformIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING");
    if (features.shaderStorageBufferArrayNonUniformIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING");
    if (features.shaderStorageImageArrayNonUniformIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_IMAGE_ARRAY_NON_UNIFORM_INDEXING");
    if (features.shaderInputAttachmentArrayNonUniformIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_INPUT_ATTACHMENT_ARRAY_NON_UNIFORM_INDEXING");
    if (features.shaderUniformTexelBufferArrayNonUniformIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_UNIFORM_TEXEL_BUFFER_ARRAY_NON_UNIFORM_INDEXING");
    if (features.shaderStorageTexelBufferArrayNonUniformIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_TEXEL_BUFFER_ARRAY_NON_UNIFORM_INDEXING");
    if (features.descriptorBindingUniformBufferUpdateAfterBind) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_UNIFORM_BUFFER_UPDATE_AFTER_BIND");
    if (features.descriptorBindingSampledImageUpdateAfterBind) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_SAMPLED_IMAGE_UPDATE_AFTER_BIND");
    if (features.descriptorBindingStorageImageUpdateAfterBind) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_STORAGE_IMAGE_UPDATE_AFTER_BIND");
    if (features.descriptorBindingStorageBufferUpdateAfterBind) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_STORAGE_BUFFER_UPDATE_AFTER_BIND");
    if (features.descriptorBindingUniformTexelBufferUpdateAfterBind) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_UNIFORM_TEXEL_BUFFER_UPDATE_AFTER_BIND");
    if (features.descriptorBindingStorageTexelBufferUpdateAfterBind) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_STORAGE_TEXEL_BUFFER_UPDATE_AFTER_BIND");
    // if (features.descriptorBindingUpdateUnusedWhilePending) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING"); // shader doesn't need to know about
    if (features.descriptorBindingPartiallyBound) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_PARTIALLY_BOUND");
    if (features.descriptorBindingVariableDescriptorCount) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT");
    if (features.runtimeDescriptorArray) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RUNTIME_DESCRIPTOR_ARRAY");
    if (features.samplerFilterMinmax) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SAMPLER_FILTER_MINMAX");
    if (features.scalarBlockLayout) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SCALAR_BLOCK_LAYOUT");
    if (features.uniformBufferStandardLayout) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_UNIFORM_BUFFER_STANDARD_LAYOUT");
    if (features.shaderSubgroupExtendedTypes) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SUBGROUP_EXTENDED_TYPES");
    if (features.separateDepthStencilLayouts) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SEPARATE_DEPTH_STENCIL_LAYOUTS");
    if (features.bufferDeviceAddress) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_BUFFER_DEVICE_ADDRESS");
    if (features.bufferDeviceAddressMultiDevice) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_BUFFER_DEVICE_ADDRESS_MULTI_DEVICE");
    if (features.vulkanMemoryModel) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_VULKAN_MEMORY_MODEL");
    if (features.vulkanMemoryModelDeviceScope) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE");
    if (features.vulkanMemoryModelAvailabilityVisibilityChains) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_VULKAN_MEMORY_MODEL_AVAILABILITY_VISIBILITY_CHAINS");
    if (features.subgroupBroadcastDynamicId) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SUBGROUP_BROADCAST_DYNAMIC_ID");
    if (features.shaderDemoteToHelperInvocation) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_DEMOTE_TO_HELPER_INVOCATION");
    if (features.shaderTerminateInvocation) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_TERMINATE_INVOCATION");
    if (features.subgroupSizeControl) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SUBGROUP_SIZE_CONTROL");
    if (features.computeFullSubgroups) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_COMPUTE_FULL_SUBGROUPS");
    if (features.shaderIntegerDotProduct) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_INTEGER_DOT_PRODUCT");
    if (features.rasterizationOrderColorAttachmentAccess) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RASTERIZATION_ORDER_COLOR_ATTACHMENT_ACCESS");
    if (features.rasterizationOrderDepthAttachmentAccess) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RASTERIZATION_ORDER_DEPTH_ATTACHMENT_ACCESS");
    if (features.rasterizationOrderStencilAttachmentAccess) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RASTERIZATION_ORDER_STENCIL_ATTACHMENT_ACCESS");
    if (features.fragmentShaderSampleInterlock) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_SHADER_SAMPLE_INTERLOCK");
    if (features.fragmentShaderPixelInterlock) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK");
    if (features.fragmentShaderShadingRateInterlock) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_SHADER_SHADING_RATE_INTERLOCK");
    if (features.indexTypeUint8) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_INDEX_TYPE_UINT8");
    if (features.shaderBufferFloat32Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT32_ATOMICS");
    if (features.shaderBufferFloat32AtomicAdd) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT32_ATOMIC_ADD");
    if (features.shaderBufferFloat64Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT64_ATOMICS");
    if (features.shaderBufferFloat64AtomicAdd) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT64_ATOMIC_ADD");
    if (features.shaderSharedFloat32Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT32_ATOMICS");
    if (features.shaderSharedFloat32AtomicAdd) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT32_ATOMIC_ADD");
    if (features.shaderSharedFloat64Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT64_ATOMICS");
    if (features.shaderSharedFloat64AtomicAdd) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT64_ATOMIC_ADD");
    if (features.shaderImageFloat32Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_IMAGE_FLOAT32_ATOMICS");
    if (features.shaderImageFloat32AtomicAdd) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_IMAGE_FLOAT32_ATOMIC_ADD");
    if (features.sparseImageFloat32Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SPARSE_IMAGE_FLOAT32_ATOMICS");
    if (features.sparseImageFloat32AtomicAdd) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SPARSE_IMAGE_FLOAT32_ATOMIC_ADD");
    if (features.shaderBufferFloat16Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT16_ATOMICS");
    if (features.shaderBufferFloat16AtomicAdd) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT16_ATOMIC_ADD");
    if (features.shaderBufferFloat16AtomicMinMax) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT16_ATOMIC_MIN_MAX");
    if (features.shaderBufferFloat32AtomicMinMax) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT32_ATOMIC_MIN_MAX");
    if (features.shaderBufferFloat64AtomicMinMax) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT64_ATOMIC_MIN_MAX");
    if (features.shaderSharedFloat16Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT16_ATOMICS");
    if (features.shaderSharedFloat16AtomicAdd) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT16_ATOMIC_ADD");
    if (features.shaderSharedFloat16AtomicMinMax) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT16_ATOMIC_MIN_MAX");
    if (features.shaderSharedFloat32AtomicMinMax) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT32_ATOMIC_MIN_MAX");
    if (features.shaderSharedFloat64AtomicMinMax) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT64_ATOMIC_MIN_MAX");
    if (features.shaderImageFloat32AtomicMinMax) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_IMAGE_FLOAT32_ATOMIC_MIN_MAX");
    if (features.sparseImageFloat32AtomicMinMax) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SPARSE_IMAGE_FLOAT32_ATOMIC_MIN_MAX");
    if (features.shaderImageInt64Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_IMAGE_INT64_ATOMICS");
    if (features.sparseImageInt64Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SPARSE_IMAGE_INT64_ATOMICS");
    if (features.accelerationStructure) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_ACCELERATION_STRUCTURE");
    if (features.accelerationStructureIndirectBuild) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_ACCELERATION_STRUCTURE_INDIRECT_BUILD");
    if (features.accelerationStructureHostCommands) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_ACCELERATION_STRUCTURE_HOST_COMMANDS");
    // if (features.descriptorBindingAccelerationStructureUpdateAfterBind) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_ACCELERATION_STRUCTURE_UPDATE_AFTER_BIND"); // shader doesn't need to know about
    if (features.rayQuery) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RAY_QUERY");
    if (features.rayTracingPipeline) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RAY_TRACING_PIPELINE");
    if (features.rayTracingPipelineTraceRaysIndirect) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RAY_TRACING_PIPELINE_TRACE_RAYS_INDIRECT");
    if (features.rayTraversalPrimitiveCulling) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RAY_TRAVERSAL_PRIMITIVE_CULLING");
    if (features.shaderDeviceClock) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_DEVICE_CLOCK");
    if (features.shaderSubgroupUniformControlFlow) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW");
    if (features.workgroupMemoryExplicitLayout) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT");
    if (features.workgroupMemoryExplicitLayoutScalarBlockLayout) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_SCALAR_BLOCK_LAYOUT");
    if (features.workgroupMemoryExplicitLayout8BitAccess) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_8BIT_ACCESS");
    if (features.workgroupMemoryExplicitLayout16BitAccess) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_16BIT_ACCESS");
    if (features.computeDerivativeGroupQuads) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_COMPUTE_DERIVATIVE_GROUP_QUADS");
    if (features.computeDerivativeGroupLinear) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_COMPUTE_DERIVATIVE_GROUP_LINEAR");
    if (features.cooperativeMatrix) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_COOPERATIVE_MATRIX");
    if (features.cooperativeMatrixRobustBufferAccess) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_COOPERATIVE_MATRIX_ROBUST_BUFFER_ACCESS");
    if (features.rayTracingMotionBlur) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RAY_TRACING_MOTION_BLUR");
    if (features.rayTracingMotionBlurPipelineTraceRaysIndirect) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RAY_TRACING_MOTION_BLUR_PIPELINE_TRACE_RAYS_INDIRECT");
    if (features.coverageReductionMode) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_COVERAGE_REDUCTION_MODE");
    if (features.deviceGeneratedCommands) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DEVICE_GENERATED_COMMANDS");
    if (features.taskShader) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_TASK_SHADER");
    if (features.meshShader) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_MESH_SHADER");
    if (features.representativeFragmentTest) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_REPRESENTATIVE_FRAGMENT_TEST");
    if (features.mixedAttachmentSamples) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_MIXED_ATTACHMENT_SAMPLES");
    if (features.hdrMetadata) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_HDR_METADATA");
    // if (features.displayTiming) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DISPLAY_TIMING"); // shader doesn't need to know about
    if (features.rasterizationOrder) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RASTERIZATION_ORDER");
    if (features.shaderExplicitVertexParameter) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_EXPLICIT_VERTEX_PARAMETER");
    if (features.shaderInfoAMD) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_INFO_AMD");
    if (features.hostQueryReset) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_HOST_QUERY_RESET");
    // if (features.pipelineCreationCacheControl) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_PIPELINE_CREATION_CACHE_CONTROL"); // shader doesn't need to know about
    if (features.colorWriteEnable) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_COLOR_WRITE_ENABLE");
    if (features.conditionalRendering) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_CONDITIONAL_RENDERING");
    if (features.inheritedConditionalRendering) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_INHERITED_CONDITIONAL_RENDERING");
    // if (features.deviceMemoryReport) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DEVICE_MEMORY_REPORT"); // shader doesn't need to know about
    if (features.fragmentDensityMap) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_DENSITY_MAP");
    if (features.fragmentDensityMapDynamic) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_DENSITY_MAP_DYNAMIC");
    if (features.fragmentDensityMapNonSubsampledImages) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_DENSITY_MAP_NON_SUBSAMPLED_IMAGES");
    if (features.fragmentDensityMapDeferred) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_DENSITY_MAP_DEFERRED");
    if (features.robustImageAccess) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_ROBUST_IMAGE_ACCESS");
    if (features.inlineUniformBlock) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_INLINE_UNIFORM_BLOCK");
    // if (features.descriptorBindingInlineUniformBlockUpdateAfterBind) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_INLINE_UNIFORM_BLOCK_UPDATE_AFTER_BIND"); // shader doesn't need to know about
    if (features.rectangularLines) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RECTANGULAR_LINES");
    if (features.bresenhamLines) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_BRESENHAM_LINES");
    if (features.smoothLines) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SMOOTH_LINES");
    if (features.stippledRectangularLines) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_STIPPLED_RECTANGULAR_LINES");
    if (features.stippledBresenhamLines) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_STIPPLED_BRESENHAM_LINES");
    if (features.stippledSmoothLines) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_STIPPLED_SMOOTH_LINES");
    // if (features.memoryPriority) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_MEMORY_PRIORITY"); // shader doesn't need to know about
    if (features.robustBufferAccess2) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_ROBUST_BUFFER_ACCESS_2");
    if (features.robustImageAccess2) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_ROBUST_IMAGE_ACCESS_2");
    if (features.nullDescriptor) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_NULL_DESCRIPTOR");
    if (features.performanceCounterQueryPools) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_PERFORMANCE_COUNTER_QUERY_POOLS");
    if (features.performanceCounterMultipleQueryPools) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_PERFORMANCE_COUNTER_MULTIPLE_QUERY_POOLS");
    if (features.pipelineExecutableInfo) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_PIPELINE_EXECUTABLE_INFO");
    // if (features.maintenance4) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_MAINTENANCE_4"); // shader doesn't need to know about
    if (features.deviceCoherentMemory) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DEVICE_COHERENT_MEMORY");
    // if (features.bufferMarkerAMD) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_BUFFER_MARKER_AMD"); // shader doesn't need to know about

    // TODO: @achal test examples 14 and 48 on all APIs and GPUs

    if (runningInRenderdoc)
        addGLSLDefineToPool(pool,"NBL_GLSL_RUNNING_IN_RENDERDOC");
}
