#include "nbl/video/IPhysicalDevice.h"

namespace nbl::video
{

IPhysicalDevice::IPhysicalDevice(core::smart_refctd_ptr<system::ISystem>&& s, core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc) :
    m_system(std::move(s)), m_GLSLCompiler(std::move(glslc))
{
    memset(&m_memoryProperties, 0, sizeof(SMemoryProperties));
    memset(&m_linearTilingUsages, 0, sizeof(SFormatImageUsages::SUsage));
    memset(&m_optimalTilingUsages, 0, sizeof(SFormatImageUsages::SUsage));
    memset(&m_bufferUsages, 0, sizeof(SFormatBufferUsages::SUsage));
}

void IPhysicalDevice::addCommonGLSLDefines(std::ostringstream& pool, const bool runningInRenderdoc)
{
    // SPhysicalDeviceLimits
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_IMAGE_DIMENSION_1D", m_properties.limits.maxImageDimension1D);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_IMAGE_DIMENSION_2D", m_properties.limits.maxImageDimension2D);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_IMAGE_DIMENSION_3D",m_properties.limits.maxImageDimension3D);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_IMAGE_DIMENSION_CUBE",m_properties.limits.maxImageDimensionCube);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_IMAGE_ARRAY_LAYERS", m_properties.limits.maxImageArrayLayers);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_BUFFER_VIEW_TEXELS", m_properties.limits.maxBufferViewTexels);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_UBO_SIZE",m_properties.limits.maxUBOSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SSBO_SIZE",m_properties.limits.maxSSBOSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PUSH_CONSTANTS_SIZE", m_properties.limits.maxPushConstantsSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_MEMORY_ALLOCATION_COUNT", m_properties.limits.maxMemoryAllocationCount);
    // addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLER_ALLOCATION_COUNT",m_properties.limits.maxSamplerAllocationCount); // shader doesn't need to know about that
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_BUFFER_IMAGE_GRANULARITY",core::min(m_properties.limits.bufferImageGranularity, std::numeric_limits<int32_t>::max()));

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_SAMPLERS", m_properties.limits.maxPerStageDescriptorSamplers);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UBOS", m_properties.limits.maxPerStageDescriptorUBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_SSBOS",m_properties.limits.maxPerStageDescriptorSSBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_IMAGES", m_properties.limits.maxPerStageDescriptorImages);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_STORAGE_IMAGES", m_properties.limits.maxPerStageDescriptorStorageImages);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_INPUT_ATTACHMENTS",m_properties.limits.maxPerStageDescriptorInputAttachments);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_RESOURCES",m_properties.limits.maxPerStageResources);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_SAMPLERS",m_properties.limits.maxDescriptorSetSamplers);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UBOS",m_properties.limits.maxDescriptorSetUBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_DYNAMIC_OFFSET_UBOS",m_properties.limits.maxDescriptorSetDynamicOffsetUBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_SSBOS",m_properties.limits.maxDescriptorSetSSBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_DYNAMIC_OFFSET_SSBOS",m_properties.limits.maxDescriptorSetDynamicOffsetSSBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_IMAGES",m_properties.limits.maxDescriptorSetImages);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_STORAGE_IMAGES",m_properties.limits.maxDescriptorSetStorageImages);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_INPUT_ATTACHMENTS",m_properties.limits.maxDescriptorSetInputAttachments);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_GENERATION_LEVEL",m_properties.limits.maxTessellationGenerationLevel);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_PATCH_SIZE",m_properties.limits.maxTessellationPatchSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_CONTROL_PER_VERTEX_INPUT_COMPONENTS",m_properties.limits.maxTessellationControlPerVertexInputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_CONTROL_PER_VERTEX_OUTPUT_COMPONENTS",m_properties.limits.maxTessellationControlPerVertexOutputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_CONTROL_PER_PATCH_OUTPUT_COMPONENTS",m_properties.limits.maxTessellationControlPerPatchOutputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_CONTROL_TOTAL_OUTPUT_COMPONENTS",m_properties.limits.maxTessellationControlTotalOutputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_EVALUATION_INPUT_COMPONENTS",m_properties.limits.maxTessellationEvaluationInputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TESSELLATION_EVALUATION_OUTPUT_COMPONENTS",m_properties.limits.maxTessellationEvaluationOutputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_GEOMETRY_SHADER_INVOCATIONS",m_properties.limits.maxGeometryShaderInvocations);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_GEOMETRY_INPUT_COMPONENTS",m_properties.limits.maxGeometryInputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_GEOMETRY_OUTPUT_COMPONENTS",m_properties.limits.maxGeometryOutputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_GEOMETRY_OUTPUT_VERTICES",m_properties.limits.maxGeometryOutputVertices);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS",m_properties.limits.maxGeometryTotalOutputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAGMENT_INPUT_COMPONENTS",m_properties.limits.maxFragmentInputComponents);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAGMENT_OUTPUT_ATTACHMENTS",m_properties.limits.maxFragmentOutputAttachments);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAGMENT_COMBINED_OUTPUT_RESOURCES",m_properties.limits.maxFragmentCombinedOutputResources);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAGMENT_DUAL_SRC_ATTACHMENTS",m_properties.limits.maxFragmentDualSrcAttachments);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_SHARED_MEMORY_SIZE",m_properties.limits.maxComputeSharedMemorySize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_X",m_properties.limits.maxComputeWorkGroupCount[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Y",m_properties.limits.maxComputeWorkGroupCount[1]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Z",m_properties.limits.maxComputeWorkGroupCount[2]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_WORKGROUP_INVOCATIONS",m_properties.limits.maxComputeWorkGroupInvocations);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_WORKGROUP_SIZE_X",m_properties.limits.maxWorkgroupSize[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_WORKGROUP_SIZE_Y",m_properties.limits.maxWorkgroupSize[1]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_WORKGROUP_SIZE_Z",m_properties.limits.maxWorkgroupSize[2]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SUB_PIXEL_PRECISION_BITS",m_properties.limits.subPixelPrecisionBits);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DRAW_INDIRECT_COUNT",m_properties.limits.maxDrawIndirectCount);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLER_LOD_BIAS",m_properties.limits.maxSamplerLodBias);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLER_ANISOTROPY_LOG2",m_properties.limits.maxSamplerAnisotropyLog2);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VIEWPORTS",m_properties.limits.maxViewports);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VIEWPORT_DIMS_X",m_properties.limits.maxViewportDims[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VIEWPORT_DIMS_Y",m_properties.limits.maxViewportDims[1]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_VIEWPORT_BOUNDS_RANGE_BEGIN",m_properties.limits.viewportBoundsRange[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_VIEWPORT_BOUNDS_RANGE_END",m_properties.limits.viewportBoundsRange[1]);
    
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_VIEWPORT_SUB_PIXEL_BITS",m_properties.limits.viewportSubPixelBits);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_MEMORY_MAP_ALIGNMENT",core::min(m_properties.limits.minMemoryMapAlignment, 1u << 30));
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_BUFFER_VIEW_ALIGNMENT",m_properties.limits.bufferViewAlignment);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_UBO_ALIGNMENT",m_properties.limits.minUBOAlignment);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_SSBO_ALIGNMENT",m_properties.limits.minSSBOAlignment);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_TEXEL_OFFSET",m_properties.limits.minTexelOffset);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TEXEL_OFFSET",m_properties.limits.maxTexelOffset);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_TEXEL_GATHER_OFFSET",m_properties.limits.minTexelGatherOffset);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TEXEL_GATHER_OFFSET",m_properties.limits.maxTexelGatherOffset);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_INTERPOLATION_OFFSET",m_properties.limits.minInterpolationOffset);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_INTERPOLATION_OFFSET",m_properties.limits.maxInterpolationOffset);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAMEBUFFER_WIDTH",m_properties.limits.maxFramebufferWidth);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAMEBUFFER_HEIGHT",m_properties.limits.maxFramebufferHeight);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAMEBUFFER_LAYERS",m_properties.limits.maxFramebufferLayers);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FRAMEBUFFER_COLOR_SAMPLE_COUNTS",m_properties.limits.framebufferColorSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FRAMEBUFFER_DEPTH_SAMPLE_COUNTS",m_properties.limits.framebufferDepthSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FRAMEBUFFER_STENCIL_SAMPLE_COUNTS",m_properties.limits.framebufferStencilSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FRAMEBUFFER_NO_ATTACHMENTS_SAMPLE_COUNTS",m_properties.limits.framebufferNoAttachmentsSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COLOR_ATTACHMENTS",m_properties.limits.maxColorAttachments);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLED_IMAGE_COLOR_SAMPLE_COUNTS",m_properties.limits.sampledImageColorSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLED_IMAGE_INTEGER_SAMPLE_COUNTS",m_properties.limits.sampledImageIntegerSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLED_IMAGE_DEPTH_SAMPLE_COUNTS",m_properties.limits.sampledImageDepthSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLED_IMAGE_STENCIL_SAMPLE_COUNTS",m_properties.limits.sampledImageStencilSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_STORAGE_IMAGE_SAMPLE_COUNTS",m_properties.limits.storageImageSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLE_MASK_WORDS",m_properties.limits.maxSampleMaskWords);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_TIMESTAMP_COMPUTE_AND_GRAPHICS",m_properties.limits.timestampComputeAndGraphics);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_TIMESTAMP_PERIOD_IN_NANO_SECONDS",m_properties.limits.timestampPeriodInNanoSeconds);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_CLIP_DISTANCES",m_properties.limits.maxClipDistances);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_CULL_DISTANCES",m_properties.limits.maxCullDistances);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMBINED_CLIP_AND_CULL_DISTANCES",m_properties.limits.maxCombinedClipAndCullDistances);
    // addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_DISCRETE_QUEUE_PRIORITIES",m_properties.limits.discreteQueuePriorities); // shader doesn't need to know about that

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_POINT_SIZE",m_properties.limits.pointSizeRange[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_POINT_SIZE",m_properties.limits.pointSizeRange[1]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_LINE_WIDTH",m_properties.limits.lineWidthRange[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_LINE_WIDTH",m_properties.limits.lineWidthRange[1]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_POINT_SIZE_GRANULARITY",m_properties.limits.pointSizeGranularity);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_LINE_WIDTH_GRANULARITY",m_properties.limits.lineWidthGranularity);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_STRICT_LINES",m_properties.limits.strictLines);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_STANDARD_SAMPLE_LOCATIONS",m_properties.limits.standardSampleLocations);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_OPTIMAL_BUFFER_COPY_OFFSET_ALIGNMENT",core::min(m_properties.limits.optimalBufferCopyOffsetAlignment, 1u << 30));
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_OPTIMAL_BUFFER_COPY_ROW_PITCH_ALIGNMENT",core::min(m_properties.limits.optimalBufferCopyRowPitchAlignment, 1u << 30));
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_NON_COHERENT_ATOM_SIZE",core::min(m_properties.limits.nonCoherentAtomSize, std::numeric_limits<int32_t>::max()));

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VERTEX_OUTPUT_COMPONENTS",m_properties.limits.maxVertexOutputComponents);
    
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SUBGROUP_SIZE",m_properties.limits.subgroupSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SUBGROUP_OPS_SHADER_STAGES",m_properties.limits.subgroupOpsShaderStages.value);
    if (m_properties.limits.shaderSubgroupBasic) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_BASIC");
    if (m_properties.limits.shaderSubgroupVote) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_VOTE");
    if (m_properties.limits.shaderSubgroupArithmetic) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_ARITHMETIC");
    if (m_properties.limits.shaderSubgroupBallot) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_BALLOT");
    if (m_properties.limits.shaderSubgroupShuffle) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_SHUFFLE");
    if (m_properties.limits.shaderSubgroupShuffleRelative) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_SHUFFLE_RELATIVE");
    if (m_properties.limits.shaderSubgroupClustered) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_CLUSTERED");
    if (m_properties.limits.shaderSubgroupQuad) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_QUAD");
    if (m_properties.limits.shaderSubgroupQuadAllStages) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_QUAD_ALL_STAGES");

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_POINT_CLIPPING_BEHAVIOR",(uint32_t)m_properties.limits.pointClippingBehavior);
    
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_SET_DESCRIPTORS",m_properties.limits.maxPerSetDescriptors);
    // addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_MEMORY_ALLOCATION_SIZE",m_properties.limits.maxMemoryAllocationSize); // shader doesn't need to know about that

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SIGNED_ZERO_INF_NAN_PRESERVE_FLOAT16",(uint32_t)m_properties.limits.shaderSignedZeroInfNanPreserveFloat16);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SIGNED_ZERO_INF_NAN_PRESERVE_FLOAT32",(uint32_t)m_properties.limits.shaderSignedZeroInfNanPreserveFloat32);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SIGNED_ZERO_INF_NAN_PRESERVE_FLOAT64",(uint32_t)m_properties.limits.shaderSignedZeroInfNanPreserveFloat64);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_DENORM_PRESERVE_FLOAT16",(uint32_t)m_properties.limits.shaderDenormPreserveFloat16);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_DENORM_PRESERVE_FLOAT32",(uint32_t)m_properties.limits.shaderDenormPreserveFloat32);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_DENORM_PRESERVE_FLOAT64",(uint32_t)m_properties.limits.shaderDenormPreserveFloat64);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_DENORM_FLUSH_TO_ZERO_FLOAT16",(uint32_t)m_properties.limits.shaderDenormFlushToZeroFloat16);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_DENORM_FLUSH_TO_ZERO_FLOAT32",(uint32_t)m_properties.limits.shaderDenormFlushToZeroFloat32);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_DENORM_FLUSH_TO_ZERO_FLOAT64",(uint32_t)m_properties.limits.shaderDenormFlushToZeroFloat64);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_ROUNDING_MODE_RTE_FLOAT16",(uint32_t)m_properties.limits.shaderRoundingModeRTEFloat16);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_ROUNDING_MODE_RTE_FLOAT32",(uint32_t)m_properties.limits.shaderRoundingModeRTEFloat32);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_ROUNDING_MODE_RTE_FLOAT64",(uint32_t)m_properties.limits.shaderRoundingModeRTEFloat64);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_ROUNDING_MODE_RTZ_FLOAT16",(uint32_t)m_properties.limits.shaderRoundingModeRTZFloat16);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_ROUNDING_MODE_RTZ_FLOAT32",(uint32_t)m_properties.limits.shaderRoundingModeRTZFloat32);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_ROUNDING_MODE_RTZ_FLOAT64",(uint32_t)m_properties.limits.shaderRoundingModeRTZFloat64);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_SAMPLERS",m_properties.limits.maxPerStageDescriptorUpdateAfterBindSamplers);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_UPDATE_AFTER_BIND_DESCRIPTORS_IN_ALL_POOLS",m_properties.limits.maxUpdateAfterBindDescriptorsInAllPools);
    if (m_properties.limits.shaderUniformBufferArrayNonUniformIndexingNative) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_UNIFORM_BUFFER_ARRAY_NON_UNIFORM_INDEXING_NATIVE");
    if (m_properties.limits.shaderSampledImageArrayNonUniformIndexingNative) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING_NATIVE");
    if (m_properties.limits.shaderStorageBufferArrayNonUniformIndexingNative) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING_NATIVE");
    if (m_properties.limits.shaderStorageImageArrayNonUniformIndexingNative) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_STORAGE_IMAGE_ARRAY_NON_UNIFORM_INDEXING_NATIVE");
    if (m_properties.limits.shaderInputAttachmentArrayNonUniformIndexingNative) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_INPUT_ATTACHMENT_ARRAY_NON_UNIFORM_INDEXING_NATIVE");
    if (m_properties.limits.robustBufferAccessUpdateAfterBind) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_ROBUST_BUFFER_ACCESS_UPDATE_AFTER_BIND");
    if (m_properties.limits.quadDivergentImplicitLod) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_QUAD_DIVERGENT_IMPLICIT_LOD");
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_SAMPLERS",m_properties.limits.maxPerStageDescriptorUpdateAfterBindSamplers);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_UBOS",m_properties.limits.maxPerStageDescriptorUpdateAfterBindUBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_SSBOS",m_properties.limits.maxPerStageDescriptorUpdateAfterBindSSBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_IMAGES",m_properties.limits.maxPerStageDescriptorUpdateAfterBindImages);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_STORAGE_IMAGES",m_properties.limits.maxPerStageDescriptorUpdateAfterBindStorageImages);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_INPUT_ATTACHMENTS",m_properties.limits.maxPerStageDescriptorUpdateAfterBindInputAttachments);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_UPDATE_AFTER_BIND_RESOURCES",m_properties.limits.maxPerStageUpdateAfterBindResources);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_SAMPLERS",m_properties.limits.maxDescriptorSetUpdateAfterBindSamplers);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_UBOS",m_properties.limits.maxDescriptorSetUpdateAfterBindUBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_DYNAMIC_OFFSET_UBOS",m_properties.limits.maxDescriptorSetUpdateAfterBindDynamicOffsetUBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_SSBOS",m_properties.limits.maxDescriptorSetUpdateAfterBindSSBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_DYNAMIC_OFFSET_SSBOS",m_properties.limits.maxDescriptorSetUpdateAfterBindDynamicOffsetSSBOs);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_IMAGES",m_properties.limits.maxDescriptorSetUpdateAfterBindImages);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_STORAGE_IMAGES",m_properties.limits.maxDescriptorSetUpdateAfterBindStorageImages);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_INPUT_ATTACHMENTS",m_properties.limits.maxDescriptorSetUpdateAfterBindInputAttachments);

    if (m_properties.limits.filterMinmaxSingleComponentFormats) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FILTER_MINMAX_SINGLE_COMPONENT_FORMATS");
    if (m_properties.limits.filterMinmaxImageComponentMapping) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FILTER_MINMAX_IMAGE_COMPONENT_MAPPING");
    
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_SUBGROUP_SIZE",m_properties.limits.minSubgroupSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SUBGROUP_SIZE",m_properties.limits.maxSubgroupSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_WORKGROUP_SUBGROUPS",m_properties.limits.maxComputeWorkgroupSubgroups);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_REQUIRED_SUBGROUP_SIZE_STAGES",m_properties.limits.requiredSubgroupSizeStages.value);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_STORAGE_TEXEL_BUFFER_OFFSET_ALIGNMENT_BYTES",m_properties.limits.minSubgroupSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_UNIFORM_TEXEL_BUFFER_OFFSET_ALIGNMENT_BYTES",m_properties.limits.maxSubgroupSize);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_BUFFER_SIZE",core::min(m_properties.limits.maxBufferSize, std::numeric_limits<int32_t>::max()));

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_PRIMITIVE_OVERESTIMATION_SIZE", m_properties.limits.primitiveOverestimationSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_EXTRA_PRIMITIVE_OVERESTIMATION_SIZE", m_properties.limits.maxExtraPrimitiveOverestimationSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_EXTRA_PRIMITIVE_OVERESTIMATION_SIZE_GRANULARITY", m_properties.limits.extraPrimitiveOverestimationSizeGranularity);
    if (m_properties.limits.primitiveUnderestimation) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_PRIMITIVE_UNDERESTIMATION");
    if (m_properties.limits.conservativePointAndLineRasterization) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_CONSERVATIVE_POINT_AND_LINE_RASTERIZATION");
    if (m_properties.limits.degenerateTrianglesRasterized) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_DEGENERATE_TRIANGLES_RASTERIZED");
    if (m_properties.limits.degenerateLinesRasterized) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_DEGENERATE_LINES_RASTERIZED");
    if (m_properties.limits.fullyCoveredFragmentShaderInputVariable) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FULLY_COVERED_FRAGMENT_SHADER_INPUT_VARIABLE");
    if (m_properties.limits.conservativeRasterizationPostDepthCoverage) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_CONSERVATIVE_RASTERIZATION_POST_DEPTH_COVERAGE");

    // addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DISCARD_RECTANGLES",m_properties.limits.maxDiscardRectangles); // shader doesn't need to know about
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_LINE_SUB_PIXEL_PRECISION_BITS",m_properties.limits.lineSubPixelPrecisionBits);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VERTEX_ATTRIB_DIVISOR",m_properties.limits.maxVertexAttribDivisor);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SUBPASS_SHADING_WORKGROUP_SIZE_ASPECT_RATIO",m_properties.limits.maxSubpassShadingWorkgroupSizeAspectRatio);

    if (m_properties.limits.integerDotProduct8BitUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_8BIT_UNSIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProduct8BitSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_8BIT_SIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProduct8BitMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_8BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (m_properties.limits.integerDotProduct4x8BitPackedUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_4X_8BIT_PACKED_UNSIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProduct4x8BitPackedSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_4X_8BIT_PACKED_SIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProduct4x8BitPackedMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_4X_8BIT_PACKED_MIXED_SIGNEDNESS_ACCELERATED");
    if (m_properties.limits.integerDotProduct16BitUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_16BIT_UNSIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProduct16BitSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_16BIT_SIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProduct16BitMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_16BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (m_properties.limits.integerDotProduct32BitUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_32BIT_UNSIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProduct32BitSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_32BIT_SIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProduct32BitMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_32BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (m_properties.limits.integerDotProduct64BitUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_64BIT_UNSIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProduct64BitSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_64BIT_SIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProduct64BitMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_64BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (m_properties.limits.integerDotProductAccumulatingSaturating8BitUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_8BIT_UNSIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProductAccumulatingSaturating8BitSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_8BIT_SIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_8BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (m_properties.limits.integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_4X_8BIT_PACKED_UNSIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_4X_8BIT_PACKED_SIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_4X_8BIT_PACKED_MIXED_SIGNEDNESS_ACCELERATED");
    if (m_properties.limits.integerDotProductAccumulatingSaturating16BitUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_16BIT_UNSIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProductAccumulatingSaturating16BitSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_16BIT_SIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_16BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (m_properties.limits.integerDotProductAccumulatingSaturating32BitUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_32BIT_UNSIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProductAccumulatingSaturating32BitSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_32BIT_SIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_32BIT_MIXED_SIGNEDNESS_ACCELERATED");
    if (m_properties.limits.integerDotProductAccumulatingSaturating64BitUnsignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_64BIT_UNSIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProductAccumulatingSaturating64BitSignedAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_64BIT_SIGNED_ACCELERATED");
    if (m_properties.limits.integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_INTEGER_DOT_PRODUCT_ACCUMULATING_SATURATING_64BIT_MIXED_SIGNEDNESS_ACCELERATED");

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_GEOMETRY_COUNT",core::min(m_properties.limits.maxGeometryCount, std::numeric_limits<int32_t>::max()));
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_INSTANCE_COUNT",core::min(m_properties.limits.maxInstanceCount, std::numeric_limits<int32_t>::max()));
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PRIMITIVE_COUNT",core::min(m_properties.limits.maxPrimitiveCount, std::numeric_limits<int32_t>::max()));
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_ACCELERATION_STRUCTURES",m_properties.limits.maxPerStageDescriptorAccelerationStructures);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_DESCRIPTOR_UPDATE_AFTER_BIND_ACCELERATION_STRUCTURES",m_properties.limits.maxPerStageDescriptorUpdateAfterBindAccelerationStructures);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_ACCELERATION_STRUCTURES",m_properties.limits.maxDescriptorSetAccelerationStructures);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_ACCELERATION_STRUCTURES",m_properties.limits.maxDescriptorSetUpdateAfterBindAccelerationStructures);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_ACCELERATION_STRUCTURE_SCRATCH_OFFSET_ALIGNMENT",m_properties.limits.minAccelerationStructureScratchOffsetAlignment);

    if (m_properties.limits.variableSampleLocations) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_VARIABLE_SAMPLE_LOCATIONS");
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLE_LOCATION_SUBPIXEL_BITS",m_properties.limits.sampleLocationSubPixelBits);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLE_LOCATION_SAMPLE_COUNTS",m_properties.limits.sampleLocationSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLE_LOCATION_GRID_SIZE_X",m_properties.limits.maxSampleLocationGridSize.width);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLE_LOCATION_GRID_SIZE_Y",m_properties.limits.maxSampleLocationGridSize.height);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLE_LOCATION_COORDINATE_RANGE_X",m_properties.limits.sampleLocationCoordinateRange[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SAMPLE_LOCATION_COORDINATE_RANGE_Y",m_properties.limits.sampleLocationCoordinateRange[1]);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_IMPORTED_HOST_POINTER_ALIGNMENT",core::min(m_properties.limits.minImportedHostPointerAlignment, 1u << 30));

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_FRAGMENT_DENSITY_TEXEL_SIZE_X",m_properties.limits.minFragmentDensityTexelSize.width);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_FRAGMENT_DENSITY_TEXEL_SIZE_Y",m_properties.limits.minFragmentDensityTexelSize.height);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAGMENT_DENSITY_TEXEL_SIZE_X",m_properties.limits.maxFragmentDensityTexelSize.width);
    addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_MAX_FRAGMENT_DENSITY_TEXEL_SIZE_Y",m_properties.limits.maxFragmentDensityTexelSize.height);
    if (m_properties.limits.fragmentDensityInvocations) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FRAGMENT_DENSITY_INVOCATIONS");

    if (m_properties.limits.subsampledLoads) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SUBSAMPLED_LOADS");
    if (m_properties.limits.subsampledCoarseReconstructionEarlyAccess) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SUBSAMPLED_COARSE_RECONSTRUCTION_EARLY_ACCESS");
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SUBSAMPLED_ARRAY_LAYERS",m_properties.limits.maxSubsampledArrayLayers);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_SUBSAMPLED_SAMPLERS",m_properties.limits.maxDescriptorSetSubsampledSamplers);

    // no need to know
    // addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_PCI_DOMAN",m_properties.limits.pciDomain);
    // addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_PCI_BUS",m_properties.limits.pciBus);
    // addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_PCI_DEVICE",m_properties.limits.pciDevice);
    // addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_PCI_FUNCTION",m_properties.limits.pciFunction);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_GROUP_HANDLE_SIZE",m_properties.limits.shaderGroupHandleSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_RAY_RECURSION_DEPTH",m_properties.limits.maxRayRecursionDepth);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SHADER_GROUP_STRIDE",m_properties.limits.maxShaderGroupStride);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_GROUP_BASE_ALIGNMENT",m_properties.limits.shaderGroupBaseAlignment);
    // addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_SIZE",m_properties.limits.shaderGroupHandleCaptureReplaySize); // [DO NOT EXPOSE] for capture tools 
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_RAY_DISPATCH_INVOCATION_COUNT",m_properties.limits.maxRayDispatchInvocationCount);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_GROUP_HANDLE_ALIGNMENT",m_properties.limits.shaderGroupHandleAlignment);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_RAY_HIT_ATTRIBUTE_SIZE",m_properties.limits.maxRayHitAttributeSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_COOPERATIVE_MATRIX_SUPPORTED_STAGES",m_properties.limits.cooperativeMatrixSupportedStages.value);
  
    if (m_properties.limits.shaderOutputViewportIndex) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_OUTPUT_VIEWPORT_INDEX");
    if (m_properties.limits.shaderOutputLayer) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_OUTPUT_LAYER");
    if (m_properties.limits.shaderIntegerFunctions2) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_INTEGER_FUNCTIONS_2");
    if (m_properties.limits.shaderSubgroupClock) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_CLOCK");
    if (m_properties.limits.imageFootprint) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_IMAGE_FOOTPRINT");
    // if (m_properties.limits.texelBufferAlignment) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_TEXEL_BUFFER_ALIGNMENT"); // shader doesn't need to know about that
    if (m_properties.limits.shaderSMBuiltins) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SM_BUILTINS");
    if (m_properties.limits.shaderSubgroupPartitioned) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_PARTITIONED");
    if (m_properties.limits.gcnShader) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_GCN_SHADER");
    if (m_properties.limits.gpuShaderHalfFloat) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_GPU_SHADER_HALF_FLOAT");
    if (m_properties.limits.shaderBallot) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_BALLOT");
    if (m_properties.limits.shaderImageLoadStoreLod) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_IMAGE_LOAD_STORE_LOD");
    if (m_properties.limits.shaderTrinaryMinmax) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_TRINARY_MINMAX");
    if (m_properties.limits.postDepthCoverage) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_POST_DEPTH_COVERAGE");
    if (m_properties.limits.shaderStencilExport) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_STENCIL_EXPORT");
    if (m_properties.limits.decorateString) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_DECORATE_STRING");
    // if (m_properties.limits.externalFence) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_EXTERNAL_FENCE"); // [TODO] requires instance extensions, add them
    // if (m_properties.limits.externalMemory) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_EXTERNAL_MEMORY"); // [TODO] requires instance extensions, add them
    // if (m_properties.limits.externalSemaphore) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_EXTERNAL_SEMAPHORE"); // [TODO] requires instance extensions, add them
    if (m_properties.limits.shaderNonSemanticInfo) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_NON_SEMANTIC_INFO");
    if (m_properties.limits.fragmentShaderBarycentric) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FRAGMENT_SHADER_BARYCENTRIC");
    if (m_properties.limits.geometryShaderPassthrough) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_GEOMETRY_SHADER_PASSTHROUGH");
    if (m_properties.limits.viewportSwizzle) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_VIEWPORT_SWIZZLE");

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_COMPUTE_UNITS",m_properties.limits.computeUnits);
    if (m_properties.limits.dispatchBase) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_DISPATCH_BASE");
    if (m_properties.limits.allowCommandBufferQueryCopies) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_ALLOW_COMMAND_BUFFER_QUERY_COPIES");
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_OPTIMALLY_RESIDENT_WORKGROUP_INVOCATIONS",m_properties.limits.maxOptimallyResidentWorkgroupInvocations);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_RESIDENT_INVOCATIONS",m_properties.limits.maxResidentInvocations);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SPIRV_VERSION",(uint32_t)m_properties.limits.spirvVersion);
    if (m_properties.limits.vertexPipelineStoresAndAtomics) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_VERTEX_PIPELINE_STORES_AND_ATOMICS");
    if (m_properties.limits.fragmentStoresAndAtomics) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_FRAGMENT_STORES_AND_ATOMICS");
    if (m_properties.limits.shaderTessellationAndGeometryPointSize) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_TESSELLATION_AND_GEOMETRY_POINT_SIZE");
    if (m_properties.limits.shaderImageGatherExtended) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_IMAGE_GATHER_EXTENDED");
    if (m_properties.limits.shaderInt64) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_INT64");
    if (m_properties.limits.shaderInt16) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_INT16");
    if (m_properties.limits.storageBuffer16BitAccess) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_STORAGE_BUFFER_16BIT_ACCESS");
    if (m_properties.limits.uniformAndStorageBuffer16BitAccess) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_UNIFORM_AND_STORAGE_BUFFER_16BIT_ACCESS");
    if (m_properties.limits.storagePushConstant16) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_STORAGE_PUSH_CONSTANT_16");
    if (m_properties.limits.storageInputOutput16) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_STORAGE_INPUT_OUTPUT_16");
    if (m_properties.limits.storageBuffer8BitAccess) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_STORAGE_BUFFER_8BIT_ACCESS");
    if (m_properties.limits.uniformAndStorageBuffer8BitAccess) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_UNIFORM_AND_STORAGE_BUFFER_8BIT_ACCESS");
    if (m_properties.limits.storagePushConstant8) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_STORAGE_PUSH_CONSTANT_8");
    if (m_properties.limits.shaderBufferInt64Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_BUFFER_INT64_ATOMICS");
    if (m_properties.limits.shaderSharedInt64Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_SHARED_INT64_ATOMICS");
    if (m_properties.limits.shaderFloat16) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_FLOAT16");
    if (m_properties.limits.shaderInt8) addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_SHADER_INT8");

    // SPhysicalDeviceFeatures
    if (m_features.robustBufferAccess) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_ROBUST_BUFFER_ACCESS");
    if (m_features.fullDrawIndexUint32) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_FULL_DRAW_INDEX_UINT32");
    if (m_features.imageCubeArray) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_IMAGE_CUBE_ARRAY");
    if (m_features.independentBlend) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_INDEPENDENT_BLEND");
    if (m_features.geometryShader) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_GEOMETRY_SHADER");
    if (m_features.tessellationShader) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_TESSELLATION_SHADER");
    if (m_features.sampleRateShading) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SAMPLE_RATE_SHADING");
    if (m_features.dualSrcBlend) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DUAL_SRC_BLEND");
    if (m_features.logicOp) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_LOGIC_OP");
    if (m_features.multiDrawIndirect) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_MULTI_DRAW_INDIRECT");
    if (m_features.drawIndirectFirstInstance) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DRAW_INDIRECT_FIRST_INSTANCE");
    if (m_features.depthClamp) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DEPTH_CLAMP");
    if (m_features.depthBiasClamp) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DEPTH_BIAS_CLAMP");
    if (m_features.fillModeNonSolid) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_FILL_MODE_NON_SOLID");
    if (m_features.depthBounds) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DEPTH_BOUNDS");
    if (m_features.wideLines) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_WIDE_LINES");
    if (m_features.largePoints) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_LARGE_POINTS");
    if (m_features.alphaToOne) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_ALPHA_TO_ONE");
    if (m_features.multiViewport) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_MULTI_VIEWPORT");
    if (m_features.samplerAnisotropy) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SAMPLER_ANISOTROPY");
    if (m_features.occlusionQueryPrecise) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_OCCLUSION_QUERY_PRECISE");
    // if (m_features.pipelineStatisticsQuery) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_PIPELINE_STATISTICS_QUERY"); // shader doesn't need to know about
    if (m_features.shaderStorageImageExtendedFormats) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_IMAGE_EXTENDED_FORMATS");
    if (m_features.shaderStorageImageMultisample) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_IMAGE_MULTISAMPLE");
    if (m_features.shaderStorageImageReadWithoutFormat) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_IMAGE_READ_WITHOUT_FORMAT");
    if (m_features.shaderStorageImageWriteWithoutFormat) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT");
    if (m_features.shaderUniformBufferArrayDynamicIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_UNIFORM_BUFFER_ARRAY_DYNAMIC_INDEXING");
    if (m_features.shaderSampledImageArrayDynamicIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SAMPLED_IMAGE_ARRAY_DYNAMIC_INDEXING");
    if (m_features.shaderStorageBufferArrayDynamicIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_BUFFER_ARRAY_DYNAMIC_INDEXING");
    if (m_features.shaderStorageImageArrayDynamicIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_IMAGE_ARRAY_DYNAMIC_INDEXING");
    if (m_features.shaderClipDistance) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_CLIP_DISTANCE");
    if (m_features.shaderCullDistance) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_CULL_DISTANCE");
    if (m_features.vertexAttributeDouble) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_VERTEX_ATTRIBUTE_DOUBLE");
    if (m_features.shaderResourceResidency) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_RESOURCE_RESIDENCY");
    if (m_features.shaderResourceMinLod) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_RESOURCE_MIN_LOD");
    if (m_features.variableMultisampleRate) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_VARIABLE_MULTISAMPLE_RATE");
    // if (m_features.inheritedQueries) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_INHERITED_QUERIES"); // shader doesn't need to know about
    if (m_features.shaderDrawParameters) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_DRAW_PARAMETERS");
    if (m_features.samplerMirrorClampToEdge) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SAMPLER_MIRROR_CLAMP_TO_EDGE");
    if (m_features.drawIndirectCount) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DRAW_INDIRECT_COUNT");
    if (m_features.descriptorIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_INDEXING");
    if (m_features.shaderInputAttachmentArrayDynamicIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_INPUT_ATTACHMENT_ARRAY_DYNAMIC_INDEXING");
    if (m_features.shaderUniformTexelBufferArrayDynamicIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_UNIFORM_TEXEL_BUFFER_ARRAY_DYNAMIC_INDEXING");
    if (m_features.shaderStorageTexelBufferArrayDynamicIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_TEXEL_BUFFER_ARRAY_DYNAMIC_INDEXING");
    if (m_features.shaderUniformBufferArrayNonUniformIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_UNIFORM_BUFFER_ARRAY_NON_UNIFORM_INDEXING");
    if (m_features.shaderSampledImageArrayNonUniformIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING");
    if (m_features.shaderStorageBufferArrayNonUniformIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING");
    if (m_features.shaderStorageImageArrayNonUniformIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_IMAGE_ARRAY_NON_UNIFORM_INDEXING");
    if (m_features.shaderInputAttachmentArrayNonUniformIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_INPUT_ATTACHMENT_ARRAY_NON_UNIFORM_INDEXING");
    if (m_features.shaderUniformTexelBufferArrayNonUniformIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_UNIFORM_TEXEL_BUFFER_ARRAY_NON_UNIFORM_INDEXING");
    if (m_features.shaderStorageTexelBufferArrayNonUniformIndexing) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_STORAGE_TEXEL_BUFFER_ARRAY_NON_UNIFORM_INDEXING");
    if (m_features.descriptorBindingUniformBufferUpdateAfterBind) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_UNIFORM_BUFFER_UPDATE_AFTER_BIND");
    if (m_features.descriptorBindingSampledImageUpdateAfterBind) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_SAMPLED_IMAGE_UPDATE_AFTER_BIND");
    if (m_features.descriptorBindingStorageImageUpdateAfterBind) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_STORAGE_IMAGE_UPDATE_AFTER_BIND");
    if (m_features.descriptorBindingStorageBufferUpdateAfterBind) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_STORAGE_BUFFER_UPDATE_AFTER_BIND");
    if (m_features.descriptorBindingUniformTexelBufferUpdateAfterBind) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_UNIFORM_TEXEL_BUFFER_UPDATE_AFTER_BIND");
    if (m_features.descriptorBindingStorageTexelBufferUpdateAfterBind) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_STORAGE_TEXEL_BUFFER_UPDATE_AFTER_BIND");
    // if (m_features.descriptorBindingUpdateUnusedWhilePending) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING"); // shader doesn't need to know about
    if (m_features.descriptorBindingPartiallyBound) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_PARTIALLY_BOUND");
    if (m_features.descriptorBindingVariableDescriptorCount) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT");
    if (m_features.runtimeDescriptorArray) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RUNTIME_DESCRIPTOR_ARRAY");
    if (m_features.samplerFilterMinmax) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SAMPLER_FILTER_MINMAX");
    if (m_features.scalarBlockLayout) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SCALAR_BLOCK_LAYOUT");
    if (m_features.uniformBufferStandardLayout) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_UNIFORM_BUFFER_STANDARD_LAYOUT");
    if (m_features.shaderSubgroupExtendedTypes) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SUBGROUP_EXTENDED_TYPES");
    if (m_features.separateDepthStencilLayouts) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SEPARATE_DEPTH_STENCIL_LAYOUTS");
    if (m_features.bufferDeviceAddress) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_BUFFER_DEVICE_ADDRESS");
    if (m_features.bufferDeviceAddressMultiDevice) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_BUFFER_DEVICE_ADDRESS_MULTI_DEVICE");
    if (m_features.vulkanMemoryModel) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_VULKAN_MEMORY_MODEL");
    if (m_features.vulkanMemoryModelDeviceScope) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE");
    if (m_features.vulkanMemoryModelAvailabilityVisibilityChains) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_VULKAN_MEMORY_MODEL_AVAILABILITY_VISIBILITY_CHAINS");
    if (m_features.subgroupBroadcastDynamicId) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SUBGROUP_BROADCAST_DYNAMIC_ID");
    if (m_features.shaderDemoteToHelperInvocation) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_DEMOTE_TO_HELPER_INVOCATION");
    if (m_features.shaderTerminateInvocation) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_TERMINATE_INVOCATION");
    if (m_features.subgroupSizeControl) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SUBGROUP_SIZE_CONTROL");
    if (m_features.computeFullSubgroups) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_COMPUTE_FULL_SUBGROUPS");
    if (m_features.shaderIntegerDotProduct) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_INTEGER_DOT_PRODUCT");
    if (m_features.rasterizationOrderColorAttachmentAccess) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RASTERIZATION_ORDER_COLOR_ATTACHMENT_ACCESS");
    if (m_features.rasterizationOrderDepthAttachmentAccess) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RASTERIZATION_ORDER_DEPTH_ATTACHMENT_ACCESS");
    if (m_features.rasterizationOrderStencilAttachmentAccess) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RASTERIZATION_ORDER_STENCIL_ATTACHMENT_ACCESS");
    if (m_features.fragmentShaderSampleInterlock) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_SHADER_SAMPLE_INTERLOCK");
    if (m_features.fragmentShaderPixelInterlock) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK");
    if (m_features.fragmentShaderShadingRateInterlock) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_SHADER_SHADING_RATE_INTERLOCK");
    if (m_features.indexTypeUint8) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_INDEX_TYPE_UINT8");
    if (m_features.shaderBufferFloat32Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT32_ATOMICS");
    if (m_features.shaderBufferFloat32AtomicAdd) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT32_ATOMIC_ADD");
    if (m_features.shaderBufferFloat64Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT64_ATOMICS");
    if (m_features.shaderBufferFloat64AtomicAdd) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT64_ATOMIC_ADD");
    if (m_features.shaderSharedFloat32Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT32_ATOMICS");
    if (m_features.shaderSharedFloat32AtomicAdd) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT32_ATOMIC_ADD");
    if (m_features.shaderSharedFloat64Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT64_ATOMICS");
    if (m_features.shaderSharedFloat64AtomicAdd) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT64_ATOMIC_ADD");
    if (m_features.shaderImageFloat32Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_IMAGE_FLOAT32_ATOMICS");
    if (m_features.shaderImageFloat32AtomicAdd) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_IMAGE_FLOAT32_ATOMIC_ADD");
    if (m_features.sparseImageFloat32Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SPARSE_IMAGE_FLOAT32_ATOMICS");
    if (m_features.sparseImageFloat32AtomicAdd) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SPARSE_IMAGE_FLOAT32_ATOMIC_ADD");
    if (m_features.shaderBufferFloat16Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT16_ATOMICS");
    if (m_features.shaderBufferFloat16AtomicAdd) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT16_ATOMIC_ADD");
    if (m_features.shaderBufferFloat16AtomicMinMax) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT16_ATOMIC_MIN_MAX");
    if (m_features.shaderBufferFloat32AtomicMinMax) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT32_ATOMIC_MIN_MAX");
    if (m_features.shaderBufferFloat64AtomicMinMax) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_BUFFER_FLOAT64_ATOMIC_MIN_MAX");
    if (m_features.shaderSharedFloat16Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT16_ATOMICS");
    if (m_features.shaderSharedFloat16AtomicAdd) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT16_ATOMIC_ADD");
    if (m_features.shaderSharedFloat16AtomicMinMax) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT16_ATOMIC_MIN_MAX");
    if (m_features.shaderSharedFloat32AtomicMinMax) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT32_ATOMIC_MIN_MAX");
    if (m_features.shaderSharedFloat64AtomicMinMax) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SHARED_FLOAT64_ATOMIC_MIN_MAX");
    if (m_features.shaderImageFloat32AtomicMinMax) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_IMAGE_FLOAT32_ATOMIC_MIN_MAX");
    if (m_features.sparseImageFloat32AtomicMinMax) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SPARSE_IMAGE_FLOAT32_ATOMIC_MIN_MAX");
    if (m_features.shaderImageInt64Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_IMAGE_INT64_ATOMICS");
    if (m_features.sparseImageInt64Atomics) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SPARSE_IMAGE_INT64_ATOMICS");
    if (m_features.accelerationStructure) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_ACCELERATION_STRUCTURE");
    if (m_features.accelerationStructureIndirectBuild) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_ACCELERATION_STRUCTURE_INDIRECT_BUILD");
    if (m_features.accelerationStructureHostCommands) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_ACCELERATION_STRUCTURE_HOST_COMMANDS");
    // if (m_features.descriptorBindingAccelerationStructureUpdateAfterBind) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_ACCELERATION_STRUCTURE_UPDATE_AFTER_BIND"); // shader doesn't need to know about
    if (m_features.rayQuery) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RAY_QUERY");
    if (m_features.rayTracingPipeline) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RAY_TRACING_PIPELINE");
    if (m_features.rayTracingPipelineTraceRaysIndirect) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RAY_TRACING_PIPELINE_TRACE_RAYS_INDIRECT");
    if (m_features.rayTraversalPrimitiveCulling) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RAY_TRAVERSAL_PRIMITIVE_CULLING");
    if (m_features.shaderDeviceClock) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_DEVICE_CLOCK");
    if (m_features.shaderSubgroupUniformControlFlow) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW");
    if (m_features.workgroupMemoryExplicitLayout) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT");
    if (m_features.workgroupMemoryExplicitLayoutScalarBlockLayout) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_SCALAR_BLOCK_LAYOUT");
    if (m_features.workgroupMemoryExplicitLayout8BitAccess) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_8BIT_ACCESS");
    if (m_features.workgroupMemoryExplicitLayout16BitAccess) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_16BIT_ACCESS");
    if (m_features.computeDerivativeGroupQuads) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_COMPUTE_DERIVATIVE_GROUP_QUADS");
    if (m_features.computeDerivativeGroupLinear) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_COMPUTE_DERIVATIVE_GROUP_LINEAR");
    if (m_features.cooperativeMatrix) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_COOPERATIVE_MATRIX");
    if (m_features.cooperativeMatrixRobustBufferAccess) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_COOPERATIVE_MATRIX_ROBUST_BUFFER_ACCESS");
    if (m_features.rayTracingMotionBlur) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RAY_TRACING_MOTION_BLUR");
    if (m_features.rayTracingMotionBlurPipelineTraceRaysIndirect) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RAY_TRACING_MOTION_BLUR_PIPELINE_TRACE_RAYS_INDIRECT");
    if (m_features.coverageReductionMode) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_COVERAGE_REDUCTION_MODE");
    if (m_features.deviceGeneratedCommands) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DEVICE_GENERATED_COMMANDS");
    if (m_features.taskShader) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_TASK_SHADER");
    if (m_features.meshShader) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_MESH_SHADER");
    if (m_features.representativeFragmentTest) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_REPRESENTATIVE_FRAGMENT_TEST");
    if (m_features.mixedAttachmentSamples) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_MIXED_ATTACHMENT_SAMPLES");
    if (m_features.hdrMetadata) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_HDR_METADATA");
    // if (m_features.displayTiming) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DISPLAY_TIMING"); // shader doesn't need to know about
    if (m_features.rasterizationOrder) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RASTERIZATION_ORDER");
    if (m_features.shaderExplicitVertexParameter) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_EXPLICIT_VERTEX_PARAMETER");
    if (m_features.shaderInfoAMD) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SHADER_INFO_AMD");
    if (m_features.hostQueryReset) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_HOST_QUERY_RESET");
    // if (m_features.pipelineCreationCacheControl) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_PIPELINE_CREATION_CACHE_CONTROL"); // shader doesn't need to know about
    if (m_features.colorWriteEnable) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_COLOR_WRITE_ENABLE");
    if (m_features.conditionalRendering) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_CONDITIONAL_RENDERING");
    if (m_features.inheritedConditionalRendering) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_INHERITED_CONDITIONAL_RENDERING");
    // if (m_features.deviceMemoryReport) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DEVICE_MEMORY_REPORT"); // shader doesn't need to know about
    if (m_features.fragmentDensityMap) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_DENSITY_MAP");
    if (m_features.fragmentDensityMapDynamic) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_DENSITY_MAP_DYNAMIC");
    if (m_features.fragmentDensityMapNonSubsampledImages) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_DENSITY_MAP_NON_SUBSAMPLED_IMAGES");
    if (m_features.fragmentDensityMapDeferred) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_FRAGMENT_DENSITY_MAP_DEFERRED");
    if (m_features.robustImageAccess) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_ROBUST_IMAGE_ACCESS");
    if (m_features.inlineUniformBlock) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_INLINE_UNIFORM_BLOCK");
    // if (m_features.descriptorBindingInlineUniformBlockUpdateAfterBind) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DESCRIPTOR_BINDING_INLINE_UNIFORM_BLOCK_UPDATE_AFTER_BIND"); // shader doesn't need to know about
    if (m_features.rectangularLines) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_RECTANGULAR_LINES");
    if (m_features.bresenhamLines) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_BRESENHAM_LINES");
    if (m_features.smoothLines) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_SMOOTH_LINES");
    if (m_features.stippledRectangularLines) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_STIPPLED_RECTANGULAR_LINES");
    if (m_features.stippledBresenhamLines) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_STIPPLED_BRESENHAM_LINES");
    if (m_features.stippledSmoothLines) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_STIPPLED_SMOOTH_LINES");
    // if (m_features.memoryPriority) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_MEMORY_PRIORITY"); // shader doesn't need to know about
    if (m_features.robustBufferAccess2) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_ROBUST_BUFFER_ACCESS_2");
    if (m_features.robustImageAccess2) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_ROBUST_IMAGE_ACCESS_2");
    if (m_features.nullDescriptor) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_NULL_DESCRIPTOR");
    if (m_features.performanceCounterQueryPools) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_PERFORMANCE_COUNTER_QUERY_POOLS");
    if (m_features.performanceCounterMultipleQueryPools) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_PERFORMANCE_COUNTER_MULTIPLE_QUERY_POOLS");
    if (m_features.pipelineExecutableInfo) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_PIPELINE_EXECUTABLE_INFO");
    // if (m_features.maintenance4) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_MAINTENANCE_4"); // shader doesn't need to know about
    if (m_features.deviceCoherentMemory) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_DEVICE_COHERENT_MEMORY");
    // if (m_features.bufferMarkerAMD) addGLSLDefineToPool(pool, "NBL_GLSL_FEATURE_BUFFER_MARKER_AMD"); // shader doesn't need to know about

    // TODO: @achal test examples 14 and 48 on all APIs and GPUs

    if (runningInRenderdoc)
        addGLSLDefineToPool(pool,"NBL_GLSL_RUNNING_IN_RENDERDOC");
}

bool IPhysicalDevice::validateLogicalDeviceCreation(const ILogicalDevice::SCreationParams& params) const
{
    using range_t = core::SRange<const ILogicalDevice::SQueueCreationParams>;
    range_t qcis(params.queueParams, params.queueParams+params.queueParamsCount);

    for (const auto& qci : qcis)
    {
        if (qci.familyIndex >= m_qfamProperties->size())
            return false;

        const auto& qfam = (*m_qfamProperties)[qci.familyIndex];
        if (qci.count == 0u)
            return false;
        if (qci.count > qfam.queueCount)
            return false;

        for (uint32_t i = 0u; i < qci.count; ++i)
        {
            const float priority = qci.priorities[i];
            if (priority < 0.f)
                return false;
            if (priority > 1.f)
                return false;
        }
    }
    
    if(!params.featuresToEnable.isSubsetOf(m_features))
        return false; // Requested features are not all supported by physical device

    return true;
}

inline core::bitflag<asset::IImage::E_ASPECT_FLAGS> getImageAspects(asset::E_FORMAT _fmt)
{
    core::bitflag<asset::IImage::E_ASPECT_FLAGS> flags;
    bool depthOrStencil = asset::isDepthOrStencilFormat(_fmt);
    bool stencilOnly = asset::isStencilOnlyFormat(_fmt);
    bool depthOnly = asset::isDepthOnlyFormat(_fmt);
    if (depthOrStencil || depthOnly) flags |= asset::IImage::EAF_DEPTH_BIT;
    if (depthOrStencil || stencilOnly) flags |= asset::IImage::EAF_STENCIL_BIT;
    if (!depthOrStencil && !stencilOnly && !depthOnly) flags |= asset::IImage::EAF_COLOR_BIT;

    return flags;
}

// Assumes no loss of precision due to block compression, only the endpoints
float getBcFormatMaxPrecision(asset::E_FORMAT format, uint32_t channel)
{
    if (channel == 3u)
    {
        switch (format)
        {
        // BC2 has 4 bit alpha
        case asset::EF_BC2_UNORM_BLOCK:
        case asset::EF_BC2_SRGB_BLOCK:
            return 1.f / 15.f;
        // BC3, BC7 and all ASTC formats have up to 8 bit alpha
        case asset::EF_BC3_UNORM_BLOCK:
        case asset::EF_BC3_SRGB_BLOCK:
        case asset::EF_BC7_UNORM_BLOCK:
        case asset::EF_BC7_SRGB_BLOCK:
        case asset::EF_ASTC_4x4_UNORM_BLOCK:
        case asset::EF_ASTC_4x4_SRGB_BLOCK:
        case asset::EF_ASTC_5x4_UNORM_BLOCK:
        case asset::EF_ASTC_5x4_SRGB_BLOCK:
        case asset::EF_ASTC_5x5_UNORM_BLOCK:
        case asset::EF_ASTC_5x5_SRGB_BLOCK:
        case asset::EF_ASTC_6x5_UNORM_BLOCK:
        case asset::EF_ASTC_6x5_SRGB_BLOCK:
        case asset::EF_ASTC_6x6_UNORM_BLOCK:
        case asset::EF_ASTC_6x6_SRGB_BLOCK:
        case asset::EF_ASTC_8x5_UNORM_BLOCK:
        case asset::EF_ASTC_8x5_SRGB_BLOCK:
        case asset::EF_ASTC_8x6_UNORM_BLOCK:
        case asset::EF_ASTC_8x6_SRGB_BLOCK:
        case asset::EF_ASTC_8x8_UNORM_BLOCK:
        case asset::EF_ASTC_8x8_SRGB_BLOCK:
        case asset::EF_ASTC_10x5_UNORM_BLOCK:
        case asset::EF_ASTC_10x5_SRGB_BLOCK:
        case asset::EF_ASTC_10x6_UNORM_BLOCK:
        case asset::EF_ASTC_10x6_SRGB_BLOCK:
        case asset::EF_ASTC_10x8_UNORM_BLOCK:
        case asset::EF_ASTC_10x8_SRGB_BLOCK:
        case asset::EF_ASTC_10x10_UNORM_BLOCK:
        case asset::EF_ASTC_10x10_SRGB_BLOCK:
        case asset::EF_ASTC_12x10_UNORM_BLOCK:
        case asset::EF_ASTC_12x10_SRGB_BLOCK:
        case asset::EF_ASTC_12x12_UNORM_BLOCK:
        case asset::EF_ASTC_12x12_SRGB_BLOCK:
            return 1.0 / 255.0;

        // Otherwise, assume binary (1 bit) alpha
        default:
            return 1.f;
        }
    }

    float rcpUnit = 0.0;
    switch (format)
    {
    case asset::EF_BC1_RGB_UNORM_BLOCK:
    case asset::EF_BC1_RGB_SRGB_BLOCK:
    case asset::EF_BC1_RGBA_UNORM_BLOCK:
    case asset::EF_BC1_RGBA_SRGB_BLOCK:
    case asset::EF_BC2_UNORM_BLOCK:
    case asset::EF_BC2_SRGB_BLOCK:
    case asset::EF_BC3_UNORM_BLOCK:
    case asset::EF_BC3_SRGB_BLOCK:
        // The color channels for BC1, BC2 & BC3 are RGB565
        rcpUnit = (channel == 1u) ? (1.0 / 63.0) : (1.0 / 31.0);
        // Weights also allow for more precision. These formats have 2 bit weights
        rcpUnit *= 1.0 / 3.0;
        break;
    case asset::EF_BC4_UNORM_BLOCK:
    case asset::EF_BC4_SNORM_BLOCK:
    case asset::EF_BC5_UNORM_BLOCK:
    case asset::EF_BC5_SNORM_BLOCK:
    case asset::EF_BC7_UNORM_BLOCK:
    case asset::EF_BC7_SRGB_BLOCK:
        rcpUnit = 1.0 / 255.0;
        break;
    case asset::EF_ASTC_4x4_UNORM_BLOCK:
    case asset::EF_ASTC_4x4_SRGB_BLOCK:
    case asset::EF_ASTC_5x4_UNORM_BLOCK:
    case asset::EF_ASTC_5x4_SRGB_BLOCK:
    case asset::EF_ASTC_5x5_UNORM_BLOCK:
    case asset::EF_ASTC_5x5_SRGB_BLOCK:
    case asset::EF_ASTC_6x5_UNORM_BLOCK:
    case asset::EF_ASTC_6x5_SRGB_BLOCK:
    case asset::EF_ASTC_6x6_UNORM_BLOCK:
    case asset::EF_ASTC_6x6_SRGB_BLOCK:
    case asset::EF_ASTC_8x5_UNORM_BLOCK:
    case asset::EF_ASTC_8x5_SRGB_BLOCK:
    case asset::EF_ASTC_8x6_UNORM_BLOCK:
    case asset::EF_ASTC_8x6_SRGB_BLOCK:
    case asset::EF_ASTC_8x8_UNORM_BLOCK:
    case asset::EF_ASTC_8x8_SRGB_BLOCK:
    case asset::EF_ASTC_10x5_UNORM_BLOCK:
    case asset::EF_ASTC_10x5_SRGB_BLOCK:
    case asset::EF_ASTC_10x6_UNORM_BLOCK:
    case asset::EF_ASTC_10x6_SRGB_BLOCK:
    case asset::EF_ASTC_10x8_UNORM_BLOCK:
    case asset::EF_ASTC_10x8_SRGB_BLOCK:
    case asset::EF_ASTC_10x10_UNORM_BLOCK:
    case asset::EF_ASTC_10x10_SRGB_BLOCK:
    case asset::EF_ASTC_12x10_UNORM_BLOCK:
    case asset::EF_ASTC_12x10_SRGB_BLOCK:
    case asset::EF_ASTC_12x12_UNORM_BLOCK:
    case asset::EF_ASTC_12x12_SRGB_BLOCK:
        // (All of these could be using HDR. Take extra flag to assume FP16 precision?)
        rcpUnit = 1.0 / 255.0;
        break;
    case asset::EF_EAC_R11_UNORM_BLOCK:
    case asset::EF_EAC_R11_SNORM_BLOCK:
    case asset::EF_EAC_R11G11_UNORM_BLOCK:
    case asset::EF_EAC_R11G11_SNORM_BLOCK:
        rcpUnit = 1.0 / 2047.0; 
        break;
    case asset::EF_ETC2_R8G8B8_UNORM_BLOCK:
    case asset::EF_ETC2_R8G8B8_SRGB_BLOCK:
    case asset::EF_ETC2_R8G8B8A1_UNORM_BLOCK:
    case asset::EF_ETC2_R8G8B8A1_SRGB_BLOCK:
    case asset::EF_ETC2_R8G8B8A8_UNORM_BLOCK:
    case asset::EF_ETC2_R8G8B8A8_SRGB_BLOCK:
        rcpUnit = 1.0 / 31.0;
        break;
    case asset::EF_BC6H_UFLOAT_BLOCK:
    case asset::EF_BC6H_SFLOAT_BLOCK:
    {
        // BC6 isn't really FP16, so this is an over-estimation
        return core::Float16Compressor::decompress(1) - 0.0;
    }
    case asset::EF_PVRTC1_2BPP_UNORM_BLOCK_IMG:
    case asset::EF_PVRTC1_4BPP_UNORM_BLOCK_IMG:
    case asset::EF_PVRTC2_2BPP_UNORM_BLOCK_IMG:
    case asset::EF_PVRTC2_4BPP_UNORM_BLOCK_IMG:
    case asset::EF_PVRTC1_2BPP_SRGB_BLOCK_IMG:
    case asset::EF_PVRTC1_4BPP_SRGB_BLOCK_IMG:
    case asset::EF_PVRTC2_2BPP_SRGB_BLOCK_IMG:
    case asset::EF_PVRTC2_4BPP_SRGB_BLOCK_IMG:
        // TODO: Use proper metrics here instead of assuming full 8 bit
        return 1.0 / 255.0;
    }

    if (isSRGBFormat(format))
    {
        return core::srgb2lin(0.0 + rcpUnit) - core::srgb2lin(0.0);
    }

    return rcpUnit;
}

double getFormatPrecisionAt(asset::E_FORMAT format, uint32_t channel, double value)
{
    if (asset::isBlockCompressionFormat(format))
        return getBcFormatMaxPrecision(format, channel);
    switch (format)
    {
    case asset::EF_E5B9G9R9_UFLOAT_PACK32:
    {
        // Minimum precision value would be a 9bit mantissa & 5bit exponent float
        // (This ignores the shared exponent)
        int bitshft = 2;

        uint16_t f16 = core::Float16Compressor::compress(value);
        uint16_t enc = f16 >> bitshft;
        uint16_t next_f16 = (enc + 1) << bitshft;

        return core::Float16Compressor::decompress(next_f16) - value;
    }
    default: return asset::getFormatPrecision(format, channel, value);
    }
}

// Returns true if 'a' is not equal to 'b' and can be promoted FROM 'b'
bool canPromoteFormat(asset::E_FORMAT a, asset::E_FORMAT b, bool srcSignedFormat, bool srcIntFormat, uint32_t srcChannels, double srcMin[], double srcMax[])
{
    // The value itself should already have been checked to not be valid before calling this
    if (a == b)
        return false;
    // Can't transcode to BC or planar
    if (asset::isBlockCompressionFormat(a))
        return false;
    if (asset::isPlanarFormat(a))
        return false;
    // Can't promote between int and normalized/float/scaled formats
    if (asset::isIntegerFormat(a) != srcIntFormat)
        return false;
    // Can't promote between signed and unsigned formats in integers
    // (this causes a different sampler type to be necessary in the shader)
    if (srcIntFormat && asset::isSignedFormat(a) != srcSignedFormat)
        return false;
    // Can't have less channels
    if (asset::getFormatChannelCount(a) < srcChannels)
        return false;

    // Can't have less precision or value range
    for (uint32_t c = 0; c < srcChannels; c++)
    {
        double mina = asset::getFormatMinValue<double>(a, c),
            minb = asset::getFormatMinValue<double>(b, c),
            maxa = asset::getFormatMaxValue<double>(a, c),
            maxb = asset::getFormatMaxValue<double>(b, c);

        // return false if a has less precision (higher precision delta) than b
        // check at 0, since precision is non-increasing
        // also check at min & max, since there's potential for cross-over with constant formats
        if (getFormatPrecisionAt(a, c, 0.0) > getFormatPrecisionAt(b, c, 0.0)
                || getFormatPrecisionAt(a, c, srcMin[c]) > getFormatPrecisionAt(b, c, srcMin[c])
                || getFormatPrecisionAt(a, c, srcMax[c]) > getFormatPrecisionAt(b, c, srcMax[c]))
            return false;
        // return false if a has less range than b
        if (mina > minb || maxa < maxb)
            return false;
    }
    return true;
}

double getFormatPrecisionMaxDt(asset::E_FORMAT f, uint32_t c, double srcMin, double srcMax)
{
    return std::max(std::max(getFormatPrecisionAt(f, c, 0.0), getFormatPrecisionAt(f, c, srcMin)), getFormatPrecisionAt(f, c, srcMax));
}

// Returns true if 'a' is a better fit than 'b' (for tie breaking)
// Tie-breaking rules:
// - RGBA vs BGRA matches srcFormat
// - Maximum precision delta is smaller
// - Value range is larger
bool isFormatBetterFit(asset::E_FORMAT a, asset::E_FORMAT b, bool srcBgra, uint32_t srcChannels, double srcMin[], double srcMax[])
{
    assert(asset::getTexelOrBlockBytesize(a) == asset::getTexelOrBlockBytesize(b));
    bool curBgraMatch = asset::isBGRALayoutFormat(a) == srcBgra;
    bool prevBgraMatch = asset::isBGRALayoutFormat(b) == srcBgra;

    // if one of the two fits the original bgra better, use that
    if (curBgraMatch != prevBgraMatch)
        return curBgraMatch;

    // Check precision deltas
    double precisionDeltasA[4];
    double precisionDeltasB[4];
    for (uint32_t c = 0; c < srcChannels; c++)
    {
        // View comments above about value selection
        // Pick the max precision delta for each format
        precisionDeltasA[c] = getFormatPrecisionMaxDt(a, c, srcMin[c], srcMax[c]);
        precisionDeltasB[c] = getFormatPrecisionMaxDt(b, c, srcMin[c], srcMax[c]);

        // if one of the two has a better max precision delta, use that
        if (precisionDeltasA[c] != precisionDeltasB[c])
            return precisionDeltasA[c] < precisionDeltasB[c];
    }

    // Check difference in quantifiable values within the ranges for a and b
    double wasteA = 0.0;
    double wasteB = 0.0;
    for (uint32_t c = 0; c < srcChannels; c++)
    {
        double mina = asset::getFormatMinValue<double>(a, c),
            minb = asset::getFormatMinValue<double>(b, c),
            maxa = asset::getFormatMaxValue<double>(a, c),
            maxb = asset::getFormatMaxValue<double>(b, c);
        assert(mina <= srcMin[c] && maxa >= srcMax[c] &&
            minb <= srcMin[c] && maxb >= srcMax[c]);

        wasteA += (srcMin[c] - mina) / precisionDeltasA[c];
        wasteA += (maxa - srcMax[c]) / precisionDeltasA[c];

        wasteB += (srcMin[c] - minb) / precisionDeltasB[c];
        wasteB += (maxb - srcMax[c]) / precisionDeltasB[c];
    }

    // if one of the two has less "waste" of quantifiable values, use that
    if (wasteA != wasteB)
        return wasteA < wasteB;

    return false;
}

// Rules for promotion:
// - Cannot convert to block or planar format
// - Aspects: Preserved or added
// - Channel count: Preserved or increased
// - Data range: Preserved or increased (per channel)
// - Data precision: Preserved or improved (per channel)
//     - Bit depth when comparing non srgb
// If there are multiple matches: Pick smallest texel block
// srcFormat can't be in validFormats (no promotion should be made if the format itself is valid)
asset::E_FORMAT narrowDownFormatPromotion(const core::unordered_set<asset::E_FORMAT>& validFormats, asset::E_FORMAT srcFormat)
{
    if (validFormats.empty()) return asset::EF_UNKNOWN;

    asset::E_FORMAT smallestTexelBlock = asset::EF_UNKNOWN;
    uint32_t smallestTexelBlockSize = -1;

    bool srcBgra = asset::isBGRALayoutFormat(srcFormat);
    auto srcChannels = asset::getFormatChannelCount(srcFormat);
    double srcMinVal[4];
    double srcMaxVal[4];
    for (uint32_t channel = 0; channel < srcChannels; channel++)
    {
        srcMinVal[channel] = asset::getFormatMinValue<double>(srcFormat, channel);
        srcMaxVal[channel] = asset::getFormatMaxValue<double>(srcFormat, channel);
    }

    for (auto iter = validFormats.begin(); iter != validFormats.end(); iter++)
    {
        asset::E_FORMAT f = *iter;

        uint32_t texelBlockSize = asset::getTexelOrBlockBytesize(f);
        // Don't promote if we have a better valid format already
        if (texelBlockSize > smallestTexelBlockSize) {
            continue;
        }

        if (texelBlockSize == smallestTexelBlockSize)
        {
            if (!isFormatBetterFit(f, smallestTexelBlock, srcBgra, srcChannels, srcMinVal, srcMaxVal))
                continue;
        }

        smallestTexelBlockSize = texelBlockSize;
        smallestTexelBlock = f;
    }

    assert(smallestTexelBlock != asset::EF_UNKNOWN);
    return smallestTexelBlock;
}

asset::E_FORMAT IPhysicalDevice::promoteBufferFormat(const SBufferFormatPromotionRequest req)
{
    assert(
        !asset::isBlockCompressionFormat(req.originalFormat) &&
        !asset::isPlanarFormat(req.originalFormat) &&
        getImageAspects(req.originalFormat).hasFlags(asset::IImage::EAF_COLOR_BIT)
    );
    auto& buf_cache = this->m_formatPromotionCache.buffers;
    auto cached = buf_cache.find(req);
    if (cached != buf_cache.end())
        return cached->second;

    if (req.usages < getBufferFormatUsages(req.originalFormat))
    {
        buf_cache.insert(cached, { req,req.originalFormat });
        return req.originalFormat;
    }

    auto srcFormat = req.originalFormat;
    bool srcIntFormat = asset::isIntegerFormat(srcFormat);
    bool srcSignedFormat = asset::isSignedFormat(srcFormat);
    auto srcChannels = asset::getFormatChannelCount(srcFormat);
    double srcMinVal[4];
    double srcMaxVal[4];
    for (uint32_t channel = 0; channel < srcChannels; channel++)
    {
        srcMinVal[channel] = asset::getFormatMinValue<double>(srcFormat, channel);
        srcMaxVal[channel] = asset::getFormatMaxValue<double>(srcFormat, channel);
    }

    // Cache valid formats per usage?
    core::unordered_set<asset::E_FORMAT> validFormats;

    for (uint32_t format = 0; format < asset::E_FORMAT::EF_UNKNOWN; format++)
    {
        auto f = static_cast<asset::E_FORMAT>(format);
        // Can't have less aspects
        if (!getImageAspects(f).hasFlags(asset::IImage::EAF_COLOR_BIT))
            continue;

        if (!canPromoteFormat(f, srcFormat, srcSignedFormat, srcIntFormat, srcChannels, srcMinVal, srcMaxVal))
            continue;

        if (req.usages < getBufferFormatUsages(f))
        {
            validFormats.insert(f);
        }
    }

    auto promoted = narrowDownFormatPromotion(validFormats, req.originalFormat);
    buf_cache.insert(cached, { req,promoted });
    return promoted;
}

asset::E_FORMAT IPhysicalDevice::promoteImageFormat(const SImageFormatPromotionRequest req, const IGPUImage::E_TILING tiling)
{
    format_image_cache_t& cache = tiling == IGPUImage::E_TILING::ET_LINEAR 
        ? this->m_formatPromotionCache.linearTilingImages 
        : this->m_formatPromotionCache.optimalTilingImages;
    auto cached = cache.find(req);
    if (cached != cache.end())
        return cached->second;

    auto getImageFormatUsagesTiling = [&](asset::E_FORMAT f)
    {
        switch (tiling)
        {
            case IGPUImage::E_TILING::ET_LINEAR:
                return getImageFormatUsagesLinear(f);
            case IGPUImage::E_TILING::ET_OPTIMAL:
                return getImageFormatUsagesOptimal(f);
            default:
                assert(false); // Invalid tiling
        }
        return SFormatImageUsages::SUsage{}; // compiler please shut up
    };

    if (req.usages < getImageFormatUsagesTiling(req.originalFormat))
    {
        cache.insert(cached, { req,req.originalFormat });
        return req.originalFormat;
    }

    auto srcFormat = req.originalFormat;
    auto srcAspects = getImageAspects(srcFormat);
    bool srcIntFormat = asset::isIntegerFormat(srcFormat);
    bool srcSignedFormat = asset::isSignedFormat(srcFormat);
    auto srcChannels = asset::getFormatChannelCount(srcFormat);
    double srcMinVal[4];
    double srcMaxVal[4];
    for (uint32_t channel = 0; channel < srcChannels; channel++)
    {
        srcMinVal[channel] = asset::getFormatMinValue<double>(srcFormat, channel);
        srcMaxVal[channel] = asset::getFormatMaxValue<double>(srcFormat, channel);
    }

    // Cache valid formats per usage?
    core::unordered_set<asset::E_FORMAT> validFormats;

    for (uint32_t format = 0; format < asset::E_FORMAT::EF_UNKNOWN; format++)
    {
        auto f = static_cast<asset::E_FORMAT>(format);
        // Can't have less aspects
        if (!getImageAspects(f).hasFlags(srcAspects))
            continue;

        if (!canPromoteFormat(f, srcFormat, srcSignedFormat, srcIntFormat, srcChannels, srcMinVal, srcMaxVal))
            continue;

        if (req.usages < getImageFormatUsagesTiling(f))
        {
            validFormats.insert(f);
        }
    }


    auto promoted = narrowDownFormatPromotion(validFormats, req.originalFormat);
    cache.insert(cached, { req,promoted });
    return promoted;
}

}
