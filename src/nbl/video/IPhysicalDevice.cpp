#include "nbl/video/IPhysicalDevice.h"

namespace nbl::video
{

IPhysicalDevice::IPhysicalDevice(core::smart_refctd_ptr<system::ISystem>&& s, core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc) :
    m_system(std::move(s)), m_GLSLCompiler(std::move(glslc))
{
    memset(&m_memoryProperties, 0, sizeof(SMemoryProperties));
    memset(&m_linearTilingUsages, 0, sizeof(SFormatImageUsage));
    memset(&m_optimalTilingUsages, 0, sizeof(SFormatImageUsage));
    memset(&m_bufferUsages, 0, sizeof(SFormatBufferUsage));
}

void IPhysicalDevice::addCommonGLSLDefines(std::ostringstream& pool, const bool runningInRenderdoc)
{
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
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLER_ALLOCATION_COUNT",m_properties.limits.maxSamplerAllocationCount);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_BUFFER_IMAGE_GRANULARITY",m_properties.limits.bufferImageGranularity);

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
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_WORKGROUP_COUNT",m_properties.limits.maxComputeWorkGroupCount[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_WORKGROUP_COUNT",m_properties.limits.maxComputeWorkGroupCount[1]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_WORKGROUP_COUNT",m_properties.limits.maxComputeWorkGroupCount[2]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMPUTE_WORKGROUP_INVOCATIONS",m_properties.limits.maxComputeWorkGroupInvocations);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_WORKGROUP_SIZE_X",m_properties.limits.maxWorkgroupSize[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_WORKGROUP_SIZE_Y",m_properties.limits.maxWorkgroupSize[1]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_WORKGROUP_SIZE_Z",m_properties.limits.maxWorkgroupSize[2]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SUB_PIXEL_PRECISION_BITS",m_properties.limits.subPixelPrecisionBits);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DRAW_INDIRECT_COUNT",m_properties.limits.maxDrawIndirectCount);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLER_LOD_BIAS",m_properties.limits.maxSamplerLodBias);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLER_ANISOTROPY_LOG2",m_properties.limits.maxSamplerAnisotropyLog2);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VIEWPORTS",m_properties.limits.maxViewports);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VIEWPORT_WIDTH",m_properties.limits.maxViewportDims[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VIEWPORT_HEIGHT",m_properties.limits.maxViewportDims[1]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_VIEWPORT_BOUNDS_RANGE",m_properties.limits.viewportBoundsRange[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_VIEWPORT_BOUNDS_RANGE",m_properties.limits.viewportBoundsRange[1]);
    
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_VIEWPORT_SUB_PIXEL-BITS",m_properties.limits.viewportSubPixelBits);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_MEMORY_MAP_ALIGNMENT",m_properties.limits.minMemoryMapAlignment);
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
    addGLSLDefineToPool(pool,"NBL_GLSL_FRAMEBUFFER_COLOR_SAMPLE_COUNTS",m_properties.limits.framebufferColorSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_FRAMEBUFFER_DEPTH_SAMPLE_COUNTS",m_properties.limits.framebufferDepthSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_FRAMEBUFFER_STENCIL_SAMPLE_COUNTS",m_properties.limits.framebufferStencilSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_FRAMEBUFFER_NO_ATTACHMENTS_SAMPLE_COUNTS",m_properties.limits.framebufferNoAttachmentsSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COLOR_ATTACHMENTS",m_properties.limits.maxColorAttachments);
    addGLSLDefineToPool(pool,"NBL_GLSL_NBL_GLSL_SAMPLED_IMAGE_COLOR_SAMPLE_COUNTS",m_properties.limits.sampledImageColorSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_NBL_GLSL_SAMPLED_IMAGE_INTEGER_SAMPLE_COUNTS",m_properties.limits.sampledImageIntegerSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_NBL_GLSL_SAMPLED_IMAGE_DEPTH_SAMPLE_COUNTS",m_properties.limits.sampledImageDepthSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_NBL_GLSL_SAMPLED_IMAGE_STENCIL_SAMPLE_COUNTS",m_properties.limits.sampledImageStencilSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_NBL_GLSL_STORAGE_IMAGE_SAMPLE_COUNTS",m_properties.limits.storageImageSampleCounts.value);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SAMPLE_MASK_WORDS",m_properties.limits.maxSampleMaskWords);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_TIMESTAMP_COMPUTE_AND_GRAPHICS",m_properties.limits.timestampComputeAndGraphics);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_TIMESTAMP_PERIOD_IN_NANO_SECONDS",m_properties.limits.timestampPeriodInNanoSeconds);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_CLIP_DISTANCES",m_properties.limits.maxClipDistances);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_CULL_DISTANCES",m_properties.limits.maxCullDistances);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_COMBINED_CLIP_AND_CULL_DISTANCES",m_properties.limits.maxCombinedClipAndCullDistances);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_DISCRETE_QUEUE_PRIORITIES",m_properties.limits.discreteQueuePriorities);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_POINT_SIZE",m_properties.limits.pointSizeRange[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_POINT_SIZE",m_properties.limits.pointSizeRange[1]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_LINE_WIDTH",m_properties.limits.lineWidthRange[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_LINE_WIDTH",m_properties.limits.lineWidthRange[1]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_POINT_SIZE_GRANULARITY",m_properties.limits.pointSizeGranularity);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_LINE_WIDTH_GRANULARITY",m_properties.limits.lineWidthGranularity);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_STRICT_LINES",m_properties.limits.strictLines);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_STANDARD_SAMPLE_LOCATIONS",m_properties.limits.standardSampleLocations);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_OPTIMAL_BUFFER_COPY_OFFSET_ALIGNMENT",m_properties.limits.optimalBufferCopyOffsetAlignment);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_OPTIMAL_BUFFER_COPY_ROW_PITCH_ALIGNMENT",m_properties.limits.optimalBufferCopyRowPitchAlignment);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_NON_COHERENT_ATOM_SIZE",m_properties.limits.nonCoherentAtomSize);

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

    // E_POINT_CLIPPING_BEHAVIOR pointClippingBehavior = EPCB_USER_CLIP_PLANES_ONLY;
    
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_SET_DESCRIPTORS",m_properties.limits.maxPerSetDescriptors);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_MEMORY_ALLOCATION_SIZE",m_properties.limits.maxMemoryAllocationSize);

    // E_TRI_BOOLEAN shaderSignedZeroInfNanPreserveFloat16 = ETB_DONT_KNOW;
    // E_TRI_BOOLEAN shaderSignedZeroInfNanPreserveFloat32 = ETB_DONT_KNOW;
    // E_TRI_BOOLEAN shaderSignedZeroInfNanPreserveFloat64 = ETB_DONT_KNOW;
    // E_TRI_BOOLEAN shaderDenormPreserveFloat16 = ETB_DONT_KNOW;
    // E_TRI_BOOLEAN shaderDenormPreserveFloat32 = ETB_DONT_KNOW;
    // E_TRI_BOOLEAN shaderDenormPreserveFloat64 = ETB_DONT_KNOW;
    // E_TRI_BOOLEAN shaderDenormFlushToZeroFloat16 = ETB_DONT_KNOW;
    // E_TRI_BOOLEAN shaderDenormFlushToZeroFloat32 = ETB_DONT_KNOW;
    // E_TRI_BOOLEAN shaderDenormFlushToZeroFloat64 = ETB_DONT_KNOW;
    // E_TRI_BOOLEAN shaderRoundingModeRTEFloat16 = ETB_DONT_KNOW;
    // E_TRI_BOOLEAN shaderRoundingModeRTEFloat32 = ETB_DONT_KNOW;
    // E_TRI_BOOLEAN shaderRoundingModeRTEFloat64 = ETB_DONT_KNOW;
    // E_TRI_BOOLEAN shaderRoundingModeRTZFloat16 = ETB_DONT_KNOW;
    // E_TRI_BOOLEAN shaderRoundingModeRTZFloat32 = ETB_DONT_KNOW;
    // E_TRI_BOOLEAN shaderRoundingModeRTZFloat64 = ETB_DONT_KNOW;

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

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_BUFFER_SIZE",core::min(m_properties.limits.maxBufferSize, ~0u));

    if (m_properties.limits.primitiveOverestimationSize) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_PRIMITIVE_OVERESTIMATION_SIZE");
    if (m_properties.limits.maxExtraPrimitiveOverestimationSize) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_EXTRA_PRIMITIVE_OVERESTIMATION_SIZE");
    if (m_properties.limits.extraPrimitiveOverestimationSizeGranularity) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_EXTRA_PRIMITIVE_OVERESTIMATION_SIZE_GRANULARITY");
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_PRIMITIVE_UNDERESTIMATION",m_properties.limits.primitiveUnderestimation);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_CONSERVATIVE_POINT_AND_LINE_RASTERIZATION",m_properties.limits.conservativePointAndLineRasterization);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_DEGENERATE_TRIANGLES_RASTERIZED",m_properties.limits.degenerateTrianglesRasterized);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_DEGENERATE_LINES_RASTERIZED",m_properties.limits.degenerateLinesRasterized);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FULLY_COVERED_FRAGMENT_SHADER_INPUT_VARIABLE",m_properties.limits.fullyCoveredFragmentShaderInputVariable);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_CONSERVATIVE_RASTERIZATION_POST_DEPTH_COVERAGE",m_properties.limits.conservativeRasterizationPostDepthCoverage);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DISCARD_RECTANGLES",m_properties.limits.maxDiscardRectangles);
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

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_GEOMETRY_COUNT",m_properties.limits.maxGeometryCount);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_INSTANCE_COUNT",m_properties.limits.maxInstanceCount);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PRIMITIVE_COUNT",m_properties.limits.maxPrimitiveCount);
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

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_IMPORTED_HOST_POINTER_ALIGNMENT",m_properties.limits.minImportedHostPointerAlignment);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_FRAGMENT_DENSITY_TEXEL_SIZE_X",m_properties.limits.minFragmentDensityTexelSize.width);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_FRAGMENT_DENSITY_TEXEL_SIZE_Y",m_properties.limits.minFragmentDensityTexelSize.height);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_FRAGMENT_DENSITY_TEXEL_SIZE_X",m_properties.limits.maxFragmentDensityTexelSize.width);
    addGLSLDefineToPool(pool, "NBL_GLSL_LIMIT_MAX_FRAGMENT_DENSITY_TEXEL_SIZE_Y",m_properties.limits.maxFragmentDensityTexelSize.height);
    if (m_properties.limits.fragmentDensityInvocations) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FRAGMENT_DENSITY_INVOCATIONS");

    if (m_properties.limits.subsampledLoads) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SUBSAMPLED_LOADS");
    if (m_properties.limits.subsampledCoarseReconstructionEarlyAccess) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SUBSAMPLED_COARSE_RECONSTRUCTION_EARLY_ACCESS");
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SUBSAMPLED_ARRAY_LAYERS",m_properties.limits.maxSubsampledArrayLayers);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_DESCRIPTOR_SET_SUBSAMPLED_SAMPLERS",m_properties.limits.maxDescriptorSetSubsampledSamplers);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_PCI_DOMAN",m_properties.limits.pciDomain);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_PCI_BUS",m_properties.limits.pciBus);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_PCI_DEVICE",m_properties.limits.pciDevice);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_PCI_FUNCTION",m_properties.limits.pciFunction);

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_GROUP_HANDLE_SIZE",m_properties.limits.shaderGroupHandleSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_RAY_RECURSION_DEPTH",m_properties.limits.maxRayRecursionDepth);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SHADER_GROUP_STRIDE",m_properties.limits.maxShaderGroupStride);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_GROUP_BASE_ALIGNMENT",m_properties.limits.shaderGroupBaseAlignment);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_SIZE",m_properties.limits.shaderGroupHandleCaptureReplaySize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_RAY_DISPATCH_INVOCATION_COUNT",m_properties.limits.maxRayDispatchInvocationCount);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_GROUP_HANDLE_ALIGNMENT",m_properties.limits.shaderGroupHandleAlignment);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_RAY_HIT_ATTRIBUTE_SIZE",m_properties.limits.maxRayHitAttributeSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_COOPERATIVE_MATRIX_SUPPORTED_STAGES",m_properties.limits.cooperativeMatrixSupportedStages.value);
  
    if (m_properties.limits.shaderOutputViewportIndex) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_OUTPUT_VIEWPORT_INDEX");
    if (m_properties.limits.shaderOutputLayer) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_OUTPUT_LAYER");
    if (m_properties.limits.shaderIntegerFunctions2) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_INTEGER_FUNCTIONS_2");
    if (m_properties.limits.shaderSubgroupClock) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_CLOCK");
    if (m_properties.limits.imageFootprint) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_IMAGE_FOOTPRINT");
    if (m_properties.limits.texelBufferAlignment) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_TEXEL_BUFFER_ALIGNMENT");
    if (m_properties.limits.shaderSMBuiltins) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_S_M_BUILTINS");
    if (m_properties.limits.shaderSubgroupPartitioned) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_SUBGROUP_PARTITIONED");
    if (m_properties.limits.gcnShader) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_GCN_SHADER");
    if (m_properties.limits.gpuShaderHalfFloat) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_GPU_SHADER_HALF_FLOAT");
    if (m_properties.limits.gpuShaderInt16) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_GPU_SHADER_INT16");
    if (m_properties.limits.shaderBallot) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_BALLOT");
    if (m_properties.limits.shaderImageLoadStoreLod) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_IMAGE_LOAD_STORE_LOD");
    if (m_properties.limits.shaderTrinaryMinmax) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_TRINARY_MINMAX");
    if (m_properties.limits.postDepthCoverage) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_POST_DEPTH_COVERAGE");
    if (m_properties.limits.shaderStencilExport) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_STENCIL_EXPORT");
    if (m_properties.limits.decorateString) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_DECORATE_STRING");
    if (m_properties.limits.externalFence) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_EXTERNAL_FENCE");
    if (m_properties.limits.externalMemory) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_EXTERNAL_MEMORY");
    if (m_properties.limits.externalSemaphore) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_EXTERNAL_SEMAPHORE");
    if (m_properties.limits.shaderNonSemanticInfo) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SHADER_NON_SEMANTIC_INFO");
    if (m_properties.limits.fragmentShaderBarycentric) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_FRAGMENT_SHADER_BARYCENTRIC");
    if (m_properties.limits.geometryShaderPassthrough) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_GEOMETRY_SHADER_PASSTHROUGH");
    if (m_properties.limits.viewportSwizzle) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_VIEWPORT_SWIZZLE");

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_COMPUTE_UNITS",m_properties.limits.computeUnits);
    if (m_properties.limits.dispatchBase) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_DISPATCH_BASE");
    if (m_properties.limits.allowCommandBufferQueryCopies) addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_ALLOW_COMMAND_BUFFER_QUERY_COPIES");
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_OPTIMALLY_RESIDENT_WORKGROUP_INVOCATIONS",m_properties.limits.maxOptimallyResidentWorkgroupInvocations);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_RESIDENT_INVOCATIONS",m_properties.limits.maxResidentInvocations);
    // asset::IGLSLCompiler::E_SPIRV_VERSION spirvVersion;


    // TODO: @achal test examples 14 and 48 on all APIs and GPUs
    

    // TODO: Add feature defines


    if (runningInRenderdoc)
        addGLSLDefineToPool(pool,"NBL_RUNNING_IN_RENDERDOC");
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

asset::E_FORMAT IPhysicalDevice::promoteBufferFormat(const FormatPromotionRequest<video::IPhysicalDevice::SFormatBufferUsage> req)
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

asset::E_FORMAT IPhysicalDevice::promoteImageFormat(const FormatPromotionRequest<video::IPhysicalDevice::SFormatImageUsage> req, const asset::IImage::E_TILING tiling)
{
    format_image_cache_t& cache = tiling == asset::IImage::E_TILING::ET_LINEAR 
        ? this->m_formatPromotionCache.linearTilingImages 
        : this->m_formatPromotionCache.optimalTilingImages;
    auto cached = cache.find(req);
    if (cached != cache.end())
        return cached->second;
    auto getImageFormatUsagesTiling = [&](asset::E_FORMAT f) {
        switch (tiling)
        {
        case asset::IImage::E_TILING::ET_LINEAR:
            return getImageFormatUsagesLinear(f);
        case asset::IImage::E_TILING::ET_OPTIMAL:
            return getImageFormatUsagesOptimal(f);
        default:
            assert(false); // Invalid tiling
        }
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
