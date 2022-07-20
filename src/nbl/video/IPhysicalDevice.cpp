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
    // uint32_t maxImageDimension1D = 0u;
    // uint32_t maxImageDimension2D = 0u;
    // uint32_t maxImageDimension3D = 0u;
    // uint32_t maxImageDimensionCube = 0u;
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_IMAGE_ARRAY_LAYERS", m_properties.limits.maxImageArrayLayers);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_BUFFER_VIEW_TEXELS", m_properties.limits.maxBufferViewTexels);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_UBO_SIZE",m_properties.limits.maxUBOSize);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SSBO_SIZE",m_properties.limits.maxSSBOSize);
    // uint32_t maxPushConstantsSize = 0u;
    // uint32_t maxMemoryAllocationCount = 0u;
    // uint32_t maxSamplerAllocationCount = 0u;
    // size_t bufferImageGranularity = 0ull;

    // uint32_t maxPerStageDescriptorSamplers = 0u;  // Descriptors with a type of EDT_COMBINED_IMAGE_SAMPLER count against this limit
    // uint32_t maxPerStageDescriptorUBOs = 0u;
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_PER_STAGE_SSBO_COUNT",m_properties.limits.maxPerStageDescriptorSSBOs);
    // uint32_t maxPerStageDescriptorImages = 0u; // Descriptors with a type of EDT_COMBINED_IMAGE_SAMPLER, EDT_UNIFORM_TEXEL_BUFFER count against this limit.
    // uint32_t maxPerStageDescriptorStorageImages = 0u;
    // uint32_t maxPerStageDescriptorInputAttachments = 0u;
    // uint32_t maxPerStageResources = 0u;

    // uint32_t maxDescriptorSetSamplers = 0u; // Descriptors with a type of EDT_COMBINED_IMAGE_SAMPLER count against this limit
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_UBO_COUNT",m_properties.limits.maxDescriptorSetUBOs);
    // uint32_t maxDescriptorSetDynamicOffsetUBOs = 0u;
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_SSBO_COUNT",m_properties.limits.maxDescriptorSetSSBOs);
    // uint32_t maxDescriptorSetDynamicOffsetSSBOs = 0u;
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_TEXTURE_COUNT",m_properties.limits.maxDescriptorSetImages);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_STORAGE_IMAGE_COUNT",m_properties.limits.maxDescriptorSetStorageImages);
    // uint32_t maxDescriptorSetInputAttachments = 0u;

    // uint32_t maxTessellationGenerationLevel = 0u;
    // uint32_t maxTessellationPatchSize = 0u;
    // uint32_t maxTessellationControlPerVertexInputComponents = 0u;
    // uint32_t maxTessellationControlPerVertexOutputComponents = 0u;
    // uint32_t maxTessellationControlPerPatchOutputComponents = 0u;
    // uint32_t maxTessellationControlTotalOutputComponents = 0u;
    // uint32_t maxTessellationEvaluationInputComponents = 0u;
    // uint32_t maxTessellationEvaluationOutputComponents = 0u;
    // uint32_t maxGeometryShaderInvocations = 0u;
    // uint32_t maxGeometryInputComponents = 0u;
    // uint32_t maxGeometryOutputComponents = 0u;
    // uint32_t maxGeometryOutputVertices = 0u;
    // uint32_t maxGeometryTotalOutputComponents = 0u;
    // uint32_t maxFragmentInputComponents = 0u;
    // uint32_t maxFragmentOutputAttachments = 0u;
    // uint32_t maxFragmentDualSrcAttachments = 0u;
    // uint32_t maxFragmentCombinedOutputResources = 0u;
    // uint32_t maxComputeSharedMemorySize;
    // uint32_t maxComputeWorkGroupCount[3];
    // uint32_t maxComputeWorkGroupInvocations = 0u;
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_WORKGROUP_SIZE_X",m_properties.limits.maxWorkgroupSize[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_WORKGROUP_SIZE_Y",m_properties.limits.maxWorkgroupSize[1]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_WORKGROUP_SIZE_Z",m_properties.limits.maxWorkgroupSize[2]);
    // uint32_t subPixelPrecisionBits = 0u;
    addGLSLDefineToPool(pool,"NBL_LIMIT_MAX_DRAW_INDIRECT_COUNT",m_properties.limits.maxDrawIndirectCount);
    // float    maxSamplerLodBias = 0.0f;
    // uint8_t  maxSamplerAnisotropyLog2 = 0.0f;
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VIEWPORTS",m_properties.limits.maxViewports);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VIEWPORT_WIDTH",m_properties.limits.maxViewportDims[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_VIEWPORT_HEIGHT",m_properties.limits.maxViewportDims[1]);
    // float    viewportBoundsRange[2]; // [min, max]
    // uint32_t viewportSubPixelBits = 0u;
    // size_t   minMemoryMapAlignment = 0ull;
    // uint32_t bufferViewAlignment = 0u;
    // uint32_t minUBOAlignment = 0u;
    // uint32_t minSSBOAlignment = 0u;
    // int32_t  minTexelOffset = 0;
    // uint32_t maxTexelOffset = 0u;
    // int32_t  minTexelGatherOffset = 0;
    // uint32_t maxTexelGatherOffset = 0u;
    // float    minInterpolationOffset = 0.0f;
    // float    maxInterpolationOffset = 0.0f;
    // uint32_t maxFramebufferWidth = 0u;
    // uint32_t maxFramebufferHeight = 0u;
    // uint32_t maxFramebufferLayers = 0u;
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferColorSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferDepthSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferStencilSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferNoAttachmentsSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    // uint32_t maxColorAttachments = 0u;
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageColorSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageIntegerSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageDepthSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageStencilSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> storageImageSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    // uint32_t maxSampleMaskWords = 0u;
    // bool timestampComputeAndGraphics = false;
    // float timestampPeriodInNanoSeconds = 0.0f; // timestampPeriod is the number of nanoseconds required for a timestamp query to be incremented by 1 (a float because vulkan reports), use core::rational in the future
    // uint32_t maxClipDistances = 0u;
    // uint32_t maxCullDistances = 0u;
    // uint32_t maxCombinedClipAndCullDistances = 0u;
    // uint32_t discreteQueuePriorities = 0u;
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_POINT_SIZE",m_properties.limits.pointSizeRange[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_POINT_SIZE",m_properties.limits.pointSizeRange[1]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MIN_LINE_WIDTH",m_properties.limits.lineWidthRange[0]);
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_LINE_WIDTH",m_properties.limits.lineWidthRange[1]);
    // float pointSizeGranularity = 0.f;
    // float lineWidthGranularity = 0.f;
    // bool strictLines = false;
    // bool standardSampleLocations = false;
    // uint64_t optimalBufferCopyOffsetAlignment = 0ull;
    // uint64_t optimalBufferCopyRowPitchAlignment = 0ull;
    // uint64_t nonCoherentAtomSize = 0ull;

    // uint32_t maxVertexOutputComponents = 0u;
    
    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_SUBGROUP_SIZE",m_properties.limits.subgroupSize);
    // core::bitflag<asset::IShader::E_SHADER_STAGE> subgroupOpsShaderStages = asset::IShader::ESS_UNKNOWN;
    // bool shaderSubgroupBasic = false;
    // bool shaderSubgroupVote = false;
    // bool shaderSubgroupArithmetic = false;
    // bool shaderSubgroupBallot = false;
    // bool shaderSubgroupShuffle = false;
    // bool shaderSubgroupShuffleRelative = false;
    // bool shaderSubgroupClustered = false;
    // bool shaderSubgroupQuad = false;
    // bool shaderSubgroupQuadAllStages = false;

    // E_POINT_CLIPPING_BEHAVIOR pointClippingBehavior = EPCB_USER_CLIP_PLANES_ONLY;
    
    // uint32_t maxPerSetDescriptors = 0u;
    // size_t maxMemoryAllocationSize = 0ull;

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

    // uint32_t maxUpdateAfterBindDescriptorsInAllPools = ~0u;
    // bool shaderUniformBufferArrayNonUniformIndexingNative = false;
    // bool shaderSampledImageArrayNonUniformIndexingNative = false;
    // bool shaderStorageBufferArrayNonUniformIndexingNative = false;
    // bool shaderStorageImageArrayNonUniformIndexingNative = false;
    // bool shaderInputAttachmentArrayNonUniformIndexingNative = false;
    // bool robustBufferAccessUpdateAfterBind = false;
    // bool quadDivergentImplicitLod = false;
    // uint32_t maxPerStageDescriptorUpdateAfterBindSamplers = 0u;
    // uint32_t maxPerStageDescriptorUpdateAfterBindUBOs = 0u;
    // uint32_t maxPerStageDescriptorUpdateAfterBindSSBOs = 0u;
    // uint32_t maxPerStageDescriptorUpdateAfterBindImages = 0u;
    // uint32_t maxPerStageDescriptorUpdateAfterBindStorageImages = 0u;
    // uint32_t maxPerStageDescriptorUpdateAfterBindInputAttachments = 0u;
    // uint32_t maxPerStageUpdateAfterBindResources = 0u;
    // uint32_t maxDescriptorSetUpdateAfterBindSamplers = 0u;
    // uint32_t maxDescriptorSetUpdateAfterBindUBOs = 0u;
    // uint32_t maxDescriptorSetUpdateAfterBindDynamicOffsetUBOs = 0u;
    // uint32_t maxDescriptorSetUpdateAfterBindSSBOs = 0u;
    // uint32_t maxDescriptorSetUpdateAfterBindDynamicOffsetSSBOs = 0u;
    // uint32_t maxDescriptorSetUpdateAfterBindImages = 0u;
    // uint32_t maxDescriptorSetUpdateAfterBindStorageImages = 0u;
    // uint32_t maxDescriptorSetUpdateAfterBindInputAttachments = 0u;

    // bool filterMinmaxSingleComponentFormats = false;
    // bool filterMinmaxImageComponentMapping = false;

    // uint32_t minSubgroupSize = 0u;
    // uint32_t maxSubgroupSize = 0u;
    // uint32_t maxComputeWorkgroupSubgroups = 0u;
    // core::bitflag<asset::IShader::E_SHADER_STAGE> requiredSubgroupSizeStages = core::bitflag<asset::IShader::E_SHADER_STAGE>(0u);

    // size_t storageTexelBufferOffsetAlignmentBytes = 0ull;
    // size_t uniformTexelBufferOffsetAlignmentBytes = 0ull;

    addGLSLDefineToPool(pool,"NBL_GLSL_LIMIT_MAX_BUFFER_SIZE",core::min(m_properties.limits.maxBufferSize, ~0u));

    // float primitiveOverestimationSize = 0.0f;
    // float maxExtraPrimitiveOverestimationSize = 0.0f;
    // float extraPrimitiveOverestimationSizeGranularity = 0.0f;
    // bool primitiveUnderestimation = false;
    // bool conservativePointAndLineRasterization = false;
    // bool degenerateTrianglesRasterized = false;
    // bool degenerateLinesRasterized = false;
    // bool fullyCoveredFragmentShaderInputVariable = false;
    // bool conservativeRasterizationPostDepthCoverage = false;

    // uint32_t maxDiscardRectangles = 0u;
    // uint32_t lineSubPixelPrecisionBits = 0;
    // uint32_t maxVertexAttribDivisor = 0;
    // uint32_t maxSubpassShadingWorkgroupSizeAspectRatio = 0;

    // bool integerDotProduct8BitUnsignedAccelerated;
    // bool integerDotProduct8BitSignedAccelerated;
    // bool integerDotProduct8BitMixedSignednessAccelerated;
    // bool integerDotProduct4x8BitPackedUnsignedAccelerated;
    // bool integerDotProduct4x8BitPackedSignedAccelerated;
    // bool integerDotProduct4x8BitPackedMixedSignednessAccelerated;
    // bool integerDotProduct16BitUnsignedAccelerated;
    // bool integerDotProduct16BitSignedAccelerated;
    // bool integerDotProduct16BitMixedSignednessAccelerated;
    // bool integerDotProduct32BitUnsignedAccelerated;
    // bool integerDotProduct32BitSignedAccelerated;
    // bool integerDotProduct32BitMixedSignednessAccelerated;
    // bool integerDotProduct64BitUnsignedAccelerated;
    // bool integerDotProduct64BitSignedAccelerated;
    // bool integerDotProduct64BitMixedSignednessAccelerated;
    // bool integerDotProductAccumulatingSaturating8BitUnsignedAccelerated;
    // bool integerDotProductAccumulatingSaturating8BitSignedAccelerated;
    // bool integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated;
    // bool integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated;
    // bool integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated;
    // bool integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated;
    // bool integerDotProductAccumulatingSaturating16BitUnsignedAccelerated;
    // bool integerDotProductAccumulatingSaturating16BitSignedAccelerated;
    // bool integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated;
    // bool integerDotProductAccumulatingSaturating32BitUnsignedAccelerated;
    // bool integerDotProductAccumulatingSaturating32BitSignedAccelerated;
    // bool integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated;
    // bool integerDotProductAccumulatingSaturating64BitUnsignedAccelerated;
    // bool integerDotProductAccumulatingSaturating64BitSignedAccelerated;
    // bool integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated;

    // uint64_t           maxGeometryCount = 0ull;
    // uint64_t           maxInstanceCount = 0ull;
    // uint64_t           maxPrimitiveCount = 0ull;
    // uint32_t           maxPerStageDescriptorAccelerationStructures = 0u;
    // uint32_t           maxPerStageDescriptorUpdateAfterBindAccelerationStructures = 0u;
    // uint32_t           maxDescriptorSetAccelerationStructures = 0u;
    // uint32_t           maxDescriptorSetUpdateAfterBindAccelerationStructures = 0u;
    // uint32_t           minAccelerationStructureScratchOffsetAlignment = 0u;

    // bool variableSampleLocations = false;
    // uint32_t        sampleLocationSubPixelBits = 0;
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampleLocationSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    // VkExtent2D      maxSampleLocationGridSize = { 0u, 0u };
    // float           sampleLocationCoordinateRange[2];

    // size_t minImportedHostPointerAlignment = 0x1ull<<63u;

    // VkExtent2D         minFragmentDensityTexelSize = {0u, 0u};
    // VkExtent2D         maxFragmentDensityTexelSize = {0u, 0u};
    // bool           fragmentDensityInvocations = false;

    // bool           subsampledLoads = false;
    // bool           subsampledCoarseReconstructionEarlyAccess = false;
    // uint32_t           maxSubsampledArrayLayers = 0u;
    // uint32_t           maxDescriptorSetSubsampledSamplers = 0u;

    // uint32_t  pciDomain = ~0u;
    // uint32_t  pciBus = ~0u;
    // uint32_t  pciDevice = ~0u;
    // uint32_t  pciFunction = ~0u;

    // uint32_t           shaderGroupHandleSize = 0u;
    // uint32_t           maxRayRecursionDepth = 0u;
    // uint32_t           maxShaderGroupStride = 0u;
    // uint32_t           shaderGroupBaseAlignment = 0u;
    // uint32_t           shaderGroupHandleCaptureReplaySize = 0u;
    // uint32_t           maxRayDispatchInvocationCount = 0u;
    // uint32_t           shaderGroupHandleAlignment = 0u;
    // uint32_t           maxRayHitAttributeSize = 0u;

    // core::bitflag<asset::IShader::E_SHADER_STAGE> cooperativeMatrixSupportedStages = asset::IShader::ESS_UNKNOWN;
  
    // bool shaderOutputViewportIndex = false;     // ALIAS: VK_EXT_shader_viewport_index_layer
    // bool shaderOutputLayer = false;             // ALIAS: VK_EXT_shader_viewport_index_layer
    // bool shaderIntegerFunctions2 = false;
    // bool shaderSubgroupClock = false;
    // bool imageFootprint = false;
    // bool texelBufferAlignment = false;
    // bool shaderSMBuiltins = false;
    // bool shaderSubgroupPartitioned = false; /* VK_NV_shader_subgroup_partitioned */
    // bool gcnShader = false; /* VK_AMD_gcn_shader */
    // bool gpuShaderHalfFloat = false; /* VK_AMD_gpu_shader_half_float */
    // bool gpuShaderInt16 = false; /* VK_AMD_gpu_shader_int16 */
    // bool shaderBallot = false; /* VK_AMD_shader_ballot */
    // bool shaderImageLoadStoreLod = false; /* VK_AMD_shader_image_load_store_lod */
    // bool shaderTrinaryMinmax = false; /* VK_AMD_shader_trinary_minmax  */
    // bool postDepthCoverage = false; /* VK_EXT_post_depth_coverage */
    // bool shaderStencilExport = false; /* VK_EXT_shader_stencil_export */
    // bool decorateString = false; /* VK_GOOGLE_decorate_string */
    // bool externalFence = false; /* VK_KHR_external_fence_fd */ /* VK_KHR_external_fence_win32 */
    // bool externalMemory = false; /* VK_KHR_external_memory_fd */ /* VK_KHR_external_memory_win32 */
    // bool externalSemaphore = false; /* VK_KHR_external_semaphore_fd */ /* VK_KHR_external_semaphore_win32 */
    // bool shaderNonSemanticInfo = false; /* VK_KHR_shader_non_semantic_info */
    // bool fragmentShaderBarycentric = false; /* VK_KHR_fragment_shader_barycentric */
    // bool geometryShaderPassthrough = false; /* VK_NV_geometry_shader_passthrough */
    // bool viewportSwizzle = false; /* VK_NV_viewport_swizzle */

    // uint32_t computeUnits = 0u;
    // bool dispatchBase = false; // true in Vk, false in GL
    // bool allowCommandBufferQueryCopies = false;
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
