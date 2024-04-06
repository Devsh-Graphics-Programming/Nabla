#ifndef _NBL_VIDEO_S_PHYSICAL_DEVICE_LIMITS_H_INCLUDED_
#define _NBL_VIDEO_S_PHYSICAL_DEVICE_LIMITS_H_INCLUDED_


#include "nbl/asset/utils/CGLSLCompiler.h" // asset::CGLSLCompiler::E_SPIRV_VERSION

#include "nbl/asset/IImage.h"
#include "nbl/asset/IRenderpass.h"

#include <type_traits>


namespace nbl::video
{

// Struct is populated with Nabla Core Profile Limit Minimums
struct SPhysicalDeviceLimits
{
    /* Vulkan 1.0 Core  */
    constexpr static inline uint32_t MinMaxImageDimension2D = 1u<<14u;
    uint32_t maxImageDimension1D = MinMaxImageDimension2D;
    uint32_t maxImageDimension2D = MinMaxImageDimension2D;
    uint32_t maxImageDimension3D = 1u<<11u;
    uint32_t maxImageDimensionCube = MinMaxImageDimension2D;
    uint32_t maxImageArrayLayers = 1u<<11u;
    uint32_t maxBufferViewTexels = 1u<<25u;
    uint32_t maxUBOSize = 1u<<16u;
    constexpr static inline uint32_t MinMaxSSBOSize = (1u<<30u)-sizeof(uint32_t);
    uint32_t maxSSBOSize = MinMaxSSBOSize;
    constexpr static inline uint16_t MaxMaxPushConstantsSize = 256u;
    uint16_t maxPushConstantsSize = 128u;
    uint32_t maxMemoryAllocationCount = 1u<<12u;
    uint32_t maxSamplerAllocationCount = 4000u;
    //! granularity, in bytes, at which buffer or linear image resources, and optimal image resources can be bound to adjacent offsets in the same allocation
    uint32_t bufferImageGranularity = 1u<<16u;
    //size_t            sparseAddressSpaceSize = 0u;         // [TODO LATER] when we support sparse
    //uint32_t          maxBoundDescriptorSets = 4;         // [DO NOT EXPOSE] we've kinda hardcoded the engine to 4 currently

    uint32_t maxPerStageDescriptorSamplers = 16u;  // Descriptors with a type of IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER count against this limit
    uint32_t maxPerStageDescriptorUBOs = 15u;
    uint32_t maxPerStageDescriptorSSBOs = 31u;
    uint32_t maxPerStageDescriptorImages = 96u; // Descriptors with a type of IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER, IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER count against this limit.
    uint32_t maxPerStageDescriptorStorageImages = 8u;
    uint32_t maxPerStageDescriptorInputAttachments = 7u;
    uint32_t maxPerStageResources = 127u;

    uint32_t maxDescriptorSetSamplers = 80u; // Descriptors with a type of IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER count against this limit
    uint32_t maxDescriptorSetUBOs = 90u;
    uint32_t maxDescriptorSetDynamicOffsetUBOs = 8u;
    uint32_t maxDescriptorSetSSBOs = 155u;
    uint32_t maxDescriptorSetDynamicOffsetSSBOs = 8u;
    uint32_t maxDescriptorSetImages = 480u; // Descriptors with a type of IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER, IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER count against this limit.
    uint32_t maxDescriptorSetStorageImages = 40u;
    uint32_t maxDescriptorSetInputAttachments = 7u;

    //uint8_t maxVertexInputAttributes = 16;
    //uint8_t maxVertexInputBindings = 16;
    //uint16_t maxVertexInputAttributeOffset = maxVertexInputBindingStride-1;
    //uint16_t maxVertexInputBindingStride = 1u<11u;
    uint16_t maxVertexOutputComponents = 124u;

    uint16_t maxTessellationGenerationLevel = 0u;
    uint16_t maxTessellationPatchSize = 0u;
    uint16_t maxTessellationControlPerVertexInputComponents = 0u;
    uint16_t maxTessellationControlPerVertexOutputComponents = 0u;
    uint16_t maxTessellationControlPerPatchOutputComponents = 0u;
    uint16_t maxTessellationControlTotalOutputComponents = 0u;
    uint16_t maxTessellationEvaluationInputComponents = 0u;
    uint16_t maxTessellationEvaluationOutputComponents = 0u;

    uint16_t maxGeometryShaderInvocations = 0u;
    uint16_t maxGeometryInputComponents = 0u;
    uint16_t maxGeometryOutputComponents = 0u;
    uint16_t maxGeometryOutputVertices = 0u;
    uint16_t maxGeometryTotalOutputComponents = 0u;

    uint32_t maxFragmentInputComponents = 116u;
    uint32_t maxFragmentOutputAttachments = 8u;
    uint32_t maxFragmentDualSrcAttachments = 1u;
    uint32_t maxFragmentCombinedOutputResources = 16u;

    uint32_t maxComputeSharedMemorySize = 1u<<15u;
    constexpr static inline uint32_t MinMaxWorkgroupCount = (1u<<16u)-1u;
    uint32_t maxComputeWorkGroupCount[3] = {MinMaxWorkgroupCount,MinMaxWorkgroupCount,MinMaxWorkgroupCount};
    constexpr static inline uint32_t MinMaxWorkgroupInvocations = 256u;
    uint16_t maxComputeWorkGroupInvocations = MinMaxWorkgroupInvocations;
    uint16_t maxWorkgroupSize[3] = {MinMaxWorkgroupInvocations,MinMaxWorkgroupInvocations,64u};

    uint8_t subPixelPrecisionBits = 4u;
    uint8_t subTexelPrecisionBits = 4u;
    uint8_t mipmapPrecisionBits = 4u;

    // [DO NOT EXPOSE] ROADMAP2022: requires fullDrawIndexUint32 so this must be 0xffFFffFFu
    //uint32_t    maxDrawIndexedIndexValue;

    // This is different to `maxDrawIndirectCount`, this is NOT about whether you can source the MDI count from a buffer, just about how many you can have
    uint32_t    maxDrawIndirectCount = 1u<<30u;

    float       maxSamplerLodBias = 4.f;
    uint8_t     maxSamplerAnisotropyLog2 = 4u;

    uint8_t     maxViewports = 16u;
    uint16_t    maxViewportDims[2] = {MinMaxImageDimension2D,MinMaxImageDimension2D};
    float       viewportBoundsRange[2] = { -MinMaxImageDimension2D*2u, MinMaxImageDimension2D*2u-1 };
    uint32_t    viewportSubPixelBits = 0u;

    uint16_t minMemoryMapAlignment = 0x1u<<6u;
    uint16_t bufferViewAlignment = 0x1u<<6u;
    uint16_t minUBOAlignment = 0x1u<<8u;
    uint16_t minSSBOAlignment = 0x1u<<6u;

    int8_t  minTexelOffset = -8;
    uint8_t maxTexelOffset = 7u;
    int8_t  minTexelGatherOffset = -8;
    uint8_t maxTexelGatherOffset = 7u;

    constexpr static inline int32_t MinSubPixelInterpolationOffsetBits = 4;
    float   minInterpolationOffset = -0.5f;
    float   maxInterpolationOffset = 0.5f-exp2f(-MinSubPixelInterpolationOffsetBits);
    uint8_t subPixelInterpolationOffsetBits = MinSubPixelInterpolationOffsetBits;

    uint32_t maxFramebufferWidth = MinMaxImageDimension2D;
    uint32_t maxFramebufferHeight = MinMaxImageDimension2D;
    uint32_t maxFramebufferLayers = 1024u;
    /*
    - Spec states minimum supported value should be at least ESCF_1_BIT
    - it might be different for each integer format, best way is to query your integer format from physical device using vkGetPhysicalDeviceImageFormatProperties and get the sampleCounts
    https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkImageFormatProperties.html
    */
    // [DO NOT EXPOSE] because it might be different for every texture format and usage
    //constexpr static inline core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> NoMSor4Samples = asset::IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT|asset::IImage::E_SAMPLE_COUNT_FLAGS::ESCF_4_BIT;
    //core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferColorSampleCounts = NoMSor4Samples;
    //core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferDepthSampleCounts = NoMSor4Samples;
    //core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferStencilSampleCounts = NoMSor4Samples;
    //core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferNoAttachmentsSampleCounts = NoMSor4Samples;
    constexpr static inline uint8_t MinMaxColorAttachments = 8u; // ROADMAP 2024 and wide reports
    uint8_t maxColorAttachments = MinMaxColorAttachments;

    // [DO NOT EXPOSE] because it might be different for every texture format and usage
    //core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageColorSampleCounts = NoMSor4Samples;
    //core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageIntegerSampleCounts = NoMSor4Samples;
    //core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageDepthSampleCounts = NoMSor4Samples;
    //core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageStencilSampleCounts = NoMSor4Samples;
    //core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> storageImageSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;

    uint8_t maxSampleMaskWords = 1u;

    // [REQUIRE] ROADMAP 2024 and good device support
    //bool timestampComputeAndGraphics = true;
    float timestampPeriodInNanoSeconds = 83.334f; // timestampPeriod is the number of nanoseconds required for a timestamp query to be incremented by 1 (a float because vulkan reports), use core::rational in the future

    uint8_t maxClipDistances = 8u;
    uint8_t maxCullDistances = 0u;
    uint8_t maxCombinedClipAndCullDistances = 8u;

    uint32_t discreteQueuePriorities = 2u;

    float pointSizeRange[2] = {1.f,64.f};
    float lineWidthRange[2] = {1.f,1.f};
    float pointSizeGranularity = 1.f;
    float lineWidthGranularity = 1.f;
    // old intels can't do this
    bool strictLines = false;

    // Had to roll back from requiring, ROADMAP 2022 but some of our targets missing
    bool standardSampleLocations = false;

    uint16_t optimalBufferCopyOffsetAlignment = 0x1u<<8u;
    uint16_t optimalBufferCopyRowPitchAlignment = 0x1u<<7u;
    uint16_t nonCoherentAtomSize = 0x1u<<8u;

    // TODO: later
    /* VkPhysicalDeviceSparseProperties */ 
    //bool    residencyStandard2DBlockShape = true;
    //bool    residencyStandard2DMultisampleBlockShape = false;
    //bool    residencyStandard3DBlockShape = true;
    //bool    residencyAlignedMipSize = false;
    //bool    residencyNonResidentStrict = true;



    /* Vulkan 1.1 Core  */
    uint16_t subgroupSize = 4u;
    core::bitflag<asset::IShader::E_SHADER_STAGE> subgroupOpsShaderStages = asset::IShader::ESS_COMPUTE|asset::IShader::ESS_ALL_GRAPHICS;
    // ROADMAP2022 mandates all but clustered and quad-all-stages, however all GPU's that we care about support basic, vote, ballot, shuffle and relative so not listing!
    bool shaderSubgroupClustered = false;
    // candidates for promotion
    bool shaderSubgroupArithmetic = false;
    bool shaderSubgroupQuad = false;
    // bad Android support
    bool shaderSubgroupQuadAllStages = false;

    enum E_POINT_CLIPPING_BEHAVIOR : uint8_t {
        EPCB_ALL_CLIP_PLANES = 0,
        EPCB_USER_CLIP_PLANES_ONLY = 1,
    };
    E_POINT_CLIPPING_BEHAVIOR pointClippingBehavior = EPCB_USER_CLIP_PLANES_ONLY;

    uint8_t     maxMultiviewViewCount = 6u;
    uint32_t    maxMultiviewInstanceIndex = (1u<<27u)-1u;

    //bool        protectedNoFault = false;
    
    uint32_t maxPerSetDescriptors = 572u;
    size_t maxMemoryAllocationSize = MinMaxSSBOSize;


    /* Vulkan 1.2 Core  */
//    VkShaderFloatControlsIndependence denormBehaviorIndependence; // TODO: need to implement ways to set them
//    VkShaderFloatControlsIndependence roundingModeIndependence;   // TODO: need to implement ways to set them
    //bool shaderSignedZeroInfNanPreserveFloat16 = true;
    //bool shaderSignedZeroInfNanPreserveFloat32 = true;
    bool shaderSignedZeroInfNanPreserveFloat64 = false;
    bool shaderDenormPreserveFloat16 = false;
    bool shaderDenormPreserveFloat32 = false;
    bool shaderDenormPreserveFloat64 = false;
    bool shaderDenormFlushToZeroFloat16 = false;
    bool shaderDenormFlushToZeroFloat32 = false;
    bool shaderDenormFlushToZeroFloat64 = false;
    // ROADMAP2024 but no good support yet
    bool shaderRoundingModeRTEFloat16 = false;
    // ROADMAP2024 but no good support yet
    bool shaderRoundingModeRTEFloat32 = false;
    bool shaderRoundingModeRTEFloat64 = false;
    bool shaderRoundingModeRTZFloat16 = false;
    bool shaderRoundingModeRTZFloat32 = false;
    bool shaderRoundingModeRTZFloat64 = false;
 
    // expose in 2 phases
    // -Update After Bindand nonUniformEXT shader qualifier:
    //      Descriptor Lifetime Tracking PR #345 will do this, cause I don't want to rewrite the tracking system again.
    // -Actual Descriptor Indexing:
    //      The whole 512k descriptor limits, runtime desc arrays, etc.will come later
    uint32_t maxUpdateAfterBindDescriptorsInAllPools = 1u<<20u;
    bool shaderUniformBufferArrayNonUniformIndexingNative = false;
    bool shaderSampledImageArrayNonUniformIndexingNative = false; // promotion candidate
    bool shaderStorageBufferArrayNonUniformIndexingNative = false;
    bool shaderStorageImageArrayNonUniformIndexingNative = false; // promotion candidate
    bool shaderInputAttachmentArrayNonUniformIndexingNative = false; // promotion candidate
    bool robustBufferAccessUpdateAfterBind = false;
    bool quadDivergentImplicitLod = false;
    uint32_t maxPerStageDescriptorUpdateAfterBindSamplers = 500000u;
    uint32_t maxPerStageDescriptorUpdateAfterBindUBOs = 15u;
    uint32_t maxPerStageDescriptorUpdateAfterBindSSBOs = 500000u;
    uint32_t maxPerStageDescriptorUpdateAfterBindImages = 500000u;
    uint32_t maxPerStageDescriptorUpdateAfterBindStorageImages = 500000u;
    uint32_t maxPerStageDescriptorUpdateAfterBindInputAttachments = MinMaxColorAttachments;
    uint32_t maxPerStageUpdateAfterBindResources = 500000u;
    uint32_t maxDescriptorSetUpdateAfterBindSamplers = 500000u;
    uint32_t maxDescriptorSetUpdateAfterBindUBOs = 72u;
    uint32_t maxDescriptorSetUpdateAfterBindDynamicOffsetUBOs = 8u;
    uint32_t maxDescriptorSetUpdateAfterBindSSBOs = 500000u;
    uint32_t maxDescriptorSetUpdateAfterBindDynamicOffsetSSBOs = 4u;
    uint32_t maxDescriptorSetUpdateAfterBindImages = 500000u;
    uint32_t maxDescriptorSetUpdateAfterBindStorageImages = 500000u;
    uint32_t maxDescriptorSetUpdateAfterBindInputAttachments = MinMaxColorAttachments;

    using RESOLVE_MODE_FLAGS = asset::IRenderpass::SCreationParams::SSubpassDescription::SDepthStencilAttachmentsRef::RESOLVE_MODE;
    core::bitflag<RESOLVE_MODE_FLAGS>   supportedDepthResolveModes = RESOLVE_MODE_FLAGS::SAMPLE_ZERO_BIT;
    core::bitflag<RESOLVE_MODE_FLAGS>   supportedStencilResolveModes = RESOLVE_MODE_FLAGS::SAMPLE_ZERO_BIT;
    bool                                independentResolveNone = false;
    bool                                independentResolve = false;

    // TODO: you'll be able to query this in format usage/feature reports
    //bool filterMinmaxSingleComponentFormats;
    bool filterMinmaxImageComponentMapping = false;

    // [DO NOT EXPOSE] its high enough (207 days of uptime at 120 FPS)
    //uint64_t maxTimelineSemaphoreValueDifference = 2147483647;

    // [DO NOT EXPOSE] because it might be different for every texture format and usage
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferIntegerColorSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);


    /* Vulkan 1.3 Core  */
    
    //      or VK_EXT_subgroup_size_control:
    uint8_t                                         minSubgroupSize = 64u;
    uint8_t                                         maxSubgroupSize = 4u;
    uint32_t                                        maxComputeWorkgroupSubgroups = 16u;
    core::bitflag<asset::IShader::E_SHADER_STAGE>   requiredSubgroupSizeStages = asset::IShader::E_SHADER_STAGE::ESS_UNKNOWN; // also None

    // [DO NOT EXPOSE]: we won't expose inline uniform blocks right now
    //constexpr static inline MinInlineUniformBlockSize = 0x1u<<8u; 
    //uint32_t              maxInlineUniformBlockSize = MinInlineUniformBlockSize;
    //uint32_t              maxPerStageDescriptorInlineUniformBlocks = 4;
    //uint32_t              maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks = 4;
    //uint32_t              maxDescriptorSetInlineUniformBlocks = 4;
    //uint32_t              maxDescriptorSetUpdateAfterBindInlineUniformBlocks = 4;
    //uint32_t              maxInlineUniformTotalSize = MinInlineUniformBlockSize;

    bool integerDotProduct8BitUnsignedAccelerated = false;
    bool integerDotProduct8BitSignedAccelerated = false;
    bool integerDotProduct8BitMixedSignednessAccelerated = false;
    bool integerDotProduct4x8BitPackedUnsignedAccelerated = false;
    bool integerDotProduct4x8BitPackedSignedAccelerated = false;
    bool integerDotProduct4x8BitPackedMixedSignednessAccelerated = false;
    bool integerDotProduct16BitUnsignedAccelerated = false;
    bool integerDotProduct16BitSignedAccelerated = false;
    bool integerDotProduct16BitMixedSignednessAccelerated = false;
    bool integerDotProduct32BitUnsignedAccelerated = false;
    bool integerDotProduct32BitSignedAccelerated = false;
    bool integerDotProduct32BitMixedSignednessAccelerated = false;
    bool integerDotProduct64BitUnsignedAccelerated = false;
    bool integerDotProduct64BitSignedAccelerated = false;
    bool integerDotProduct64BitMixedSignednessAccelerated = false;
    bool integerDotProductAccumulatingSaturating8BitUnsignedAccelerated = false;
    bool integerDotProductAccumulatingSaturating8BitSignedAccelerated = false;
    bool integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated = false;
    bool integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated = false;
    bool integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated = false;
    bool integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated = false;
    bool integerDotProductAccumulatingSaturating16BitUnsignedAccelerated = false;
    bool integerDotProductAccumulatingSaturating16BitSignedAccelerated = false;
    bool integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated = false;
    bool integerDotProductAccumulatingSaturating32BitUnsignedAccelerated = false;
    bool integerDotProductAccumulatingSaturating32BitSignedAccelerated = false;
    bool integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated = false;
    bool integerDotProductAccumulatingSaturating64BitUnsignedAccelerated = false;
    bool integerDotProductAccumulatingSaturating64BitSignedAccelerated = false;
    bool integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated = false;
    
    // or VK_EXT_texel_buffer_alignment:
    // [DO NOT EXPOSE]: the single texel alignments, let people just overalign
    size_t storageTexelBufferOffsetAlignmentBytes = std::numeric_limits<size_t>::max();
    //bool              storageTexelBufferOffsetSingleTexelAlignment;
    size_t uniformTexelBufferOffsetAlignmentBytes = std::numeric_limits<size_t>::max();
    //bool              uniformTexelBufferOffsetSingleTexelAlignment;
    
    size_t maxBufferSize = MinMaxSSBOSize; // or VK_KHR_maintenance4


    /* Nabla Core Profile Extensions*/

    /* VK_EXT_external_memory_host */
    /* ExternalMemoryHostPropertiesEXT */
    uint32_t minImportedHostPointerAlignment = 0x1u<<31u;

    /* ShaderAtomicFloatFeaturesEXT *//* VK_EXT_shader_atomic_float */
    // [REQUIRE] Nabla Core Profile
    //bool shaderBufferFloat32Atomics = true;
    bool shaderBufferFloat32AtomicAdd = false;
    bool shaderBufferFloat64Atomics = false;
    bool shaderBufferFloat64AtomicAdd = false;
    // [REQUIRE] Nabla Core Profile
    //bool shaderSharedFloat32Atomics = true;
    bool shaderSharedFloat32AtomicAdd = false;
    bool shaderSharedFloat64Atomics = false;
    bool shaderSharedFloat64AtomicAdd = false;
    // [REQUIRE] Nabla Core Profile
    //bool shaderImageFloat32Atomics = true;
    bool shaderImageFloat32AtomicAdd = false;
    bool sparseImageFloat32Atomics = false;
    bool sparseImageFloat32AtomicAdd = false;

    /* Robustness2PropertiesEXT *//* provided by VK_EXT_robustness2 */
    size_t robustStorageBufferAccessSizeAlignment = 0x1ull << 63;
    size_t robustUniformBufferAccessSizeAlignment = 0x1ull << 63;


    /* Vulkan Extensions */

    /* VK_AMD_shader_trinary_minmax  */
    bool shaderTrinaryMinmax = false;

    /* VK_AMD_shader_explicit_vertex_parameter */
    bool shaderExplicitVertexParameter = false;

    /* VK_AMD_gpu_shader_half_float */
    bool gpuShaderHalfFloatAMD = false;

    /* VK_AMD_shader_image_load_store_lod */
    bool shaderImageLoadStoreLod = false;

    // [TODO LATER] to expose but contingent on the TODO to implement one day
    /* PushDescriptorPropertiesKHR *//* provided by VK_KHR_push_descriptor */
    //uint32_t           maxPushDescriptors = 0u;

    // [TODO] need impl
    /* VK_GOOGLE_display_timing */
    bool displayTiming = false;

    /* VK_EXT_discard_rectangles */
    /* DiscardRectanglePropertiesEXT */
    uint32_t maxDiscardRectangles = 0u;

    /* VK_EXT_conservative_rasterization */
    /* ConservativeRasterizationPropertiesEXT */
    float   primitiveOverestimationSize = 0.0f;
    float   maxExtraPrimitiveOverestimationSize = 0.0f;
    float   extraPrimitiveOverestimationSizeGranularity = std::numeric_limits<float>::infinity();
    bool    primitiveUnderestimation = false;
    bool    conservativePointAndLineRasterization = false;
    bool    degenerateTrianglesRasterized = false;
    bool    degenerateLinesRasterized = false;
    bool    fullyCoveredFragmentShaderInputVariable = false;
    bool    conservativeRasterizationPostDepthCoverage = false;

    /* VK_EXT_queue_family_foreign */
    bool queueFamilyForeign = false;

    /* VK_EXT_shader_stencil_export */
    bool shaderStencilExport = false;

    /* VK_EXT_sample_locations */
    /* SampleLocationsPropertiesEXT */
    bool                                                variableSampleLocations = false;
    uint8_t                                             sampleLocationSubPixelBits = 0;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS>  sampleLocationSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    hlsl::uint32_t2                                     maxSampleLocationGridSize = { 0u, 0u };
    float                                               sampleLocationCoordinateRange[2] = { 1.f, 0.f };

    /* VK_KHR_acceleration_structure */
    /* AccelerationStructurePropertiesKHR */
    uint64_t           maxAccelerationStructureGeometryCount = 0ull;
    uint64_t           maxAccelerationStructureInstanceCount = 0ull;
    uint64_t           maxAccelerationStructurePrimitiveCount = 0ull;
    uint32_t           maxPerStageDescriptorAccelerationStructures = 0u;
    uint32_t           maxPerStageDescriptorUpdateAfterBindAccelerationStructures = 0u;
    uint32_t           maxDescriptorSetAccelerationStructures = 0u;
    uint32_t           maxDescriptorSetUpdateAfterBindAccelerationStructures = 0u;
    uint32_t           minAccelerationStructureScratchOffsetAlignment = 0x1u << 31u;

    /* VK_KHR_ray_tracing_pipeline */
    /* RayTracingPipelinePropertiesKHR */
    //uint32_t           shaderGroupHandleSize = 32u; // `exact` limit type
    uint32_t           maxRayRecursionDepth = 0u;
    uint32_t           maxShaderGroupStride = 0u;
    uint32_t           shaderGroupBaseAlignment = 0x1u << 31u;
    uint32_t           maxRayDispatchInvocationCount = 0u;
    uint32_t           shaderGroupHandleAlignment = 0x1u << 31u;
    uint32_t           maxRayHitAttributeSize = 0u;

    /* VK_NV_shader_sm_builtins */
    /* ShaderSMBuiltinsFeaturesNV */
    bool shaderSMBuiltins = false;

    /* VK_EXT_post_depth_coverage */
    bool postDepthCoverage = false;

    /* VK_KHR_shader_clock */
    /* ShaderClockFeaturesKHR */
    bool shaderDeviceClock = false;

    /* VK_NV_compute_shader_derivatives */
    /* ComputeShaderDerivativesFeaturesNV */
    bool computeDerivativeGroupQuads = false;
    bool computeDerivativeGroupLinear = false;

    /* VK_NV_shader_image_footprint */
    /* ShaderImageFootprintFeaturesNV */
    bool imageFootprint = false;

    /* VK_INTEL_shader_integer_functions2 */
    /* ShaderIntegerFunctions2FeaturesINTEL */
    bool shaderIntegerFunctions2 = false;

    /* VK_EXT_pci_bus_info */
    /* PCIBusInfoPropertiesEXT */
    uint32_t  pciDomain = ~0u;
    uint32_t  pciBus = ~0u;
    uint32_t  pciDevice = ~0u;
    uint32_t  pciFunction = ~0u;

    /* VK_EXT_fragment_density_map */
    /* FragmentDensityMapPropertiesEXT */
    hlsl::uint32_t2 minFragmentDensityTexelSize = { ~0u, ~0u };
    hlsl::uint32_t2 maxFragmentDensityTexelSize = { 0u, 0u };
    bool            fragmentDensityInvocations = false;

    /* VK_GOOGLE_decorate_string */
    bool decorateString = false;

    /* VK_EXT_shader_image_atomic_int64 */
    /* ShaderImageAtomicInt64FeaturesEXT */
    bool shaderImageInt64Atomics = false;
    bool sparseImageInt64Atomics = false;

    // [TODO] this feature introduces new/more pipeline state with VkPipelineRasterizationLineStateCreateInfoEXT
    /* VK_EXT_line_rasterization */
    /* LineRasterizationPropertiesEXT */
    uint32_t lineSubPixelPrecisionBits = 0;

    /* VK_EXT_shader_atomic_float2 */
    /* ShaderAtomicFloat2FeaturesEXT */
    bool shaderBufferFloat16Atomics = false;
    bool shaderBufferFloat16AtomicAdd = false;
    bool shaderBufferFloat16AtomicMinMax = false;
    bool shaderBufferFloat32AtomicMinMax = false;
    bool shaderBufferFloat64AtomicMinMax = false;
    bool shaderSharedFloat16Atomics = false;
    bool shaderSharedFloat16AtomicAdd = false;
    bool shaderSharedFloat16AtomicMinMax = false;
    bool shaderSharedFloat32AtomicMinMax = false;
    bool shaderSharedFloat64AtomicMinMax = false;
    bool shaderImageFloat32AtomicMinMax = false;
    bool sparseImageFloat32AtomicMinMax = false;

    // [DO NOT EXPOSE] won't expose right now, will do if we implement the extension
    /* VK_NV_device_generated_commands */
    /* DeviceGeneratedCommandsPropertiesNV */
    //uint32_t           maxGraphicsShaderGroupCount = 0;
    //uint32_t           maxIndirectSequenceCount = 0;
    //uint32_t           maxIndirectCommandsTokenCount = 0;
    //uint32_t           maxIndirectCommandsStreamCount = 0;
    //uint32_t           maxIndirectCommandsTokenOffset = 0;
    //uint32_t           maxIndirectCommandsStreamStride = 0;
    //uint32_t           minSequencesCountBufferOffsetAlignment = 0x1u<<31;
    //uint32_t           minSequencesIndexBufferOffsetAlignment = 0x1u<<31;
    //uint32_t           minIndirectCommandsBufferOffsetAlignment = 0x1u<<31;

    // [TODO] need impl
    /* VK_EXT_device_memory_report */
    /* DeviceMemoryReportFeaturesEXT */
    bool deviceMemoryReport = false;

    /* VK_KHR_shader_non_semantic_info */
    bool shaderNonSemanticInfo = false;

    // [TODO LATER] not in header (previous comment: too much effort)
    /* GraphicsPipelineLibraryPropertiesEXT *//* provided by VK_EXT_graphics_pipeline_library */
    //bool           graphicsPipelineLibraryFastLinking = false;
    //bool           graphicsPipelineLibraryIndependentInterpolationDecoration = false;

    /* VK_AMD_shader_early_and_late_fragment_tests */
    bool shaderEarlyAndLateFragmentTests = false;

    /* VK_KHR_fragment_shader_barycentric */
    bool fragmentShaderBarycentric = false;

    /* VK_KHR_shader_subgroup_uniform_control_flow */
    /* ShaderSubgroupUniformControlFlowFeaturesKHR */
    bool shaderSubgroupUniformControlFlow = false;

    /* provided by VK_EXT_fragment_density_map2 */
    /* FragmentDensityMap2PropertiesEXT */
    bool                subsampledLoads = false;
    bool                subsampledCoarseReconstructionEarlyAccess = false;
    uint32_t            maxSubsampledArrayLayers = 0u;
    uint32_t            maxDescriptorSetSubsampledSamplers = 0u;

    /* VK_KHR_workgroup_memory_explicit_layout */
    /* WorkgroupMemoryExplicitLayoutFeaturesKHR */
    bool workgroupMemoryExplicitLayout = false;
    bool workgroupMemoryExplicitLayoutScalarBlockLayout = false;
    bool workgroupMemoryExplicitLayout8BitAccess = false;
    bool workgroupMemoryExplicitLayout16BitAccess = false;

    // [TODO] need new commandbuffer methods, etc
    /* VK_EXT_color_write_enable */
    /* ColorWriteEnableFeaturesEXT */
    bool colorWriteEnable = false;

    /* CooperativeMatrixPropertiesKHR  *//* VK_KHR_cooperative_matrix */
    core::bitflag<asset::IShader::E_SHADER_STAGE> cooperativeMatrixSupportedStages = asset::IShader::ESS_UNKNOWN;


    /*  Always enabled if available, reported as limits */

    // Core 1.0 Features

    // mostly just desktops support this
    bool logicOp = false;

    // All iOS GPUs don't support
    bool vertexPipelineStoresAndAtomics = false;

    // ROADMAP 2022 no support on iOS GPUs
    bool fragmentStoresAndAtomics = false;

    // Candidate for promotion, just need to look into Linux and Android
    bool shaderTessellationAndGeometryPointSize = false;

    // Apple GPUs and some Intels don't support
    bool shaderStorageImageMultisample = false;

    // Intel is a special boy and doesn't support
    bool shaderStorageImageReadWithoutFormat = false;

    // ROADMAP 2022 but no iOS GPU supports
    bool shaderStorageImageArrayDynamicIndexing = false;
    
    // Intel Gen12 and ARC are special-boy drivers (TM)
    bool shaderFloat64 = false;

    // poor support on Apple GPUs
    bool variableMultisampleRate = false;

    // Core 1.1 Features or VK_KHR_16bit_storage
    bool storagePushConstant16 = false;
    bool storageInputOutput16 = false;

    // Core 1.1 Features or VK_KHR_multiview, normally would be required but MoltenVK mismatches these
    bool multiviewGeometryShader = false;
    bool multiviewTessellationShader = false;

    // Vulkan 1.2 Core or VK_KHR_draw_indirect_count:
    bool drawIndirectCount = false;

    // Vulkan 1.2 Core or VK_KHR_8bit_storage:
    bool storagePushConstant8 = false;

    // Vulkan 1.2 Core or VK_KHR_shader_atomic_int64:
    bool shaderBufferInt64Atomics = false;
    bool shaderSharedInt64Atomics = false;

    // Vulkan 1.2 Core or VK_KHR_shader_float16_int8:
    bool shaderFloat16 = false;

    // Vulkan 1.2 Core or VK_EXT_descriptor_indexing
    bool shaderInputAttachmentArrayDynamicIndexing = false;
    bool shaderUniformBufferArrayNonUniformIndexing = false;
    bool shaderInputAttachmentArrayNonUniformIndexing = false;
    bool descriptorBindingUniformBufferUpdateAfterBind = false;
    
    // Vulkan 1.2 or VK_EXT_sampler_filter_minmax
    bool samplerFilterMinmax = false; // TODO: Actually implement the sampler flag enums
    
    // Vulkan 1.3 requires but we make concessions for MoltenVK
    bool vulkanMemoryModelAvailabilityVisibilityChains = false;

    // Vulkan 1.2 Core or VK_EXT_shader_viewport_index_layer
    bool shaderOutputViewportIndex = false; // ALIAS: VK_EXT_shader_viewport_index_layer
    bool shaderOutputLayer = false; // ALIAS: VK_EXT_shader_viewport_index_layer

    // Vulkan 1.3 non-optional requires but poor support
    bool shaderDemoteToHelperInvocation = false; // or VK_EXT_shader_demote_to_helper_invocation
    bool shaderTerminateInvocation = false; // or VK_KHR_shader_terminate_invocation

    // Vulkan 1.3 non-optional requires but poor support
    bool shaderZeroInitializeWorkgroupMemory = false; // or VK_KHR_zero_initialize_workgroup_memory




    /* Nabla */
    uint32_t computeUnits = 0u;
    bool dispatchBase = false; // true in Vk, false in GL
    bool allowCommandBufferQueryCopies = false;
    uint32_t maxOptimallyResidentWorkgroupInvocations = 0u; //  its 1D because multidimensional workgroups are an illusion
    uint32_t maxResidentInvocations = 0u; //  These are maximum number of invocations you could expect to execute simultaneously on this device.
    asset::CGLSLCompiler::E_SPIRV_VERSION spirvVersion = asset::CGLSLCompiler::E_SPIRV_VERSION::ESV_1_6;

    // utility functions
    // In the cases where the workgroups synchronise with each other such as work DAGs (i.e. `CScanner`),
    // `workgroupSpinningProtection` is meant to protect against launching a dispatch so wide that
    // a workgroup of the next cut of the DAG spins for an extended time to wait on a workgroup from a previous one.
    inline uint32_t computeOptimalPersistentWorkgroupDispatchSize(const uint64_t elementCount, const uint32_t workgroupSize, const uint32_t workgroupSpinningProtection=1u) const
    {
        assert(elementCount!=0ull && "Input element count can't be 0!");
        const uint64_t infinitelyWideDeviceWGCount = (elementCount-1ull)/(static_cast<uint64_t>(workgroupSize)*static_cast<uint64_t>(workgroupSpinningProtection))+1ull;
        const uint32_t maxResidentWorkgroups = maxResidentInvocations/workgroupSize;
        return static_cast<uint32_t>(core::min<uint64_t>(infinitelyWideDeviceWGCount,maxResidentWorkgroups));
    }

    // we don't compare certain capabilities because they don't mean better/worse
    inline bool isSubsetOf(const SPhysicalDeviceLimits& _rhs) const
    {
        if (maxImageDimension1D > _rhs.maxImageDimension1D) return false;
        if (maxImageDimension2D > _rhs.maxImageDimension2D) return false;
        if (maxImageDimension3D > _rhs.maxImageDimension3D) return false;
        if (maxImageDimensionCube > _rhs.maxImageDimensionCube) return false;
        if (maxImageArrayLayers > _rhs.maxImageArrayLayers) return false;
        if (maxBufferViewTexels > _rhs.maxBufferViewTexels) return false;
        if (maxUBOSize > _rhs.maxUBOSize) return false;
        if (maxSSBOSize > _rhs.maxSSBOSize) return false;
        if (maxPushConstantsSize > _rhs.maxPushConstantsSize) return false;
        if (maxMemoryAllocationCount > _rhs.maxMemoryAllocationCount) return false;
        if (maxSamplerAllocationCount > _rhs.maxSamplerAllocationCount) return false;
        
        if (bufferImageGranularity < _rhs.bufferImageGranularity) return false;

        if (maxPerStageDescriptorSamplers > _rhs.maxPerStageDescriptorSamplers) return false;
        if (maxPerStageDescriptorUBOs > _rhs.maxPerStageDescriptorUBOs) return false;
        if (maxPerStageDescriptorSSBOs > _rhs.maxPerStageDescriptorSSBOs) return false;
        if (maxPerStageDescriptorImages > _rhs.maxPerStageDescriptorImages) return false;
        if (maxPerStageDescriptorStorageImages > _rhs.maxPerStageDescriptorStorageImages) return false;
        if (maxPerStageDescriptorInputAttachments > _rhs.maxPerStageDescriptorInputAttachments) return false;
        if (maxPerStageResources > _rhs.maxPerStageResources) return false;

        if (maxDescriptorSetSamplers > _rhs.maxDescriptorSetSamplers) return false;
        if (maxDescriptorSetUBOs > _rhs.maxDescriptorSetUBOs) return false;
        if (maxDescriptorSetDynamicOffsetUBOs > _rhs.maxDescriptorSetDynamicOffsetUBOs) return false;
        if (maxDescriptorSetSSBOs > _rhs.maxDescriptorSetSSBOs) return false;
        if (maxDescriptorSetDynamicOffsetSSBOs > _rhs.maxDescriptorSetDynamicOffsetSSBOs) return false;
        if (maxDescriptorSetImages > _rhs.maxDescriptorSetImages) return false;
        if (maxDescriptorSetStorageImages > _rhs.maxDescriptorSetStorageImages) return false;
        if (maxDescriptorSetInputAttachments > _rhs.maxDescriptorSetInputAttachments) return false;

        if (maxVertexOutputComponents > _rhs.maxVertexOutputComponents) return false;

        if (maxTessellationGenerationLevel > _rhs.maxTessellationGenerationLevel) return false;
        if (maxTessellationPatchSize > _rhs.maxTessellationPatchSize) return false;
        if (maxTessellationControlPerVertexInputComponents > _rhs.maxTessellationControlPerVertexInputComponents) return false;
        if (maxTessellationControlPerVertexOutputComponents > _rhs.maxTessellationControlPerVertexOutputComponents) return false;
        if (maxTessellationControlPerPatchOutputComponents > _rhs.maxTessellationControlPerPatchOutputComponents) return false;
        if (maxTessellationControlTotalOutputComponents > _rhs.maxTessellationControlTotalOutputComponents) return false;
        if (maxTessellationEvaluationInputComponents > _rhs.maxTessellationEvaluationInputComponents) return false;
        if (maxTessellationEvaluationOutputComponents > _rhs.maxTessellationEvaluationOutputComponents) return false;

        if (maxGeometryShaderInvocations > _rhs.maxGeometryShaderInvocations) return false;
        if (maxGeometryInputComponents > _rhs.maxGeometryInputComponents) return false;
        if (maxGeometryOutputComponents > _rhs.maxGeometryOutputComponents) return false;
        if (maxGeometryOutputVertices > _rhs.maxGeometryOutputVertices) return false;
        if (maxGeometryTotalOutputComponents > _rhs.maxGeometryTotalOutputComponents) return false;

        if (maxFragmentInputComponents > _rhs.maxFragmentInputComponents) return false;
        if (maxFragmentOutputAttachments > _rhs.maxFragmentOutputAttachments) return false;
        if (maxFragmentDualSrcAttachments > _rhs.maxFragmentDualSrcAttachments) return false;
        if (maxFragmentCombinedOutputResources > _rhs.maxFragmentCombinedOutputResources) return false;

        if (maxComputeSharedMemorySize > _rhs.maxComputeSharedMemorySize) return false;
        if (maxComputeWorkGroupCount[0] > _rhs.maxComputeWorkGroupCount[0]) return false;
        if (maxComputeWorkGroupCount[1] > _rhs.maxComputeWorkGroupCount[1]) return false;
        if (maxComputeWorkGroupCount[2] > _rhs.maxComputeWorkGroupCount[2]) return false;
        if (maxComputeWorkGroupInvocations > _rhs.maxComputeWorkGroupInvocations) return false;
        if (maxWorkgroupSize[0] > _rhs.maxWorkgroupSize[0]) return false;
        if (maxWorkgroupSize[1] > _rhs.maxWorkgroupSize[1]) return false;
        if (maxWorkgroupSize[2] > _rhs.maxWorkgroupSize[2]) return false;

        if (subPixelPrecisionBits > _rhs.subPixelPrecisionBits) return false;
        if (subTexelPrecisionBits > _rhs.subTexelPrecisionBits) return false;
        if (mipmapPrecisionBits > _rhs.mipmapPrecisionBits) return false;

        if (maxDrawIndirectCount > _rhs.maxDrawIndirectCount) return false;

        if (maxSamplerLodBias > _rhs.maxSamplerLodBias) return false;
        if (maxSamplerAnisotropyLog2 > _rhs.maxSamplerAnisotropyLog2) return false;

        if (maxViewports > _rhs.maxViewports) return false;
        if (maxViewportDims[0] > _rhs.maxViewportDims[0]) return false;
        if (maxViewportDims[1] > _rhs.maxViewportDims[1]) return false;
        if (viewportBoundsRange[0] < _rhs.viewportBoundsRange[0] || viewportBoundsRange[1] > _rhs.viewportBoundsRange[1]) return false;
        if (viewportSubPixelBits > _rhs.viewportSubPixelBits) return false;

        // the `>` is on purpose, its not a restriction its a guarantee
        if (minMemoryMapAlignment > _rhs.minMemoryMapAlignment) return false;
        if (bufferViewAlignment < _rhs.bufferViewAlignment) return false;
        if (minUBOAlignment < _rhs.minUBOAlignment) return false;
        if (minSSBOAlignment < _rhs.minSSBOAlignment) return false;

        if (minTexelOffset < _rhs.minTexelOffset || maxTexelOffset > _rhs.maxTexelOffset) return false;
        if (minTexelGatherOffset < _rhs.minTexelGatherOffset || maxTexelGatherOffset > _rhs.maxTexelGatherOffset) return false;

        if (minInterpolationOffset < _rhs.minInterpolationOffset || maxInterpolationOffset > _rhs.maxInterpolationOffset) return false;
        if (subPixelInterpolationOffsetBits > _rhs.subPixelInterpolationOffsetBits) return false;
        if (maxFramebufferWidth > _rhs.maxFramebufferWidth) return false;
        if (maxFramebufferHeight > _rhs.maxFramebufferHeight) return false;
        if (maxFramebufferLayers > _rhs.maxFramebufferLayers) return false;

        if (maxColorAttachments > _rhs.maxColorAttachments) return false;

        if (maxSampleMaskWords > _rhs.maxSampleMaskWords) return false;

        // don't compare certain things, they don't make your device better or worse
        //if (timestampPeriodInNanoSeconds < _rhs.timestampPeriodInNanoSeconds) return false;

        if (maxClipDistances > _rhs.maxClipDistances) return false;
        if (maxCullDistances > _rhs.maxCullDistances) return false;
        if (maxCombinedClipAndCullDistances > _rhs.maxCombinedClipAndCullDistances) return false;

        if (discreteQueuePriorities > _rhs.discreteQueuePriorities) return false;

        if (pointSizeRange[0] < _rhs.pointSizeRange[0] || pointSizeRange[1] > _rhs.pointSizeRange[1]) return false;
        if (lineWidthRange[0] < _rhs.lineWidthRange[0] || lineWidthRange[1] > _rhs.lineWidthRange[1]) return false;
        if (pointSizeGranularity < _rhs.pointSizeGranularity) return false;
        if (lineWidthGranularity < _rhs.lineWidthGranularity) return false;

        if (strictLines && !_rhs.strictLines) return false;

        if (standardSampleLocations && !_rhs.standardSampleLocations) return false;

        if (optimalBufferCopyOffsetAlignment < _rhs.optimalBufferCopyOffsetAlignment) return false;
        if (optimalBufferCopyRowPitchAlignment < _rhs.optimalBufferCopyRowPitchAlignment) return false;
        if (nonCoherentAtomSize < _rhs.nonCoherentAtomSize) return false;

        // don't compare certain things, they don't make your device better or worse
        //if (subgroupSize > _rhs.subgroupSize) return false;
        if (!_rhs.subgroupOpsShaderStages.hasFlags(subgroupOpsShaderStages)) return false;
        if (shaderSubgroupClustered && !_rhs.shaderSubgroupClustered) return false;
        if (shaderSubgroupArithmetic && !_rhs.shaderSubgroupArithmetic) return false;
        if (shaderSubgroupQuad && !_rhs.shaderSubgroupQuad) return false;
        if (shaderSubgroupQuadAllStages && !_rhs.shaderSubgroupQuadAllStages) return false;

        if (pointClippingBehavior==EPCB_ALL_CLIP_PLANES && _rhs.pointClippingBehavior==EPCB_USER_CLIP_PLANES_ONLY) return false;

        if (maxMultiviewViewCount > _rhs.maxMultiviewViewCount) return false;
        if (maxMultiviewInstanceIndex > _rhs.maxMultiviewInstanceIndex) return false;
        
        if (maxPerSetDescriptors > _rhs.maxPerSetDescriptors) return false;
        if (maxMemoryAllocationSize > _rhs.maxMemoryAllocationSize) return false;

        if (shaderSignedZeroInfNanPreserveFloat64 && !_rhs.shaderSignedZeroInfNanPreserveFloat64) return false;
        if (shaderDenormPreserveFloat16 && !_rhs.shaderDenormPreserveFloat16) return false;
        if (shaderDenormPreserveFloat32 && !_rhs.shaderDenormPreserveFloat32) return false;
        if (shaderDenormPreserveFloat64 && !_rhs.shaderDenormPreserveFloat64) return false;
        if (shaderDenormFlushToZeroFloat16 && !_rhs.shaderDenormFlushToZeroFloat16) return false;
        if (shaderDenormFlushToZeroFloat32 && !_rhs.shaderDenormFlushToZeroFloat32) return false;
        if (shaderDenormFlushToZeroFloat64 && !_rhs.shaderDenormFlushToZeroFloat64) return false;
        if (shaderRoundingModeRTEFloat16 && !_rhs.shaderRoundingModeRTEFloat16) return false;
        if (shaderRoundingModeRTEFloat32 && !_rhs.shaderRoundingModeRTEFloat32) return false;
        if (shaderRoundingModeRTEFloat64 && !_rhs.shaderRoundingModeRTEFloat64) return false;
        if (shaderRoundingModeRTZFloat16 && !_rhs.shaderRoundingModeRTZFloat16) return false;
        if (shaderRoundingModeRTZFloat32 && !_rhs.shaderRoundingModeRTZFloat32) return false;
        if (shaderRoundingModeRTZFloat64 && !_rhs.shaderRoundingModeRTZFloat64) return false;

        if (maxUpdateAfterBindDescriptorsInAllPools > _rhs.maxUpdateAfterBindDescriptorsInAllPools) return false;
        if (shaderUniformBufferArrayNonUniformIndexingNative && !_rhs.shaderUniformBufferArrayNonUniformIndexingNative) return false;
        if (shaderSampledImageArrayNonUniformIndexingNative && !_rhs.shaderSampledImageArrayNonUniformIndexingNative) return false;
        if (shaderStorageBufferArrayNonUniformIndexingNative && !_rhs.shaderStorageBufferArrayNonUniformIndexingNative) return false;
        if (shaderStorageImageArrayNonUniformIndexingNative && !_rhs.shaderStorageImageArrayNonUniformIndexingNative) return false;
        if (shaderInputAttachmentArrayNonUniformIndexingNative && !_rhs.shaderInputAttachmentArrayNonUniformIndexingNative) return false;
        if (robustBufferAccessUpdateAfterBind && !_rhs.robustBufferAccessUpdateAfterBind) return false;
        if (quadDivergentImplicitLod && !_rhs.quadDivergentImplicitLod) return false;

        if (maxPerStageDescriptorUpdateAfterBindSamplers > _rhs.maxPerStageDescriptorUpdateAfterBindSamplers) return false;
        if (maxPerStageDescriptorUpdateAfterBindUBOs > _rhs.maxPerStageDescriptorUpdateAfterBindUBOs) return false;
        if (maxPerStageDescriptorUpdateAfterBindSSBOs > _rhs.maxPerStageDescriptorUpdateAfterBindSSBOs) return false;
        if (maxPerStageDescriptorUpdateAfterBindImages > _rhs.maxPerStageDescriptorUpdateAfterBindImages) return false;
        if (maxPerStageDescriptorUpdateAfterBindStorageImages > _rhs.maxPerStageDescriptorUpdateAfterBindStorageImages) return false;
        if (maxPerStageDescriptorUpdateAfterBindInputAttachments > _rhs.maxPerStageDescriptorUpdateAfterBindInputAttachments) return false;
        if (maxPerStageUpdateAfterBindResources > _rhs.maxPerStageUpdateAfterBindResources) return false;
        if (maxDescriptorSetUpdateAfterBindSamplers > _rhs.maxDescriptorSetUpdateAfterBindSamplers) return false;
        if (maxDescriptorSetUpdateAfterBindUBOs > _rhs.maxDescriptorSetUpdateAfterBindUBOs) return false;
        if (maxDescriptorSetUpdateAfterBindDynamicOffsetUBOs > _rhs.maxDescriptorSetUpdateAfterBindDynamicOffsetUBOs) return false;
        if (maxDescriptorSetUpdateAfterBindSSBOs > _rhs.maxDescriptorSetUpdateAfterBindSSBOs) return false;
        if (maxDescriptorSetUpdateAfterBindDynamicOffsetSSBOs > _rhs.maxDescriptorSetUpdateAfterBindDynamicOffsetSSBOs) return false;
        if (maxDescriptorSetUpdateAfterBindImages > _rhs.maxDescriptorSetUpdateAfterBindImages) return false;
        if (maxDescriptorSetUpdateAfterBindStorageImages > _rhs.maxDescriptorSetUpdateAfterBindStorageImages) return false;
        if (maxDescriptorSetUpdateAfterBindInputAttachments > _rhs.maxDescriptorSetUpdateAfterBindInputAttachments) return false;

        if (!_rhs.supportedDepthResolveModes.hasFlags(supportedDepthResolveModes)) return false;
        if (!_rhs.supportedStencilResolveModes.hasFlags(supportedStencilResolveModes)) return false;
        if (independentResolveNone && !_rhs.independentResolveNone) return false;
        if (independentResolve && !_rhs.independentResolve) return false;

        if (filterMinmaxImageComponentMapping && !_rhs.filterMinmaxImageComponentMapping) return false;

        if (minSubgroupSize < _rhs.minSubgroupSize || maxSubgroupSize > _rhs.maxSubgroupSize) return false;
        if (maxComputeWorkgroupSubgroups > _rhs.maxComputeWorkgroupSubgroups) return false;
        if (!_rhs.requiredSubgroupSizeStages.hasFlags(requiredSubgroupSizeStages)) return false;

        if (integerDotProduct8BitUnsignedAccelerated && !_rhs.integerDotProduct8BitUnsignedAccelerated) return false;
        if (integerDotProduct8BitSignedAccelerated && !_rhs.integerDotProduct8BitSignedAccelerated) return false;
        if (integerDotProduct8BitMixedSignednessAccelerated && !_rhs.integerDotProduct8BitMixedSignednessAccelerated) return false;
        if (integerDotProduct4x8BitPackedUnsignedAccelerated && !_rhs.integerDotProduct4x8BitPackedUnsignedAccelerated) return false;
        if (integerDotProduct4x8BitPackedSignedAccelerated && !_rhs.integerDotProduct4x8BitPackedSignedAccelerated) return false;
        if (integerDotProduct4x8BitPackedMixedSignednessAccelerated && !_rhs.integerDotProduct4x8BitPackedMixedSignednessAccelerated) return false;
        if (integerDotProduct16BitUnsignedAccelerated && !_rhs.integerDotProduct16BitUnsignedAccelerated) return false;
        if (integerDotProduct16BitSignedAccelerated && !_rhs.integerDotProduct16BitSignedAccelerated) return false;
        if (integerDotProduct16BitMixedSignednessAccelerated && !_rhs.integerDotProduct16BitMixedSignednessAccelerated) return false;
        if (integerDotProduct32BitUnsignedAccelerated && !_rhs.integerDotProduct32BitUnsignedAccelerated) return false;
        if (integerDotProduct32BitSignedAccelerated && !_rhs.integerDotProduct32BitSignedAccelerated) return false;
        if (integerDotProduct32BitMixedSignednessAccelerated && !_rhs.integerDotProduct32BitMixedSignednessAccelerated) return false;
        if (integerDotProduct64BitUnsignedAccelerated && !_rhs.integerDotProduct64BitUnsignedAccelerated) return false;
        if (integerDotProduct64BitSignedAccelerated && !_rhs.integerDotProduct64BitSignedAccelerated) return false;
        if (integerDotProduct64BitMixedSignednessAccelerated && !_rhs.integerDotProduct64BitMixedSignednessAccelerated) return false;
        if (integerDotProductAccumulatingSaturating8BitUnsignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating8BitUnsignedAccelerated) return false;
        if (integerDotProductAccumulatingSaturating8BitSignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating8BitSignedAccelerated) return false;
        if (integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated && !_rhs.integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated) return false;
        if (integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated) return false;
        if (integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated) return false;
        if (integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated && !_rhs.integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated) return false;
        if (integerDotProductAccumulatingSaturating16BitUnsignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating16BitUnsignedAccelerated) return false;
        if (integerDotProductAccumulatingSaturating16BitSignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating16BitSignedAccelerated) return false;
        if (integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated && !_rhs.integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated) return false;
        if (integerDotProductAccumulatingSaturating32BitUnsignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating32BitUnsignedAccelerated) return false;
        if (integerDotProductAccumulatingSaturating32BitSignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating32BitSignedAccelerated) return false;
        if (integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated && !_rhs.integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated) return false;
        if (integerDotProductAccumulatingSaturating64BitUnsignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating64BitUnsignedAccelerated) return false;
        if (integerDotProductAccumulatingSaturating64BitSignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating64BitSignedAccelerated) return false;
        if (integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated && !_rhs.integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated) return false;

        if (storageTexelBufferOffsetAlignmentBytes < _rhs.storageTexelBufferOffsetAlignmentBytes) return false;
        if (uniformTexelBufferOffsetAlignmentBytes < _rhs.uniformTexelBufferOffsetAlignmentBytes) return false;

        if (maxBufferSize > _rhs.maxBufferSize) return false;

        if (minImportedHostPointerAlignment < _rhs.minImportedHostPointerAlignment) return false;

        if (shaderBufferFloat32AtomicAdd && !_rhs.shaderBufferFloat32AtomicAdd) return false;
        if (shaderBufferFloat64Atomics && !_rhs.shaderBufferFloat64Atomics) return false;
        if (shaderBufferFloat64AtomicAdd && !_rhs.shaderBufferFloat64AtomicAdd) return false;
        if (shaderSharedFloat32AtomicAdd && !_rhs.shaderSharedFloat32AtomicAdd) return false;
        if (shaderSharedFloat64Atomics && !_rhs.shaderSharedFloat64Atomics) return false;
        if (shaderSharedFloat64AtomicAdd && !_rhs.shaderSharedFloat64AtomicAdd) return false;
        if (shaderImageFloat32AtomicAdd && !_rhs.shaderImageFloat32AtomicAdd) return false;
        if (sparseImageFloat32Atomics && !_rhs.sparseImageFloat32Atomics) return false;
        if (sparseImageFloat32AtomicAdd && !_rhs.sparseImageFloat32AtomicAdd) return false;

        if (robustStorageBufferAccessSizeAlignment < _rhs.robustStorageBufferAccessSizeAlignment) return false;
        if (robustUniformBufferAccessSizeAlignment < _rhs.robustUniformBufferAccessSizeAlignment) return false;

        if (shaderTrinaryMinmax && !_rhs.shaderTrinaryMinmax) return false;

        if (shaderExplicitVertexParameter && !_rhs.shaderExplicitVertexParameter) return false;

        if (gpuShaderHalfFloatAMD && !_rhs.gpuShaderHalfFloatAMD) return false;

        if (shaderImageLoadStoreLod && !_rhs.shaderImageLoadStoreLod) return false;

        if (displayTiming && !_rhs.displayTiming) return false;

        if (maxDiscardRectangles > _rhs.maxDiscardRectangles) return false;

        // don't compare certain things, they don't make your device better or worse
        //if (primitiveOverestimationSize > _rhs.primitiveOverestimationSize) return false;
        if (maxExtraPrimitiveOverestimationSize > _rhs.maxExtraPrimitiveOverestimationSize) return false;
        if (extraPrimitiveOverestimationSizeGranularity < _rhs.extraPrimitiveOverestimationSizeGranularity) return false;
        if (primitiveUnderestimation && !_rhs.primitiveUnderestimation) return false;
        if (conservativePointAndLineRasterization && !_rhs.conservativePointAndLineRasterization) return false;
        if (degenerateTrianglesRasterized && !_rhs.degenerateTrianglesRasterized) return false;
        if (degenerateLinesRasterized && !_rhs.degenerateLinesRasterized) return false;
        if (fullyCoveredFragmentShaderInputVariable && !_rhs.fullyCoveredFragmentShaderInputVariable) return false;
        if (conservativeRasterizationPostDepthCoverage && !_rhs.conservativeRasterizationPostDepthCoverage) return false;

        if (queueFamilyForeign && !_rhs.queueFamilyForeign) return false;

        if (shaderStencilExport && !_rhs.shaderStencilExport) return false;

        if (variableSampleLocations && !_rhs.variableSampleLocations) return false;
        if (sampleLocationSubPixelBits > _rhs.sampleLocationSubPixelBits) return false;
        if (!_rhs.sampleLocationSampleCounts.hasFlags(sampleLocationSampleCounts)) return false;
        if (maxSampleLocationGridSize.x > _rhs.maxSampleLocationGridSize.x) return false;
        if (maxSampleLocationGridSize.y > _rhs.maxSampleLocationGridSize.y) return false;
        if (sampleLocationCoordinateRange[0] < _rhs.sampleLocationCoordinateRange[0] || sampleLocationCoordinateRange[1] > _rhs.sampleLocationCoordinateRange[1]) return false;

        if (maxAccelerationStructureGeometryCount > _rhs.maxAccelerationStructureGeometryCount) return false;
        if (maxAccelerationStructureInstanceCount > _rhs.maxAccelerationStructureInstanceCount) return false;
        if (maxAccelerationStructurePrimitiveCount > _rhs.maxAccelerationStructurePrimitiveCount) return false;
        if (maxPerStageDescriptorAccelerationStructures > _rhs.maxPerStageDescriptorAccelerationStructures) return false;
        if (maxPerStageDescriptorUpdateAfterBindAccelerationStructures > _rhs.maxPerStageDescriptorUpdateAfterBindAccelerationStructures) return false;
        if (maxDescriptorSetAccelerationStructures > _rhs.maxDescriptorSetAccelerationStructures) return false;
        if (maxDescriptorSetUpdateAfterBindAccelerationStructures > _rhs.maxDescriptorSetUpdateAfterBindAccelerationStructures) return false;
        if (minAccelerationStructureScratchOffsetAlignment < _rhs.minAccelerationStructureScratchOffsetAlignment) return false;

        if (maxRayRecursionDepth > _rhs.maxRayRecursionDepth) return false;
        if (maxShaderGroupStride > _rhs.maxShaderGroupStride) return false;
        if (shaderGroupBaseAlignment < _rhs.shaderGroupBaseAlignment) return false;
        if (maxRayDispatchInvocationCount > _rhs.maxRayDispatchInvocationCount) return false;
        if (shaderGroupHandleAlignment < _rhs.shaderGroupHandleAlignment) return false;
        if (maxRayHitAttributeSize > _rhs.maxRayHitAttributeSize) return false;

        if (shaderSMBuiltins && !_rhs.shaderSMBuiltins) return false;

        if (postDepthCoverage && !_rhs.postDepthCoverage) return false;

        if (shaderDeviceClock && !_rhs.shaderDeviceClock) return false;

        if (computeDerivativeGroupQuads && !_rhs.computeDerivativeGroupQuads) return false;
        if (computeDerivativeGroupLinear && !_rhs.computeDerivativeGroupLinear) return false;

        if (imageFootprint && !_rhs.imageFootprint) return false;

        if (shaderIntegerFunctions2 && !_rhs.shaderIntegerFunctions2) return false;

        // don't compare certain things, they don't make your device better or worse
        // uint32_t  pciDomain = ~0u;
        // uint32_t  pciBus = ~0u;
        // uint32_t  pciDevice = ~0u;
        // uint32_t  pciFunction = ~0u;

        if (minFragmentDensityTexelSize.x < _rhs.minFragmentDensityTexelSize.x) return false;
        if (minFragmentDensityTexelSize.y < _rhs.minFragmentDensityTexelSize.y) return false;
        if (fragmentDensityInvocations && !_rhs.fragmentDensityInvocations) return false;

        if (decorateString && !_rhs.decorateString) return false;

        if (shaderImageInt64Atomics && !_rhs.shaderImageInt64Atomics) return false;
        if (sparseImageInt64Atomics && !_rhs.sparseImageInt64Atomics) return false;

        if (lineSubPixelPrecisionBits > _rhs.lineSubPixelPrecisionBits) return false;

        if (shaderBufferFloat16Atomics && !_rhs.shaderBufferFloat16Atomics) return false;
        if (shaderBufferFloat16AtomicAdd && !_rhs.shaderBufferFloat16AtomicAdd) return false;
        if (shaderBufferFloat16AtomicMinMax && !_rhs.shaderBufferFloat16AtomicMinMax) return false;
        if (shaderBufferFloat32AtomicMinMax && !_rhs.shaderBufferFloat32AtomicMinMax) return false;
        if (shaderBufferFloat64AtomicMinMax && !_rhs.shaderBufferFloat64AtomicMinMax) return false;
        if (shaderSharedFloat16Atomics && !_rhs.shaderSharedFloat16Atomics) return false;
        if (shaderSharedFloat16AtomicAdd && !_rhs.shaderSharedFloat16AtomicAdd) return false;
        if (shaderSharedFloat16AtomicMinMax && !_rhs.shaderSharedFloat16AtomicMinMax) return false;
        if (shaderSharedFloat32AtomicMinMax && !_rhs.shaderSharedFloat32AtomicMinMax) return false;
        if (shaderSharedFloat64AtomicMinMax && !_rhs.shaderSharedFloat64AtomicMinMax) return false;
        if (shaderImageFloat32AtomicMinMax && !_rhs.shaderImageFloat32AtomicMinMax) return false;
        if (sparseImageFloat32AtomicMinMax && !_rhs.sparseImageFloat32AtomicMinMax) return false;
        
        if (deviceMemoryReport && !_rhs.deviceMemoryReport) return false;
     
        if (shaderNonSemanticInfo && !_rhs.shaderNonSemanticInfo) return false;
     
        if (shaderEarlyAndLateFragmentTests && !_rhs.shaderEarlyAndLateFragmentTests) return false;
     
        if (fragmentShaderBarycentric && !_rhs.fragmentShaderBarycentric) return false;
     
        if (shaderSubgroupUniformControlFlow && !_rhs.shaderSubgroupUniformControlFlow) return false;

        if (subsampledLoads && !_rhs.subsampledLoads) return false;
        if (subsampledCoarseReconstructionEarlyAccess && !_rhs.subsampledCoarseReconstructionEarlyAccess) return false;
        if (maxSubsampledArrayLayers > _rhs.maxSubsampledArrayLayers) return false;
        if (maxDescriptorSetSubsampledSamplers > _rhs.maxDescriptorSetSubsampledSamplers) return false;

        if (workgroupMemoryExplicitLayout && !_rhs.workgroupMemoryExplicitLayout) return false;
        if (workgroupMemoryExplicitLayoutScalarBlockLayout && !_rhs.workgroupMemoryExplicitLayoutScalarBlockLayout) return false;
        if (workgroupMemoryExplicitLayout8BitAccess && !_rhs.workgroupMemoryExplicitLayout8BitAccess) return false;
        if (workgroupMemoryExplicitLayout16BitAccess && !_rhs.workgroupMemoryExplicitLayout16BitAccess) return false;
        
        if (colorWriteEnable && !_rhs.colorWriteEnable) return false;

        if (!_rhs.cooperativeMatrixSupportedStages.hasFlags(cooperativeMatrixSupportedStages)) return false;

        if (logicOp && !_rhs.logicOp) return false;
        
        if (vertexPipelineStoresAndAtomics && !_rhs.vertexPipelineStoresAndAtomics) return false;
        
        if (fragmentStoresAndAtomics && !_rhs.fragmentStoresAndAtomics) return false;
        
        if (shaderTessellationAndGeometryPointSize && !_rhs.shaderTessellationAndGeometryPointSize) return false;
        
        if (shaderStorageImageMultisample && !_rhs.shaderStorageImageMultisample) return false;
        
        if (shaderStorageImageReadWithoutFormat && !_rhs.shaderStorageImageReadWithoutFormat) return false;
        
        if (shaderStorageImageArrayDynamicIndexing && !_rhs.shaderStorageImageArrayDynamicIndexing) return false;
        
        if (shaderFloat64 && !_rhs.shaderFloat64) return false;
        
        if (variableMultisampleRate && !_rhs.variableMultisampleRate) return false;
        
        if (storagePushConstant16 && !_rhs.storagePushConstant16) return false;
        if (storageInputOutput16 && !_rhs.storageInputOutput16) return false;
        
        if (multiviewGeometryShader && !_rhs.multiviewGeometryShader) return false;
        if (multiviewTessellationShader && !_rhs.multiviewTessellationShader) return false;
        
        if (drawIndirectCount && !_rhs.drawIndirectCount) return false;
        
        if (storagePushConstant8 && !_rhs.storagePushConstant8) return false;
        
        if (shaderBufferInt64Atomics && !_rhs.shaderBufferInt64Atomics) return false;
        if (shaderSharedInt64Atomics && !_rhs.shaderSharedInt64Atomics) return false;
        
        if (shaderFloat16 && !_rhs.shaderFloat16) return false;
        
        if (shaderInputAttachmentArrayDynamicIndexing && !_rhs.shaderInputAttachmentArrayDynamicIndexing) return false;
        if (shaderUniformBufferArrayNonUniformIndexing && !_rhs.shaderUniformBufferArrayNonUniformIndexing) return false;
        if (shaderInputAttachmentArrayNonUniformIndexing && !_rhs.shaderInputAttachmentArrayNonUniformIndexing) return false;
        if (descriptorBindingUniformBufferUpdateAfterBind && !_rhs.descriptorBindingUniformBufferUpdateAfterBind) return false;
        
        if (samplerFilterMinmax && !_rhs.samplerFilterMinmax) return false;
        
        if (vulkanMemoryModelAvailabilityVisibilityChains && !_rhs.vulkanMemoryModelAvailabilityVisibilityChains) return false;
        
        if (shaderOutputViewportIndex && !_rhs.shaderOutputViewportIndex) return false;
        if (shaderOutputLayer && !_rhs.shaderOutputLayer) return false;
        
        if (shaderDemoteToHelperInvocation && !_rhs.shaderDemoteToHelperInvocation) return false;
        if (shaderTerminateInvocation && !_rhs.shaderTerminateInvocation) return false;
        
        if (shaderZeroInitializeWorkgroupMemory && !_rhs.shaderZeroInitializeWorkgroupMemory) return false;

        // Nabla
        if (computeUnits > _rhs.computeUnits) return false;
        if (dispatchBase && !_rhs.dispatchBase) return false;
        if (allowCommandBufferQueryCopies && !_rhs.allowCommandBufferQueryCopies) return false;
        if (maxOptimallyResidentWorkgroupInvocations > _rhs.maxOptimallyResidentWorkgroupInvocations) return false;
        if (maxResidentInvocations > _rhs.maxResidentInvocations) return false;
        if (spirvVersion > _rhs.spirvVersion) return false;

        return true;
    }
};

} // nbl::video

#endif
