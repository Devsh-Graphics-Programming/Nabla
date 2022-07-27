#ifndef _NBL_VIDEO_S_PHYSICAL_DEVICE_LIMITS_H_INCLUDED_
#define _NBL_VIDEO_S_PHYSICAL_DEVICE_LIMITS_H_INCLUDED_

#include <type_traits>
#include "nbl/asset/utils/IGLSLCompiler.h" // asset::IGLSLCompiler::E_SPIRV_VERSION
#include "nbl/asset/IImage.h"

namespace nbl::video
{

struct SPhysicalDeviceLimits
{
    enum E_TRI_BOOLEAN
    {
        ETB_FALSE,
        ETB_DONT_KNOW,
        ETB_TRUE
    };

    /* Vulkan 1.0 Core  */
    uint32_t maxImageDimension1D = 0u;
    uint32_t maxImageDimension2D = 0u;
    uint32_t maxImageDimension3D = 0u;
    uint32_t maxImageDimensionCube = 0u;
    uint32_t maxImageArrayLayers = 0u;
    uint32_t maxBufferViewTexels = 0u;
    uint32_t maxUBOSize = 0u;
    uint32_t maxSSBOSize = 0u;
    uint32_t maxPushConstantsSize = 0u;
    uint32_t maxMemoryAllocationCount = 0u;
    uint32_t maxSamplerAllocationCount = 0u;
    size_t bufferImageGranularity = 0ull;
    //size_t          sparseAddressSpaceSize;         // [TODO LATER] when we support sparse
    //uint32_t              maxBoundDescriptorSets;         // [DO NOT EXPOSE] we've kinda hardcoded the engine to 4 currently

    uint32_t maxPerStageDescriptorSamplers = 0u;  // Descriptors with a type of EDT_COMBINED_IMAGE_SAMPLER count against this limit
    uint32_t maxPerStageDescriptorUBOs = 0u;
    uint32_t maxPerStageDescriptorSSBOs = 0u;
    uint32_t maxPerStageDescriptorImages = 0u; // Descriptors with a type of EDT_COMBINED_IMAGE_SAMPLER, EDT_UNIFORM_TEXEL_BUFFER count against this limit.
    uint32_t maxPerStageDescriptorStorageImages = 0u;
    uint32_t maxPerStageDescriptorInputAttachments = 0u;
    uint32_t maxPerStageResources = 0u;

    uint32_t maxDescriptorSetSamplers = 0u; // Descriptors with a type of EDT_COMBINED_IMAGE_SAMPLER count against this limit
    uint32_t maxDescriptorSetUBOs = 0u;
    uint32_t maxDescriptorSetDynamicOffsetUBOs = 0u;
    uint32_t maxDescriptorSetSSBOs = 0u;
    uint32_t maxDescriptorSetDynamicOffsetSSBOs = 0u;
    uint32_t maxDescriptorSetImages = 0u; // Descriptors with a type of EDT_COMBINED_IMAGE_SAMPLER, EDT_UNIFORM_TEXEL_BUFFER count against this limit.
    uint32_t maxDescriptorSetStorageImages = 0u;
    uint32_t maxDescriptorSetInputAttachments = 0u;

    uint32_t maxVertexOutputComponents = 0u;

    uint32_t maxTessellationGenerationLevel = 0u;
    uint32_t maxTessellationPatchSize = 0u;
    uint32_t maxTessellationControlPerVertexInputComponents = 0u;
    uint32_t maxTessellationControlPerVertexOutputComponents = 0u;
    uint32_t maxTessellationControlPerPatchOutputComponents = 0u;
    uint32_t maxTessellationControlTotalOutputComponents = 0u;
    uint32_t maxTessellationEvaluationInputComponents = 0u;
    uint32_t maxTessellationEvaluationOutputComponents = 0u;
    uint32_t maxGeometryShaderInvocations = 0u;
    uint32_t maxGeometryInputComponents = 0u;
    uint32_t maxGeometryOutputComponents = 0u;
    uint32_t maxGeometryOutputVertices = 0u;
    uint32_t maxGeometryTotalOutputComponents = 0u;
    uint32_t maxFragmentInputComponents = 0u;
    uint32_t maxFragmentOutputAttachments = 0u;
    uint32_t maxFragmentDualSrcAttachments = 0u;
    uint32_t maxFragmentCombinedOutputResources = 0u;
    uint32_t maxComputeSharedMemorySize;
    uint32_t maxComputeWorkGroupCount[3];
    uint32_t maxComputeWorkGroupInvocations = 0u;
    uint32_t maxWorkgroupSize[3] = {};
    uint32_t subPixelPrecisionBits = 0u;
    //uint32_t              subTexelPrecisionBits;
    //uint32_t              mipmapPrecisionBits; // [TODO] require investigation GL+ES spec
    //uint32_t              maxDrawIndexedIndexValue;
    uint32_t maxDrawIndirectCount = 0u;
    float    maxSamplerLodBias = 0.0f;
    uint8_t  maxSamplerAnisotropyLog2 = 0.0f;
    uint32_t maxViewports = 0u;
    uint32_t maxViewportDims[2] = {};
    float    viewportBoundsRange[2]; // [min, max]
    uint32_t viewportSubPixelBits = 0u;
    size_t   minMemoryMapAlignment = 0ull;
    uint32_t bufferViewAlignment = 0u;
    uint32_t minUBOAlignment = 0u;
    uint32_t minSSBOAlignment = 0u;
    int32_t  minTexelOffset = 0;
    uint32_t maxTexelOffset = 0u;
    int32_t  minTexelGatherOffset = 0;
    uint32_t maxTexelGatherOffset = 0u;
    float    minInterpolationOffset = 0.0f;
    float    maxInterpolationOffset = 0.0f;
    //uint32_t              subPixelInterpolationOffsetBits;
    uint32_t maxFramebufferWidth = 0u;
    uint32_t maxFramebufferHeight = 0u;
    uint32_t maxFramebufferLayers = 0u;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferColorSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferDepthSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferStencilSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferNoAttachmentsSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    uint32_t maxColorAttachments = 0u;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageColorSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageIntegerSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageDepthSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageStencilSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> storageImageSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    uint32_t maxSampleMaskWords = 0u;
    bool timestampComputeAndGraphics = false;
    float timestampPeriodInNanoSeconds = 0.0f; // timestampPeriod is the number of nanoseconds required for a timestamp query to be incremented by 1 (a float because vulkan reports), use core::rational in the future
    uint32_t maxClipDistances = 0u;
    uint32_t maxCullDistances = 0u;
    uint32_t maxCombinedClipAndCullDistances = 0u;
    uint32_t discreteQueuePriorities = 0u;
    float pointSizeRange[2];
    float lineWidthRange[2];
    float pointSizeGranularity = 0.f;
    float lineWidthGranularity = 0.f;
    bool strictLines = false;
    bool standardSampleLocations = false;
    uint64_t optimalBufferCopyOffsetAlignment = 0ull;
    uint64_t optimalBufferCopyRowPitchAlignment = 0ull;
    uint64_t nonCoherentAtomSize = 0ull;

    /* VkPhysicalDeviceSparseProperties */ 
    //bool    residencyStandard2DBlockShape;
    //bool    residencyStandard2DMultisampleBlockShape;
    //bool    residencyStandard3DBlockShape;
    //bool    residencyAlignedMipSize;
    //bool    residencyNonResidentStrict;




    /* Vulkan 1.1 Core  */
    uint32_t subgroupSize = 0u;
    core::bitflag<asset::IShader::E_SHADER_STAGE> subgroupOpsShaderStages = asset::IShader::ESS_UNKNOWN;
    bool shaderSubgroupBasic = false;
    bool shaderSubgroupVote = false;
    bool shaderSubgroupArithmetic = false;
    bool shaderSubgroupBallot = false;
    bool shaderSubgroupShuffle = false;
    bool shaderSubgroupShuffleRelative = false;
    bool shaderSubgroupClustered = false;
    bool shaderSubgroupQuad = false;
    bool shaderSubgroupQuadAllStages = false; //quadOperationsInAllStages;

    
    enum E_POINT_CLIPPING_BEHAVIOR : uint8_t {
        EPCB_ALL_CLIP_PLANES = 0,
        EPCB_USER_CLIP_PLANES_ONLY = 1,
    };
    E_POINT_CLIPPING_BEHAVIOR pointClippingBehavior = EPCB_USER_CLIP_PLANES_ONLY;
    
    uint32_t maxPerSetDescriptors = 0u;
    size_t maxMemoryAllocationSize = 0ull;




    /* Vulkan 1.2 Core  */

    //      or VK_KHR_shader_float_controls:
    //VkShaderFloatControlsIndependence    denormBehaviorIndependence; // TODO: need to implement ways to set them
    //VkShaderFloatControlsIndependence    roundingModeIndependence;   // TODO: need to implement ways to set them
    E_TRI_BOOLEAN shaderSignedZeroInfNanPreserveFloat16 = ETB_DONT_KNOW;
    E_TRI_BOOLEAN shaderSignedZeroInfNanPreserveFloat32 = ETB_DONT_KNOW;
    E_TRI_BOOLEAN shaderSignedZeroInfNanPreserveFloat64 = ETB_DONT_KNOW;
    E_TRI_BOOLEAN shaderDenormPreserveFloat16 = ETB_DONT_KNOW;
    E_TRI_BOOLEAN shaderDenormPreserveFloat32 = ETB_DONT_KNOW;
    E_TRI_BOOLEAN shaderDenormPreserveFloat64 = ETB_DONT_KNOW;
    E_TRI_BOOLEAN shaderDenormFlushToZeroFloat16 = ETB_DONT_KNOW;
    E_TRI_BOOLEAN shaderDenormFlushToZeroFloat32 = ETB_DONT_KNOW;
    E_TRI_BOOLEAN shaderDenormFlushToZeroFloat64 = ETB_DONT_KNOW;
    E_TRI_BOOLEAN shaderRoundingModeRTEFloat16 = ETB_DONT_KNOW;
    E_TRI_BOOLEAN shaderRoundingModeRTEFloat32 = ETB_DONT_KNOW;
    E_TRI_BOOLEAN shaderRoundingModeRTEFloat64 = ETB_DONT_KNOW;
    E_TRI_BOOLEAN shaderRoundingModeRTZFloat16 = ETB_DONT_KNOW;
    E_TRI_BOOLEAN shaderRoundingModeRTZFloat32 = ETB_DONT_KNOW;
    E_TRI_BOOLEAN shaderRoundingModeRTZFloat64 = ETB_DONT_KNOW;
 
    // expose in 2 phases
    // -Update After Bindand nonUniformEXT shader qualifier:
    //      Descriptor Lifetime Tracking PR #345 will do this, cause I don't want to rewrite the tracking system again.
    // -Actual Descriptor Indexing:
    //      The whole 512k descriptor limits, runtime desc arrays, etc.will come later
    uint32_t maxUpdateAfterBindDescriptorsInAllPools = ~0u;
    bool shaderUniformBufferArrayNonUniformIndexingNative = false;
    bool shaderSampledImageArrayNonUniformIndexingNative = false;
    bool shaderStorageBufferArrayNonUniformIndexingNative = false;
    bool shaderStorageImageArrayNonUniformIndexingNative = false;
    bool shaderInputAttachmentArrayNonUniformIndexingNative = false;
    bool robustBufferAccessUpdateAfterBind = false;
    bool quadDivergentImplicitLod = false;
    uint32_t maxPerStageDescriptorUpdateAfterBindSamplers = 0u;
    uint32_t maxPerStageDescriptorUpdateAfterBindUBOs = 0u;
    uint32_t maxPerStageDescriptorUpdateAfterBindSSBOs = 0u;
    uint32_t maxPerStageDescriptorUpdateAfterBindImages = 0u;
    uint32_t maxPerStageDescriptorUpdateAfterBindStorageImages = 0u;
    uint32_t maxPerStageDescriptorUpdateAfterBindInputAttachments = 0u;
    uint32_t maxPerStageUpdateAfterBindResources = 0u;
    uint32_t maxDescriptorSetUpdateAfterBindSamplers = 0u;
    uint32_t maxDescriptorSetUpdateAfterBindUBOs = 0u;
    uint32_t maxDescriptorSetUpdateAfterBindDynamicOffsetUBOs = 0u;
    uint32_t maxDescriptorSetUpdateAfterBindSSBOs = 0u;
    uint32_t maxDescriptorSetUpdateAfterBindDynamicOffsetSSBOs = 0u;
    uint32_t maxDescriptorSetUpdateAfterBindImages = 0u;
    uint32_t maxDescriptorSetUpdateAfterBindStorageImages = 0u;
    uint32_t maxDescriptorSetUpdateAfterBindInputAttachments = 0u;

    //      or VK_EXT_sampler_filter_minmax:
    bool filterMinmaxSingleComponentFormats = false;
    bool filterMinmaxImageComponentMapping = false;

    /* Vulkan 1.3 Core  */
    
    //      or VK_EXT_subgroup_size_control:
    uint32_t                                        minSubgroupSize = 0u;
    uint32_t                                        maxSubgroupSize = 0u;
    uint32_t                                        maxComputeWorkgroupSubgroups = 0u;
    core::bitflag<asset::IShader::E_SHADER_STAGE>   requiredSubgroupSizeStages = core::bitflag<asset::IShader::E_SHADER_STAGE>(0u);
    
    //      Only Core 1.3 -> VkPhysicalDeviceVulkan13Properties should be used for this variable
    //uint32_t              maxInlineUniformTotalSize;
    
    // or VK_EXT_texel_buffer_alignment:
    size_t storageTexelBufferOffsetAlignmentBytes = 0ull;
    //bool              storageTexelBufferOffsetSingleTexelAlignment;
    size_t uniformTexelBufferOffsetAlignmentBytes = 0ull;
    //bool              uniformTexelBufferOffsetSingleTexelAlignment;
    
    size_t maxBufferSize = 0ull; // or VK_KHR_maintenance4




    /* Vulkan Extensions */

    /* ConservativeRasterizationPropertiesEXT *//* provided by VK_EXT_conservative_rasterization */
    float   primitiveOverestimationSize = 0.0f;
    float   maxExtraPrimitiveOverestimationSize = 0.0f;
    float   extraPrimitiveOverestimationSizeGranularity = 0.0f;
    bool    primitiveUnderestimation = false;
    bool    conservativePointAndLineRasterization = false;
    bool    degenerateTrianglesRasterized = false;
    bool    degenerateLinesRasterized = false;
    bool    fullyCoveredFragmentShaderInputVariable = false;
    bool    conservativeRasterizationPostDepthCoverage = false;

    /* DiscardRectanglePropertiesEXT *//* provided by VK_EXT_discard_rectangles */
    uint32_t maxDiscardRectangles = 0u;

    // [TODO] this feature introduces new/more pipeline state with VkPipelineRasterizationLineStateCreateInfoEXT
    /* LineRasterizationPropertiesEXT *//* provided by VK_EXT_line_rasterization */
    uint32_t lineSubPixelPrecisionBits = 0;

    // [TODO] we would have to change the API
    /* VertexAttributeDivisorPropertiesEXT *//* provided by VK_EXT_vertex_attribute_divisor */
    uint32_t maxVertexAttribDivisor = 0;

    /* SubpassShadingPropertiesHUAWEI *//* VK_HUAWEI_subpass_shading */
    uint32_t maxSubpassShadingWorkgroupSizeAspectRatio = 0;

    /* ShaderIntegerDotProductProperties *//* VK_KHR_shader_integer_dot_product */
    bool integerDotProduct8BitUnsignedAccelerated;
    bool integerDotProduct8BitSignedAccelerated;
    bool integerDotProduct8BitMixedSignednessAccelerated;
    bool integerDotProduct4x8BitPackedUnsignedAccelerated;
    bool integerDotProduct4x8BitPackedSignedAccelerated;
    bool integerDotProduct4x8BitPackedMixedSignednessAccelerated;
    bool integerDotProduct16BitUnsignedAccelerated;
    bool integerDotProduct16BitSignedAccelerated;
    bool integerDotProduct16BitMixedSignednessAccelerated;
    bool integerDotProduct32BitUnsignedAccelerated;
    bool integerDotProduct32BitSignedAccelerated;
    bool integerDotProduct32BitMixedSignednessAccelerated;
    bool integerDotProduct64BitUnsignedAccelerated;
    bool integerDotProduct64BitSignedAccelerated;
    bool integerDotProduct64BitMixedSignednessAccelerated;
    bool integerDotProductAccumulatingSaturating8BitUnsignedAccelerated;
    bool integerDotProductAccumulatingSaturating8BitSignedAccelerated;
    bool integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated;
    bool integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated;
    bool integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated;
    bool integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated;
    bool integerDotProductAccumulatingSaturating16BitUnsignedAccelerated;
    bool integerDotProductAccumulatingSaturating16BitSignedAccelerated;
    bool integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated;
    bool integerDotProductAccumulatingSaturating32BitUnsignedAccelerated;
    bool integerDotProductAccumulatingSaturating32BitSignedAccelerated;
    bool integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated;
    bool integerDotProductAccumulatingSaturating64BitUnsignedAccelerated;
    bool integerDotProductAccumulatingSaturating64BitSignedAccelerated;
    bool integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated;

    /* AccelerationStructurePropertiesKHR *//* provided by VK_KHR_acceleration_structure */
    uint64_t           maxGeometryCount = 0ull;
    uint64_t           maxInstanceCount = 0ull;
    uint64_t           maxPrimitiveCount = 0ull;
    uint32_t           maxPerStageDescriptorAccelerationStructures = 0u;
    uint32_t           maxPerStageDescriptorUpdateAfterBindAccelerationStructures = 0u;
    uint32_t           maxDescriptorSetAccelerationStructures = 0u;
    uint32_t           maxDescriptorSetUpdateAfterBindAccelerationStructures = 0u;
    uint32_t           minAccelerationStructureScratchOffsetAlignment = 0u;

    /* SampleLocationsPropertiesEXT *//* provided by VK_EXT_sample_locations */
    bool variableSampleLocations = false;
    uint32_t        sampleLocationSubPixelBits = 0;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampleLocationSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    VkExtent2D      maxSampleLocationGridSize = { 0u, 0u };
    float           sampleLocationCoordinateRange[2];

    /* ExternalMemoryHostPropertiesEXT *//* provided by VK_EXT_external_memory_host */
    size_t minImportedHostPointerAlignment = 0; // 0x1ull << 63u; // causes issues with GLSL define
    
    /* FragmentDensityMapPropertiesEXT *//* provided by VK_EXT_fragment_density_map */
    VkExtent2D         minFragmentDensityTexelSize = {0u, 0u};
    VkExtent2D         maxFragmentDensityTexelSize = {0u, 0u};
    bool           fragmentDensityInvocations = false;
    
    /* FragmentDensityMap2PropertiesEXT *//* provided by VK_EXT_fragment_density_map2 */
    bool           subsampledLoads = false;
    bool           subsampledCoarseReconstructionEarlyAccess = false;
    uint32_t           maxSubsampledArrayLayers = 0u;
    uint32_t           maxDescriptorSetSubsampledSamplers = 0u;

    /* PCIBusInfoPropertiesEXT *//* provided by VK_EXT_pci_bus_info */
    uint32_t  pciDomain = ~0u;
    uint32_t  pciBus = ~0u;
    uint32_t  pciDevice = ~0u;
    uint32_t  pciFunction = ~0u;

    /* RayTracingPipelinePropertiesKHR *//* provided by VK_KHR_ray_tracing_pipeline */
    uint32_t           shaderGroupHandleSize = 0u;
    uint32_t           maxRayRecursionDepth = 0u;
    uint32_t           maxShaderGroupStride = 0u;
    uint32_t           shaderGroupBaseAlignment = 0u;
    uint32_t           shaderGroupHandleCaptureReplaySize = 0u;
    uint32_t           maxRayDispatchInvocationCount = 0u;
    uint32_t           shaderGroupHandleAlignment = 0u;
    uint32_t           maxRayHitAttributeSize = 0u;

    /* CooperativeMatrixPropertiesNV *//* VK_NV_cooperative_matrix */
    core::bitflag<asset::IShader::E_SHADER_STAGE> cooperativeMatrixSupportedStages = asset::IShader::ESS_UNKNOWN;

    // [TODO LATER] not in header (previous comment: too much effort)
    // GLHint: Report false for both on GL
    /* GraphicsPipelineLibraryPropertiesEXT *//* provided by VK_EXT_graphics_pipeline_library */
    //bool           graphicsPipelineLibraryFastLinking;
    //bool           graphicsPipelineLibraryIndependentInterpolationDecoration;

    // [TODO LATER] to expose but contingent on the TODO to implement one day
    /* PushDescriptorPropertiesKHR *//* provided by VK_KHR_push_descriptor */
    //uint32_t           maxPushDescriptors;

    // [TODO LATER] no such struct?
    /* Maintenance2PropertiesKHR *//* provided by VK_KHR_maintenance2 *//* MOVED TO Vulkan 1.1 Core  */

    // [TODO LATER] If needed 
    /* MultiviewPropertiesKHR    *//* provided by VK_KHR_multiview    *//* MOVED TO Vulkan 1.1 Core  */
    //uint32_t                   maxMultiviewViewCount;
    //uint32_t                   maxMultiviewInstanceIndex;
    //bool                   protectedNoFault;

    // [TODO LATER] Needs API work to expose -> https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkSubpassDescriptionDepthStencilResolve.html
    //      or VK_KHR_depth_stencil_resolve:
    //VkResolveModeFlags                   supportedDepthResolveModes;
    //VkResolveModeFlags                   supportedStencilResolveModes;
    //bool                             independentResolveNone;
    //bool                             independentResolve;

    // [TODO LATER]: we don't expose inline uniform blocks right now
    /* InlineUniformBlockPropertiesEXT ---> MOVED TO Vulkan 1.3 Core  */
    //      or VK_EXT_inline_uniform_block: 
    //uint32_t              maxInlineUniformBlockSize; 
    //uint32_t              maxPerStageDescriptorInlineUniformBlocks;
    //uint32_t              maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks;
    //uint32_t              maxDescriptorSetInlineUniformBlocks;
    //uint32_t              maxDescriptorSetUpdateAfterBindInlineUniformBlocks;

    // [DO NOT EXPOSE] right now, no idea if we'll ever expose and implement those but they'd all be false for OpenGL
    /* BlendOperationAdvancedPropertiesEXT *//* provided by VK_EXT_blend_operation_advanced */
    //uint32_t           advancedBlendMaxColorAttachments;
    //bool           advancedBlendIndependentBlend;
    //bool           advancedBlendNonPremultipliedSrcColor;
    //bool           advancedBlendNonPremultipliedDstColor;
    //bool           advancedBlendCorrelatedOverlap;
    //bool           advancedBlendAllOperations;

    // [DO NOT EXPOSE] not going to expose custom border colors for now
    /* CustomBorderColorPropertiesEXT *//* provided by VK_EXT_custom_border_color */
    //uint32_t           maxCustomBorderColorSamplers;

    // [DO NOT EXPOSE] this extension is dumb, if we're recording that many draws we will be using Multi Draw INDIRECT which is better supported
    /* MultiDrawPropertiesEXT *//* provided by VK_EXT_multi_draw */
    //uint32_t           maxMultiDrawCount;

    // [DO NOT EXPOSE] wont expose in near or far future
    /* DrmPropertiesEXT *//* provided by VK_EXT_physical_device_drm */
    //bool           hasPrimary;
    //bool           hasRender;
    //int64_t            primaryMajor;
    //int64_t            primaryMinor;
    //int64_t            renderMajor;
    //int64_t            renderMinor;

    // [DO NOT EXPOSE] we don't expose or want timeline semaphore currently
    /* TimelineSemaphorePropertiesKHR *//* VK_KHR_timeline_semaphore *//* MOVED TO Vulkan 1.2 Core  */

    // [DO NOT EXPOSE] we will never expose provoking vertex control, we will always set the provoking vertex to the LAST (vulkan default) convention also because of never exposing Xform Feedback, we'll never expose this as well
    /* ProvokingVertexPropertiesEXT *//* provided by VK_EXT_provoking_vertex */
    //bool           provokingVertexModePerPipeline;
    //bool           transformFeedbackPreservesTriangleFanProvokingVertex;

    // [DO NOT EXPOSE] yet
    /* Robustness2PropertiesEXT *//* provided by VK_EXT_robustness2 */
    //size_t       robustStorageBufferAccessSizeAlignment;
    //size_t       robustUniformBufferAccessSizeAlignment;

    // [DO NOT EXPOSE] replaced by VK_KHR_multiview
    /* VK_KHX_multiview */

    // [DO NOT EXPOSE] Coverage 0%, no structs defined anywhere in vulkan headers
    /* VK_KHR_fragment_shader_barycentric */

    // [DO NOT EXPOSE] we've decided to never expose transform feedback in the engine 
    /* TransformFeedbackPropertiesEXT *//* provided by VK_EXT_transform_feedback */
    //uint32_t           maxTransformFeedbackStreams;
    //uint32_t           maxTransformFeedbackBuffers;
    //size_t       maxTransformFeedbackBufferSize;
    //uint32_t           maxTransformFeedbackStreamDataSize;
    //uint32_t           maxTransformFeedbackBufferDataSize;
    //uint32_t           maxTransformFeedbackBufferDataStride;
    //bool           transformFeedbackQueries;
    //bool           transformFeedbackStreamsLinesTriangles;
    //bool           transformFeedbackRasterizationStreamSelect;
    //bool           transformFeedbackDraw;

    // [DO NOT EXPOSE] not implementing or exposing VRS in near or far future
    /* FragmentShadingRatePropertiesKHR *//* provided by VK_KHR_fragment_shading_rate */
    //VkExtent2D               minFragmentShadingRateAttachmentTexelSize;
    //VkExtent2D               maxFragmentShadingRateAttachmentTexelSize;
    //uint32_t                 maxFragmentShadingRateAttachmentTexelSizeAspectRatio;
    //bool                 primitiveFragmentShadingRateWithMultipleViewports;
    //bool                 layeredShadingRateAttachments;
    //bool                 fragmentShadingRateNonTrivialCombinerOps;
    //VkExtent2D               maxFragmentSize;
    //uint32_t                 maxFragmentSizeAspectRatio;
    //uint32_t                 maxFragmentShadingRateCoverageSamples;
    //VkSampleCountFlagBits    maxFragmentShadingRateRasterizationSamples;
    //bool                 fragmentShadingRateWithShaderDepthStencilWrites;
    //bool                 fragmentShadingRateWithSampleMask;
    //bool                 fragmentShadingRateWithShaderSampleMask;
    //bool                 fragmentShadingRateWithConservativeRasterization;
    //bool                 fragmentShadingRateWithFragmentShaderInterlock;
    //bool                 fragmentShadingRateWithCustomSampleLocations;
    //bool                 fragmentShadingRateStrictMultiplyCombiner;

    // [DO NOT EXPOSE] Provisional/not available anymore
    /* VK_KHR_portability_subset */ 

    // [DO NOT EXPOSE] We don't support PerformanceQueries at the moment;
    // But we have a bool with the same name below under Nabla and that is mostly for GL when NBL_ARB_query_buffer_object is reported and that holds for every query 
    /* PerformanceQueryPropertiesKHR *//* provided by VK_KHR_performance_query */
    // bool           allowCommandBufferQueryCopies;

    // [DO NOT EXPOSE] means beta/experimental, lets not expose any of those
    /* MultiviewPerViewAttributesPropertiesNVX *//* VK_NVX_multiview_per_view_attributes */
    // bool           perViewPositionAllComponents;

    // [DO NOT EXPOSE] Preview Extension of raytracing, useless
    /* VK_NVX_raytracing */
    
    // [DO NOT EXPOSE] won't expose right now, will do if we implement the extension
    /* DeviceGeneratedCommandsPropertiesNV *//* VK_NV_device_generated_commands */
    //uint32_t           maxGraphicsShaderGroupCount;
    //uint32_t           maxIndirectSequenceCount;
    //uint32_t           maxIndirectCommandsTokenCount;
    //uint32_t           maxIndirectCommandsStreamCount;
    //uint32_t           maxIndirectCommandsTokenOffset;
    //uint32_t           maxIndirectCommandsStreamStride;
    //uint32_t           minSequencesCountBufferOffsetAlignment;
    //uint32_t           minSequencesIndexBufferOffsetAlignment;
    //uint32_t           minIndirectCommandsBufferOffsetAlignment;

    
    // [DO NOT EXPOSE] not implementing or exposing VRS in near or far future
    /* FragmentShadingRateEnumsPropertiesNV *//* VK_NV_fragment_shading_rate_enums */
    // VkSampleCountFlagBits    maxFragmentShadingRateInvocationCount;
    
    // [DO NOT EXPOSE] wont expose right now, may in the future
    /* MeshShaderPropertiesNV *//* VK_NV_mesh_shader */
    //uint32_t           maxDrawMeshTasksCount;
    //uint32_t           maxTaskWorkGroupInvocations;
    //uint32_t           maxTaskWorkGroupSize[3];
    //uint32_t           maxTaskTotalMemorySize;
    //uint32_t           maxTaskOutputCount;
    //uint32_t           maxMeshWorkGroupInvocations;
    //uint32_t           maxMeshWorkGroupSize[3];
    //uint32_t           maxMeshTotalMemorySize;
    //uint32_t           maxMeshOutputVertices;
    //uint32_t           maxMeshOutputPrimitives;
    //uint32_t           maxMeshMultiviewViewCount;
    //uint32_t           meshOutputPerVertexGranularity;
    //uint32_t           meshOutputPerPrimitiveGranularity;

    // [DO NOT EXPOSE] MOVED TO Vulkan 1.1 Core
    /* Maintenance3PropertiesKHR *//* provided by VK_KHR_maintenance3 */

    // [DO NOT EXPOSE] useless because of VK_KHR_ray_tracing_pipeline
    /* VK_NV_ray_tracing */

    // [DO NOT EXPOSE] not implementing or exposing VRS in near or far future
    /* ShadingRateImagePropertiesNV *//* VK_NV_shading_rate_image */
    //VkExtent2D         shadingRateTexelSize;
    //uint32_t           shadingRatePaletteSize;
    //uint32_t           shadingRateMaxCoarseSamples;

    // [DO NOT EXPOSE] not implementing or exposing VRS in near or far future
    /* FragmentDensityMapOffsetPropertiesQCOM *//* VK_QCOM_fragment_density_map_offset */
    //VkExtent2D         fragmentDensityOffsetGranularity;

    //! [DO NOT EXPOSE]
    //! maxVertexInputAttributes and maxVertexInputBindings: In OpenGL (and ES) the de-jure (legal) minimum is 16, and de-facto (in practice) Vulkan reports begin at 16.
    //! maxVertexInputAttributeOffset and maxVertexInputBindingStride: In OpenGL (and ES) the de-jure (legal) minimum is 2047 for both, and de-facto (in practice) Vulkan reports begin at 2047.
    //! Asset Conversion:
    //! An ICPUMeshBuffer is an IAsset and for reasons of serialization and conversion we've hardcoded the attribute and binding count to 16 (the bitfields, array sizes, etc.)
    //! variable attribute count meshes would be a mess.
    //! uint32_t              maxVertexInputAttributes;
    //! uint32_t              maxVertexInputBindings;
    //! uint32_t              maxVertexInputAttributeOffset;
    //! uint32_t              maxVertexInputBindingStride;

    /*
    - Spec states minimum supported value should be at least ESCF_1_BIT
    - it might be different for each integer format, best way is to query your integer format from physical device using vkGetPhysicalDeviceImageFormatProperties and get the sampleCounts
    https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkImageFormatProperties.html
    */
    // [DO NOT EXPOSE] because it might be different for every texture format and usage
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferIntegerColorSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);

    /*  Always enabled, reported as limits */
    bool shaderOutputViewportIndex = false;     // ALIAS: VK_EXT_shader_viewport_index_layer
    bool shaderOutputLayer = false;             // ALIAS: VK_EXT_shader_viewport_index_layer

    /* ShaderIntegerFunctions2FeaturesINTEL *//* VK_INTEL_shader_integer_functions2 */
    bool shaderIntegerFunctions2 = false;

    /* ShaderClockFeaturesKHR *//* VK_KHR_shader_clock */
    bool shaderSubgroupClock = false;

    /* ShaderImageFootprintFeaturesNV *//* VK_NV_shader_image_footprint */
    bool imageFootprint = false;

    /* TexelBufferAlignmentFeaturesEXT *//* VK_EXT_texel_buffer_alignment */
    bool texelBufferAlignment = false;

    // [TODO] implement the enable by default and expose behaviour for ones below once API changes

    /* ShaderSMBuiltinsFeaturesNV *//* VK_NV_shader_sm_builtins */
    bool shaderSMBuiltins = false;

    bool shaderSubgroupPartitioned = false; /* VK_NV_shader_subgroup_partitioned */
    bool gcnShader = false; /* VK_AMD_gcn_shader */
    bool gpuShaderHalfFloat = false; /* VK_AMD_gpu_shader_half_float */
    bool gpuShaderInt16 = false; /* VK_AMD_gpu_shader_int16 */
    bool shaderBallot = false; /* VK_AMD_shader_ballot */
    bool shaderImageLoadStoreLod = false; /* VK_AMD_shader_image_load_store_lod */
    bool shaderTrinaryMinmax = false; /* VK_AMD_shader_trinary_minmax  */
    bool postDepthCoverage = false; /* VK_EXT_post_depth_coverage */
    bool shaderStencilExport = false; /* VK_EXT_shader_stencil_export */
    bool decorateString = false; /* VK_GOOGLE_decorate_string */
    bool externalFence = false; /* VK_KHR_external_fence_fd */ /* VK_KHR_external_fence_win32 */ // [TODO] requires instance extensions, add them
    bool externalMemory = false; /* VK_KHR_external_memory_fd */ /* VK_KHR_external_memory_win32 */ // [TODO] requires instance extensions, add them
    bool externalSemaphore = false; /* VK_KHR_external_semaphore_fd */ /* VK_KHR_external_semaphore_win32 */ // [TODO] requires instance extensions, add them
    bool shaderNonSemanticInfo = false; /* VK_KHR_shader_non_semantic_info */
    bool fragmentShaderBarycentric = false; /* VK_KHR_fragment_shader_barycentric */
    bool geometryShaderPassthrough = false; /* VK_NV_geometry_shader_passthrough */
    bool viewportSwizzle = false; /* VK_NV_viewport_swizzle */

    /* Nabla */
    uint32_t computeUnits = 0u;
    bool dispatchBase = false; // true in Vk, false in GL
    bool allowCommandBufferQueryCopies = false;
    uint32_t maxOptimallyResidentWorkgroupInvocations = 0u; //  its 1D because multidimensional workgroups are an illusion
    uint32_t maxResidentInvocations = 0u; //  These are maximum number of invocations you could expect to execute simultaneously on this device.
    asset::IGLSLCompiler::E_SPIRV_VERSION spirvVersion;

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
};

} // nbl::video

#endif
