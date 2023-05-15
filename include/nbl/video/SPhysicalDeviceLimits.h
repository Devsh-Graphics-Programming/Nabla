#ifndef _NBL_VIDEO_S_PHYSICAL_DEVICE_LIMITS_H_INCLUDED_
#define _NBL_VIDEO_S_PHYSICAL_DEVICE_LIMITS_H_INCLUDED_

#include <type_traits>
#include "nbl/asset/utils/CGLSLCompiler.h" // asset::CGLSLCompiler::E_SPIRV_VERSION
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
    //! granularity, in bytes, at which buffer or linear image resources, and optimal image resources can be bound to adjacent offsets in the same allocation
    size_t bufferImageGranularity = std::numeric_limits<size_t>::max();
    //size_t            sparseAddressSpaceSize;         // [TODO LATER] when we support sparse
    //uint32_t          maxBoundDescriptorSets;         // [DO NOT EXPOSE] we've kinda hardcoded the engine to 4 currently

    uint32_t maxPerStageDescriptorSamplers = 0u;  // Descriptors with a type of IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER count against this limit
    uint32_t maxPerStageDescriptorUBOs = 0u;
    uint32_t maxPerStageDescriptorSSBOs = 0u;
    uint32_t maxPerStageDescriptorImages = 0u; // Descriptors with a type of IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER, IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER count against this limit.
    uint32_t maxPerStageDescriptorStorageImages = 0u;
    uint32_t maxPerStageDescriptorInputAttachments = 0u;
    uint32_t maxPerStageResources = 0u;

    uint32_t maxDescriptorSetSamplers = 0u; // Descriptors with a type of IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER count against this limit
    uint32_t maxDescriptorSetUBOs = 0u;
    uint32_t maxDescriptorSetDynamicOffsetUBOs = 0u;
    uint32_t maxDescriptorSetSSBOs = 0u;
    uint32_t maxDescriptorSetDynamicOffsetSSBOs = 0u;
    uint32_t maxDescriptorSetImages = 0u; // Descriptors with a type of IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER, IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER count against this limit.
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
    uint8_t  maxSamplerAnisotropyLog2 = 0u;
    uint32_t maxViewports = 0u;
    uint32_t maxViewportDims[2] = {};
    float    viewportBoundsRange[2] = { 0.0f, 0.0f};
    uint32_t viewportSubPixelBits = 0u;
    size_t   minMemoryMapAlignment = std::numeric_limits<size_t>::max();
    uint32_t bufferViewAlignment = 0x1u << 31u;
    uint32_t minUBOAlignment = 0x1u << 31u;
    uint32_t minSSBOAlignment = 0x1u << 31u;
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
    float pointSizeRange[2] = { std::numeric_limits<float>::max(), 0.0f};
    float lineWidthRange[2] = { std::numeric_limits<float>::max(), 0.0f};
    float pointSizeGranularity = 1.f;
    float lineWidthGranularity = 1.f;
    bool strictLines = false;
    bool standardSampleLocations = false;
    uint64_t optimalBufferCopyOffsetAlignment = std::numeric_limits<size_t>::max();
    uint64_t optimalBufferCopyRowPitchAlignment = std::numeric_limits<size_t>::max();
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
    uint32_t                                        minSubgroupSize = ~0u;
    uint32_t                                        maxSubgroupSize = 0u;
    uint32_t                                        maxComputeWorkgroupSubgroups = 0u;
    core::bitflag<asset::IShader::E_SHADER_STAGE>   requiredSubgroupSizeStages = core::bitflag<asset::IShader::E_SHADER_STAGE>(0u);
    
    //      Only Core 1.3 -> VkPhysicalDeviceVulkan13Properties should be used for this variable
    //uint32_t              maxInlineUniformTotalSize;
    
    // or VK_EXT_texel_buffer_alignment:
    size_t storageTexelBufferOffsetAlignmentBytes = std::numeric_limits<size_t>::max();
    //bool              storageTexelBufferOffsetSingleTexelAlignment;
    size_t uniformTexelBufferOffsetAlignmentBytes = std::numeric_limits<size_t>::max();
    //bool              uniformTexelBufferOffsetSingleTexelAlignment;
    
    size_t maxBufferSize = 0ull; // or VK_KHR_maintenance4




    /* Vulkan Extensions */

    /* ConservativeRasterizationPropertiesEXT *//* provided by VK_EXT_conservative_rasterization */
    float   primitiveOverestimationSize = 0.0f;
    float   maxExtraPrimitiveOverestimationSize = 0.0f;
    float   extraPrimitiveOverestimationSizeGranularity = 1.0f;
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
    uint32_t           minAccelerationStructureScratchOffsetAlignment = 0x1u << 31u;

    /* SampleLocationsPropertiesEXT *//* provided by VK_EXT_sample_locations */
    bool            variableSampleLocations = false;
    uint32_t        sampleLocationSubPixelBits = 0;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampleLocationSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    VkExtent2D      maxSampleLocationGridSize = { 0u, 0u };
    float           sampleLocationCoordinateRange[2] = {1.f, 0.f};

    /* ExternalMemoryHostPropertiesEXT *//* provided by VK_EXT_external_memory_host */
    size_t minImportedHostPointerAlignment = std::numeric_limits<size_t>::max();
    
    /* FragmentDensityMapPropertiesEXT *//* provided by VK_EXT_fragment_density_map */
    VkExtent2D          minFragmentDensityTexelSize = {~0u, ~0u};
    VkExtent2D          maxFragmentDensityTexelSize = {0u, 0u};
    bool                fragmentDensityInvocations = false;
    
    /* FragmentDensityMap2PropertiesEXT *//* provided by VK_EXT_fragment_density_map2 */
    bool                subsampledLoads = false;
    bool                subsampledCoarseReconstructionEarlyAccess = false;
    uint32_t            maxSubsampledArrayLayers = 0u;
    uint32_t            maxDescriptorSetSubsampledSamplers = 0u;

    /* PCIBusInfoPropertiesEXT *//* provided by VK_EXT_pci_bus_info */
    uint32_t  pciDomain = ~0u;
    uint32_t  pciBus = ~0u;
    uint32_t  pciDevice = ~0u;
    uint32_t  pciFunction = ~0u;

    /* RayTracingPipelinePropertiesKHR *//* provided by VK_KHR_ray_tracing_pipeline */
    uint32_t           shaderGroupHandleSize = 0u;
    uint32_t           maxRayRecursionDepth = 0u;
    uint32_t           maxShaderGroupStride = 0u;
    uint32_t           shaderGroupBaseAlignment = 0x1u << 31u;
    uint32_t           maxRayDispatchInvocationCount = 0u;
    uint32_t           shaderGroupHandleAlignment = 0x1u << 31u;
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

    // Core 1.0 Features
    bool vertexPipelineStoresAndAtomics = false;
    bool fragmentStoresAndAtomics = false;
    bool shaderTessellationAndGeometryPointSize = false;
    bool shaderImageGatherExtended = false;
    bool shaderInt64 = false;
    bool shaderInt16 = false;
    bool samplerAnisotropy = false;

    // Core 1.1 Features or VK_KHR_16bit_storage */
    bool storageBuffer16BitAccess = false;
    bool uniformAndStorageBuffer16BitAccess = false;
    bool storagePushConstant16 = false;
    bool storageInputOutput16 = false;

    // Vulkan 1.2 Core or VK_KHR_8bit_storage:
    bool storageBuffer8BitAccess = false;
    bool uniformAndStorageBuffer8BitAccess = false;
    bool storagePushConstant8 = false;
    // Vulkan 1.2 Core or VK_KHR_shader_atomic_int64:
    bool shaderBufferInt64Atomics = false;
    bool shaderSharedInt64Atomics = false;
    // Vulkan 1.2 Core or VK_KHR_shader_float16_int8:
    bool shaderFloat16 = false;
    bool shaderInt8 = false;

    // Vulkan 1.2 Struct Or
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

    /* ShaderSMBuiltinsFeaturesNV *//* VK_NV_shader_sm_builtins */
    bool shaderSMBuiltins = false;

    // [TODO] MORE: Use multiple booleans that represent what `VK_KHR_maintenance4` adds support for, instead of single bool; see description in https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_maintenance4.html
    /* VkPhysicalDeviceMaintenance4Features *//* VK_KHR_maintenance4 */
    bool workgroupSizeFromSpecConstant = false;

    bool shaderSubgroupPartitioned = false; /* VK_NV_shader_subgroup_partitioned */
    bool gcnShader = false; /* VK_AMD_gcn_shader */
    bool gpuShaderHalfFloat = false; /* VK_AMD_gpu_shader_half_float */
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
    asset::CGLSLCompiler::E_SPIRV_VERSION spirvVersion;

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
        if (maxDrawIndirectCount > _rhs.maxDrawIndirectCount) return false;
        if (maxSamplerLodBias > _rhs.maxSamplerLodBias) return false;
        if (maxSamplerAnisotropyLog2 > _rhs.maxSamplerAnisotropyLog2) return false;
        if (maxViewports > _rhs.maxViewports) return false;
        if (maxViewportDims[0] > _rhs.maxViewportDims[0]) return false;
        if (maxViewportDims[1] > _rhs.maxViewportDims[1]) return false;
        if (viewportBoundsRange[0] < _rhs.viewportBoundsRange[0] || viewportBoundsRange[1] > _rhs.viewportBoundsRange[1]) return false;
        if (viewportSubPixelBits > _rhs.viewportSubPixelBits) return false;
        if (minMemoryMapAlignment < _rhs.minMemoryMapAlignment) return false;
        if (bufferViewAlignment < _rhs.bufferViewAlignment) return false;
        if (minUBOAlignment < _rhs.minUBOAlignment) return false;
        if (minSSBOAlignment < _rhs.minSSBOAlignment) return false;
        if (minTexelOffset < _rhs.minTexelOffset || maxTexelOffset > _rhs.maxTexelOffset) return false;
        if (minTexelGatherOffset < _rhs.minTexelGatherOffset || maxTexelGatherOffset > _rhs.maxTexelGatherOffset) return false;
        if (minInterpolationOffset < _rhs.minInterpolationOffset || maxInterpolationOffset > _rhs.maxInterpolationOffset) return false;
        if (maxFramebufferWidth > _rhs.maxFramebufferWidth) return false;
        if (maxFramebufferHeight > _rhs.maxFramebufferHeight) return false;
        if (maxFramebufferLayers > _rhs.maxFramebufferLayers) return false;
        if (maxColorAttachments > _rhs.maxColorAttachments) return false;
        if (!_rhs.framebufferColorSampleCounts.hasFlags(framebufferColorSampleCounts)) return false;
        if (!_rhs.framebufferDepthSampleCounts.hasFlags(framebufferDepthSampleCounts)) return false;
        if (!_rhs.framebufferStencilSampleCounts.hasFlags(framebufferStencilSampleCounts)) return false;
        if (!_rhs.framebufferNoAttachmentsSampleCounts.hasFlags(framebufferNoAttachmentsSampleCounts)) return false;
        if (!_rhs.sampledImageColorSampleCounts.hasFlags(sampledImageColorSampleCounts)) return false;
        if (!_rhs.sampledImageIntegerSampleCounts.hasFlags(sampledImageIntegerSampleCounts)) return false;
        if (!_rhs.sampledImageDepthSampleCounts.hasFlags(sampledImageDepthSampleCounts)) return false;
        if (!_rhs.sampledImageStencilSampleCounts.hasFlags(sampledImageStencilSampleCounts)) return false;
        if (!_rhs.storageImageSampleCounts.hasFlags(storageImageSampleCounts)) return false;
        if (maxSampleMaskWords > _rhs.maxSampleMaskWords) return false;
        if (timestampComputeAndGraphics && !_rhs.timestampComputeAndGraphics) return false;
        if (timestampPeriodInNanoSeconds > _rhs.timestampPeriodInNanoSeconds) return false;
        if (maxClipDistances > _rhs.maxClipDistances) return false;
        if (maxCullDistances > _rhs.maxCullDistances) return false;
        if (maxCombinedClipAndCullDistances > _rhs.maxCombinedClipAndCullDistances) return false;
        if (discreteQueuePriorities > _rhs.discreteQueuePriorities) return false;
        if (pointSizeRange[0] < _rhs.pointSizeRange[0] || pointSizeRange[1] > _rhs.pointSizeRange[1]) return false;
        if (lineWidthRange[0] < _rhs.lineWidthRange[0] || lineWidthRange[1] > _rhs.lineWidthRange[1]) return false;
        if (pointSizeGranularity < _rhs.pointSizeGranularity) return false;
        if (lineWidthGranularity < _rhs.lineWidthGranularity) return false;
        if (strictLines > _rhs.strictLines) return false;
        if (standardSampleLocations > _rhs.standardSampleLocations) return false;
        if (optimalBufferCopyOffsetAlignment < _rhs.optimalBufferCopyOffsetAlignment) return false;
        if (optimalBufferCopyRowPitchAlignment < _rhs.optimalBufferCopyRowPitchAlignment) return false;
        if (nonCoherentAtomSize > _rhs.nonCoherentAtomSize) return false;
        if (subgroupSize > _rhs.subgroupSize) return false;
        if (!_rhs.subgroupOpsShaderStages.hasFlags(subgroupOpsShaderStages)) return false;
        if (shaderSubgroupBasic && !_rhs.shaderSubgroupBasic) return false;
        if (shaderSubgroupVote && !_rhs.shaderSubgroupVote) return false;
        if (shaderSubgroupArithmetic && !_rhs.shaderSubgroupArithmetic) return false;
        if (shaderSubgroupBallot && !_rhs.shaderSubgroupBallot) return false;
        if (shaderSubgroupShuffle && !_rhs.shaderSubgroupShuffle) return false;
        if (shaderSubgroupShuffleRelative && !_rhs.shaderSubgroupShuffleRelative) return false;
        if (shaderSubgroupClustered && !_rhs.shaderSubgroupClustered) return false;
        if (shaderSubgroupQuad && !_rhs.shaderSubgroupQuad) return false;
        if (shaderSubgroupQuadAllStages && !_rhs.shaderSubgroupQuadAllStages) return false;
        if (maxPerSetDescriptors > _rhs.maxPerSetDescriptors) return false;
        if (maxMemoryAllocationSize > _rhs.maxMemoryAllocationSize) return false;
        if (shaderSignedZeroInfNanPreserveFloat16 == ETB_TRUE && _rhs.shaderSignedZeroInfNanPreserveFloat16 == ETB_FALSE) return false;
        if (shaderSignedZeroInfNanPreserveFloat32 == ETB_TRUE && _rhs.shaderSignedZeroInfNanPreserveFloat32 == ETB_FALSE) return false;
        if (shaderSignedZeroInfNanPreserveFloat64 == ETB_TRUE && _rhs.shaderSignedZeroInfNanPreserveFloat64 == ETB_FALSE) return false;
        if (shaderDenormPreserveFloat16 == ETB_TRUE && _rhs.shaderDenormPreserveFloat16 == ETB_FALSE) return false;
        if (shaderDenormPreserveFloat32 == ETB_TRUE && _rhs.shaderDenormPreserveFloat32 == ETB_FALSE) return false;
        if (shaderDenormPreserveFloat64 == ETB_TRUE && _rhs.shaderDenormPreserveFloat64 == ETB_FALSE) return false;
        if (shaderDenormFlushToZeroFloat16 == ETB_TRUE && _rhs.shaderDenormFlushToZeroFloat16 == ETB_FALSE) return false;
        if (shaderDenormFlushToZeroFloat32 == ETB_TRUE && _rhs.shaderDenormFlushToZeroFloat32 == ETB_FALSE) return false;
        if (shaderDenormFlushToZeroFloat64 == ETB_TRUE && _rhs.shaderDenormFlushToZeroFloat64 == ETB_FALSE) return false;
        if (shaderRoundingModeRTEFloat16 == ETB_TRUE && _rhs.shaderRoundingModeRTEFloat16 == ETB_FALSE) return false;
        if (shaderRoundingModeRTEFloat32 == ETB_TRUE && _rhs.shaderRoundingModeRTEFloat32 == ETB_FALSE) return false;
        if (shaderRoundingModeRTEFloat64 == ETB_TRUE && _rhs.shaderRoundingModeRTEFloat64 == ETB_FALSE) return false;
        if (shaderRoundingModeRTZFloat16 == ETB_TRUE && _rhs.shaderRoundingModeRTZFloat16 == ETB_FALSE) return false;
        if (shaderRoundingModeRTZFloat32 == ETB_TRUE && _rhs.shaderRoundingModeRTZFloat32 == ETB_FALSE) return false;
        if (shaderRoundingModeRTZFloat64 == ETB_TRUE && _rhs.shaderRoundingModeRTZFloat64 == ETB_FALSE) return false;
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
        if (filterMinmaxSingleComponentFormats && !_rhs.filterMinmaxSingleComponentFormats) return false;
        if (filterMinmaxImageComponentMapping && !_rhs.filterMinmaxImageComponentMapping) return false;
        
        if (minSubgroupSize < _rhs.minSubgroupSize || maxSubgroupSize > _rhs.maxSubgroupSize) return false;

        if (maxComputeWorkgroupSubgroups > _rhs.maxComputeWorkgroupSubgroups) return false;
        if (!_rhs.requiredSubgroupSizeStages.hasFlags(requiredSubgroupSizeStages)) return false;
        if (storageTexelBufferOffsetAlignmentBytes < _rhs.storageTexelBufferOffsetAlignmentBytes) return false;
        if (uniformTexelBufferOffsetAlignmentBytes < _rhs.uniformTexelBufferOffsetAlignmentBytes) return false;
        if (maxBufferSize > _rhs.maxBufferSize) return false;
        if (primitiveOverestimationSize > _rhs.primitiveOverestimationSize) return false;
        if (maxExtraPrimitiveOverestimationSize > _rhs.maxExtraPrimitiveOverestimationSize) return false;
        if (extraPrimitiveOverestimationSizeGranularity < _rhs.extraPrimitiveOverestimationSizeGranularity) return false;
        if (primitiveUnderestimation && !_rhs.primitiveUnderestimation) return false;
        if (conservativePointAndLineRasterization && !_rhs.conservativePointAndLineRasterization) return false;
        if (degenerateTrianglesRasterized && !_rhs.degenerateTrianglesRasterized) return false;
        if (degenerateLinesRasterized && !_rhs.degenerateLinesRasterized) return false;
        if (fullyCoveredFragmentShaderInputVariable && !_rhs.fullyCoveredFragmentShaderInputVariable) return false;
        if (conservativeRasterizationPostDepthCoverage && !_rhs.conservativeRasterizationPostDepthCoverage) return false;
        if (maxDiscardRectangles > _rhs.maxDiscardRectangles) return false;
        if (lineSubPixelPrecisionBits > _rhs.lineSubPixelPrecisionBits) return false;
        if (maxVertexAttribDivisor > _rhs.maxVertexAttribDivisor) return false;
        if (maxSubpassShadingWorkgroupSizeAspectRatio > _rhs.maxSubpassShadingWorkgroupSizeAspectRatio) return false;
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
        if (maxGeometryCount > _rhs.maxGeometryCount) return false;
        if (maxInstanceCount > _rhs.maxInstanceCount) return false;
        if (maxPrimitiveCount > _rhs.maxPrimitiveCount) return false;
        if (maxPerStageDescriptorAccelerationStructures > _rhs.maxPerStageDescriptorAccelerationStructures) return false;
        if (maxPerStageDescriptorUpdateAfterBindAccelerationStructures > _rhs.maxPerStageDescriptorUpdateAfterBindAccelerationStructures) return false;
        if (maxDescriptorSetAccelerationStructures > _rhs.maxDescriptorSetAccelerationStructures) return false;
        if (maxDescriptorSetUpdateAfterBindAccelerationStructures > _rhs.maxDescriptorSetUpdateAfterBindAccelerationStructures) return false;
        if (minAccelerationStructureScratchOffsetAlignment < _rhs.minAccelerationStructureScratchOffsetAlignment) return false;
        if (variableSampleLocations && !_rhs.variableSampleLocations) return false;
        if (sampleLocationSubPixelBits > _rhs.sampleLocationSubPixelBits) return false;
        if (!_rhs.sampleLocationSampleCounts.hasFlags(sampleLocationSampleCounts)) return false;
        if (maxSampleLocationGridSize.width > _rhs.maxSampleLocationGridSize.width) return false;
        if (maxSampleLocationGridSize.height > _rhs.maxSampleLocationGridSize.height) return false;
        if (sampleLocationCoordinateRange[0] < _rhs.sampleLocationCoordinateRange[0] || sampleLocationCoordinateRange[1] > _rhs.sampleLocationCoordinateRange[1]) return false;
        if (minImportedHostPointerAlignment < _rhs.minImportedHostPointerAlignment) return false;

        // TODO: Revise min/maxFragmentDensityTexelSize
        if (minFragmentDensityTexelSize.width < _rhs.minFragmentDensityTexelSize.width) return false;
        if (minFragmentDensityTexelSize.height < _rhs.minFragmentDensityTexelSize.height) return false;
        if (maxFragmentDensityTexelSize.width > _rhs.maxFragmentDensityTexelSize.width) return false;
        if (maxFragmentDensityTexelSize.height > _rhs.maxFragmentDensityTexelSize.height) return false;

        if (fragmentDensityInvocations && !_rhs.fragmentDensityInvocations) return false;
        if (subsampledLoads && !_rhs.subsampledLoads) return false;
        if (subsampledCoarseReconstructionEarlyAccess && !_rhs.subsampledCoarseReconstructionEarlyAccess) return false;
        if (maxSubsampledArrayLayers > _rhs.maxSubsampledArrayLayers) return false;
        if (maxDescriptorSetSubsampledSamplers > _rhs.maxDescriptorSetSubsampledSamplers) return false;
        // uint32_t  pciDomain = ~0u;
        // uint32_t  pciBus = ~0u;
        // uint32_t  pciDevice = ~0u;
        // uint32_t  pciFunction = ~0u;
        if (shaderGroupHandleSize > _rhs.shaderGroupHandleSize) return false;
        if (maxRayRecursionDepth > _rhs.maxRayRecursionDepth) return false;
        if (maxShaderGroupStride > _rhs.maxShaderGroupStride) return false;
        if (shaderGroupBaseAlignment < _rhs.shaderGroupBaseAlignment) return false;
        if (maxRayDispatchInvocationCount > _rhs.maxRayDispatchInvocationCount) return false;
        if (shaderGroupHandleAlignment < _rhs.shaderGroupHandleAlignment) return false;
        if (maxRayHitAttributeSize > _rhs.maxRayHitAttributeSize) return false;
        if (!_rhs.cooperativeMatrixSupportedStages.hasFlags(cooperativeMatrixSupportedStages)) return false;
        if (shaderOutputViewportIndex && !_rhs.shaderOutputViewportIndex) return false;
        if (shaderOutputLayer && !_rhs.shaderOutputLayer) return false;
        if (shaderIntegerFunctions2 && !_rhs.shaderIntegerFunctions2) return false;
        if (shaderSubgroupClock && !_rhs.shaderSubgroupClock) return false;
        if (imageFootprint && !_rhs.imageFootprint) return false;
        if (texelBufferAlignment && !_rhs.texelBufferAlignment) return false;
        if (shaderSMBuiltins && !_rhs.shaderSMBuiltins) return false;
        if (workgroupSizeFromSpecConstant && !_rhs.workgroupSizeFromSpecConstant) return false;
        if (shaderSubgroupPartitioned && !_rhs.shaderSubgroupPartitioned) return false;
        if (gcnShader && !_rhs.gcnShader) return false;
        if (gpuShaderHalfFloat && !_rhs.gpuShaderHalfFloat) return false;
        if (shaderBallot && !_rhs.shaderBallot) return false;
        if (shaderImageLoadStoreLod && !_rhs.shaderImageLoadStoreLod) return false;
        if (shaderTrinaryMinmax && !_rhs.shaderTrinaryMinmax) return false;
        if (postDepthCoverage && !_rhs.postDepthCoverage) return false;
        if (shaderStencilExport && !_rhs.shaderStencilExport) return false;
        if (decorateString && !_rhs.decorateString) return false;
        if (externalFence && !_rhs.externalFence) return false;
        if (externalMemory && !_rhs.externalMemory) return false;
        if (externalSemaphore && !_rhs.externalSemaphore) return false;
        if (shaderNonSemanticInfo && !_rhs.shaderNonSemanticInfo) return false;
        if (fragmentShaderBarycentric && !_rhs.fragmentShaderBarycentric) return false;
        if (geometryShaderPassthrough && !_rhs.geometryShaderPassthrough) return false;
        if (viewportSwizzle && !_rhs.viewportSwizzle) return false;
        if (computeUnits > _rhs.computeUnits) return false;
        if (dispatchBase && !_rhs.dispatchBase) return false;
        if (allowCommandBufferQueryCopies && !_rhs.allowCommandBufferQueryCopies) return false;
        if (maxOptimallyResidentWorkgroupInvocations > _rhs.maxOptimallyResidentWorkgroupInvocations) return false;
        if (maxResidentInvocations > _rhs.maxResidentInvocations) return false;
        if (spirvVersion > _rhs.spirvVersion) return false;
        if (vertexPipelineStoresAndAtomics && !_rhs.vertexPipelineStoresAndAtomics) return false;
        if (fragmentStoresAndAtomics && !_rhs.fragmentStoresAndAtomics) return false;
        if (storageBuffer8BitAccess && !_rhs.storageBuffer8BitAccess) return false;
        if (uniformAndStorageBuffer8BitAccess && !_rhs.uniformAndStorageBuffer8BitAccess) return false;
        if (storagePushConstant8 && !_rhs.storagePushConstant8) return false;
        if (shaderBufferInt64Atomics && !_rhs.shaderBufferInt64Atomics) return false;
        if (shaderSharedInt64Atomics && !_rhs.shaderSharedInt64Atomics) return false;
        if (shaderFloat16 && !_rhs.shaderFloat16) return false;
        if (shaderInt8 && !_rhs.shaderInt8) return false;
        if (shaderTessellationAndGeometryPointSize && !_rhs.shaderTessellationAndGeometryPointSize) return false;
        if (shaderImageGatherExtended && !_rhs.shaderImageGatherExtended) return false;
        if (shaderInt64 && !_rhs.shaderInt64) return false;
        if (shaderInt16 && !_rhs.shaderInt16) return false;
        if (samplerAnisotropy && !_rhs.samplerAnisotropy) return false;
        if (uniformAndStorageBuffer16BitAccess && !_rhs.uniformAndStorageBuffer16BitAccess) return false;
        if (storagePushConstant16 && !_rhs.storagePushConstant16) return false;
        if (storageInputOutput16 && !_rhs.storageInputOutput16) return false;
        if (storageBuffer16BitAccess && !_rhs.storageBuffer16BitAccess) return false;
        
        return true;
    }
};

} // nbl::video

#endif
