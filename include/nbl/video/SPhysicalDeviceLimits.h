#ifndef _NBL_VIDEO_S_PHYSICAL_DEVICE_LIMITS_H_INCLUDED_
#define _NBL_VIDEO_S_PHYSICAL_DEVICE_LIMITS_H_INCLUDED_

#include <type_traits>
#include "nbl/asset/utils/IGLSLCompiler.h" // asset::IGLSLCompiler::E_SPIRV_VERSION
#include "nbl/asset/IImage.h"

namespace nbl::video
{

struct SPhysicalDeviceLimits
{
    /* Vulkan 1.0 Core  */
    uint32_t maxImageDimension1D;
    uint32_t maxImageDimension2D;
    uint32_t maxImageDimension3D;
    uint32_t maxImageDimensionCube;
    uint32_t maxImageArrayLayers;
    uint32_t maxBufferViewTexels;
    uint32_t maxUBOSize;
    uint32_t maxSSBOSize;
    uint32_t maxPushConstantsSize;
    uint32_t maxMemoryAllocationCount;
    uint32_t maxSamplerAllocationCount;
    size_t bufferImageGranularity;
    //VkDeviceSize          sparseAddressSpaceSize;         // We support none of the sparse memory operations
    //uint32_t              maxBoundDescriptorSets;         // DO NOT EXPOSE: we've kinda hardcoded the engine to 4 currently

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

    //! DO NOT EXPOSE 
    //! maxVertexInputAttributes and maxVertexInputBindings: In OpenGL (and ES) the de-jure (legal) minimum is 16, and de-facto (in practice) Vulkan reports begin at 16.
    //! maxVertexInputAttributeOffset and maxVertexInputBindingStride: In OpenGL (and ES) the de-jure (legal) minimum is 2047 for both, and de-facto (in practice) Vulkan reports begin at 2047.
    //! Asset Conversion:
    //! An ICPUMeshBuffer is an IAsset and for reasons of serialization and conversion we've hardcoded the attribute and binding count to 16 (the bitfields, array sizes, etc.)
    //! variable attribute count meshes would be a mess.
    //! uint32_t              maxVertexInputAttributes;
    //! uint32_t              maxVertexInputBindings;
    //! uint32_t              maxVertexInputAttributeOffset;
    //! uint32_t              maxVertexInputBindingStride;
    
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
    //uint32_t              mipmapPrecisionBits; // TODO: require investigation GL+ES spec
    //uint32_t              maxDrawIndexedIndexValue;
    uint32_t maxDrawIndirectCount = 0u;
    float    maxSamplerLodBias = 0.0f;
    uint8_t  maxSamplerAnisotropyLog2 = 0.0f;
    uint32_t maxViewports = 0u;
    uint32_t maxViewportDims[2] = {};
    float    viewportBoundsRange[2]; // [min, max]
    uint32_t viewportSubPixelBits = 0u;
    size_t   minMemoryMapAlignment = 0ull;
    uint32_t bufferViewAlignment;
    uint32_t minUBOAlignment;
    uint32_t minSSBOAlignment;
    int32_t  minTexelOffset;
    uint32_t maxTexelOffset;
    int32_t  minTexelGatherOffset;
    uint32_t maxTexelGatherOffset;
    float    minInterpolationOffset;
    float    maxInterpolationOffset;
    //uint32_t              subPixelInterpolationOffsetBits;
    uint32_t maxFramebufferWidth;
    uint32_t maxFramebufferHeight;
    uint32_t maxFramebufferLayers;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferColorSampleCounts;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferDepthSampleCounts;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferStencilSampleCounts;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferNoAttachmentsSampleCounts;
    uint32_t maxColorAttachments = 0u;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageColorSampleCounts;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageIntegerSampleCounts;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageDepthSampleCounts;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageStencilSampleCounts;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> storageImageSampleCounts;
    uint32_t maxSampleMaskWords = 0u;
    bool timestampComputeAndGraphics;
    float timestampPeriodInNanoSeconds; // timestampPeriod is the number of nanoseconds required for a timestamp query to be incremented by 1 (a float because vulkan reports), use core::rational in the future
    uint32_t maxClipDistances;
    uint32_t maxCullDistances;
    uint32_t maxCombinedClipAndCullDistances;
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
    //VkBool32    residencyStandard2DBlockShape;
    //VkBool32    residencyStandard2DMultisampleBlockShape;
    //VkBool32    residencyStandard3DBlockShape;
    //VkBool32    residencyAlignedMipSize;
    //VkBool32    residencyNonResidentStrict;




    /* Vulkan 1.1 Core  */
    uint32_t subgroupSize;
    core::bitflag<asset::IShader::E_SHADER_STAGE> subgroupOpsShaderStages;
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
    E_POINT_CLIPPING_BEHAVIOR pointClippingBehavior;
    
    // TODO: If needed 
    //uint32_t                   maxMultiviewViewCount;
    //uint32_t                   maxMultiviewInstanceIndex;
    //VkBool32                   protectedNoFault;
    
    uint32_t maxPerSetDescriptors = 0u;
    size_t maxMemoryAllocationSize = 0ull;




    /* Vulkan 1.2 Core  */

    //      or VK_KHR_shader_float_controls:
    //VkShaderFloatControlsIndependence    denormBehaviorIndependence; // TODO: need to implement ways to set them
    //VkShaderFloatControlsIndependence    roundingModeIndependence;   // TODO: need to implement ways to set them
    bool shaderSignedZeroInfNanPreserveFloat16;
    bool shaderSignedZeroInfNanPreserveFloat32;
    bool shaderSignedZeroInfNanPreserveFloat64;
    bool shaderDenormPreserveFloat16;
    bool shaderDenormPreserveFloat32;
    bool shaderDenormPreserveFloat64;
    bool shaderDenormFlushToZeroFloat16;
    bool shaderDenormFlushToZeroFloat32;
    bool shaderDenormFlushToZeroFloat64;
    bool shaderRoundingModeRTEFloat16;
    bool shaderRoundingModeRTEFloat32;
    bool shaderRoundingModeRTEFloat64;
    bool shaderRoundingModeRTZFloat16;
    bool shaderRoundingModeRTZFloat32;
    bool shaderRoundingModeRTZFloat64;
 
    //      or VK_EXT_descriptor_indexing:
    uint32_t maxUpdateAfterBindDescriptorsInAllPools;
    bool shaderUniformBufferArrayNonUniformIndexingNative;
    bool shaderSampledImageArrayNonUniformIndexingNative;
    bool shaderStorageBufferArrayNonUniformIndexingNative;
    bool shaderStorageImageArrayNonUniformIndexingNative;
    bool shaderInputAttachmentArrayNonUniformIndexingNative;
    bool robustBufferAccessUpdateAfterBind;
    bool quadDivergentImplicitLod;
    uint32_t maxPerStageDescriptorUpdateAfterBindSamplers;
    uint32_t maxPerStageDescriptorUpdateAfterBindUBOs;
    uint32_t maxPerStageDescriptorUpdateAfterBindSSBOs;
    uint32_t maxPerStageDescriptorUpdateAfterBindImages;
    uint32_t maxPerStageDescriptorUpdateAfterBindStorageImages;
    uint32_t maxPerStageDescriptorUpdateAfterBindInputAttachments;
    uint32_t maxPerStageUpdateAfterBindResources;
    uint32_t maxDescriptorSetUpdateAfterBindSamplers;
    uint32_t maxDescriptorSetUpdateAfterBindUBOs;
    uint32_t maxDescriptorSetUpdateAfterBindDynamicOffsetUBOs;
    uint32_t maxDescriptorSetUpdateAfterBindSSBOs;
    uint32_t maxDescriptorSetUpdateAfterBindDynamicOffsetSSBOs;
    uint32_t maxDescriptorSetUpdateAfterBindImages;
    uint32_t maxDescriptorSetUpdateAfterBindStorageImages;
    uint32_t maxDescriptorSetUpdateAfterBindInputAttachments;
    
    // TODO: Needs API work to expose -> https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkSubpassDescriptionDepthStencilResolve.html
    //      or VK_KHR_depth_stencil_resolve:
    //VkResolveModeFlags                   supportedDepthResolveModes;
    //VkResolveModeFlags                   supportedStencilResolveModes;
    //VkBool32                             independentResolveNone;
    //VkBool32                             independentResolve;

    //      or VK_EXT_sampler_filter_minmax:
    bool filterMinmaxSingleComponentFormats;
    bool filterMinmaxImageComponentMapping;
 
    //      or VK_KHR_timeline_semaphore:
    //uint64_t                             maxTimelineSemaphoreValueDifference; //  we don't expose or want timeline semaphore currently

    //      Only Core 1.2 -> VkPhysicalDeviceVulkan12Properties should be used for this variable
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferIntegerColorSampleCounts;




    /* Vulkan 1.3 Core  */
    
    //      or VK_EXT_subgroup_size_control:
    uint32_t                                        minSubgroupSize = 0u;
    uint32_t                                        maxSubgroupSize = 0u;
    uint32_t                                        maxComputeWorkgroupSubgroups = 0u;
    core::bitflag<asset::IShader::E_SHADER_STAGE>   requiredSubgroupSizeStages = core::bitflag<asset::IShader::E_SHADER_STAGE>(0u);
    
    // [Future TODO]: we don't expose inline uniform blocks right now
    //      or VK_EXT_inline_uniform_block: 
    //uint32_t              maxInlineUniformBlockSize; 
    //uint32_t              maxPerStageDescriptorInlineUniformBlocks;
    //uint32_t              maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks;
    //uint32_t              maxDescriptorSetInlineUniformBlocks;
    //uint32_t              maxDescriptorSetUpdateAfterBindInlineUniformBlocks;
    
    //      Only Core 1.3 -> VkPhysicalDeviceVulkan13Properties should be used for this variable
    //uint32_t              maxInlineUniformTotalSize;
    
    // or VK_KHR_shader_integer_dot_product
    //VkBool32              integerDotProduct8BitUnsignedAccelerated;
    //VkBool32              integerDotProduct8BitSignedAccelerated;
    //VkBool32              integerDotProduct8BitMixedSignednessAccelerated;
    //VkBool32              integerDotProduct4x8BitPackedUnsignedAccelerated;
    //VkBool32              integerDotProduct4x8BitPackedSignedAccelerated;
    //VkBool32              integerDotProduct4x8BitPackedMixedSignednessAccelerated;
    //VkBool32              integerDotProduct16BitUnsignedAccelerated;
    //VkBool32              integerDotProduct16BitSignedAccelerated;
    //VkBool32              integerDotProduct16BitMixedSignednessAccelerated;
    //VkBool32              integerDotProduct32BitUnsignedAccelerated;
    //VkBool32              integerDotProduct32BitSignedAccelerated;
    //VkBool32              integerDotProduct32BitMixedSignednessAccelerated;
    //VkBool32              integerDotProduct64BitUnsignedAccelerated;
    //VkBool32              integerDotProduct64BitSignedAccelerated;
    //VkBool32              integerDotProduct64BitMixedSignednessAccelerated;
    //VkBool32              integerDotProductAccumulatingSaturating8BitUnsignedAccelerated;
    //VkBool32              integerDotProductAccumulatingSaturating8BitSignedAccelerated;
    //VkBool32              integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated;
    //VkBool32              integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated;
    //VkBool32              integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated;
    //VkBool32              integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated;
    //VkBool32              integerDotProductAccumulatingSaturating16BitUnsignedAccelerated;
    //VkBool32              integerDotProductAccumulatingSaturating16BitSignedAccelerated;
    //VkBool32              integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated;
    //VkBool32              integerDotProductAccumulatingSaturating32BitUnsignedAccelerated;
    //VkBool32              integerDotProductAccumulatingSaturating32BitSignedAccelerated;
    //VkBool32              integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated;
    //VkBool32              integerDotProductAccumulatingSaturating64BitUnsignedAccelerated;
    //VkBool32              integerDotProductAccumulatingSaturating64BitSignedAccelerated;
    //VkBool32              integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated;
    
    // or VK_EXT_texel_buffer_alignment:
    VkDeviceSize          storageTexelBufferOffsetAlignmentBytes;
    //VkBool32              storageTexelBufferOffsetSingleTexelAlignment;
    //VkDeviceSize          uniformTexelBufferOffsetAlignmentBytes;
    //VkBool32              uniformTexelBufferOffsetSingleTexelAlignment;
    
    size_t                  maxBufferSize = 0ull; // or VK_KHR_maintenance4




    /* Vulkan Extensions */

    /* ShaderCorePropertiesAMD *//* provided by VK_AMD_shader_core_properties */
    //uint32_t           shaderEngineCount;
    //uint32_t           shaderArraysPerEngineCount;
    //uint32_t           computeUnitsPerShaderArray;
    //uint32_t           simdPerComputeUnit;
    //uint32_t           wavefrontsPerSimd;
    //uint32_t           wavefrontSize;
    //uint32_t           sgprsPerSimd;
    //uint32_t           minSgprAllocation;
    //uint32_t           maxSgprAllocation;
    //uint32_t           sgprAllocationGranularity;
    //uint32_t           vgprsPerSimd;
    //uint32_t           minVgprAllocation;
    //uint32_t           maxVgprAllocation;
    //uint32_t           vgprAllocationGranularity;

    // We will never expose this vendor specific meta-data (no new feature) to the user, but might use the extension to provide some cross platform meta-info in the Nabla section
    /* ShaderCoreProperties2AMD *//* provided by VK_AMD_shader_core_properties2 */
    //VkShaderCorePropertiesFlagsAMD    shaderCoreFeatures;
    //uint32_t                          activeComputeUnitCount;
    
    // DO NOT EXPOSE right now, no idea if we'll ever expose and implement those but they'd all be false for OpenGL
    /* BlendOperationAdvancedPropertiesEXT *//* provided by VK_EXT_blend_operation_advanced */
    //uint32_t           advancedBlendMaxColorAttachments;
    //VkBool32           advancedBlendIndependentBlend;
    //VkBool32           advancedBlendNonPremultipliedSrcColor;
    //VkBool32           advancedBlendNonPremultipliedDstColor;
    //VkBool32           advancedBlendCorrelatedOverlap;
    //VkBool32           advancedBlendAllOperations;
            
    /* ConservativeRasterizationPropertiesEXT *//* provided by VK_EXT_conservative_rasterization */
    float   primitiveOverestimationSize;
    float   maxExtraPrimitiveOverestimationSize;
    float   extraPrimitiveOverestimationSizeGranularity;
    bool    primitiveUnderestimation;
    bool    conservativePointAndLineRasterization;
    bool    degenerateTrianglesRasterized;
    bool    degenerateLinesRasterized;
    bool    fullyCoveredFragmentShaderInputVariable;
    bool    conservativeRasterizationPostDepthCoverage;
          
    // [DO NOT EXPOSE] not going to expose custom border colors for now
    /* CustomBorderColorPropertiesEXT *//* provided by VK_EXT_custom_border_color */
    //uint32_t           maxCustomBorderColorSamplers;

    /* DescriptorIndexingPropertiesEXT ---> MOVED TO Vulkan 1.2 Core  *//* provided by VK_AMD_shader_core_properties */

    /* DiscardRectanglePropertiesEXT *//* provided by VK_EXT_discard_rectangles */
    //uint32_t           maxDiscardRectangles;
            
    /* ExternalMemoryHostPropertiesEXT *//* provided by VK_EXT_external_memory_host */
    //VkDeviceSize       minImportedHostPointerAlignment;
    
    /* FragmentDensityMapPropertiesEXT *//* provided by VK_EXT_fragment_density_map */
    //VkExtent2D         minFragmentDensityTexelSize;
    //VkExtent2D         maxFragmentDensityTexelSize;
    //VkBool32           fragmentDensityInvocations;
    
    /* FragmentDensityMap2PropertiesEXT *//* provided by VK_EXT_fragment_density_map2 */
    //VkBool32           subsampledLoads;
    //VkBool32           subsampledCoarseReconstructionEarlyAccess;
    //uint32_t           maxSubsampledArrayLayers;
    //uint32_t           maxDescriptorSetSubsampledSamplers;
    
    /* GraphicsPipelineLibraryPropertiesEXT *//* provided by VK_EXT_graphics_pipeline_library */
    //VkBool32           graphicsPipelineLibraryFastLinking;
    //VkBool32           graphicsPipelineLibraryIndependentInterpolationDecoration;

    /* InlineUniformBlockPropertiesEXT ---> MOVED TO Vulkan 1.3 Core  */

    /* LineRasterizationPropertiesEXT *//* provided by VK_EXT_line_rasterization */
    //uint32_t           lineSubPixelPrecisionBits;

    /* MultiDrawPropertiesEXT *//* provided by VK_EXT_multi_draw */
    //uint32_t           maxMultiDrawCount;

    /* PCIBusInfoPropertiesEXT *//* provided by VK_EXT_pci_bus_info */
    //uint32_t           pciDomain;
    //uint32_t           pciBus;
    //uint32_t           pciDevice;
    //uint32_t           pciFunction;

    // [DO NOT EXPOSE] wont expose in near or far future
    /* DrmPropertiesEXT *//* provided by VK_EXT_physical_device_drm */
    //VkBool32           hasPrimary;
    //VkBool32           hasRender;
    //int64_t            primaryMajor;
    //int64_t            primaryMinor;
    //int64_t            renderMajor;
    //int64_t            renderMinor;

    // [DO NOT EXPOSE] we will never expose provoking vertex control, we will always set the provoking vertex to the LAST (vulkan default) convention also because of never exposing Xform Feedback, we'll never expose this as well
    /* ProvokingVertexPropertiesEXT *//* provided by VK_EXT_provoking_vertex */
    //VkBool32           provokingVertexModePerPipeline;
    //VkBool32           transformFeedbackPreservesTriangleFanProvokingVertex;

    // [DO NOT EXPOSE] yet
    /* Robustness2PropertiesEXT *//* provided by VK_EXT_robustness2 */
    //VkDeviceSize       robustStorageBufferAccessSizeAlignment;
    //VkDeviceSize       robustUniformBufferAccessSizeAlignment;

    /* SamplerFilterMinmaxPropertiesEXT ---> MOVED TO Vulkan 1.2 Core  */

    /* SampleLocationsPropertiesEXT *//* provided by VK_EXT_sample_locations */
    //VkSampleCountFlags    sampleLocationSampleCounts;
    //VkExtent2D            maxSampleLocationGridSize;
    //float                 sampleLocationCoordinateRange[2];
    //uint32_t              sampleLocationSubPixelBits;
    //VkBool32              variableSampleLocations;

    /* SubgroupSizeControlPropertiesEXT ---> MOVED TO Vulkan 1.3 Core  */

    /* TexelBufferAlignmentPropertiesEXT ---> MOVED TO Vulkan 1.3 Core  */
    
    // [DO NOT EXPOSE] we've decided to never expose transform feedback in the engine 
    /* TransformFeedbackPropertiesEXT *//* provided by VK_EXT_transform_feedback */
    //uint32_t           maxTransformFeedbackStreams;
    //uint32_t           maxTransformFeedbackBuffers;
    //VkDeviceSize       maxTransformFeedbackBufferSize;
    //uint32_t           maxTransformFeedbackStreamDataSize;
    //uint32_t           maxTransformFeedbackBufferDataSize;
    //uint32_t           maxTransformFeedbackBufferDataStride;
    //VkBool32           transformFeedbackQueries;
    //VkBool32           transformFeedbackStreamsLinesTriangles;
    //VkBool32           transformFeedbackRasterizationStreamSelect;
    //VkBool32           transformFeedbackDraw;

    /* VertexAttributeDivisorPropertiesEXT *//* provided by VK_EXT_vertex_attribute_divisor */
    //uint32_t           maxVertexAttribDivisor;

    /* AccelerationStructurePropertiesKHR *//* provided by VK_KHR_acceleration_structure */
    uint64_t           maxGeometryCount;
    uint64_t           maxInstanceCount;
    uint64_t           maxPrimitiveCount;
    uint32_t           maxPerStageDescriptorAccelerationStructures;
    uint32_t           maxPerStageDescriptorUpdateAfterBindAccelerationStructures;
    uint32_t           maxDescriptorSetAccelerationStructures;
    uint32_t           maxDescriptorSetUpdateAfterBindAccelerationStructures;
    uint32_t           minAccelerationStructureScratchOffsetAlignment;
 
    /* DepthStencilResolvePropertiesKHR ---> MOVED TO Vulkan 1.2 Core  */
    /* DriverPropertiesKHR ---> MOVED TO Vulkan 1.2 Core  */
    /* VK_KHR_fragment_shader_barycentric --> Coverage 0% --> no structs defined anywhere in vulkan headers */

    // [DO NOT EXPOSE] not implementing or exposing VRS in near or far future
    /* FragmentShadingRatePropertiesKHR *//* provided by VK_KHR_fragment_shading_rate */
    //VkExtent2D               minFragmentShadingRateAttachmentTexelSize;
    //VkExtent2D               maxFragmentShadingRateAttachmentTexelSize;
    //uint32_t                 maxFragmentShadingRateAttachmentTexelSizeAspectRatio;
    //VkBool32                 primitiveFragmentShadingRateWithMultipleViewports;
    //VkBool32                 layeredShadingRateAttachments;
    //VkBool32                 fragmentShadingRateNonTrivialCombinerOps;
    //VkExtent2D               maxFragmentSize;
    //uint32_t                 maxFragmentSizeAspectRatio;
    //uint32_t                 maxFragmentShadingRateCoverageSamples;
    //VkSampleCountFlagBits    maxFragmentShadingRateRasterizationSamples;
    //VkBool32                 fragmentShadingRateWithShaderDepthStencilWrites;
    //VkBool32                 fragmentShadingRateWithSampleMask;
    //VkBool32                 fragmentShadingRateWithShaderSampleMask;
    //VkBool32                 fragmentShadingRateWithConservativeRasterization;
    //VkBool32                 fragmentShadingRateWithFragmentShaderInterlock;
    //VkBool32                 fragmentShadingRateWithCustomSampleLocations;
    //VkBool32                 fragmentShadingRateStrictMultiplyCombiner;
    
    /* Maintenance2PropertiesKHR *//* provided by VK_KHR_maintenance2 *//* MOVED TO Vulkan 1.1 Core  */
    /* Maintenance3PropertiesKHR *//* provided by VK_KHR_maintenance3 *//* MOVED TO Vulkan 1.1 Core  */
    /* Maintenance4PropertiesKHR *//* provided by VK_KHR_maintenance4 *//* MOVED TO Vulkan 1.3 Core  */
    /* MultiviewPropertiesKHR    *//* provided by VK_KHR_multiview    *//* MOVED TO Vulkan 1.1 Core  */

    /* PerformanceQueryPropertiesKHR *//* provided by VK_KHR_performance_query */
    // ! We don't support PerformanceQueries at the moment;
    // ! But we have a bool with the same name below under Nabla and that is mostly for GL when NBL_ARB_query_buffer_object is reported and that holds for every query 
    // VkBool32           allowCommandBufferQueryCopies;

    /* VK_KHR_portability_subset - PROVISINAL/NOTAVAILABLEANYMORE */

    // [TODO] to expose but contingent on the TODO to implement one day
    /* PushDescriptorPropertiesKHR *//* provided by VK_KHR_push_descriptor */
    //uint32_t           maxPushDescriptors;

    /* RayTracingPipelinePropertiesKHR *//* provided by VK_KHR_ray_tracing_pipeline */
    uint32_t           shaderGroupHandleSize;
    uint32_t           maxRayRecursionDepth;
    uint32_t           maxShaderGroupStride;
    uint32_t           shaderGroupBaseAlignment;
    uint32_t           shaderGroupHandleCaptureReplaySize;
    uint32_t           maxRayDispatchInvocationCount;
    uint32_t           shaderGroupHandleAlignment;
    uint32_t           maxRayHitAttributeSize;

    /* FloatControlsPropertiesKHR *//* VK_KHR_shader_float_controls *//* MOVED TO Vulkan 1.2 Core  */
    /* ShaderIntegerDotProductFeaturesKHR *//* VK_KHR_shader_integer_dot_product *//* MOVED TO Vulkan 1.3 Core  */
    /* TimelineSemaphorePropertiesKHR *//* VK_KHR_timeline_semaphore *//* MOVED TO Vulkan 1.2 Core  */
    /* VK_KHX_multiview *//* replaced by VK_KHR_multiview */

    // [DO NOT EXPOSE] means beta/experimental, lets not expose any of those
    /* MultiviewPerViewAttributesPropertiesNVX *//* VK_NVX_multiview_per_view_attributes */
    // VkBool32           perViewPositionAllComponents;

    /* VK_NVX_raytracing *//* Preview Extension of raytracing, useless*/
    
    /* CooperativeMatrixPropertiesNV *//* VK_NV_cooperative_matrix */
    // VkShaderStageFlags    cooperativeMatrixSupportedStages;

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

    /* VK_NV_ray_tracing *//* useless because of VK_KHR_ray_tracing_pipeline */

    /* ShaderSMBuiltinsPropertiesNV *//* VK_NV_shader_sm_builtins */
    //uint32_t           shaderSMCount;
    //uint32_t           shaderWarpsPerSM;
    
    // [DO NOT EXPOSE] not implementing or exposing VRS in near or far future
    /* ShadingRateImagePropertiesNV *//* VK_NV_shading_rate_image */
    //VkExtent2D         shadingRateTexelSize;
    //uint32_t           shadingRatePaletteSize;
    //uint32_t           shadingRateMaxCoarseSamples;

    // [DO NOT EXPOSE] not implementing or exposing VRS in near or far future
    /* FragmentDensityMapOffsetPropertiesQCOM *//* VK_QCOM_fragment_density_map_offset */
    //VkExtent2D         fragmentDensityOffsetGranularity;

    /* SubpassShadingPropertiesHUAWEI *//* VK_HUAWEI_subpass_shading */
    // uint32_t           maxSubpassShadingWorkgroupSizeAspectRatio;

    /* Nabla */
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
