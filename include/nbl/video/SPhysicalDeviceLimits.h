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
    uint32_t maxBufferViewSizeTexels;
    uint32_t maxUBOSize;
    uint32_t maxSSBOSize;
    uint32_t maxPushConstantsSize;
    //uint32_t              maxMemoryAllocationCount;
    //uint32_t              maxSamplerAllocationCount;
    //VkDeviceSize          bufferImageGranularity;
    //VkDeviceSize          sparseAddressSpaceSize;
    //uint32_t              maxBoundDescriptorSets;
    //uint32_t              maxPerStageDescriptorSamplers;
    //uint32_t              maxPerStageDescriptorUniformBuffers;
    uint32_t maxPerStageDescriptorSSBOs;
    //uint32_t              maxPerStageDescriptorSampledImages;
    //uint32_t              maxPerStageDescriptorStorageImages;
    //uint32_t              maxPerStageDescriptorInputAttachments;
    //uint32_t              maxPerStageResources;
    //uint32_t              maxDescriptorSetSamplers;
    uint32_t maxDescriptorSetUBOs;
    uint32_t maxDescriptorSetDynamicOffsetUBOs;
    uint32_t maxDescriptorSetSSBOs;
    uint32_t maxDescriptorSetDynamicOffsetSSBOs;
    uint32_t maxDescriptorSetImages;
    uint32_t maxDescriptorSetStorageImages;
    //! uint32_t              maxDescriptorSetInputAttachments;

    //! DON'T EXPOSE 
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
    //uint32_t              maxFragmentInputComponents;
    //uint32_t              maxFragmentOutputAttachments;
    //uint32_t              maxFragmentDualSrcAttachments;
    //uint32_t              maxFragmentCombinedOutputResources;
    uint32_t maxComputeSharedMemorySize;
    //uint32_t              maxComputeWorkGroupCount[3];
    //uint32_t              maxComputeWorkGroupInvocations;
    uint32_t maxWorkgroupSize[3];
    //uint32_t              subPixelPrecisionBits;
    //uint32_t              subTexelPrecisionBits;
    //uint32_t              mipmapPrecisionBits;
    //uint32_t              maxDrawIndexedIndexValue;
    uint32_t maxDrawIndirectCount;
    //float                 maxSamplerLodBias;
    float    maxSamplerAnisotropyLog2;
    uint32_t maxViewports;
    uint32_t maxViewportDims[2];
    //float                 viewportBoundsRange[2];
    //uint32_t              viewportSubPixelBits;
    size_t   minMemoryMapAlignment = 0ull;
    uint32_t bufferViewAlignment;
    uint32_t UBOAlignment;
    uint32_t SSBOAlignment;
    int32_t  minTexelOffset;
    uint32_t maxTexelOffset;
    int32_t  minTexelGatherOffset;
    uint32_t maxTexelGatherOffset;
    //float                 minInterpolationOffset;
    //float                 maxInterpolationOffset;
    //uint32_t              subPixelInterpolationOffsetBits;
    uint32_t maxFramebufferWidth;
    uint32_t maxFramebufferHeight;
    uint32_t maxFramebufferLayers;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferColorSampleCounts;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferDepthSampleCounts;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferStencilSampleCounts;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferNoAttachmentsSampleCounts;
    uint32_t maxColorAttachments = 0u;
    //VkSampleCountFlags    sampledImageColorSampleCounts;
    //VkSampleCountFlags    sampledImageIntegerSampleCounts;
    //VkSampleCountFlags    sampledImageDepthSampleCounts;
    //VkSampleCountFlags    sampledImageStencilSampleCounts;
    //VkSampleCountFlags    storageImageSampleCounts;
    //uint32_t              maxSampleMaskWords;
    //VkBool32              timestampComputeAndGraphics;
    float    timestampPeriodInNanoSeconds; // timestampPeriod is the number of nanoseconds required for a timestamp query to be incremented by 1 (a float because vulkan reports), use core::rational in the future
    uint32_t maxClipDistances;
    uint32_t maxCullDistances;
    uint32_t maxCombinedClipAndCullDistances;
    //uint32_t              discreteQueuePriorities;
    float pointSizeRange[2];
    float lineWidthRange[2];
    float pointSizeGranularity = 0.f;
    float lineWidthGranularity = 0.f;
    //VkBool32              strictLines;
    //VkBool32              standardSampleLocations;
    //VkDeviceSize          optimalBufferCopyOffsetAlignment;
    //VkDeviceSize          optimalBufferCopyRowPitchAlignment;
    uint64_t nonCoherentAtomSize;

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
    //VkPointClippingBehavior    pointClippingBehavior;
    //uint32_t                   maxMultiviewViewCount;
    //uint32_t                   maxMultiviewInstanceIndex;
    //VkBool32                   protectedNoFault;
    //uint32_t                   maxPerSetDescriptors;
    //VkDeviceSize               maxMemoryAllocationSize;
            
    /* Vulkan 1.2 Core  */
    //VkShaderFloatControlsIndependence    denormBehaviorIndependence;
    //VkShaderFloatControlsIndependence    roundingModeIndependence;
    //VkBool32                             shaderSignedZeroInfNanPreserveFloat16;
    //VkBool32                             shaderSignedZeroInfNanPreserveFloat32;
    //VkBool32                             shaderSignedZeroInfNanPreserveFloat64;
    //VkBool32                             shaderDenormPreserveFloat16;
    //VkBool32                             shaderDenormPreserveFloat32;
    //VkBool32                             shaderDenormPreserveFloat64;
    //VkBool32                             shaderDenormFlushToZeroFloat16;
    //VkBool32                             shaderDenormFlushToZeroFloat32;
    //VkBool32                             shaderDenormFlushToZeroFloat64;
    //VkBool32                             shaderRoundingModeRTEFloat16;
    //VkBool32                             shaderRoundingModeRTEFloat32;
    //VkBool32                             shaderRoundingModeRTEFloat64;
    //VkBool32                             shaderRoundingModeRTZFloat16;
    //VkBool32                             shaderRoundingModeRTZFloat32;
    //VkBool32                             shaderRoundingModeRTZFloat64;
    //uint32_t                             maxUpdateAfterBindDescriptorsInAllPools;
    //VkBool32                             shaderUniformBufferArrayNonUniformIndexingNative;
    //VkBool32                             shaderSampledImageArrayNonUniformIndexingNative;
    //VkBool32                             shaderStorageBufferArrayNonUniformIndexingNative;
    //VkBool32                             shaderStorageImageArrayNonUniformIndexingNative;
    //VkBool32                             shaderInputAttachmentArrayNonUniformIndexingNative;
    //VkBool32                             robustBufferAccessUpdateAfterBind;
    //VkBool32                             quadDivergentImplicitLod;
    //uint32_t                             maxPerStageDescriptorUpdateAfterBindSamplers;
    //uint32_t                             maxPerStageDescriptorUpdateAfterBindUniformBuffers;
    //uint32_t                             maxPerStageDescriptorUpdateAfterBindStorageBuffers;
    //uint32_t                             maxPerStageDescriptorUpdateAfterBindSampledImages;
    //uint32_t                             maxPerStageDescriptorUpdateAfterBindStorageImages;
    //uint32_t                             maxPerStageDescriptorUpdateAfterBindInputAttachments;
    //uint32_t                             maxPerStageUpdateAfterBindResources;
    //uint32_t                             maxDescriptorSetUpdateAfterBindSamplers;
    //uint32_t                             maxDescriptorSetUpdateAfterBindUniformBuffers;
    //uint32_t                             maxDescriptorSetUpdateAfterBindUniformBuffersDynamic;
    //uint32_t                             maxDescriptorSetUpdateAfterBindStorageBuffers;
    //uint32_t                             maxDescriptorSetUpdateAfterBindStorageBuffersDynamic;
    //uint32_t                             maxDescriptorSetUpdateAfterBindSampledImages;
    //uint32_t                             maxDescriptorSetUpdateAfterBindStorageImages;
    //uint32_t                             maxDescriptorSetUpdateAfterBindInputAttachments;
    //VkResolveModeFlags                   supportedDepthResolveModes;
    //VkResolveModeFlags                   supportedStencilResolveModes;
    //VkBool32                             independentResolveNone;
    //VkBool32                             independentResolve;
    //VkBool32                             filterMinmaxSingleComponentFormats;
    //VkBool32                             filterMinmaxImageComponentMapping;
    //uint64_t                             maxTimelineSemaphoreValueDifference;
    //VkSampleCountFlags                   framebufferIntegerColorSampleCounts;
            
    /* Vulkan 1.3 Core  */
    //uint32_t              minSubgroupSize;
    //uint32_t              maxSubgroupSize;
    //uint32_t              maxComputeWorkgroupSubgroups;
    //VkShaderStageFlags    requiredSubgroupSizeStages;
    //uint32_t              maxInlineUniformBlockSize;
    //uint32_t              maxPerStageDescriptorInlineUniformBlocks;
    //uint32_t              maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks;
    //uint32_t              maxDescriptorSetInlineUniformBlocks;
    //uint32_t              maxDescriptorSetUpdateAfterBindInlineUniformBlocks;
    //uint32_t              maxInlineUniformTotalSize;
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
    //VkDeviceSize          storageTexelBufferOffsetAlignmentBytes;
    //VkBool32              storageTexelBufferOffsetSingleTexelAlignment;
    //VkDeviceSize          uniformTexelBufferOffsetAlignmentBytes;
    //VkBool32              uniformTexelBufferOffsetSingleTexelAlignment;
    //VkDeviceSize          maxBufferSize;

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

    /* ShaderCoreProperties2AMD *//* provided by VK_AMD_shader_core_properties2 */
    //VkShaderCorePropertiesFlagsAMD    shaderCoreFeatures;
    //uint32_t                          activeComputeUnitCount;

    /* BlendOperationAdvancedPropertiesEXT *//* provided by VK_EXT_blend_operation_advanced */
    //uint32_t           advancedBlendMaxColorAttachments;
    //VkBool32           advancedBlendIndependentBlend;
    //VkBool32           advancedBlendNonPremultipliedSrcColor;
    //VkBool32           advancedBlendNonPremultipliedDstColor;
    //VkBool32           advancedBlendCorrelatedOverlap;
    //VkBool32           advancedBlendAllOperations;
            
    /* ConservativeRasterizationPropertiesEXT *//* provided by VK_EXT_conservative_rasterization */
    //float              primitiveOverestimationSize;
    //float              maxExtraPrimitiveOverestimationSize;
    //float              extraPrimitiveOverestimationSizeGranularity;
    //VkBool32           primitiveUnderestimation;
    //VkBool32           conservativePointAndLineRasterization;
    //VkBool32           degenerateTrianglesRasterized;
    //VkBool32           degenerateLinesRasterized;
    //VkBool32           fullyCoveredFragmentShaderInputVariable;
    //VkBool32           conservativeRasterizationPostDepthCoverage;
            
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

    /* DrmPropertiesEXT *//* provided by VK_EXT_physical_device_drm */
    //VkBool32           hasPrimary;
    //VkBool32           hasRender;
    //int64_t            primaryMajor;
    //int64_t            primaryMinor;
    //int64_t            renderMajor;
    //int64_t            renderMinor;

    /* ProvokingVertexPropertiesEXT *//* provided by VK_EXT_provoking_vertex */
    //VkBool32           provokingVertexModePerPipeline;
    //VkBool32           transformFeedbackPreservesTriangleFanProvokingVertex;

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
    // ! But we have a bool with the same name in SFeatures and that is mostly for GL when NBL_ARB_query_buffer_object is reported and that holds for every query 
    // VkBool32           allowCommandBufferQueryCopies;

    /* VK_KHR_portability_subset - PROVISINAL/NOTAVAILABLEANYMORE */

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

    /* MultiviewPerViewAttributesPropertiesNVX *//* VK_NVX_multiview_per_view_attributes */
    // VkBool32           perViewPositionAllComponents;

    /* VK_NVX_raytracing *//* Preview Extension of raytracing, useless*/
    
    /* CooperativeMatrixPropertiesNV *//* VK_NV_cooperative_matrix */
    // VkShaderStageFlags    cooperativeMatrixSupportedStages;

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

    /* FragmentShadingRateEnumsPropertiesNV *//* VK_NV_fragment_shading_rate_enums */
    // VkSampleCountFlagBits    maxFragmentShadingRateInvocationCount;

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

    /* ShadingRateImagePropertiesNV *//* VK_NV_shading_rate_image */
    //VkExtent2D         shadingRateTexelSize;
    //uint32_t           shadingRatePaletteSize;
    //uint32_t           shadingRateMaxCoarseSamples;

    /* FragmentDensityMapOffsetPropertiesQCOM *//* VK_QCOM_fragment_density_map_offset */
    //VkExtent2D         fragmentDensityOffsetGranularity;

    /* Nabla */
    uint32_t maxBufferSize;
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
