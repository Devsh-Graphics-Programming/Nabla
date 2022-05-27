#ifndef _NBL_VIDEO_S_PHYSICAL_DEVICE_LIMITS_H_INCLUDED_
#define _NBL_VIDEO_S_PHYSICAL_DEVICE_LIMITS_H_INCLUDED_

#include <type_traits>
#include "nbl/asset/utils/IGLSLCompiler.h" // asset::IGLSLCompiler::E_SPIRV_VERSION


namespace nbl::video
{

struct SPhysicalDeviceLimits
{
    /* Vulkan Core 1.0 */
    uint32_t maxImageDimension1D;
    uint32_t maxImageDimension2D;
    uint32_t maxImageDimension3D;
    uint32_t maxImageDimensionCube;
    uint32_t maxImageArrayLayers;
    uint32_t maxBufferViewSizeTexels;
    uint32_t maxUBOSize;
    uint32_t maxSSBOSize;
    //uint32_t              maxPushConstantsSize;
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
    //uint32_t              maxDescriptorSetInputAttachments;
    //uint32_t              maxVertexInputAttributes;
    //uint32_t              maxVertexInputBindings;
    //uint32_t              maxVertexInputAttributeOffset;
    //uint32_t              maxVertexInputBindingStride;
    //uint32_t              maxVertexOutputComponents;
    //uint32_t              maxTessellationGenerationLevel;
    //uint32_t              maxTessellationPatchSize;
    //uint32_t              maxTessellationControlPerVertexInputComponents;
    //uint32_t              maxTessellationControlPerVertexOutputComponents;
    //uint32_t              maxTessellationControlPerPatchOutputComponents;
    //uint32_t              maxTessellationControlTotalOutputComponents;
    //uint32_t              maxTessellationEvaluationInputComponents;
    //uint32_t              maxTessellationEvaluationOutputComponents;
    //uint32_t              maxGeometryShaderInvocations;
    //uint32_t              maxGeometryInputComponents;
    //uint32_t              maxGeometryOutputComponents;
    //uint32_t              maxGeometryOutputVertices;
    //uint32_t              maxGeometryTotalOutputComponents;
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
    //size_t                minMemoryMapAlignment;
    uint32_t bufferViewAlignment;
    uint32_t UBOAlignment;
    uint32_t SSBOAlignment;
    //int32_t               minTexelOffset;
    //uint32_t              maxTexelOffset;
    //int32_t               minTexelGatherOffset;
    //uint32_t              maxTexelGatherOffset;
    //float                 minInterpolationOffset;
    //float                 maxInterpolationOffset;
    //uint32_t              subPixelInterpolationOffsetBits;
    //uint32_t              maxFramebufferWidth;
    //uint32_t              maxFramebufferHeight;
    //uint32_t              maxFramebufferLayers;
    //VkSampleCountFlags    framebufferColorSampleCounts;
    //VkSampleCountFlags    framebufferDepthSampleCounts;
    //VkSampleCountFlags    framebufferStencilSampleCounts;
    //VkSampleCountFlags    framebufferNoAttachmentsSampleCounts;
    //uint32_t              maxColorAttachments;
    //VkSampleCountFlags    sampledImageColorSampleCounts;
    //VkSampleCountFlags    sampledImageIntegerSampleCounts;
    //VkSampleCountFlags    sampledImageDepthSampleCounts;
    //VkSampleCountFlags    sampledImageStencilSampleCounts;
    //VkSampleCountFlags    storageImageSampleCounts;
    //uint32_t              maxSampleMaskWords;
    //VkBool32              timestampComputeAndGraphics;
    float    timestampPeriodInNanoSeconds; // timestampPeriod is the number of nanoseconds required for a timestamp query to be incremented by 1 (a float because vulkan reports), use core::rational in the future
    //uint32_t              maxClipDistances;
    //uint32_t              maxCullDistances;
    //uint32_t              maxCombinedClipAndCullDistances;
    //uint32_t              discreteQueuePriorities;
    float pointSizeRange[2];
    float lineWidthRange[2];
    //float                 pointSizeGranularity;
    //float                 lineWidthGranularity;
    //VkBool32              strictLines;
    //VkBool32              standardSampleLocations;
    //VkDeviceSize          optimalBufferCopyOffsetAlignment;
    //VkDeviceSize          optimalBufferCopyRowPitchAlignment;
    uint64_t nonCoherentAtomSize;
            
    /* Vulkan Core 1.1 */
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
            
    /* Vulkan Core 1.2 */
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
            
    /* Vulkan Core 1.3 */
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

    /* ShaderCorePropertiesAMD */
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
    /* ShaderCoreProperties2AMD */
    //VkShaderCorePropertiesFlagsAMD    shaderCoreFeatures;
    //uint32_t                          activeComputeUnitCount;

    /* BlendOperationAdvancedPropertiesEXT */
    //uint32_t           advancedBlendMaxColorAttachments;
    //VkBool32           advancedBlendIndependentBlend;
    //VkBool32           advancedBlendNonPremultipliedSrcColor;
    //VkBool32           advancedBlendNonPremultipliedDstColor;
    //VkBool32           advancedBlendCorrelatedOverlap;
    //VkBool32           advancedBlendAllOperations;
            
    /* ConservativeRasterizationPropertiesEXT */
    //float              primitiveOverestimationSize;
    //float              maxExtraPrimitiveOverestimationSize;
    //float              extraPrimitiveOverestimationSizeGranularity;
    //VkBool32           primitiveUnderestimation;
    //VkBool32           conservativePointAndLineRasterization;
    //VkBool32           degenerateTrianglesRasterized;
    //VkBool32           degenerateLinesRasterized;
    //VkBool32           fullyCoveredFragmentShaderInputVariable;
    //VkBool32           conservativeRasterizationPostDepthCoverage;
            
    /* CustomBorderColorPropertiesEXT */
    //uint32_t           maxCustomBorderColorSamplers;

    /* DescriptorIndexingPropertiesEXT ---> MOVED TO Vulkan 1.2 Core */

    /* DiscardRectanglePropertiesEXT */
    //uint32_t           maxDiscardRectangles;
            
    /* ExternalMemoryHostPropertiesEXT */
    //VkDeviceSize       minImportedHostPointerAlignment;
    
    /* AccelerationStructurePropertiesKHR */
    uint64_t           maxGeometryCount;
    uint64_t           maxInstanceCount;
    uint64_t           maxPrimitiveCount;
    uint32_t           maxPerStageDescriptorAccelerationStructures;
    uint32_t           maxPerStageDescriptorUpdateAfterBindAccelerationStructures;
    uint32_t           maxDescriptorSetAccelerationStructures;
    uint32_t           maxDescriptorSetUpdateAfterBindAccelerationStructures;
    uint32_t           minAccelerationStructureScratchOffsetAlignment;
            
    /* RayTracingPipelinePropertiesKHR */
    uint32_t           shaderGroupHandleSize;
    uint32_t           maxRayRecursionDepth;
    uint32_t           maxShaderGroupStride;
    uint32_t           shaderGroupBaseAlignment;
    uint32_t           shaderGroupHandleCaptureReplaySize;
    uint32_t           maxRayDispatchInvocationCount;
    uint32_t           shaderGroupHandleAlignment;
    uint32_t           maxRayHitAttributeSize;
            
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
