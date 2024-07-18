   // VK 1.0 Core

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

    if (minMemoryMapAlignment > _rhs.minMemoryMapAlignment) return false;
    if (bufferViewAlignment < _rhs.bufferViewAlignment) return false;
    if (minUBOAlignment < _rhs.minUBOAlignment) return false;
    if (minSSBOAlignment < _rhs.minSSBOAlignment) return false;

    if (minTexelOffset < _rhs.minTexelOffset) return false;
    if (maxTexelOffset > _rhs.maxTexelOffset) return false;
    if (minTexelGatherOffset < _rhs.minTexelGatherOffset) return false;
    if (maxTexelGatherOffset > _rhs.maxTexelGatherOffset) return false;

    if (minInterpolationOffset < _rhs.minInterpolationOffset) return false;
    if (maxInterpolationOffset > _rhs.maxInterpolationOffset) return false;
    if (subPixelInterpolationOffsetBits > _rhs.subPixelInterpolationOffsetBits) return false;

    if (maxFramebufferWidth > _rhs.maxFramebufferWidth) return false;
    if (maxFramebufferHeight > _rhs.maxFramebufferHeight) return false;
    if (maxFramebufferLayers > _rhs.maxFramebufferLayers) return false;

    if (maxColorAttachments > _rhs.maxColorAttachments) return false;


    if (maxSampleMaskWords > _rhs.maxSampleMaskWords) return false;

    // if (timestampPeriodInNanoSeconds < _rhs.timestampPeriodInNanoSeconds) return false;

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


   // VK 1.1 Everything is either a Limit or Required

    // if (subgroupSize > _rhs.subgroupSize) return false;
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

   // VK 1.2

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



   // VK 1.3

    if (minSubgroupSize < _rhs.minSubgroupSize) return false;
    if (maxSubgroupSize > _rhs.maxSubgroupSize) return false;
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

   // Nabla Core Extensions

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

   // Extensions

    if (shaderTrinaryMinmax && !_rhs.shaderTrinaryMinmax) return false;

    if (shaderExplicitVertexParameter && !_rhs.shaderExplicitVertexParameter) return false;

    if (gpuShaderHalfFloatAMD && !_rhs.gpuShaderHalfFloatAMD) return false;

    if (shaderImageLoadStoreLod && !_rhs.shaderImageLoadStoreLod) return false;


    if (displayTiming && !_rhs.displayTiming) return false;

    if (maxDiscardRectangles > _rhs.maxDiscardRectangles) return false;

    // if (primitiveOverestimationSize > _rhs.primitiveOverestimationSize) return false;
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

    if (computeDerivativeGroupQuads && !_rhs.computeDerivativeGroupQuads) return false;
    if (computeDerivativeGroupLinear && !_rhs.computeDerivativeGroupLinear) return false;

    if (imageFootprint && !_rhs.imageFootprint) return false;

    // if (pciDomain > _rhs.pciDomain) return false;
    // if (pciBus > _rhs.pciBus) return false;
    // if (pciDevice > _rhs.pciDevice) return false;
    // if (pciFunction > _rhs.pciFunction) return false;

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

   // Nabla

    if (computeUnits > _rhs.computeUnits) return false;
    if (dispatchBase && !_rhs.dispatchBase) return false;
    if (allowCommandBufferQueryCopies && !_rhs.allowCommandBufferQueryCopies) return false;
    if (maxOptimallyResidentWorkgroupInvocations > _rhs.maxOptimallyResidentWorkgroupInvocations) return false;
    if (maxResidentInvocations > _rhs.maxResidentInvocations) return false;
    if (spirvVersion > _rhs.spirvVersion) return false;

   // Core 1.0
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
    if (shaderDeviceClock && !_rhs.shaderDeviceClock) return false;
    if (shaderSubgroupClock && !_rhs.shaderSubgroupClock) return false;
    if (imageFootPrint && !_rhs.imageFootPrint) return false;
    if (shaderIntegerFunctions2 && !_rhs.shaderIntegerFunctions2) return false;
    if (shaderEarlyAndLateFragmentTests && !_rhs.shaderEarlyAndLateFragmentTests) return false;

