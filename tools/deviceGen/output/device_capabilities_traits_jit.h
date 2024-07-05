std::string jit_traits = R"===(
// Limits JIT Members
// VK 1.0
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxImageDimension1D = )===" + CJITIncludeLoader::to_string(limits.maxImageDimension1D) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxImageDimension2D = )===" + CJITIncludeLoader::to_string(limits.maxImageDimension2D) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxImageDimension3D = )===" + CJITIncludeLoader::to_string(limits.maxImageDimension3D) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxImageDimensionCube = )===" + CJITIncludeLoader::to_string(limits.maxImageDimensionCube) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxImageArrayLayers = )===" + CJITIncludeLoader::to_string(limits.maxImageArrayLayers) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxBufferViewTexels = )===" + CJITIncludeLoader::to_string(limits.maxBufferViewTexels) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxUBOSize = )===" + CJITIncludeLoader::to_string(limits.maxUBOSize) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxSSBOSize = )===" + CJITIncludeLoader::to_string(limits.maxSSBOSize) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxPushConstantsSize = )===" + CJITIncludeLoader::to_string(limits.maxPushConstantsSize) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxMemoryAllocationCount = )===" + CJITIncludeLoader::to_string(limits.maxMemoryAllocationCount) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxSamplerAllocationCount = )===" + CJITIncludeLoader::to_string(limits.maxSamplerAllocationCount) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t bufferImageGranularity = )===" + CJITIncludeLoader::to_string(limits.bufferImageGranularity) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorSamplers = )===" + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorSamplers) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorUBOs = )===" + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorUBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorSSBOs = )===" + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorSSBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorImages = )===" + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorImages) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorStorageImages = )===" + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorStorageImages) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorInputAttachments = )===" + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorInputAttachments) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageResources = )===" + CJITIncludeLoader::to_string(limits.maxPerStageResources) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetSamplers = )===" + CJITIncludeLoader::to_string(limits.maxDescriptorSetSamplers) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetUBOs = )===" + CJITIncludeLoader::to_string(limits.maxDescriptorSetUBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetDynamicOffsetUBOs = )===" + CJITIncludeLoader::to_string(limits.maxDescriptorSetDynamicOffsetUBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetSSBOs = )===" + CJITIncludeLoader::to_string(limits.maxDescriptorSetSSBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetDynamicOffsetSSBOs = )===" + CJITIncludeLoader::to_string(limits.maxDescriptorSetDynamicOffsetSSBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetImages = )===" + CJITIncludeLoader::to_string(limits.maxDescriptorSetImages) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetStorageImages = )===" + CJITIncludeLoader::to_string(limits.maxDescriptorSetStorageImages) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetInputAttachments = )===" + CJITIncludeLoader::to_string(limits.maxDescriptorSetInputAttachments) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t maxVertexOutputComponents = )===" + CJITIncludeLoader::to_string(limits.maxVertexOutputComponents) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTessellationGenerationLevel = )===" + CJITIncludeLoader::to_string(limits.maxTessellationGenerationLevel) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTessellationPatchSize = )===" + CJITIncludeLoader::to_string(limits.maxTessellationPatchSize) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTessellationControlPerVertexInputComponents = )===" + CJITIncludeLoader::to_string(limits.maxTessellationControlPerVertexInputComponents) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTessellationControlPerVertexOutputComponents = )===" + CJITIncludeLoader::to_string(limits.maxTessellationControlPerVertexOutputComponents) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTessellationControlPerPatchOutputComponents = )===" + CJITIncludeLoader::to_string(limits.maxTessellationControlPerPatchOutputComponents) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTessellationControlTotalOutputComponents = )===" + CJITIncludeLoader::to_string(limits.maxTessellationControlTotalOutputComponents) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTessellationEvaluationInputComponents = )===" + CJITIncludeLoader::to_string(limits.maxTessellationEvaluationInputComponents) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTessellationEvaluationOutputComponents = )===" + CJITIncludeLoader::to_string(limits.maxTessellationEvaluationOutputComponents) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t maxGeometryShaderInvocations = )===" + CJITIncludeLoader::to_string(limits.maxGeometryShaderInvocations) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxGeometryInputComponents = )===" + CJITIncludeLoader::to_string(limits.maxGeometryInputComponents) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxGeometryOutputComponents = )===" + CJITIncludeLoader::to_string(limits.maxGeometryOutputComponents) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxGeometryOutputVertices = )===" + CJITIncludeLoader::to_string(limits.maxGeometryOutputVertices) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxGeometryTotalOutputComponents = )===" + CJITIncludeLoader::to_string(limits.maxGeometryTotalOutputComponents) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxFragmentInputComponents = )===" + CJITIncludeLoader::to_string(limits.maxFragmentInputComponents) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxFragmentOutputAttachments = )===" + CJITIncludeLoader::to_string(limits.maxFragmentOutputAttachments) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxFragmentDualSrcAttachments = )===" + CJITIncludeLoader::to_string(limits.maxFragmentDualSrcAttachments) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxFragmentCombinedOutputResources = )===" + CJITIncludeLoader::to_string(limits.maxFragmentCombinedOutputResources) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxComputeSharedMemorySize = )===" + CJITIncludeLoader::to_string(limits.maxComputeSharedMemorySize) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxComputeWorkGroupCountX = )===" + CJITIncludeLoader::to_string(limits.maxComputeWorkGroupCount[0]) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxComputeWorkGroupCountY = )===" + CJITIncludeLoader::to_string(limits.maxComputeWorkGroupCount[1]) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxComputeWorkGroupCountZ = )===" + CJITIncludeLoader::to_string(limits.maxComputeWorkGroupCount[2]) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxComputeWorkGroupInvocations = )===" + CJITIncludeLoader::to_string(limits.maxComputeWorkGroupInvocations) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxWorkgroupSizeX = )===" + CJITIncludeLoader::to_string(limits.maxWorkgroupSize[0]) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxWorkgroupSizeY = )===" + CJITIncludeLoader::to_string(limits.maxWorkgroupSize[1]) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxWorkgroupSizeZ = )===" + CJITIncludeLoader::to_string(limits.maxWorkgroupSize[2]) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t subPixelPrecisionBits = )===" + CJITIncludeLoader::to_string(limits.subPixelPrecisionBits) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t subTexelPrecisionBits = )===" + CJITIncludeLoader::to_string(limits.subTexelPrecisionBits) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t mipmapPrecisionBits = )===" + CJITIncludeLoader::to_string(limits.mipmapPrecisionBits) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDrawIndirectCount = )===" + CJITIncludeLoader::to_string(limits.maxDrawIndirectCount) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxSamplerLodBiasBitPattern = )===" + CJITIncludeLoader::to_string(*reinterpret_cast<const uint32_t *>(&limits.maxSamplerLodBias)) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxSamplerAnisotropyLog2 = )===" + CJITIncludeLoader::to_string(limits.maxSamplerAnisotropyLog2) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t maxViewports = )===" + CJITIncludeLoader::to_string(limits.maxViewports) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxViewportDimsX = )===" + CJITIncludeLoader::to_string(limits.maxViewportDims[0]) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxViewportDimsY = )===" + CJITIncludeLoader::to_string(limits.maxViewportDims[1]) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t viewportBoundsRangeBitPatternMin = )===" + CJITIncludeLoader::to_string(*reinterpret_cast<const uint32_t *>(&limits.viewportBoundsRange[0])) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t viewportBoundsRangeBitPatternMax = )===" + CJITIncludeLoader::to_string(*reinterpret_cast<const uint32_t *>(&limits.viewportBoundsRange[1])) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t viewportSubPixelBits = )===" + CJITIncludeLoader::to_string(limits.viewportSubPixelBits) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t minMemoryMapAlignment = )===" + CJITIncludeLoader::to_string(limits.minMemoryMapAlignment) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t bufferViewAlignment = )===" + CJITIncludeLoader::to_string(limits.bufferViewAlignment) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t minUBOAlignment = )===" + CJITIncludeLoader::to_string(limits.minUBOAlignment) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t minSSBOAlignment = )===" + CJITIncludeLoader::to_string(limits.minSSBOAlignment) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE int16_t minTexelOffset = )===" + CJITIncludeLoader::to_string(limits.minTexelOffset) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTexelOffset = )===" + CJITIncludeLoader::to_string(limits.maxTexelOffset) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE int16_t minTexelGatherOffset = )===" + CJITIncludeLoader::to_string(limits.minTexelGatherOffset) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTexelGatherOffset = )===" + CJITIncludeLoader::to_string(limits.maxTexelGatherOffset) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t minInterpolationOffsetBitPattern = )===" + CJITIncludeLoader::to_string(*reinterpret_cast<const uint32_t *>(&limits.minInterpolationOffset)) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxInterpolationOffsetBitPattern = )===" + CJITIncludeLoader::to_string(*reinterpret_cast<const uint32_t *>(&limits.maxInterpolationOffset)) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t subPixelInterpolationOffsetBits = )===" + CJITIncludeLoader::to_string(limits.subPixelInterpolationOffsetBits) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxFramebufferWidth = )===" + CJITIncludeLoader::to_string(limits.maxFramebufferWidth) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxFramebufferHeight = )===" + CJITIncludeLoader::to_string(limits.maxFramebufferHeight) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxFramebufferLayers = )===" + CJITIncludeLoader::to_string(limits.maxFramebufferLayers) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t maxColorAttachments = )===" + CJITIncludeLoader::to_string(limits.maxColorAttachments) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t maxSampleMaskWords = )===" + CJITIncludeLoader::to_string(limits.maxSampleMaskWords) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t timestampPeriodInNanoSecondsBitPattern = )===" + CJITIncludeLoader::to_string(*reinterpret_cast<const uint32_t *>(&limits.timestampPeriodInNanoSeconds)) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t maxClipDistances = )===" + CJITIncludeLoader::to_string(limits.maxClipDistances) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxCullDistances = )===" + CJITIncludeLoader::to_string(limits.maxCullDistances) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxCombinedClipAndCullDistances = )===" + CJITIncludeLoader::to_string(limits.maxCombinedClipAndCullDistances) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t discreteQueuePriorities = )===" + CJITIncludeLoader::to_string(limits.discreteQueuePriorities) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t pointSizeRangeBitPatternMin = )===" + CJITIncludeLoader::to_string(*reinterpret_cast<const uint32_t *>(&limits.pointSizeRange[0])) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t pointSizeRangeBitPatternMax = )===" + CJITIncludeLoader::to_string(*reinterpret_cast<const uint32_t *>(&limits.pointSizeRange[1])) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t lineWidthRangeBitPatternMin = )===" + CJITIncludeLoader::to_string(*reinterpret_cast<const uint32_t *>(&limits.lineWidthRange[0])) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t lineWidthRangeBitPatternMax = )===" + CJITIncludeLoader::to_string(*reinterpret_cast<const uint32_t *>(&limits.lineWidthRange[1])) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t pointSizeGranularityBitPattern = )===" + CJITIncludeLoader::to_string(*reinterpret_cast<const uint32_t *>(&limits.pointSizeGranularity)) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t lineWidthGranularityBitPattern = )===" + CJITIncludeLoader::to_string(*reinterpret_cast<const uint32_t *>(&limits.lineWidthGranularity)) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool strictLines = )===" + CJITIncludeLoader::to_string(limits.strictLines) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool standardSampleLocations = )===" + CJITIncludeLoader::to_string(limits.standardSampleLocations) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t optimalBufferCopyOffsetAlignment = )===" + CJITIncludeLoader::to_string(limits.optimalBufferCopyOffsetAlignment) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t optimalBufferCopyRowPitchAlignment = )===" + CJITIncludeLoader::to_string(limits.optimalBufferCopyRowPitchAlignment) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t nonCoherentAtomSize = )===" + CJITIncludeLoader::to_string(limits.nonCoherentAtomSize) + R"===(;

// VK 1.1
NBL_CONSTEXPR_STATIC_INLINE uint16_t subgroupSize = )===" + CJITIncludeLoader::to_string(limits.subgroupSize) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE ShaderStage subgroupOpsShaderStages = )===" + CJITIncludeLoader::to_string(limits.subgroupOpsShaderStages) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupClustered = )===" + CJITIncludeLoader::to_string(limits.shaderSubgroupClustered) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupArithmetic = )===" + CJITIncludeLoader::to_string(limits.shaderSubgroupArithmetic) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupQuad = )===" + CJITIncludeLoader::to_string(limits.shaderSubgroupQuad) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupQuadAllStages = )===" + CJITIncludeLoader::to_string(limits.shaderSubgroupQuadAllStages) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE PointClippingBehavior pointClippingBehavior = )===" + CJITIncludeLoader::to_string(limits.pointClippingBehavior) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t maxMultiviewViewCount = )===" + CJITIncludeLoader::to_string(limits.maxMultiviewViewCount) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxMultiviewInstanceIndex = )===" + CJITIncludeLoader::to_string(limits.maxMultiviewInstanceIndex) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerSetDescriptors = )===" + CJITIncludeLoader::to_string(limits.maxPerSetDescriptors) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t maxMemoryAllocationSize = )===" + CJITIncludeLoader::to_string(limits.maxMemoryAllocationSize) + R"===(;

// VK 1.2
NBL_CONSTEXPR_STATIC_INLINE bool shaderSignedZeroInfNanPreserveFloat64 = )===" + CJITIncludeLoader::to_string(limits.shaderSignedZeroInfNanPreserveFloat64) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderDenormPreserveFloat16 = )===" + CJITIncludeLoader::to_string(limits.shaderDenormPreserveFloat16) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderDenormPreserveFloat32 = )===" + CJITIncludeLoader::to_string(limits.shaderDenormPreserveFloat32) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderDenormPreserveFloat64 = )===" + CJITIncludeLoader::to_string(limits.shaderDenormPreserveFloat64) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderDenormFlushToZeroFloat16 = )===" + CJITIncludeLoader::to_string(limits.shaderDenormFlushToZeroFloat16) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderDenormFlushToZeroFloat32 = )===" + CJITIncludeLoader::to_string(limits.shaderDenormFlushToZeroFloat32) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderDenormFlushToZeroFloat64 = )===" + CJITIncludeLoader::to_string(limits.shaderDenormFlushToZeroFloat64) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderRoundingModeRTEFloat16 = )===" + CJITIncludeLoader::to_string(limits.shaderRoundingModeRTEFloat16) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderRoundingModeRTEFloat32 = )===" + CJITIncludeLoader::to_string(limits.shaderRoundingModeRTEFloat32) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderRoundingModeRTEFloat64 = )===" + CJITIncludeLoader::to_string(limits.shaderRoundingModeRTEFloat64) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderRoundingModeRTZFloat16 = )===" + CJITIncludeLoader::to_string(limits.shaderRoundingModeRTZFloat16) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderRoundingModeRTZFloat32 = )===" + CJITIncludeLoader::to_string(limits.shaderRoundingModeRTZFloat32) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderRoundingModeRTZFloat64 = )===" + CJITIncludeLoader::to_string(limits.shaderRoundingModeRTZFloat64) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxUpdateAfterBindDescriptorsInAllPools = )===" + CJITIncludeLoader::to_string(limits.maxUpdateAfterBindDescriptorsInAllPools) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderUniformBufferArrayNonUniformIndexingNative = )===" + CJITIncludeLoader::to_string(limits.shaderUniformBufferArrayNonUniformIndexingNative) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSampledImageArrayNonUniformIndexingNative = )===" + CJITIncludeLoader::to_string(limits.shaderSampledImageArrayNonUniformIndexingNative) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderStorageBufferArrayNonUniformIndexingNative = )===" + CJITIncludeLoader::to_string(limits.shaderStorageBufferArrayNonUniformIndexingNative) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderStorageImageArrayNonUniformIndexingNative = )===" + CJITIncludeLoader::to_string(limits.shaderStorageImageArrayNonUniformIndexingNative) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderInputAttachmentArrayNonUniformIndexingNative = )===" + CJITIncludeLoader::to_string(limits.shaderInputAttachmentArrayNonUniformIndexingNative) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool robustBufferAccessUpdateAfterBind = )===" + CJITIncludeLoader::to_string(limits.robustBufferAccessUpdateAfterBind) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool quadDivergentImplicitLod = )===" + CJITIncludeLoader::to_string(limits.quadDivergentImplicitLod) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorUpdateAfterBindSamplers = )===" + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorUpdateAfterBindSamplers) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorUpdateAfterBindUBOs = )===" + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorUpdateAfterBindUBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorUpdateAfterBindSSBOs = )===" + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorUpdateAfterBindSSBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorUpdateAfterBindImages = )===" + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorUpdateAfterBindImages) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorUpdateAfterBindStorageImages = )===" + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorUpdateAfterBindStorageImages) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorUpdateAfterBindInputAttachments = )===" + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorUpdateAfterBindInputAttachments) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageUpdateAfterBindResources = )===" + CJITIncludeLoader::to_string(limits.maxPerStageUpdateAfterBindResources) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetUpdateAfterBindSamplers = )===" + CJITIncludeLoader::to_string(limits.maxDescriptorSetUpdateAfterBindSamplers) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetUpdateAfterBindUBOs = )===" + CJITIncludeLoader::to_string(limits.maxDescriptorSetUpdateAfterBindUBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetUpdateAfterBindDynamicOffsetUBOs = )===" + CJITIncludeLoader::to_string(limits.maxDescriptorSetUpdateAfterBindDynamicOffsetUBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetUpdateAfterBindSSBOs = )===" + CJITIncludeLoader::to_string(limits.maxDescriptorSetUpdateAfterBindSSBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetUpdateAfterBindDynamicOffsetSSBOs = )===" + CJITIncludeLoader::to_string(limits.maxDescriptorSetUpdateAfterBindDynamicOffsetSSBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetUpdateAfterBindImages = )===" + CJITIncludeLoader::to_string(limits.maxDescriptorSetUpdateAfterBindImages) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetUpdateAfterBindStorageImages = )===" + CJITIncludeLoader::to_string(limits.maxDescriptorSetUpdateAfterBindStorageImages) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetUpdateAfterBindInputAttachments = )===" + CJITIncludeLoader::to_string(limits.maxDescriptorSetUpdateAfterBindInputAttachments) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE ResolveModeFlags supportedDepthResolveModes = )===" + CJITIncludeLoader::to_string(limits.supportedDepthResolveModes) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE ResolveModeFlags supportedStencilResolveModes = )===" + CJITIncludeLoader::to_string(limits.supportedStencilResolveModes) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool independentResolveNone = )===" + CJITIncludeLoader::to_string(limits.independentResolveNone) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool independentResolve = )===" + CJITIncludeLoader::to_string(limits.independentResolve) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool filterMinmaxImageComponentMapping = )===" + CJITIncludeLoader::to_string(limits.filterMinmaxImageComponentMapping) + R"===(;

// VK 1.3
NBL_CONSTEXPR_STATIC_INLINE uint16_t minSubgroupSize = )===" + CJITIncludeLoader::to_string(limits.minSubgroupSize) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxSubgroupSize = )===" + CJITIncludeLoader::to_string(limits.maxSubgroupSize) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxComputeWorkgroupSubgroups = )===" + CJITIncludeLoader::to_string(limits.maxComputeWorkgroupSubgroups) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE ShaderStage requiredSubgroupSizeStages = )===" + CJITIncludeLoader::to_string(limits.requiredSubgroupSizeStages) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct8BitUnsignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProduct8BitUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct8BitSignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProduct8BitSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct8BitMixedSignednessAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProduct8BitMixedSignednessAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct4x8BitPackedUnsignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProduct4x8BitPackedUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct4x8BitPackedSignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProduct4x8BitPackedSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct4x8BitPackedMixedSignednessAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProduct4x8BitPackedMixedSignednessAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct16BitUnsignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProduct16BitUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct16BitSignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProduct16BitSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct16BitMixedSignednessAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProduct16BitMixedSignednessAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct32BitUnsignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProduct32BitUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct32BitSignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProduct32BitSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct32BitMixedSignednessAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProduct32BitMixedSignednessAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct64BitUnsignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProduct64BitUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct64BitSignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProduct64BitSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct64BitMixedSignednessAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProduct64BitMixedSignednessAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating8BitUnsignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating8BitUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating8BitSignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating8BitSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating16BitUnsignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating16BitUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating16BitSignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating16BitSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating32BitUnsignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating32BitUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating32BitSignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating32BitSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating64BitUnsignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating64BitUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating64BitSignedAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating64BitSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated = )===" + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint64_t storageTexelBufferOffsetAlignmentBytes = )===" + CJITIncludeLoader::to_string(limits.storageTexelBufferOffsetAlignmentBytes) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t uniformTexelBufferOffsetAlignmentBytes = )===" + CJITIncludeLoader::to_string(limits.uniformTexelBufferOffsetAlignmentBytes) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint64_t maxBufferSize = )===" + CJITIncludeLoader::to_string(limits.maxBufferSize) + R"===(;

// Nabla Core Extensions
NBL_CONSTEXPR_STATIC_INLINE uint32_t minImportedHostPointerAlignment = )===" + CJITIncludeLoader::to_string(limits.minImportedHostPointerAlignment) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderBufferFloat32AtomicAdd = )===" + CJITIncludeLoader::to_string(limits.shaderBufferFloat32AtomicAdd) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderBufferFloat64Atomics = )===" + CJITIncludeLoader::to_string(limits.shaderBufferFloat64Atomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderBufferFloat64AtomicAdd = )===" + CJITIncludeLoader::to_string(limits.shaderBufferFloat64AtomicAdd) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSharedFloat32AtomicAdd = )===" + CJITIncludeLoader::to_string(limits.shaderSharedFloat32AtomicAdd) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSharedFloat64Atomics = )===" + CJITIncludeLoader::to_string(limits.shaderSharedFloat64Atomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSharedFloat64AtomicAdd = )===" + CJITIncludeLoader::to_string(limits.shaderSharedFloat64AtomicAdd) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderImageFloat32AtomicAdd = )===" + CJITIncludeLoader::to_string(limits.shaderImageFloat32AtomicAdd) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool sparseImageFloat32Atomics = )===" + CJITIncludeLoader::to_string(limits.sparseImageFloat32Atomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool sparseImageFloat32AtomicAdd = )===" + CJITIncludeLoader::to_string(limits.sparseImageFloat32AtomicAdd) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint64_t robustStorageBufferAccessSizeAlignment = )===" + CJITIncludeLoader::to_string(limits.robustStorageBufferAccessSizeAlignment) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t robustUniformBufferAccessSizeAlignment = )===" + CJITIncludeLoader::to_string(limits.robustUniformBufferAccessSizeAlignment) + R"===(;

// Extensions
NBL_CONSTEXPR_STATIC_INLINE bool shaderTrinaryMinmax = )===" + CJITIncludeLoader::to_string(limits.shaderTrinaryMinmax) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderExplicitVertexParameter = )===" + CJITIncludeLoader::to_string(limits.shaderExplicitVertexParameter) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool gpuShaderHalfFloatAMD = )===" + CJITIncludeLoader::to_string(limits.gpuShaderHalfFloatAMD) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderImageLoadStoreLod = )===" + CJITIncludeLoader::to_string(limits.shaderImageLoadStoreLod) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool displayTiming = )===" + CJITIncludeLoader::to_string(limits.displayTiming) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDiscardRectangles = )===" + CJITIncludeLoader::to_string(limits.maxDiscardRectangles) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t primitiveOverestimationSizeBitPattern = )===" + CJITIncludeLoader::to_string(*reinterpret_cast<const uint32_t *>(&limits.primitiveOverestimationSize)) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxExtraPrimitiveOverestimationSizeBitPattern = )===" + CJITIncludeLoader::to_string(*reinterpret_cast<const uint32_t *>(&limits.maxExtraPrimitiveOverestimationSize)) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t extraPrimitiveOverestimationSizeGranularityBitPattern = )===" + CJITIncludeLoader::to_string(*reinterpret_cast<const uint32_t *>(&limits.extraPrimitiveOverestimationSizeGranularity)) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool primitiveUnderestimation = )===" + CJITIncludeLoader::to_string(limits.primitiveUnderestimation) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool conservativePointAndLineRasterization = )===" + CJITIncludeLoader::to_string(limits.conservativePointAndLineRasterization) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool degenerateTrianglesRasterized = )===" + CJITIncludeLoader::to_string(limits.degenerateTrianglesRasterized) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool degenerateLinesRasterized = )===" + CJITIncludeLoader::to_string(limits.degenerateLinesRasterized) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool fullyCoveredFragmentShaderInputVariable = )===" + CJITIncludeLoader::to_string(limits.fullyCoveredFragmentShaderInputVariable) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool conservativeRasterizationPostDepthCoverage = )===" + CJITIncludeLoader::to_string(limits.conservativeRasterizationPostDepthCoverage) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool queueFamilyForeign = )===" + CJITIncludeLoader::to_string(limits.queueFamilyForeign) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderStencilExport = )===" + CJITIncludeLoader::to_string(limits.shaderStencilExport) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool variableSampleLocations = )===" + CJITIncludeLoader::to_string(limits.variableSampleLocations) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t sampleLocationSubPixelBits = )===" + CJITIncludeLoader::to_string(limits.sampleLocationSubPixelBits) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE SampleCountFlags sampleLocationSampleCounts = )===" + CJITIncludeLoader::to_string(limits.sampleLocationSampleCounts) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxSampleLocationGridSizeX = )===" + CJITIncludeLoader::to_string(limits.maxSampleLocationGridSize.x) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxSampleLocationGridSizeY = )===" + CJITIncludeLoader::to_string(limits.maxSampleLocationGridSize.y) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t sampleLocationCoordinateRangeBitPatternMin = )===" + CJITIncludeLoader::to_string(*reinterpret_cast<const uint32_t *>(&limits.sampleLocationCoordinateRange[0])) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t sampleLocationCoordinateRangeBitPatternMax = )===" + CJITIncludeLoader::to_string(*reinterpret_cast<const uint32_t *>(&limits.sampleLocationCoordinateRange[1])) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint64_t maxAccelerationStructureGeometryCount = )===" + CJITIncludeLoader::to_string(limits.maxAccelerationStructureGeometryCount) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t maxAccelerationStructureInstanceCount = )===" + CJITIncludeLoader::to_string(limits.maxAccelerationStructureInstanceCount) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t maxAccelerationStructurePrimitiveCount = )===" + CJITIncludeLoader::to_string(limits.maxAccelerationStructurePrimitiveCount) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t maxPerStageDescriptorAccelerationStructures = )===" + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorAccelerationStructures) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t maxPerStageDescriptorUpdateAfterBindAccelerationStructures = )===" + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorUpdateAfterBindAccelerationStructures) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t maxDescriptorSetAccelerationStructures = )===" + CJITIncludeLoader::to_string(limits.maxDescriptorSetAccelerationStructures) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t maxDescriptorSetUpdateAfterBindAccelerationStructures = )===" + CJITIncludeLoader::to_string(limits.maxDescriptorSetUpdateAfterBindAccelerationStructures) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t minAccelerationStructureScratchOffsetAlignment = )===" + CJITIncludeLoader::to_string(limits.minAccelerationStructureScratchOffsetAlignment) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxRayRecursionDepth = )===" + CJITIncludeLoader::to_string(limits.maxRayRecursionDepth) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxShaderGroupStride = )===" + CJITIncludeLoader::to_string(limits.maxShaderGroupStride) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t shaderGroupBaseAlignment = )===" + CJITIncludeLoader::to_string(limits.shaderGroupBaseAlignment) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxRayDispatchInvocationCount = )===" + CJITIncludeLoader::to_string(limits.maxRayDispatchInvocationCount) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t shaderGroupHandleAlignment = )===" + CJITIncludeLoader::to_string(limits.shaderGroupHandleAlignment) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxRayHitAttributeSize = )===" + CJITIncludeLoader::to_string(limits.maxRayHitAttributeSize) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderSMBuiltins = )===" + CJITIncludeLoader::to_string(limits.shaderSMBuiltins) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool postDepthCoverage = )===" + CJITIncludeLoader::to_string(limits.postDepthCoverage) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool computeDerivativeGroupQuads = )===" + CJITIncludeLoader::to_string(limits.computeDerivativeGroupQuads) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool computeDerivativeGroupLinear = )===" + CJITIncludeLoader::to_string(limits.computeDerivativeGroupLinear) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool imageFootprint = )===" + CJITIncludeLoader::to_string(limits.imageFootprint) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t pciDomain = )===" + CJITIncludeLoader::to_string(limits.pciDomain) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t pciBus = )===" + CJITIncludeLoader::to_string(limits.pciBus) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t pciDevice = )===" + CJITIncludeLoader::to_string(limits.pciDevice) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t pciFunction = )===" + CJITIncludeLoader::to_string(limits.pciFunction) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t minFragmentDensityTexelSizeX = )===" + CJITIncludeLoader::to_string(limits.minFragmentDensityTexelSize.x) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t minFragmentDensityTexelSizeY = )===" + CJITIncludeLoader::to_string(limits.minFragmentDensityTexelSize.y) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxFragmentDensityTexelSizeX = )===" + CJITIncludeLoader::to_string(limits.maxFragmentDensityTexelSize.x) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxFragmentDensityTexelSizeY = )===" + CJITIncludeLoader::to_string(limits.maxFragmentDensityTexelSize.y) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool fragmentDensityInvocations = )===" + CJITIncludeLoader::to_string(limits.fragmentDensityInvocations) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool decorateString = )===" + CJITIncludeLoader::to_string(limits.decorateString) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderImageInt64Atomics = )===" + CJITIncludeLoader::to_string(limits.shaderImageInt64Atomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool sparseImageInt64Atomics = )===" + CJITIncludeLoader::to_string(limits.sparseImageInt64Atomics) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t lineSubPixelPrecisionBits = )===" + CJITIncludeLoader::to_string(limits.lineSubPixelPrecisionBits) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderBufferFloat16Atomics = )===" + CJITIncludeLoader::to_string(limits.shaderBufferFloat16Atomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderBufferFloat16AtomicAdd = )===" + CJITIncludeLoader::to_string(limits.shaderBufferFloat16AtomicAdd) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderBufferFloat16AtomicMinMax = )===" + CJITIncludeLoader::to_string(limits.shaderBufferFloat16AtomicMinMax) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderBufferFloat32AtomicMinMax = )===" + CJITIncludeLoader::to_string(limits.shaderBufferFloat32AtomicMinMax) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderBufferFloat64AtomicMinMax = )===" + CJITIncludeLoader::to_string(limits.shaderBufferFloat64AtomicMinMax) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSharedFloat16Atomics = )===" + CJITIncludeLoader::to_string(limits.shaderSharedFloat16Atomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSharedFloat16AtomicAdd = )===" + CJITIncludeLoader::to_string(limits.shaderSharedFloat16AtomicAdd) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSharedFloat16AtomicMinMax = )===" + CJITIncludeLoader::to_string(limits.shaderSharedFloat16AtomicMinMax) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSharedFloat32AtomicMinMax = )===" + CJITIncludeLoader::to_string(limits.shaderSharedFloat32AtomicMinMax) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSharedFloat64AtomicMinMax = )===" + CJITIncludeLoader::to_string(limits.shaderSharedFloat64AtomicMinMax) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderImageFloat32AtomicMinMax = )===" + CJITIncludeLoader::to_string(limits.shaderImageFloat32AtomicMinMax) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool sparseImageFloat32AtomicMinMax = )===" + CJITIncludeLoader::to_string(limits.sparseImageFloat32AtomicMinMax) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool deviceMemoryReport = )===" + CJITIncludeLoader::to_string(limits.deviceMemoryReport) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderNonSemanticInfo = )===" + CJITIncludeLoader::to_string(limits.shaderNonSemanticInfo) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool fragmentShaderBarycentric = )===" + CJITIncludeLoader::to_string(limits.fragmentShaderBarycentric) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupUniformControlFlow = )===" + CJITIncludeLoader::to_string(limits.shaderSubgroupUniformControlFlow) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool subsampledLoads = )===" + CJITIncludeLoader::to_string(limits.subsampledLoads) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool subsampledCoarseReconstructionEarlyAccess = )===" + CJITIncludeLoader::to_string(limits.subsampledCoarseReconstructionEarlyAccess) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxSubsampledArrayLayers = )===" + CJITIncludeLoader::to_string(limits.maxSubsampledArrayLayers) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetSubsampledSamplers = )===" + CJITIncludeLoader::to_string(limits.maxDescriptorSetSubsampledSamplers) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool workgroupMemoryExplicitLayout = )===" + CJITIncludeLoader::to_string(limits.workgroupMemoryExplicitLayout) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool workgroupMemoryExplicitLayoutScalarBlockLayout = )===" + CJITIncludeLoader::to_string(limits.workgroupMemoryExplicitLayoutScalarBlockLayout) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool workgroupMemoryExplicitLayout8BitAccess = )===" + CJITIncludeLoader::to_string(limits.workgroupMemoryExplicitLayout8BitAccess) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool workgroupMemoryExplicitLayout16BitAccess = )===" + CJITIncludeLoader::to_string(limits.workgroupMemoryExplicitLayout16BitAccess) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool colorWriteEnable = )===" + CJITIncludeLoader::to_string(limits.colorWriteEnable) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE ShaderStage cooperativeMatrixSupportedStages = )===" + CJITIncludeLoader::to_string(limits.cooperativeMatrixSupportedStages) + R"===(;

// Nabla
NBL_CONSTEXPR_STATIC_INLINE uint32_t computeUnits = )===" + CJITIncludeLoader::to_string(limits.computeUnits) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool dispatchBase = )===" + CJITIncludeLoader::to_string(limits.dispatchBase) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool allowCommandBufferQueryCopies = )===" + CJITIncludeLoader::to_string(limits.allowCommandBufferQueryCopies) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxOptimallyResidentWorkgroupInvocations = )===" + CJITIncludeLoader::to_string(limits.maxOptimallyResidentWorkgroupInvocations) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxResidentInvocations = )===" + CJITIncludeLoader::to_string(limits.maxResidentInvocations) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE SpirvVersion spirvVersion = )===" + CJITIncludeLoader::to_string(limits.spirvVersion) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool logicOp = )===" + CJITIncludeLoader::to_string(limits.logicOp) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool vertexPipelineStoresAndAtomics = )===" + CJITIncludeLoader::to_string(limits.vertexPipelineStoresAndAtomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool fragmentStoresAndAtomics = )===" + CJITIncludeLoader::to_string(limits.fragmentStoresAndAtomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderTessellationAndGeometryPointSize = )===" + CJITIncludeLoader::to_string(limits.shaderTessellationAndGeometryPointSize) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderStorageImageMultisample = )===" + CJITIncludeLoader::to_string(limits.shaderStorageImageMultisample) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderStorageImageReadWithoutFormat = )===" + CJITIncludeLoader::to_string(limits.shaderStorageImageReadWithoutFormat) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderStorageImageArrayDynamicIndexing = )===" + CJITIncludeLoader::to_string(limits.shaderStorageImageArrayDynamicIndexing) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderFloat64 = )===" + CJITIncludeLoader::to_string(limits.shaderFloat64) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool variableMultisampleRate = )===" + CJITIncludeLoader::to_string(limits.variableMultisampleRate) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool storagePushConstant16 = )===" + CJITIncludeLoader::to_string(limits.storagePushConstant16) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool storageInputOutput16 = )===" + CJITIncludeLoader::to_string(limits.storageInputOutput16) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool multiviewGeometryShader = )===" + CJITIncludeLoader::to_string(limits.multiviewGeometryShader) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool multiviewTessellationShader = )===" + CJITIncludeLoader::to_string(limits.multiviewTessellationShader) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool drawIndirectCount = )===" + CJITIncludeLoader::to_string(limits.drawIndirectCount) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool storagePushConstant8 = )===" + CJITIncludeLoader::to_string(limits.storagePushConstant8) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderBufferInt64Atomics = )===" + CJITIncludeLoader::to_string(limits.shaderBufferInt64Atomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSharedInt64Atomics = )===" + CJITIncludeLoader::to_string(limits.shaderSharedInt64Atomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderFloat16 = )===" + CJITIncludeLoader::to_string(limits.shaderFloat16) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderInputAttachmentArrayDynamicIndexing = )===" + CJITIncludeLoader::to_string(limits.shaderInputAttachmentArrayDynamicIndexing) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderUniformBufferArrayNonUniformIndexing = )===" + CJITIncludeLoader::to_string(limits.shaderUniformBufferArrayNonUniformIndexing) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderInputAttachmentArrayNonUniformIndexing = )===" + CJITIncludeLoader::to_string(limits.shaderInputAttachmentArrayNonUniformIndexing) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool descriptorBindingUniformBufferUpdateAfterBind = )===" + CJITIncludeLoader::to_string(limits.descriptorBindingUniformBufferUpdateAfterBind) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool samplerFilterMinmax = )===" + CJITIncludeLoader::to_string(limits.samplerFilterMinmax) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool vulkanMemoryModelAvailabilityVisibilityChains = )===" + CJITIncludeLoader::to_string(limits.vulkanMemoryModelAvailabilityVisibilityChains) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderOutputViewportIndex = )===" + CJITIncludeLoader::to_string(limits.shaderOutputViewportIndex) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderOutputLayer = )===" + CJITIncludeLoader::to_string(limits.shaderOutputLayer) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderDemoteToHelperInvocation = )===" + CJITIncludeLoader::to_string(limits.shaderDemoteToHelperInvocation) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderTerminateInvocation = )===" + CJITIncludeLoader::to_string(limits.shaderTerminateInvocation) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderZeroInitializeWorkgroupMemory = )===" + CJITIncludeLoader::to_string(limits.shaderZeroInitializeWorkgroupMemory) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderDeviceClock = )===" + CJITIncludeLoader::to_string(limits.shaderDeviceClock) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupClock = )===" + CJITIncludeLoader::to_string(limits.shaderSubgroupClock) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool imageFootPrint = )===" + CJITIncludeLoader::to_string(limits.imageFootPrint) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderIntegerFunctions2 = )===" + CJITIncludeLoader::to_string(limits.shaderIntegerFunctions2) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderEarlyAndLateFragmentTests = )===" + CJITIncludeLoader::to_string(limits.shaderEarlyAndLateFragmentTests) + R"===(;

// Features JIT Members
// VK 1.0
NBL_CONSTEXPR_STATIC_INLINE bool robustBufferAccess = )===" + CJITIncludeLoader::to_string(features.robustBufferAccess) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool geometryShader = )===" + CJITIncludeLoader::to_string(features.geometryShader) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool tessellationShader = )===" + CJITIncludeLoader::to_string(features.tessellationShader) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool depthBounds = )===" + CJITIncludeLoader::to_string(features.depthBounds) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool wideLines = )===" + CJITIncludeLoader::to_string(features.wideLines) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool largePoints = )===" + CJITIncludeLoader::to_string(features.largePoints) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool alphaToOne = )===" + CJITIncludeLoader::to_string(features.alphaToOne) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool pipelineStatisticsQuery = )===" + CJITIncludeLoader::to_string(features.pipelineStatisticsQuery) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderCullDistance = )===" + CJITIncludeLoader::to_string(features.shaderCullDistance) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderResourceResidency = )===" + CJITIncludeLoader::to_string(features.shaderResourceResidency) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderResourceMinLod = )===" + CJITIncludeLoader::to_string(features.shaderResourceMinLod) + R"===(;

// VK 1.1
// VK 1.2
NBL_CONSTEXPR_STATIC_INLINE bool bufferDeviceAddressMultiDevice = )===" + CJITIncludeLoader::to_string(features.bufferDeviceAddressMultiDevice) + R"===(;

// VK 1.3
NBL_CONSTEXPR_STATIC_INLINE bool robustImageAccess = )===" + CJITIncludeLoader::to_string(features.robustImageAccess) + R"===(;

// Nabla Core Extensions
NBL_CONSTEXPR_STATIC_INLINE bool robustBufferAccess2 = )===" + CJITIncludeLoader::to_string(features.robustBufferAccess2) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool robustImageAccess2 = )===" + CJITIncludeLoader::to_string(features.robustImageAccess2) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool nullDescriptor = )===" + CJITIncludeLoader::to_string(features.nullDescriptor) + R"===(;

// Extensions
NBL_CONSTEXPR_STATIC_INLINE SwapchainMode swapchainMode = )===" + CJITIncludeLoader::to_string(features.swapchainMode) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderInfoAMD = )===" + CJITIncludeLoader::to_string(features.shaderInfoAMD) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool conditionalRendering = )===" + CJITIncludeLoader::to_string(features.conditionalRendering) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool inheritedConditionalRendering = )===" + CJITIncludeLoader::to_string(features.inheritedConditionalRendering) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool geometryShaderPassthrough = )===" + CJITIncludeLoader::to_string(features.geometryShaderPassthrough) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool hdrMetadata = )===" + CJITIncludeLoader::to_string(features.hdrMetadata) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool performanceCounterQueryPools = )===" + CJITIncludeLoader::to_string(features.performanceCounterQueryPools) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool performanceCounterMultipleQueryPools = )===" + CJITIncludeLoader::to_string(features.performanceCounterMultipleQueryPools) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool mixedAttachmentSamples = )===" + CJITIncludeLoader::to_string(features.mixedAttachmentSamples) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool accelerationStructure = )===" + CJITIncludeLoader::to_string(features.accelerationStructure) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool accelerationStructureIndirectBuild = )===" + CJITIncludeLoader::to_string(features.accelerationStructureIndirectBuild) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool accelerationStructureHostCommands = )===" + CJITIncludeLoader::to_string(features.accelerationStructureHostCommands) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool rayTracingPipeline = )===" + CJITIncludeLoader::to_string(features.rayTracingPipeline) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool rayTraversalPrimitiveCulling = )===" + CJITIncludeLoader::to_string(features.rayTraversalPrimitiveCulling) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool rayQuery = )===" + CJITIncludeLoader::to_string(features.rayQuery) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool representativeFragmentTest = )===" + CJITIncludeLoader::to_string(features.representativeFragmentTest) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool bufferMarkerAMD = )===" + CJITIncludeLoader::to_string(features.bufferMarkerAMD) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool fragmentDensityMap = )===" + CJITIncludeLoader::to_string(features.fragmentDensityMap) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool fragmentDensityMapDynamic = )===" + CJITIncludeLoader::to_string(features.fragmentDensityMapDynamic) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool fragmentDensityMapNonSubsampledImages = )===" + CJITIncludeLoader::to_string(features.fragmentDensityMapNonSubsampledImages) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool deviceCoherentMemory = )===" + CJITIncludeLoader::to_string(features.deviceCoherentMemory) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool memoryPriority = )===" + CJITIncludeLoader::to_string(features.memoryPriority) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool fragmentShaderSampleInterlock = )===" + CJITIncludeLoader::to_string(features.fragmentShaderSampleInterlock) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool fragmentShaderPixelInterlock = )===" + CJITIncludeLoader::to_string(features.fragmentShaderPixelInterlock) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool fragmentShaderShadingRateInterlock = )===" + CJITIncludeLoader::to_string(features.fragmentShaderShadingRateInterlock) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool rectangularLines = )===" + CJITIncludeLoader::to_string(features.rectangularLines) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool bresenhamLines = )===" + CJITIncludeLoader::to_string(features.bresenhamLines) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool smoothLines = )===" + CJITIncludeLoader::to_string(features.smoothLines) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool stippledRectangularLines = )===" + CJITIncludeLoader::to_string(features.stippledRectangularLines) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool stippledBresenhamLines = )===" + CJITIncludeLoader::to_string(features.stippledBresenhamLines) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool stippledSmoothLines = )===" + CJITIncludeLoader::to_string(features.stippledSmoothLines) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool indexTypeUint8 = )===" + CJITIncludeLoader::to_string(features.indexTypeUint8) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool deferredHostOperations = )===" + CJITIncludeLoader::to_string(features.deferredHostOperations) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool pipelineExecutableInfo = )===" + CJITIncludeLoader::to_string(features.pipelineExecutableInfo) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool deviceGeneratedCommands = )===" + CJITIncludeLoader::to_string(features.deviceGeneratedCommands) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool rayTracingMotionBlur = )===" + CJITIncludeLoader::to_string(features.rayTracingMotionBlur) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool rayTracingMotionBlurPipelineTraceRaysIndirect = )===" + CJITIncludeLoader::to_string(features.rayTracingMotionBlurPipelineTraceRaysIndirect) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool fragmentDensityMapDeferred = )===" + CJITIncludeLoader::to_string(features.fragmentDensityMapDeferred) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool rasterizationOrderColorAttachmentAccess = )===" + CJITIncludeLoader::to_string(features.rasterizationOrderColorAttachmentAccess) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool rasterizationOrderDepthAttachmentAccess = )===" + CJITIncludeLoader::to_string(features.rasterizationOrderDepthAttachmentAccess) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool rasterizationOrderStencilAttachmentAccess = )===" + CJITIncludeLoader::to_string(features.rasterizationOrderStencilAttachmentAccess) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool cooperativeMatrixRobustBufferAccess = )===" + CJITIncludeLoader::to_string(features.cooperativeMatrixRobustBufferAccess) + R"===(;

// Nabla
)===";
