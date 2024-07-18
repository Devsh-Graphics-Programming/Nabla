std::string jit_traits = R"===(
// Limits JIT Members
// VK 1.0
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxImageDimension1D = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxImageDimension1D) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxImageDimension2D = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxImageDimension2D) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxImageDimension3D = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxImageDimension3D) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxImageDimensionCube = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxImageDimensionCube) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxImageArrayLayers = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxImageArrayLayers) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxBufferViewTexels = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxBufferViewTexels) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxUBOSize = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxUBOSize) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxSSBOSize = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxSSBOSize) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxPushConstantsSize = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxPushConstantsSize) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxMemoryAllocationCount = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxMemoryAllocationCount) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxSamplerAllocationCount = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxSamplerAllocationCount) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t bufferImageGranularity = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.bufferImageGranularity) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorSamplers = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorSamplers) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorUBOs = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorUBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorSSBOs = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorSSBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorImages = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorImages) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorStorageImages = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorStorageImages) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorInputAttachments = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorInputAttachments) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageResources = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxPerStageResources) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetSamplers = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxDescriptorSetSamplers) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetUBOs = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxDescriptorSetUBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetDynamicOffsetUBOs = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxDescriptorSetDynamicOffsetUBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetSSBOs = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxDescriptorSetSSBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetDynamicOffsetSSBOs = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxDescriptorSetDynamicOffsetSSBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetImages = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxDescriptorSetImages) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetStorageImages = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxDescriptorSetStorageImages) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetInputAttachments = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxDescriptorSetInputAttachments) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t maxVertexOutputComponents = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxVertexOutputComponents) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTessellationGenerationLevel = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxTessellationGenerationLevel) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTessellationPatchSize = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxTessellationPatchSize) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTessellationControlPerVertexInputComponents = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxTessellationControlPerVertexInputComponents) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTessellationControlPerVertexOutputComponents = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxTessellationControlPerVertexOutputComponents) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTessellationControlPerPatchOutputComponents = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxTessellationControlPerPatchOutputComponents) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTessellationControlTotalOutputComponents = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxTessellationControlTotalOutputComponents) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTessellationEvaluationInputComponents = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxTessellationEvaluationInputComponents) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTessellationEvaluationOutputComponents = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxTessellationEvaluationOutputComponents) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t maxGeometryShaderInvocations = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxGeometryShaderInvocations) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxGeometryInputComponents = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxGeometryInputComponents) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxGeometryOutputComponents = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxGeometryOutputComponents) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxGeometryOutputVertices = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxGeometryOutputVertices) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxGeometryTotalOutputComponents = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxGeometryTotalOutputComponents) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxFragmentInputComponents = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxFragmentInputComponents) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxFragmentOutputAttachments = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxFragmentOutputAttachments) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxFragmentDualSrcAttachments = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxFragmentDualSrcAttachments) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxFragmentCombinedOutputResources = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxFragmentCombinedOutputResources) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxComputeSharedMemorySize = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxComputeSharedMemorySize) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxComputeWorkGroupCountX = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxComputeWorkGroupCount[0]) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxComputeWorkGroupCountY = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxComputeWorkGroupCount[1]) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxComputeWorkGroupCountZ = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxComputeWorkGroupCount[2]) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxComputeWorkGroupInvocations = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxComputeWorkGroupInvocations) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxWorkgroupSizeX = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxWorkgroupSize[0]) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxWorkgroupSizeY = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxWorkgroupSize[1]) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxWorkgroupSizeZ = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxWorkgroupSize[2]) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t subPixelPrecisionBits = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.subPixelPrecisionBits) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t subTexelPrecisionBits = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.subTexelPrecisionBits) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t mipmapPrecisionBits = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.mipmapPrecisionBits) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDrawIndirectCount = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxDrawIndirectCount) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxSamplerLodBiasBitPattern = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(std::bit_cast<uint32_t>(limits.maxSamplerLodBias)) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxSamplerAnisotropyLog2 = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxSamplerAnisotropyLog2) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t maxViewports = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxViewports) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxViewportDimsX = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxViewportDims[0]) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxViewportDimsY = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxViewportDims[1]) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t viewportBoundsRangeBitPatternMin = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(std::bit_cast<uint32_t>(limits.viewportBoundsRange[0])) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t viewportBoundsRangeBitPatternMax = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(std::bit_cast<uint32_t>(limits.viewportBoundsRange[1])) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t viewportSubPixelBits = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.viewportSubPixelBits) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t minMemoryMapAlignment = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.minMemoryMapAlignment) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t bufferViewAlignment = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.bufferViewAlignment) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t minUBOAlignment = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.minUBOAlignment) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t minSSBOAlignment = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.minSSBOAlignment) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE int16_t minTexelOffset = )===" + std::string("(int16_t)") + CJITIncludeLoader::to_string(limits.minTexelOffset) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTexelOffset = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxTexelOffset) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE int16_t minTexelGatherOffset = )===" + std::string("(int16_t)") + CJITIncludeLoader::to_string(limits.minTexelGatherOffset) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxTexelGatherOffset = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxTexelGatherOffset) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t minInterpolationOffsetBitPattern = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(std::bit_cast<uint32_t>(limits.minInterpolationOffset)) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxInterpolationOffsetBitPattern = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(std::bit_cast<uint32_t>(limits.maxInterpolationOffset)) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t subPixelInterpolationOffsetBits = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.subPixelInterpolationOffsetBits) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxFramebufferWidth = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxFramebufferWidth) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxFramebufferHeight = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxFramebufferHeight) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxFramebufferLayers = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxFramebufferLayers) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t maxColorAttachments = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxColorAttachments) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t maxSampleMaskWords = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxSampleMaskWords) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t timestampPeriodInNanoSecondsBitPattern = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(std::bit_cast<uint32_t>(limits.timestampPeriodInNanoSeconds)) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t maxClipDistances = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxClipDistances) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxCullDistances = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxCullDistances) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxCombinedClipAndCullDistances = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxCombinedClipAndCullDistances) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t discreteQueuePriorities = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.discreteQueuePriorities) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t pointSizeRangeBitPatternMin = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(std::bit_cast<uint32_t>(limits.pointSizeRange[0])) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t pointSizeRangeBitPatternMax = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(std::bit_cast<uint32_t>(limits.pointSizeRange[1])) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t lineWidthRangeBitPatternMin = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(std::bit_cast<uint32_t>(limits.lineWidthRange[0])) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t lineWidthRangeBitPatternMax = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(std::bit_cast<uint32_t>(limits.lineWidthRange[1])) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t pointSizeGranularityBitPattern = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(std::bit_cast<uint32_t>(limits.pointSizeGranularity)) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t lineWidthGranularityBitPattern = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(std::bit_cast<uint32_t>(limits.lineWidthGranularity)) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool strictLines = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.strictLines) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool standardSampleLocations = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.standardSampleLocations) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t optimalBufferCopyOffsetAlignment = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.optimalBufferCopyOffsetAlignment) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t optimalBufferCopyRowPitchAlignment = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.optimalBufferCopyRowPitchAlignment) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t nonCoherentAtomSize = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.nonCoherentAtomSize) + R"===(;

// VK 1.1
NBL_CONSTEXPR_STATIC_INLINE uint16_t subgroupSize = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.subgroupSize) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t subgroupOpsShaderStagesBitPattern = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.subgroupOpsShaderStages) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupClustered = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderSubgroupClustered) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupArithmetic = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderSubgroupArithmetic) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupQuad = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderSubgroupQuad) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupQuadAllStages = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderSubgroupQuadAllStages) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint64_t pointClippingBehaviorBitPattern = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.pointClippingBehavior) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint16_t maxMultiviewViewCount = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxMultiviewViewCount) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxMultiviewInstanceIndex = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxMultiviewInstanceIndex) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerSetDescriptors = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxPerSetDescriptors) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t maxMemoryAllocationSize = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.maxMemoryAllocationSize) + R"===(;

// VK 1.2
NBL_CONSTEXPR_STATIC_INLINE bool shaderSignedZeroInfNanPreserveFloat64 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderSignedZeroInfNanPreserveFloat64) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderDenormPreserveFloat16 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderDenormPreserveFloat16) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderDenormPreserveFloat32 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderDenormPreserveFloat32) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderDenormPreserveFloat64 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderDenormPreserveFloat64) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderDenormFlushToZeroFloat16 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderDenormFlushToZeroFloat16) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderDenormFlushToZeroFloat32 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderDenormFlushToZeroFloat32) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderDenormFlushToZeroFloat64 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderDenormFlushToZeroFloat64) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderRoundingModeRTEFloat16 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderRoundingModeRTEFloat16) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderRoundingModeRTEFloat32 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderRoundingModeRTEFloat32) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderRoundingModeRTEFloat64 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderRoundingModeRTEFloat64) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderRoundingModeRTZFloat16 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderRoundingModeRTZFloat16) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderRoundingModeRTZFloat32 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderRoundingModeRTZFloat32) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderRoundingModeRTZFloat64 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderRoundingModeRTZFloat64) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxUpdateAfterBindDescriptorsInAllPools = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxUpdateAfterBindDescriptorsInAllPools) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderUniformBufferArrayNonUniformIndexingNative = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderUniformBufferArrayNonUniformIndexingNative) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSampledImageArrayNonUniformIndexingNative = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderSampledImageArrayNonUniformIndexingNative) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderStorageBufferArrayNonUniformIndexingNative = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderStorageBufferArrayNonUniformIndexingNative) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderStorageImageArrayNonUniformIndexingNative = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderStorageImageArrayNonUniformIndexingNative) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderInputAttachmentArrayNonUniformIndexingNative = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderInputAttachmentArrayNonUniformIndexingNative) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool robustBufferAccessUpdateAfterBind = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.robustBufferAccessUpdateAfterBind) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool quadDivergentImplicitLod = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.quadDivergentImplicitLod) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorUpdateAfterBindSamplers = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorUpdateAfterBindSamplers) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorUpdateAfterBindUBOs = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorUpdateAfterBindUBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorUpdateAfterBindSSBOs = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorUpdateAfterBindSSBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorUpdateAfterBindImages = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorUpdateAfterBindImages) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorUpdateAfterBindStorageImages = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorUpdateAfterBindStorageImages) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageDescriptorUpdateAfterBindInputAttachments = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorUpdateAfterBindInputAttachments) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxPerStageUpdateAfterBindResources = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxPerStageUpdateAfterBindResources) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetUpdateAfterBindSamplers = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxDescriptorSetUpdateAfterBindSamplers) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetUpdateAfterBindUBOs = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxDescriptorSetUpdateAfterBindUBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetUpdateAfterBindDynamicOffsetUBOs = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxDescriptorSetUpdateAfterBindDynamicOffsetUBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetUpdateAfterBindSSBOs = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxDescriptorSetUpdateAfterBindSSBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetUpdateAfterBindDynamicOffsetSSBOs = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxDescriptorSetUpdateAfterBindDynamicOffsetSSBOs) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetUpdateAfterBindImages = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxDescriptorSetUpdateAfterBindImages) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetUpdateAfterBindStorageImages = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxDescriptorSetUpdateAfterBindStorageImages) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetUpdateAfterBindInputAttachments = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxDescriptorSetUpdateAfterBindInputAttachments) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint64_t supportedDepthResolveModesBitPattern = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.supportedDepthResolveModes) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t supportedStencilResolveModesBitPattern = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.supportedStencilResolveModes) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool independentResolveNone = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.independentResolveNone) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool independentResolve = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.independentResolve) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool filterMinmaxImageComponentMapping = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.filterMinmaxImageComponentMapping) + R"===(;

// VK 1.3
NBL_CONSTEXPR_STATIC_INLINE uint16_t minSubgroupSize = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.minSubgroupSize) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t maxSubgroupSize = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.maxSubgroupSize) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxComputeWorkgroupSubgroups = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxComputeWorkgroupSubgroups) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t requiredSubgroupSizeStagesBitPattern = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.requiredSubgroupSizeStages) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct8BitUnsignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProduct8BitUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct8BitSignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProduct8BitSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct8BitMixedSignednessAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProduct8BitMixedSignednessAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct4x8BitPackedUnsignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProduct4x8BitPackedUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct4x8BitPackedSignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProduct4x8BitPackedSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct4x8BitPackedMixedSignednessAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProduct4x8BitPackedMixedSignednessAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct16BitUnsignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProduct16BitUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct16BitSignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProduct16BitSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct16BitMixedSignednessAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProduct16BitMixedSignednessAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct32BitUnsignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProduct32BitUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct32BitSignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProduct32BitSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct32BitMixedSignednessAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProduct32BitMixedSignednessAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct64BitUnsignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProduct64BitUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct64BitSignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProduct64BitSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProduct64BitMixedSignednessAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProduct64BitMixedSignednessAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating8BitUnsignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating8BitUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating8BitSignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating8BitSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating16BitUnsignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating16BitUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating16BitSignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating16BitSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating32BitUnsignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating32BitUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating32BitSignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating32BitSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating64BitUnsignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating64BitUnsignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating64BitSignedAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating64BitSignedAccelerated) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint64_t storageTexelBufferOffsetAlignmentBytes = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.storageTexelBufferOffsetAlignmentBytes) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t uniformTexelBufferOffsetAlignmentBytes = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.uniformTexelBufferOffsetAlignmentBytes) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint64_t maxBufferSize = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.maxBufferSize) + R"===(;

// Nabla Core Extensions
NBL_CONSTEXPR_STATIC_INLINE uint32_t minImportedHostPointerAlignment = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.minImportedHostPointerAlignment) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderBufferFloat32AtomicAdd = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderBufferFloat32AtomicAdd) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderBufferFloat64Atomics = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderBufferFloat64Atomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderBufferFloat64AtomicAdd = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderBufferFloat64AtomicAdd) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSharedFloat32AtomicAdd = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderSharedFloat32AtomicAdd) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSharedFloat64Atomics = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderSharedFloat64Atomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSharedFloat64AtomicAdd = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderSharedFloat64AtomicAdd) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderImageFloat32AtomicAdd = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderImageFloat32AtomicAdd) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool sparseImageFloat32Atomics = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.sparseImageFloat32Atomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool sparseImageFloat32AtomicAdd = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.sparseImageFloat32AtomicAdd) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint64_t robustStorageBufferAccessSizeAlignment = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.robustStorageBufferAccessSizeAlignment) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t robustUniformBufferAccessSizeAlignment = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.robustUniformBufferAccessSizeAlignment) + R"===(;

// Extensions
NBL_CONSTEXPR_STATIC_INLINE bool shaderTrinaryMinmax = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderTrinaryMinmax) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderExplicitVertexParameter = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderExplicitVertexParameter) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool gpuShaderHalfFloatAMD = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.gpuShaderHalfFloatAMD) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderImageLoadStoreLod = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderImageLoadStoreLod) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool displayTiming = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.displayTiming) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDiscardRectangles = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxDiscardRectangles) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t primitiveOverestimationSizeBitPattern = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(std::bit_cast<uint32_t>(limits.primitiveOverestimationSize)) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxExtraPrimitiveOverestimationSizeBitPattern = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(std::bit_cast<uint32_t>(limits.maxExtraPrimitiveOverestimationSize)) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t extraPrimitiveOverestimationSizeGranularityBitPattern = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(std::bit_cast<uint32_t>(limits.extraPrimitiveOverestimationSizeGranularity)) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool primitiveUnderestimation = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.primitiveUnderestimation) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool conservativePointAndLineRasterization = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.conservativePointAndLineRasterization) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool degenerateTrianglesRasterized = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.degenerateTrianglesRasterized) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool degenerateLinesRasterized = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.degenerateLinesRasterized) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool fullyCoveredFragmentShaderInputVariable = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.fullyCoveredFragmentShaderInputVariable) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool conservativeRasterizationPostDepthCoverage = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.conservativeRasterizationPostDepthCoverage) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool queueFamilyForeign = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.queueFamilyForeign) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderStencilExport = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderStencilExport) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool variableSampleLocations = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.variableSampleLocations) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint16_t sampleLocationSubPixelBits = )===" + std::string("(uint16_t)") + CJITIncludeLoader::to_string(limits.sampleLocationSubPixelBits) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t sampleLocationSampleCountsBitPattern = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.sampleLocationSampleCounts) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxSampleLocationGridSizeX = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxSampleLocationGridSize.x) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxSampleLocationGridSizeY = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxSampleLocationGridSize.y) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t sampleLocationCoordinateRangeBitPatternMin = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(std::bit_cast<uint32_t>(limits.sampleLocationCoordinateRange[0])) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t sampleLocationCoordinateRangeBitPatternMax = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(std::bit_cast<uint32_t>(limits.sampleLocationCoordinateRange[1])) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint64_t maxAccelerationStructureGeometryCount = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.maxAccelerationStructureGeometryCount) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t maxAccelerationStructureInstanceCount = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.maxAccelerationStructureInstanceCount) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t maxAccelerationStructurePrimitiveCount = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.maxAccelerationStructurePrimitiveCount) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t maxPerStageDescriptorAccelerationStructures = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorAccelerationStructures) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t maxPerStageDescriptorUpdateAfterBindAccelerationStructures = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.maxPerStageDescriptorUpdateAfterBindAccelerationStructures) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t maxDescriptorSetAccelerationStructures = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.maxDescriptorSetAccelerationStructures) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t maxDescriptorSetUpdateAfterBindAccelerationStructures = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.maxDescriptorSetUpdateAfterBindAccelerationStructures) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t minAccelerationStructureScratchOffsetAlignment = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.minAccelerationStructureScratchOffsetAlignment) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t maxRayRecursionDepth = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxRayRecursionDepth) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxShaderGroupStride = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxShaderGroupStride) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t shaderGroupBaseAlignment = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.shaderGroupBaseAlignment) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxRayDispatchInvocationCount = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxRayDispatchInvocationCount) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t shaderGroupHandleAlignment = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.shaderGroupHandleAlignment) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxRayHitAttributeSize = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxRayHitAttributeSize) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderSMBuiltins = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderSMBuiltins) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool postDepthCoverage = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.postDepthCoverage) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool computeDerivativeGroupQuads = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.computeDerivativeGroupQuads) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool computeDerivativeGroupLinear = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.computeDerivativeGroupLinear) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t pciDomain = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.pciDomain) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t pciBus = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.pciBus) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t pciDevice = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.pciDevice) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t pciFunction = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.pciFunction) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t minFragmentDensityTexelSizeX = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.minFragmentDensityTexelSize.x) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t minFragmentDensityTexelSizeY = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.minFragmentDensityTexelSize.y) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxFragmentDensityTexelSizeX = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxFragmentDensityTexelSize.x) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxFragmentDensityTexelSizeY = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxFragmentDensityTexelSize.y) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool fragmentDensityInvocations = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.fragmentDensityInvocations) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool decorateString = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.decorateString) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderImageInt64Atomics = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderImageInt64Atomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool sparseImageInt64Atomics = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.sparseImageInt64Atomics) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint32_t lineSubPixelPrecisionBits = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.lineSubPixelPrecisionBits) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderBufferFloat16Atomics = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderBufferFloat16Atomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderBufferFloat16AtomicAdd = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderBufferFloat16AtomicAdd) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderBufferFloat16AtomicMinMax = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderBufferFloat16AtomicMinMax) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderBufferFloat32AtomicMinMax = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderBufferFloat32AtomicMinMax) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderBufferFloat64AtomicMinMax = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderBufferFloat64AtomicMinMax) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSharedFloat16Atomics = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderSharedFloat16Atomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSharedFloat16AtomicAdd = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderSharedFloat16AtomicAdd) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSharedFloat16AtomicMinMax = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderSharedFloat16AtomicMinMax) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSharedFloat32AtomicMinMax = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderSharedFloat32AtomicMinMax) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSharedFloat64AtomicMinMax = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderSharedFloat64AtomicMinMax) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderImageFloat32AtomicMinMax = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderImageFloat32AtomicMinMax) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool sparseImageFloat32AtomicMinMax = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.sparseImageFloat32AtomicMinMax) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool deviceMemoryReport = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.deviceMemoryReport) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderNonSemanticInfo = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderNonSemanticInfo) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool fragmentShaderBarycentric = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.fragmentShaderBarycentric) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupUniformControlFlow = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderSubgroupUniformControlFlow) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool subsampledLoads = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.subsampledLoads) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool subsampledCoarseReconstructionEarlyAccess = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.subsampledCoarseReconstructionEarlyAccess) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxSubsampledArrayLayers = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxSubsampledArrayLayers) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxDescriptorSetSubsampledSamplers = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxDescriptorSetSubsampledSamplers) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool workgroupMemoryExplicitLayout = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.workgroupMemoryExplicitLayout) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool workgroupMemoryExplicitLayoutScalarBlockLayout = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.workgroupMemoryExplicitLayoutScalarBlockLayout) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool workgroupMemoryExplicitLayout8BitAccess = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.workgroupMemoryExplicitLayout8BitAccess) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool workgroupMemoryExplicitLayout16BitAccess = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.workgroupMemoryExplicitLayout16BitAccess) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool colorWriteEnable = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.colorWriteEnable) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE uint64_t cooperativeMatrixSupportedStagesBitPattern = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.cooperativeMatrixSupportedStages) + R"===(;

// Nabla
NBL_CONSTEXPR_STATIC_INLINE uint32_t computeUnits = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.computeUnits) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool dispatchBase = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.dispatchBase) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool allowCommandBufferQueryCopies = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.allowCommandBufferQueryCopies) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxOptimallyResidentWorkgroupInvocations = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxOptimallyResidentWorkgroupInvocations) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint32_t maxResidentInvocations = )===" + std::string("(uint32_t)") + CJITIncludeLoader::to_string(limits.maxResidentInvocations) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE uint64_t spirvVersionBitPattern = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(limits.spirvVersion) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool logicOp = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.logicOp) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool vertexPipelineStoresAndAtomics = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.vertexPipelineStoresAndAtomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool fragmentStoresAndAtomics = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.fragmentStoresAndAtomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderTessellationAndGeometryPointSize = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderTessellationAndGeometryPointSize) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderStorageImageMultisample = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderStorageImageMultisample) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderStorageImageReadWithoutFormat = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderStorageImageReadWithoutFormat) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderStorageImageArrayDynamicIndexing = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderStorageImageArrayDynamicIndexing) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderFloat64 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderFloat64) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool variableMultisampleRate = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.variableMultisampleRate) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool storagePushConstant16 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.storagePushConstant16) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool storageInputOutput16 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.storageInputOutput16) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool multiviewGeometryShader = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.multiviewGeometryShader) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool multiviewTessellationShader = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.multiviewTessellationShader) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool drawIndirectCount = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.drawIndirectCount) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool storagePushConstant8 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.storagePushConstant8) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderBufferInt64Atomics = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderBufferInt64Atomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSharedInt64Atomics = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderSharedInt64Atomics) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderFloat16 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderFloat16) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderInputAttachmentArrayDynamicIndexing = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderInputAttachmentArrayDynamicIndexing) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderUniformBufferArrayNonUniformIndexing = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderUniformBufferArrayNonUniformIndexing) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderInputAttachmentArrayNonUniformIndexing = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderInputAttachmentArrayNonUniformIndexing) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool descriptorBindingUniformBufferUpdateAfterBind = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.descriptorBindingUniformBufferUpdateAfterBind) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool samplerFilterMinmax = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.samplerFilterMinmax) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool vulkanMemoryModelAvailabilityVisibilityChains = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.vulkanMemoryModelAvailabilityVisibilityChains) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderOutputViewportIndex = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderOutputViewportIndex) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderOutputLayer = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderOutputLayer) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderDemoteToHelperInvocation = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderDemoteToHelperInvocation) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderTerminateInvocation = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderTerminateInvocation) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderZeroInitializeWorkgroupMemory = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderZeroInitializeWorkgroupMemory) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderDeviceClock = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderDeviceClock) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupClock = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderSubgroupClock) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool imageFootprint = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.imageFootprint) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderIntegerFunctions2 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderIntegerFunctions2) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderEarlyAndLateFragmentTests = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(limits.shaderEarlyAndLateFragmentTests) + R"===(;

// Features JIT Members
// VK 1.0
NBL_CONSTEXPR_STATIC_INLINE bool robustBufferAccess = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.robustBufferAccess) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool geometryShader = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.geometryShader) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool tessellationShader = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.tessellationShader) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool depthBounds = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.depthBounds) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool wideLines = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.wideLines) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool largePoints = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.largePoints) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool alphaToOne = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.alphaToOne) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool pipelineStatisticsQuery = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.pipelineStatisticsQuery) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderCullDistance = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.shaderCullDistance) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderResourceResidency = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.shaderResourceResidency) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool shaderResourceMinLod = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.shaderResourceMinLod) + R"===(;

// VK 1.1
// VK 1.2
NBL_CONSTEXPR_STATIC_INLINE bool bufferDeviceAddressMultiDevice = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.bufferDeviceAddressMultiDevice) + R"===(;

// VK 1.3
NBL_CONSTEXPR_STATIC_INLINE bool robustImageAccess = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.robustImageAccess) + R"===(;

// Nabla Core Extensions
NBL_CONSTEXPR_STATIC_INLINE bool robustBufferAccess2 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.robustBufferAccess2) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool robustImageAccess2 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.robustImageAccess2) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool nullDescriptor = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.nullDescriptor) + R"===(;

// Extensions
NBL_CONSTEXPR_STATIC_INLINE uint64_t swapchainModeBitPattern = )===" + std::string("(uint64_t)") + CJITIncludeLoader::to_string(features.swapchainMode) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool shaderInfoAMD = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.shaderInfoAMD) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool conditionalRendering = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.conditionalRendering) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool inheritedConditionalRendering = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.inheritedConditionalRendering) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool geometryShaderPassthrough = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.geometryShaderPassthrough) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool hdrMetadata = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.hdrMetadata) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool performanceCounterQueryPools = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.performanceCounterQueryPools) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool performanceCounterMultipleQueryPools = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.performanceCounterMultipleQueryPools) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool mixedAttachmentSamples = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.mixedAttachmentSamples) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool accelerationStructure = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.accelerationStructure) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool accelerationStructureIndirectBuild = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.accelerationStructureIndirectBuild) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool accelerationStructureHostCommands = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.accelerationStructureHostCommands) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool rayTracingPipeline = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.rayTracingPipeline) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool rayTraversalPrimitiveCulling = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.rayTraversalPrimitiveCulling) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool rayQuery = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.rayQuery) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool representativeFragmentTest = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.representativeFragmentTest) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool bufferMarkerAMD = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.bufferMarkerAMD) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool fragmentDensityMap = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.fragmentDensityMap) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool fragmentDensityMapDynamic = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.fragmentDensityMapDynamic) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool fragmentDensityMapNonSubsampledImages = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.fragmentDensityMapNonSubsampledImages) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool deviceCoherentMemory = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.deviceCoherentMemory) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool memoryPriority = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.memoryPriority) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool fragmentShaderSampleInterlock = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.fragmentShaderSampleInterlock) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool fragmentShaderPixelInterlock = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.fragmentShaderPixelInterlock) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool fragmentShaderShadingRateInterlock = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.fragmentShaderShadingRateInterlock) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool rectangularLines = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.rectangularLines) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool bresenhamLines = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.bresenhamLines) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool smoothLines = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.smoothLines) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool stippledRectangularLines = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.stippledRectangularLines) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool stippledBresenhamLines = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.stippledBresenhamLines) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool stippledSmoothLines = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.stippledSmoothLines) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool indexTypeUint8 = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.indexTypeUint8) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool deferredHostOperations = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.deferredHostOperations) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool pipelineExecutableInfo = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.pipelineExecutableInfo) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool deviceGeneratedCommands = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.deviceGeneratedCommands) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool rayTracingMotionBlur = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.rayTracingMotionBlur) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool rayTracingMotionBlurPipelineTraceRaysIndirect = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.rayTracingMotionBlurPipelineTraceRaysIndirect) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool fragmentDensityMapDeferred = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.fragmentDensityMapDeferred) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool rasterizationOrderColorAttachmentAccess = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.rasterizationOrderColorAttachmentAccess) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool rasterizationOrderDepthAttachmentAccess = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.rasterizationOrderDepthAttachmentAccess) + R"===(;
NBL_CONSTEXPR_STATIC_INLINE bool rasterizationOrderStencilAttachmentAccess = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.rasterizationOrderStencilAttachmentAccess) + R"===(;

NBL_CONSTEXPR_STATIC_INLINE bool cooperativeMatrixRobustBufferAccess = )===" + std::string("(bool)") + CJITIncludeLoader::to_string(features.cooperativeMatrixRobustBufferAccess) + R"===(;

// Nabla
)===";
