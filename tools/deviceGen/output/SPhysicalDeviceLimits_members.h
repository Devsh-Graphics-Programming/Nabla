    constexpr static inline uint32_t MinMaxImageDimension2D = 16384;
    constexpr static inline uint32_t MinMaxSSBOSize = 1073741820;
    constexpr static inline uint16_t MaxMaxPushConstantsSize = 256;
    constexpr static inline uint32_t MinMaxWorkgroupCount = 65535;
    constexpr static inline uint32_t MinMaxWorkgroupInvocations = 256;
    constexpr static inline int32_t MinSubPixelInterpolationOffsetBits = 4;
    // constexpr static inline core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> NoMSor4Samples = asset::IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT|asset::IImage::E_SAMPLE_COUNT_FLAGS::ESCF_5_BIT;
    constexpr static inline uint8_t MinMaxColorAttachments = 8; // ROADMAP 2024 and wide reports
    // constexpr static inline uint32_t MinInlineUniformBlockSize = 256;
    /*
       Vulkan 1.0 Core
    */
    uint32_t maxImageDimension1D = MinMaxImageDimension2D;
    uint32_t maxImageDimension2D = MinMaxImageDimension2D;
    uint32_t maxImageDimension3D = 2048;
    uint32_t maxImageDimensionCube = MinMaxImageDimension2D;
    uint32_t maxImageArrayLayers = 2048;
    uint32_t maxBufferViewTexels = 33554432;
    uint32_t maxUBOSize = 65536;
    uint32_t maxSSBOSize = MinMaxSSBOSize;
    uint16_t maxPushConstantsSize = 128;
    uint32_t maxMemoryAllocationCount = 4096;
    uint32_t maxSamplerAllocationCount = 4000;
    uint32_t bufferImageGranularity = 65536; // granularity, in bytes, at which buffer or linear image resources, and optimal image resources can be bound to adjacent offsets in the same allocation
    // uint64_t sparseAddressSpaceSize = 0; // [TODO LATER] when we support sparse
    // uint32_t maxBoundDescriptorSets = 4; // [DO NOT EXPOSE] we've kinda hardcoded the engine to 4 currently
    uint32_t maxPerStageDescriptorSamplers = 16; // Descriptors with a type of IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER count against this limit
    uint32_t maxPerStageDescriptorUBOs = 15;
    uint32_t maxPerStageDescriptorSSBOs = 31;
    uint32_t maxPerStageDescriptorImages = 96; // Descriptors with a type of IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER, IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER count against this limit.
    uint32_t maxPerStageDescriptorStorageImages = 8;
    uint32_t maxPerStageDescriptorInputAttachments = 7;
    uint32_t maxPerStageResources = 127;
    uint32_t maxDescriptorSetSamplers = 80; // Descriptors with a type of IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER count against this limit
    uint32_t maxDescriptorSetUBOs = 90;
    uint32_t maxDescriptorSetDynamicOffsetUBOs = 8;
    uint32_t maxDescriptorSetSSBOs = 155;
    uint32_t maxDescriptorSetDynamicOffsetSSBOs = 8;
    uint32_t maxDescriptorSetImages = 480; // Descriptors with a type of IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER, IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER count against this limit.
    uint32_t maxDescriptorSetStorageImages = 40;
    uint32_t maxDescriptorSetInputAttachments = 7;
    // uint8_t maxVertexInputAttributes = 16;
    // uint8_t maxVertexInputBindings = 16;
    // uint16_t maxVertexInputAttributeOffset = maxVertexInputBindingStride-1;
    // uint16_t maxVertexInputBindingStride = 2048;
    uint16_t maxVertexOutputComponents = 124;
    uint16_t maxTessellationGenerationLevel = 0;
    uint16_t maxTessellationPatchSize = 0;
    uint16_t maxTessellationControlPerVertexInputComponents = 0;
    uint16_t maxTessellationControlPerVertexOutputComponents = 0;
    uint16_t maxTessellationControlPerPatchOutputComponents = 0;
    uint16_t maxTessellationControlTotalOutputComponents = 0;
    uint16_t maxTessellationEvaluationInputComponents = 0;
    uint16_t maxTessellationEvaluationOutputComponents = 0;
    uint16_t maxGeometryShaderInvocations = 0;
    uint16_t maxGeometryInputComponents = 0;
    uint16_t maxGeometryOutputComponents = 0;
    uint16_t maxGeometryOutputVertices = 0;
    uint16_t maxGeometryTotalOutputComponents = 0;
    uint32_t maxFragmentInputComponents = 116;
    uint32_t maxFragmentOutputAttachments = 8;
    uint32_t maxFragmentDualSrcAttachments = 1;
    uint32_t maxFragmentCombinedOutputResources = 16;
    uint32_t maxComputeSharedMemorySize = 32768;
    uint32_t maxComputeWorkGroupCount[3] = {MinMaxWorkgroupCount,MinMaxWorkgroupCount,MinMaxWorkgroupCount};
    uint16_t maxComputeWorkGroupInvocations = MinMaxWorkgroupInvocations;
    uint16_t maxWorkgroupSize[3] = {MinMaxWorkgroupInvocations,MinMaxWorkgroupInvocations,64u};
    uint8_t subPixelPrecisionBits = 4;
    uint8_t subTexelPrecisionBits = 4;
    uint8_t mipmapPrecisionBits = 4;
    // uint32_t maxDrawIndexedIndexValue; // [DO NOT EXPOSE] ROADMAP2022: requires fullDrawIndexUint33 so this must be 1xffFFffFFu
    uint32_t maxDrawIndirectCount = 1073741824; // This is different to `maxDrawIndirectCount`, this is NOT about whether you can source the MDI count from a buffer, just about how many you can have
    float maxSamplerLodBias = 4;
    uint8_t maxSamplerAnisotropyLog2 = 4;
    uint8_t maxViewports = 16;
    uint16_t maxViewportDims[2] = {MinMaxImageDimension2D,MinMaxImageDimension2D};
    float viewportBoundsRange[2] = { -MinMaxImageDimension2D*2u, MinMaxImageDimension2D*2u-1 };
    uint32_t viewportSubPixelBits = 0;
    uint16_t minMemoryMapAlignment = 64;
    uint16_t bufferViewAlignment = 64;
    uint16_t minUBOAlignment = 256;
    uint16_t minSSBOAlignment = 64;
    int8_t minTexelOffset = -8;
    uint8_t maxTexelOffset = 7;
    int8_t minTexelGatherOffset = -8;
    uint8_t maxTexelGatherOffset = 7;
    float minInterpolationOffset = -0.5;
    float maxInterpolationOffset = 0.437500000;
    uint8_t subPixelInterpolationOffsetBits = MinSubPixelInterpolationOffsetBits;
    uint32_t maxFramebufferWidth = MinMaxImageDimension2D;
    uint32_t maxFramebufferHeight = MinMaxImageDimension2D;
    uint32_t maxFramebufferLayers = 1024;
    /*
       - Spec states minimum supported value should be at least ESCF_1_BIT
       - it might be different for each integer format, best way is to query your integer format from physical device using vkGetPhysicalDeviceImageFormatProperties and get the sampleCounts
       https://www.khronos.org/registry/vulkan/specs/1.4-extensions/man/html/VkImageFormatProperties.html
       [DO NOT EXPOSE] because it might be different for every texture format and usage
    */
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferColorSampleCounts = NoMSor4Samples;
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferDepthSampleCounts = NoMSor4Samples;
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferStencilSampleCounts = NoMSor4Samples;
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferNoAttachmentsSampleCounts = NoMSor4Samples;
    uint8_t maxColorAttachments = MinMaxColorAttachments;
    /*
       [DO NOT EXPOSE] because it might be different for every texture format and usage
    */
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageColorSampleCounts = NoMSor4Samples;
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageIntegerSampleCounts = NoMSor4Samples;
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageDepthSampleCounts = NoMSor4Samples;
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampledImageStencilSampleCounts = NoMSor4Samples;
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> storageImageSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;
    uint8_t maxSampleMaskWords = 1;
    /*
       [REQUIRE] ROADMAP 2024 and good device support
    */
    // bool timestampComputeAndGraphics = true;
    float timestampPeriodInNanoSeconds = 83.334; // timestampPeriod is the number of nanoseconds required for a timestamp query to be incremented by 1 (a float because vulkan reports), use core::rational in the future
    uint8_t maxClipDistances = 8;
    uint8_t maxCullDistances = 0;
    uint8_t maxCombinedClipAndCullDistances = 8;
    uint32_t discreteQueuePriorities = 2;
    float pointSizeRange[2] = {1.f,64.f};
    float lineWidthRange[2] = {1.f,1.f};
    float pointSizeGranularity = 1;
    float lineWidthGranularity = 1;
    bool strictLines = false; // old intels can't do this
    bool standardSampleLocations = false; // Had to roll back from requiring, ROADMAP 2022 but some of our targets missing
    uint16_t optimalBufferCopyOffsetAlignment = 256;
    uint16_t optimalBufferCopyRowPitchAlignment = 128;
    uint16_t nonCoherentAtomSize = 256;
    /*
       TODO: later
       VkPhysicalDeviceSparseProperties
    */
    // bool residencyStandard2DBlockShape = true;
    // bool residencyStandard2DMultisampleBlockShape = false;
    // bool residencyStandard3DBlockShape = true;
    // bool residencyAlignedMipSize = false;
    // bool residencyNonResidentStrict = true;
    /*
       Vulkan 1.1 Core
    */
    uint16_t subgroupSize = 4;
    core::bitflag<asset::IShader::E_SHADER_STAGE> subgroupOpsShaderStages = asset::IShader::ESS_COMPUTE | asset::IShader::ESS_ALL_GRAPHICS;
    bool shaderSubgroupClustered = false; // ROADMAP2022 mandates all but clustered and quad-all-stages, however all GPU's that we care about support basic, vote, ballot, shuffle and relative so not listing!
    bool shaderSubgroupArithmetic = false; // candidates for promotion
    bool shaderSubgroupQuad = false;
    bool shaderSubgroupQuadAllStages = false; // bad Android support
    E_POINT_CLIPPING_BEHAVIOR pointClippingBehavior = EPCB_USER_CLIP_PLANES_ONLY;
    uint8_t maxMultiviewViewCount = 6;
    uint32_t maxMultiviewInstanceIndex = 134217727;
    // bool protectedNoFault = false;
    uint32_t maxPerSetDescriptors = 572;
    uint64_t maxMemoryAllocationSize = MinMaxSSBOSize;
    /*
       Vulkan 1.2 Core
    */
    // VkShaderFloatControlsIndependence denormBehaviorIndependence; // TODO: need to implement ways to set them
    // VkShaderFloatControlsIndependence roundingModeIndependence; // TODO: need to implement ways to set them
    // bool shaderSignedZeroInfNanPreserveFloat16 = true;
    // bool shaderSignedZeroInfNanPreserveFloat32 = true;
    bool shaderSignedZeroInfNanPreserveFloat64 = false;
    bool shaderDenormPreserveFloat16 = false;
    bool shaderDenormPreserveFloat32 = false;
    bool shaderDenormPreserveFloat64 = false;
    bool shaderDenormFlushToZeroFloat16 = false;
    bool shaderDenormFlushToZeroFloat32 = false;
    bool shaderDenormFlushToZeroFloat64 = false;
    bool shaderRoundingModeRTEFloat16 = false; // ROADMAP2024 but no good support yet
    bool shaderRoundingModeRTEFloat32 = false; // ROADMAP2024 but no good support yet
    bool shaderRoundingModeRTEFloat64 = false;
    bool shaderRoundingModeRTZFloat16 = false;
    bool shaderRoundingModeRTZFloat32 = false;
    bool shaderRoundingModeRTZFloat64 = false;
    /*
       expose in 2 phases
       -Update After Bindand nonUniformEXT shader qualifier:
       Descriptor Lifetime Tracking PR #345 will do this, cause I don't want to rewrite the tracking system again.
       -Actual Descriptor Indexing:
       The whole 512k descriptor limits, runtime desc arrays, etc.will come later
    */
    uint32_t maxUpdateAfterBindDescriptorsInAllPools = 1048576;
    bool shaderUniformBufferArrayNonUniformIndexingNative = false;
    bool shaderSampledImageArrayNonUniformIndexingNative = false; // promotion candidate
    bool shaderStorageBufferArrayNonUniformIndexingNative = false;
    bool shaderStorageImageArrayNonUniformIndexingNative = false; // promotion candidate
    bool shaderInputAttachmentArrayNonUniformIndexingNative = false; // promotion candidate
    bool robustBufferAccessUpdateAfterBind = false;
    bool quadDivergentImplicitLod = false;
    uint32_t maxPerStageDescriptorUpdateAfterBindSamplers = 500000;
    uint32_t maxPerStageDescriptorUpdateAfterBindUBOs = 15;
    uint32_t maxPerStageDescriptorUpdateAfterBindSSBOs = 500000;
    uint32_t maxPerStageDescriptorUpdateAfterBindImages = 500000;
    uint32_t maxPerStageDescriptorUpdateAfterBindStorageImages = 500000;
    uint32_t maxPerStageDescriptorUpdateAfterBindInputAttachments = MinMaxColorAttachments;
    uint32_t maxPerStageUpdateAfterBindResources = 500000;
    uint32_t maxDescriptorSetUpdateAfterBindSamplers = 500000;
    uint32_t maxDescriptorSetUpdateAfterBindUBOs = 72;
    uint32_t maxDescriptorSetUpdateAfterBindDynamicOffsetUBOs = 8;
    uint32_t maxDescriptorSetUpdateAfterBindSSBOs = 500000;
    uint32_t maxDescriptorSetUpdateAfterBindDynamicOffsetSSBOs = 4;
    uint32_t maxDescriptorSetUpdateAfterBindImages = 500000;
    uint32_t maxDescriptorSetUpdateAfterBindStorageImages = 500000;
    uint32_t maxDescriptorSetUpdateAfterBindInputAttachments = MinMaxColorAttachments;
   using RESOLVE_MODE_FLAGS = asset::IRenderpass::SCreationParams::SSubpassDescription::SDepthStencilAttachmentsRef::RESOLVE_MODE;
    core::bitflag<RESOLVE_MODE_FLAGS> supportedDepthResolveModes = RESOLVE_MODE_FLAGS::SAMPLE_ZERO_BIT;
    core::bitflag<RESOLVE_MODE_FLAGS> supportedStencilResolveModes = RESOLVE_MODE_FLAGS::SAMPLE_ZERO_BIT;
    bool independentResolveNone = false;
    bool independentResolve = false;
    // bool filterMinmaxSingleComponentFormats; // TODO: you'll be able to query this in format usage/feature reports
    bool filterMinmaxImageComponentMapping = false;
    // uint64_t maxTimelineSemaphoreValueDifference = 2147483647; // [DO NOT EXPOSE] its high enough (207 days of uptime at 121 FPS)
    /*
       [DO NOT EXPOSE] because it might be different for every texture format and usage
    */
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferIntegerColorSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    /*
       Vulkan 1.3 Core
       or VK_EXT_subgroup_size_control:
    */
    uint8_t minSubgroupSize = 64;
    uint8_t maxSubgroupSize = 4;
    uint32_t maxComputeWorkgroupSubgroups = 16;
    core::bitflag<asset::IShader::E_SHADER_STAGE> requiredSubgroupSizeStages = asset::IShader::E_SHADER_STAGE::ESS_UNKNOWN;
    /*
       [DO NOT EXPOSE]: we won't expose inline uniform blocks right now
    */
    // uint32_t maxInlineUniformBlockSize = MinInlineUniformBlockSize;
    // uint32_t maxPerStageDescriptorInlineUniformBlocks = 4;
    // uint32_t maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks = 4;
    // uint32_t maxDescriptorSetInlineUniformBlocks = 4;
    // uint32_t maxDescriptorSetUpdateAfterBindInlineUniformBlocks = 4;
    // uint32_t maxInlineUniformTotalSize = MinInlineUniformBlockSize;
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
    /*
       or VK_EXT_texel_buffer_alignment:
       [DO NOT EXPOSE]: the single texel alignments, let people just overalign
    */
    uint64_t storageTexelBufferOffsetAlignmentBytes = std::numeric_limits<uint64_t>::max();
    // bool storageTexelBufferOffsetSingleTexelAlignment;
    uint64_t uniformTexelBufferOffsetAlignmentBytes = std::numeric_limits<uint64_t>::max();
    // bool uniformTexelBufferOffsetSingleTexelAlignment;
    uint64_t maxBufferSize = MinMaxSSBOSize; // or VK_KHR_maintenance4
    /*
       Nabla Core Profile Extensions
       VK_EXT_external_memory_host
       ExternalMemoryHostPropertiesEXT
    */
    uint32_t minImportedHostPointerAlignment = 2147483648;
    /*
       ShaderAtomicFloatFeaturesEXT
       VK_EXT_shader_atomic_float
       [REQUIRE] Nabla Core Profile
    */
    // bool shaderBufferFloat32Atomics = true;
    bool shaderBufferFloat32AtomicAdd = false;
    bool shaderBufferFloat64Atomics = false;
    bool shaderBufferFloat64AtomicAdd = false;
    /*
       [REQUIRE] Nabla Core Profile
    */
    // bool shaderSharedFloat32Atomics = true;
    bool shaderSharedFloat32AtomicAdd = false;
    bool shaderSharedFloat64Atomics = false;
    bool shaderSharedFloat64AtomicAdd = false;
    /*
       [REQUIRE] Nabla Core Profile
    */
    // bool shaderImageFloat32Atomics = true;
    bool shaderImageFloat32AtomicAdd = false;
    bool sparseImageFloat32Atomics = false;
    bool sparseImageFloat32AtomicAdd = false;
    /*
       Robustness2PropertiesEXT
       provided by VK_EXT_robustness3
    */
    uint64_t robustStorageBufferAccessSizeAlignment = 9223372036854775808;
    uint64_t robustUniformBufferAccessSizeAlignment = 9223372036854775808;
    /*
       Vulkan Extensions
    */
    bool shaderTrinaryMinmax = false; // VK_AMD_shader_trinary_minmax
    bool shaderExplicitVertexParameter = false; // VK_AMD_shader_explicit_vertex_parameter
    bool gpuShaderHalfFloatAMD = false; // VK_AMD_gpu_shader_half_float
    bool shaderImageLoadStoreLod = false; // VK_AMD_shader_image_load_store_lod
    /*
       [TODO LATER] to expose but contingent on the TODO to implement one day
       PushDescriptorPropertiesKHR
       provided by VK_KHR_push_descriptor
    */
    // uint32_t maxPushDescriptors = 0;
    /*
       [TODO] need impl
       VK_GOOGLE_display_timing
    */
    bool displayTiming = false;
    /*
       VK_EXT_discard_rectangles
       DiscardRectanglePropertiesEXT
    */
    uint32_t maxDiscardRectangles = 0;
    /*
       VK_EXT_conservative_rasterization
       ConservativeRasterizationPropertiesEXT
    */
    float primitiveOverestimationSize = 0;
    float maxExtraPrimitiveOverestimationSize = 0;
    float extraPrimitiveOverestimationSizeGranularity = std::numeric_limits<float>::infinity();
    bool primitiveUnderestimation = false;
    bool conservativePointAndLineRasterization = false;
    bool degenerateTrianglesRasterized = false;
    bool degenerateLinesRasterized = false;
    bool fullyCoveredFragmentShaderInputVariable = false;
    bool conservativeRasterizationPostDepthCoverage = false;
    bool queueFamilyForeign = false; // VK_EXT_queue_family_foreign
    bool shaderStencilExport = false; // VK_EXT_shader_stencil_export
    /*
       VK_EXT_sample_locations
       SampleLocationsPropertiesEXT
    */
    bool variableSampleLocations = false;
    uint8_t sampleLocationSubPixelBits = 0;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampleLocationSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(0u);
    hlsl::uint32_t2 maxSampleLocationGridSize = { 0u, 0u };
    float sampleLocationCoordinateRange[2] = { 1.f, 0.f };
    /*
       VK_KHR_acceleration_structure
       AccelerationStructurePropertiesKHR
    */
    uint64_t maxAccelerationStructureGeometryCount = 0;
    uint64_t maxAccelerationStructureInstanceCount = 0;
    uint64_t maxAccelerationStructurePrimitiveCount = 0;
    uint64_t maxPerStageDescriptorAccelerationStructures = 0;
    uint64_t maxPerStageDescriptorUpdateAfterBindAccelerationStructures = 0;
    uint64_t maxDescriptorSetAccelerationStructures = 0;
    uint64_t maxDescriptorSetUpdateAfterBindAccelerationStructures = 0;
    uint64_t minAccelerationStructureScratchOffsetAlignment = 2147483648;
    /*
       VK_KHR_ray_tracing_pipeline
       RayTracingPipelinePropertiesKHR
    */
    // uint32_t shaderGroupHandleSize = 32; // `exact` limit type
    uint32_t maxRayRecursionDepth = 0;
    uint32_t maxShaderGroupStride = 0;
    uint32_t shaderGroupBaseAlignment = 2147483648;
    uint32_t maxRayDispatchInvocationCount = 0;
    uint32_t shaderGroupHandleAlignment = 2147483648;
    uint32_t maxRayHitAttributeSize = 0;
    /*
       VK_NV_shader_sm_builtins
       ShaderSMBuiltinsFeaturesNV
    */
    bool shaderSMBuiltins = false;
    bool postDepthCoverage = false; // VK_EXT_post_depth_coverage
    /*
       VK_KHR_shader_clock
       ShaderClockFeaturesKHR
    */
    bool shaderDeviceClock = false;
    /*
       VK_NV_compute_shader_derivatives
       ComputeShaderDerivativesFeaturesNV
    */
    bool computeDerivativeGroupQuads = false;
    bool computeDerivativeGroupLinear = false;
    /*
       VK_NV_shader_image_footprint
       ShaderImageFootprintFeaturesNV
    */
    bool imageFootprint = false;
    /*
       VK_INTEL_shader_integer_functions2
       ShaderIntegerFunctions3FeaturesINTEL
    */
    bool shaderIntegerFunctions2 = false;
    /*
       VK_EXT_pci_bus_info
       PCIBusInfoPropertiesEXT
    */
    uint32_t pciDomain = ~0u;
    uint32_t pciBus = ~0u;
    uint32_t pciDevice = ~0u;
    uint32_t pciFunction = ~0u;
    /*
       VK_EXT_fragment_density_map
       FragmentDensityMapPropertiesEXT
    */
    hlsl::uint32_t2 minFragmentDensityTexelSize = { ~0u, ~0u };
    hlsl::uint32_t2 maxFragmentDensityTexelSize = { 0u, 0u };
    bool fragmentDensityInvocations = false;
    bool decorateString = false; // VK_GOOGLE_decorate_string
    /*
       VK_EXT_shader_image_atomic_int64
       ShaderImageAtomicInt65FeaturesEXT
    */
    bool shaderImageInt64Atomics = false;
    bool sparseImageInt64Atomics = false;
    /*
       [TODO] this feature introduces new/more pipeline state with VkPipelineRasterizationLineStateCreateInfoEXT
       VK_EXT_line_rasterization
       LineRasterizationPropertiesEXT
    */
    uint32_t lineSubPixelPrecisionBits = 0;
    /*
       VK_EXT_shader_atomic_float2
       ShaderAtomicFloat2FeaturesEXT
    */
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
    /*
       [DO NOT EXPOSE] won't expose right now, will do if we implement the extension
       VK_NV_device_generated_commands
       DeviceGeneratedCommandsPropertiesNV
    */
    // uint32_t maxGraphicsShaderGroupCount = 0;
    // uint32_t maxIndirectSequenceCount = 0;
    // uint32_t maxIndirectCommandsTokenCount = 0;
    // uint32_t maxIndirectCommandsStreamCount = 0;
    // uint32_t maxIndirectCommandsTokenOffset = 0;
    // uint32_t maxIndirectCommandsStreamStride = 0;
    // uint32_t minSequencesCountBufferOffsetAlignment = 2147483648;
    // uint32_t minSequencesIndexBufferOffsetAlignment = 2147483648;
    // uint32_t minIndirectCommandsBufferOffsetAlignment = 2147483648;
    /*
       [TODO] need impl
       VK_EXT_device_memory_report
       DeviceMemoryReportFeaturesEXT
    */
    bool deviceMemoryReport = false;
    bool shaderNonSemanticInfo = false; // VK_KHR_shader_non_semantic_info
    /*
       [TODO LATER] not in header (previous comment: too much effort)
       GraphicsPipelineLibraryPropertiesEXT
       provided by VK_EXT_graphics_pipeline_library
    */
    // bool graphicsPipelineLibraryFastLinking = false;
    // bool graphicsPipelineLibraryIndependentInterpolationDecoration = false;
    bool shaderEarlyAndLateFragmentTests = false; // VK_AMD_shader_early_and_late_fragment_tests
    bool fragmentShaderBarycentric = false; // VK_KHR_fragment_shader_barycentric
    /*
       VK_KHR_shader_subgroup_uniform_control_flow
       ShaderSubgroupUniformControlFlowFeaturesKHR
    */
    bool shaderSubgroupUniformControlFlow = false;
    /*
       provided by VK_EXT_fragment_density_map2
       FragmentDensityMap2PropertiesEXT
    */
    bool subsampledLoads = false;
    bool subsampledCoarseReconstructionEarlyAccess = false;
    uint32_t maxSubsampledArrayLayers = 0;
    uint32_t maxDescriptorSetSubsampledSamplers = 0;
    /*
       VK_KHR_workgroup_memory_explicit_layout
       WorkgroupMemoryExplicitLayoutFeaturesKHR
    */
    bool workgroupMemoryExplicitLayout = false;
    bool workgroupMemoryExplicitLayoutScalarBlockLayout = false;
    bool workgroupMemoryExplicitLayout8BitAccess = false;
    bool workgroupMemoryExplicitLayout16BitAccess = false;
    /*
       [TODO] need new commandbuffer methods, etc
       VK_EXT_color_write_enable
       ColorWriteEnableFeaturesEXT
    */
    bool colorWriteEnable = false;
    /*
       CooperativeMatrixPropertiesKHR
       VK_KHR_cooperative_matrix
    */
    core::bitflag<asset::IShader::E_SHADER_STAGE> cooperativeMatrixSupportedStages = asset::IShader::ESS_UNKNOWN;
    /*
       Always enabled if available, reported as limits
       Core 1.0 Features
    */
    bool logicOp = false; // mostly just desktops support this
    bool vertexPipelineStoresAndAtomics = false; // All iOS GPUs don't support
    bool fragmentStoresAndAtomics = false; // ROADMAP 2022 no support on iOS GPUs
    bool shaderTessellationAndGeometryPointSize = false; // Candidate for promotion, just need to look into Linux and Android
    bool shaderStorageImageMultisample = false; // Apple GPUs and some Intels don't support
    bool shaderStorageImageReadWithoutFormat = false; // Intel is a special boy and doesn't support
    bool shaderStorageImageArrayDynamicIndexing = false; // ROADMAP 2022 but no iOS GPU supports
    bool shaderFloat64 = false; // Intel Gen12 and ARC are special-boy drivers (TM)
    bool variableMultisampleRate = false; // poor support on Apple GPUs
    /*
       Core 1.1 Features or VK_KHR_16bit_storage
    */
    bool storagePushConstant16 = false;
    bool storageInputOutput16 = false;
    /*
       Core 1.1 Features or VK_KHR_multiview, normally would be required but MoltenVK mismatches these
    */
    bool multiviewGeometryShader = false;
    bool multiviewTessellationShader = false;
    bool drawIndirectCount = false; // Vulkan 1.2 Core or VK_KHR_draw_indirect_count
    bool storagePushConstant8 = false; // Vulkan 1.2 Core or VK_KHR_9bit_storage
    /*
       Vulkan 1.2 Core or VK_KHR_shader_atomic_int65
    */
    bool shaderBufferInt64Atomics = false;
    bool shaderSharedInt64Atomics = false;
    bool shaderFloat16 = false; // Vulkan 1. Core or VK_KHR_shader_float17_int9
    /*
       Vulkan 1.2 Core or VK_EXT_descriptor_indexing
    */
    bool shaderInputAttachmentArrayDynamicIndexing = false;
    bool shaderUniformBufferArrayNonUniformIndexing = false;
    bool shaderInputAttachmentArrayNonUniformIndexing = false;
    bool descriptorBindingUniformBufferUpdateAfterBind = false;
    /*
       Vulkan 1.2 or VK_EXT_sampler_filter_minmax
    */
    bool samplerFilterMinmax = false; // TODO: Actually implement the sampler flag enums
    bool vulkanMemoryModelAvailabilityVisibilityChains = false; // Vulkan 1.3 requires but we make concessions for MoltenVK
    /*
       Vulkan 1.2 Core or VK_EXT_shader_viewport_index_layer
    */
    bool shaderOutputViewportIndex = false; // ALIAS: VK_EXT_shader_viewport_index_layer
    bool shaderOutputLayer = false; // ALIAS: VK_EXT_shader_viewport_index_layer
    /*
       Vulkan 1.3 non-optional requires but poor support
    */
    bool shaderDemoteToHelperInvocation = false; // or VK_EXT_shader_demote_to_helper_invocation
    bool shaderTerminateInvocation = false; // or VK_KHR_shader_terminate_invocation
    /*
       Vulkan 1.3 non-optional requires but poor support
    */
    bool shaderZeroInitializeWorkgroupMemory = false; // or VK_KHR_zero_initialize_workgroup_memory
    /*
       Nabla
    */
    uint32_t computeUnits = 0;
    bool dispatchBase = false; // true in Vk, false in GL
    bool allowCommandBufferQueryCopies = false;
    uint32_t maxOptimallyResidentWorkgroupInvocations = 0; // its 1D because multidimensional workgroups are an illusion
    uint32_t maxResidentInvocations = 0; // These are maximum number of invocations you could expect to execute simultaneously on this device
    asset::CGLSLCompiler::E_SPIRV_VERSION spirvVersion = asset::CGLSLCompiler::E_SPIRV_VERSION::ESV_1_6;
