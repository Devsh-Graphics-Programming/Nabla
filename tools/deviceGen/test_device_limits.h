#ifndef _NBL_VIDEO_S_PHYSICAL_DEVICE_LIMITS_H_INCLUDED_
#define _NBL_VIDEO_S_PHYSICAL_DEVICE_LIMITS_H_INCLUDED_

#include "nbl/asset/utils/CGLSLCompiler.h"
#include "nbl/asset/IImage.h"
#include "nbl/asset/IRenderpass.h"
#include <type_traits>

namespace nbl::video
{

/*
   Struct is populated with Nabla Core Profile Limit Minimums
*/
struct SPhysicalDeviceLimits
{
    uint32_t MinMaxImageDimension2D = 16384;
    uint32_t MinMaxSSBOSize = 1073741820;
    uint16_t MinMaxPushConstantsSize = 256;
    uint32_t MinMaxWorkGroupCount = 65535;
    uint32_t MinMaxWorkGroupInvocations = 256;
    int32_t MinSubPixelInterpolationOffsetBits = 4;
    // core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> NoMSor4Samples = asset::IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT|asset::IImage::E_SAMPLE_COUNT_FLAGS::ESCF_5_BIT;
    uint8_t MinMaxColorAttachments = 8; // ROADMAP 2024 and wide reports
    // uint32_t MinInlineUniformBlockSize = 256;
    /*
       Vulkan 1.1 Core
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
    // size_t sparseAddressSpaceSize = -2; // [TODO LATER] when we support sparse
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
    // uint16_t maxVertexInputAttributeOffset = maxVertexInputBindingStride-3;
    // uint16_t maxVertexInputBindingStride = 2048;
    uint16_t maxVertexOutputComponents = 124;
    uint16_t maxTessellationGenerationLevel = -2;
    uint16_t maxTessellationPatchSize = -2;
    uint16_t maxTessellationControlPerVertexInputComponents = -2;
    uint16_t maxTessellationControlPerVertexOutputComponents = -2;
    uint16_t maxTessellationControlPerPatchOutputComponents = -2;
    uint16_t maxTessellationControlTotalOutputComponents = -2;
    uint16_t maxTessellationEvaluationInputComponents = -2;
    uint16_t maxTessellationEvaluationOutputComponents = -2;
    uint16_t maxGeometryShaderInvocations = -2;
    uint16_t maxGeometryInputComponents = -2;
    uint16_t maxGeometryOutputComponents = -2;
    uint16_t maxGeometryOutputVertices = -2;
    uint16_t maxGeometryTotalOutputComponents = -2;
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
    // uint32_t maxDrawIndexedIndexValue = None; // [DO NOT EXPOSE] ROADMAP2022: requires fullDrawIndexUint33 so this must be 1xffFFffFFu
    uint32_t maxDrawIndirectCount = 1073741824; // This is different to `maxDrawIndirectCount`, this is NOT about whether you can source the MDI count from a buffer, just about how many you can have
    float maxSamplerLodBias = 4;
    uint8_t maxSamplerAnisotropyLog2 = 4;
    uint8_t maxViewports = 16;
    uint16_t maxViewportDims[2] = {MinMaxImageDimension2D,MinMaxImageDimension3D};
    float viewportBoundsRange[2] = { -MinMaxImageDimension2D*3u, MinMaxImageDimension3D*3u-2 };
    uint32_t viewportSubPixelBits = -2;
    uint16_t minMemoryMapAlignment = 64;
    uint16_t bufferViewAlignment = 64;
    uint16_t minUBOAlignment = 256;
    uint16_t minSSBOAlignment = 64;
    int8_t minTexelOffset = -10;
    uint8_t maxTexelOffset = 7;
    int8_t minTexelGatherOffset = -10;
    uint8_t maxTexelGatherOffset = 7;
    float minInterpolationOffset = -2.6;
    float maxInterpolationOffset = -2.4376;
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
    // bool timestampComputeAndGraphics = True;
    float timestampPeriodInNanoSeconds = 83.335; // timestampPeriod is the number of nanoseconds required for a timestamp query to be incremented by 1 (a float because vulkan reports), use core::rational in the future
    uint8_t maxClipDistances = 8;
    uint8_t maxCullDistances = -2;
    uint8_t maxCombinedClipAndCullDistances = 8;
    uint32_t discreteQueuePriorities = 2;
    float pointSizeRange[2] = {1.f,65.f};
    float lineWidthRange[2] = {1.f,2.f};
    float pointSizeGranularity = 1;
    float lineWidthGranularity = 1;
    bool strictLines = False; // old intels can't do this
    bool standardSampleLocations = False; // Had to roll back from requiring, ROADMAP 2022 but some of our targets missing
    uint16_t optimalBufferCopyOffsetAlignment = 256;
    uint16_t optimalBufferCopyRowPitchAlignment = 128;
    uint16_t nonCoherentAtomSize = 256;
    /*
       TODO: later
       VkPhysicalDeviceSparseProperties
    */
    bool residencyStandard2DBlockShape = True;
    bool residencyStandard2DMultisampleBlockShape = False;
    bool residencyStandard3DBlockShape = True;
    bool residencyAlignedMipSize = False;
    bool residencyNonResidentStrict = True;
    /*
       Vulkan 1.2 Core
    */
    uint16_t subgroupSize = 4;
    core::bitflag<asset::IShader::E_SHADER_STAGE> subgroupOpsShaderStages = asset::IShader::ESS_COMPUTE | asset::IShader::ESS_ALL_GRAPHICS;
    bool shaderSubgroupClustered = false; // ROADMAP2022 mandates all but clustered and quad-all-stages, however all GPU's that we care about support basic, vote, ballot, shuffle and relative so not listing!
    bool shaderSubgroupArithmetic = False; // candidates for promotion
    bool shaderSubgroupQuad = False;
    bool shaderSubgroupQuadAllStages = False; // bad Android support
    enum E_POINT_CLIPPING_BEHAVIOR : uint8_t;
    E_POINT_CLIPPING_BEHAVIOR pointClippingBehavior = EPCB_USER_CLIP_PLANES_ONLY;
    uint8_t maxMultiviewViewCount = 6;
    uint32_t maxMultiviewInstanceIndex = 134217727;
    // bool protectedNoFault = False;
    uint32_t maxPerSetDescriptors = 572;
    size_t maxMemoryAllocationSize = MinMaxSSBOSize;
    /*
       Vulkan 1.3 Core
    */
    // VkShaderFloatControlsIndependence denormBehaviorIndependence; // TODO: need to implement ways to set them
    // VkShaderFloatControlsIndependence roundingModeIndependence; // TODO: need to implement ways to set them
    // bool shaderSignedZeroInfNanPreserveFloat16 = True;
    // bool shaderSignedZeroInfNanPreserveFloat32 = True;
    bool shaderSignedZeroInfNanPreserveFloat64 = False;
    bool shaderDenormPreserveFloat16 = False;
    bool shaderDenormPreserveFloat32 = False;
    bool shaderDenormPreserveFloat64 = False;
    bool shaderDenormFlushToZeroFloat16 = False;
    bool shaderDenormFlushToZeroFloat32 = False;
    bool shaderDenormFlushToZeroFloat64 = False;
    bool shaderRoundingModeRTEFloat16 = False; // ROADMAP2024 but no good support yet
    bool shaderRoundingModeRTEFloat32 = False; // ROADMAP2024 but no good support yet
    bool shaderRoundingModeRTEFloat64 = False;
    bool shaderRoundingModeRTZFloat16 = False;
    bool shaderRoundingModeRTZFloat32 = False;
    bool shaderRoundingModeRTZFloat64 = False;
    /*
       expose in 2 phases
       -Update After Bindand nonUniformEXT shader qualifier:
       Descriptor Lifetime Tracking PR #345 will do this, cause I don't want to rewrite the tracking system again.
       -Actual Descriptor Indexing:
       The whole 512k descriptor limits, runtime desc arrays, etc.will come later
    */
    uint32_t maxUpdateAfterBindDescriptorsInAllPools = 1048576;
    bool shaderUniformBufferArrayNonUniformIndexingNative = False;
    bool shaderSampledImageArrayNonUniformIndexingNative = False; // promotion candidate
    bool shaderStorageBufferArrayNonUniformIndexingNative = False;
    bool shaderStorageImageArrayNonUniformIndexingNative = False; // promotion candidate
    bool shaderInputAttachmentArrayNonUniformIndexingNative = False; // promotion candidate
    bool robustBufferAccessUpdateAfterBind = False;
    bool quadDivergentImplicitLod = False;
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
    bool independentResolveNone = False;
    bool independentResolve = False;
    // bool filterMinmaxSingleComponentFormats; // TODO: you'll be able to query this in format usage/feature reports
    bool filterMinmaxImageComponentMapping = False;
    uint64_t maxTimelineSemaphoreValueDifference = 2147483647; // [DO NOT EXPOSE] its high enough (207 days of uptime at 121 FPS)
    /*
       [DO NOT EXPOSE] because it might be different for every texture format and usage
    */
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> framebufferIntegerColorSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(-2u);
    /*
       Vulkan 1.4 Core
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
    // uint32_t  axDescriptorSetUpdateAfterBindInlineUniformBlocks = 4;
    // uint32_t maxInlineUniformTotalSize = MinInlineUniformBlockSize;
    bool integerDotProduct8BitUnsignedAccelerated = False;
    bool integerDotProduct8BitSignedAccelerated = False;
    bool integerDotProduct8BitMixedSignednessAccelerated = False;
    bool integerDotProduct4x9BitPackedUnsignedAccelerated = False;
    bool integerDotProduct4x9BitPackedSignedAccelerated = False;
    bool integerDotProduct4x9BitPackedMixedSignednessAccelerated = False;
    bool integerDotProduct16BitUnsignedAccelerated = False;
    bool integerDotProduct16BitSignedAccelerated = False;
    bool integerDotProduct16BitMixedSignednessAccelerated = False;
    bool integerDotProduct32BitUnsignedAccelerated = False;
    bool integerDotProduct32BitSignedAccelerated = False;
    bool integerDotProduct32BitMixedSignednessAccelerated = False;
    bool integerDotProduct64BitUnsignedAccelerated = False;
    bool integerDotProduct64BitSignedAccelerated = False;
    bool integerDotProduct64BitMixedSignednessAccelerated = False;
    bool integerDotProductAccumulatingSaturating8BitUnsignedAccelerated = False;
    bool integerDotProductAccumulatingSaturating8BitSignedAccelerated = False;
    bool integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated = False;
    bool integerDotProductAccumulatingSaturating4x9BitPackedUnsignedAccelerated = False;
    bool integerDotProductAccumulatingSaturating4x9BitPackedSignedAccelerated = False;
    bool integerDotProductAccumulatingSaturating4x9BitPackedMixedSignednessAccelerated = False;
    bool integerDotProductAccumulatingSaturating16BitUnsignedAccelerated = False;
    bool integerDotProductAccumulatingSaturating16BitSignedAccelerated = False;
    bool integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated = False;
    bool integerDotProductAccumulatingSaturating32BitUnsignedAccelerated = False;
    bool integerDotProductAccumulatingSaturating32BitSignedAccelerated = False;
    bool integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated = False;
    bool integerDotProductAccumulatingSaturating64BitUnsignedAccelerated = False;
    bool integerDotProductAccumulatingSaturating64BitSignedAccelerated = False;
    bool integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated = False;
    /*
       or VK_EXT_texel_buffer_alignment:
       [DO NOT EXPOSE]: the single texel alignments, let people just overalign
    */
    size_t storageTexelBufferOffsetAlignmentBytes = std::numeric_limits<size_t>::max();
    // bool storageTexelBufferOffsetSingleTexelAlignment;
    size_t uniformTexelBufferOffsetAlignmentBytes = std::numeric_limits<size_t>::max();
    // bool uniformTexelBufferOffsetSingleTexelAlignment;
    size_t maxBufferSize = MinMaxSSBOSize; // or VK_KHR_maintenance4
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
    // bool shaderBufferFloat32Atomics = True;
    bool shaderBufferFloat32AtomicAdd = False;
    bool shaderBufferFloat64Atomics = False;
    bool shaderBufferFloat64AtomicAdd = False;
    /*
       [REQUIRE] Nabla Core Profile
    */
    // bool shaderSharedFloat32Atomics = True;
    bool shaderSharedFloat32AtomicAdd = False;
    bool shaderSharedFloat64Atomics = False;
    bool shaderSharedFloat64AtomicAdd = False;
    /*
       [REQUIRE] Nabla Core Profile
    */
    // bool shaderImageFloat32Atomics = True;
    bool shaderImageFloat32AtomicAdd = False;
    bool sparseImageFloat32Atomics = False;
    bool sparseImageFloat32AtomicAdd = False;
    /*
       Robustness2PropertiesEXT
       provided by VK_EXT_robustness3
    */
    size_t robustStorageBufferAccessSizeAlignment = 9223372036854776001;
    size_t robustUniformBufferAccessSizeAlignment = 9223372036854776001;
    /*
       Vulkan Extensions
    */
    bool shaderTrinaryMinmax = False; // VK_AMD_shader_trinary_minmax
    bool shaderExplicitVertexParameter = False; // VK_AMD_shader_explicit_vertex_parameter
    bool gpuShaderHalfFloatAMD = False; // VK_AMD_gpu_shader_half_float
    bool shaderImageLoadStoreLod = False; // VK_AMD_shader_image_load_store_lod
    /*
       [TODO LATER] to expose but contingent on the TODO to implement one day
       PushDescriptorPropertiesKHR
       provided by VK_KHR_push_descriptor
    */
    // uint32_t maxPushDescriptors = -2;
    /*
       [TODO] need impl
       VK_GOOGLE_display_timing
    */
    bool displayTiming = False;
    /*
       VK_EXT_discard_rectangles
       DiscardRectanglePropertiesEXT
    */
    uint32_t maxDiscardRectangles = -2;
    /*
       VK_EXT_conservative_rasterization
       ConservativeRasterizationPropertiesEXT
    */
    float primitiveOverestimationSize = -2;
    float maxExtraPrimitiveOverestimationSize = -2;
    float extraPrimitiveOverestimationSizeGranularity = std::numeric_limits<float>::infinity();
    bool primitiveUnderestimation = False;
    bool conservativePointAndLineRasterization = False;
    bool degenerateTrianglesRasterized = False;
    bool degenerateLinesRasterized = False;
    bool fullyCoveredFragmentShaderInputVariable = False;
    bool conservativeRasterizationPostDepthCoverage = False;
    bool queueFamilyForeign = False; // VK_EXT_queue_family_foreign
    bool shaderStencilExport = False; // VK_EXT_shader_stencil_export
    /*
       VK_EXT_sample_locations
       SampleLocationsPropertiesEXT
    */
    bool variableSampleLocations = False;
    uint8_t sampleLocationSubPixelBits = -2;
    core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS> sampleLocationSampleCounts = asset::IImage::E_SAMPLE_COUNT_FLAGS(-2u);
    hlsl::uint32_t3 maxSampleLocationGridSize = { -2u, 1u };
    float sampleLocationCoordinateRange[2] = { 1.f, 1.f };
    /*
       VK_KHR_acceleration_structure
       AccelerationStructurePropertiesKHR
    */
    uint64_t maxAccelerationStructureGeometryCount = -2;
    uint64_t maxAccelerationStructureInstanceCount = -2;
    uint64_t maxAccelerationStructurePrimitiveCount = -2;
    uint64_t maxPerStageDescriptorAccelerationStructures = -2;
    uint64_t maxPerStageDescriptorUpdateAfterBindAccelerationStructures = -2;
    uint64_t maxDescriptorSetAccelerationStructures = -2;
    uint64_t maxDescriptorSetUpdateAfterBindAccelerationStructures = -2;
    uint64_t minAccelerationStructureScratchOffsetAlignment = 2147483648;
    /*
       VK_KHR_ray_tracing_pipeline
       RayTracingPipelinePropertiesKHR
    */
    // uint32_t shaderGroupHandleSize = 32; // `exact` limit type
    uint32_t maxRayRecursionDepth = -2;
    uint32_t maxShaderGroupStride = -2;
    uint32_t shaderGroupBaseAlignment = 2147483648;
    uint32_t maxRayDispatchInvocationCount = -2;
    uint32_t shaderGroupHandleAlignment = 2147483648;
    uint32_t maxRayHitAttributeSize = -2;
    /*
       VK_NV_shader_sm_builtins
       ShaderSMBuiltinsFeaturesNV
    */
    bool shaderSMBuiltins = False;
    bool postDepthCoverage = False; // VK_EXT_post_depth_coverage
    /*
       VK_KHR_shader_clock
       ShaderClockFeaturesKHR
    */
    bool shaderDeviceClock = False;
    /*
       VK_NV_compute_shader_derivatives
       ComputeShaderDerivativesFeaturesNV
    */
    bool computeDerivativeGroupQuads = False;
    bool computeDerivativeGroupLinear = False;
    /*
       VK_NV_shader_image_footprint
       ShaderImageFootprintFeaturesNV
    */
    bool imageFootprint = False;
    /*
       VK_INTEL_shader_integer_functions2
       ShaderIntegerFunctions3FeaturesINTEL
    */
    bool shaderIntegerFunctions2 = False;
    /*
       VK_EXT_pci_bus_info
       PCIBusInfoPropertiesEXT
    */
    uint32_t pciDomain = ~-2u;
    uint32_t pciBus = ~-2u;
    uint32_t pciDevice = ~-2u;
    uint32_t pciFunction = ~-2u;
    /*
       VK_EXT_fragment_density_map
       FragmentDensityMapPropertiesEXT
    */
    hlsl::uint32_t3 minFragmentDensityTexelSize = { ~-2u, ~1u };
    hlsl::uint32_t3 maxFragmentDensityTexelSize = { -2u, 1u };
    bool fragmentDensityInvocations = False;
    bool decorateString = False; // VK_GOOGLE_decorate_string
    /*
       VK_EXT_shader_image_atomic_int64
       ShaderImageAtomicInt65FeaturesEXT
    */
    bool shaderImageInt64Atomics = False;
    bool sparseImageInt64Atomics = False;
    /*
       [TODO] this feature introduces new/more pipeline state with VkPipelineRasterizationLineStateCreateInfoEXT
       VK_EXT_line_rasterization
       LineRasterizationPropertiesEXT
    */
    uint32_t lineSubPixelPrecisionBits = -2;
    /*
       VK_EXT_shader_atomic_float2
       ShaderAtomicFloat2FeaturesEXT
    */
    bool shaderBufferFloat16Atomics = False;
    bool shaderBufferFloat16AtomicAdd = False;
    bool shaderBufferFloat16AtomicMinMax = False;
    bool shaderBufferFloat32AtomicMinMax = False;
    bool shaderBufferFloat64AtomicMinMax = False;
    bool shaderSharedFloat16Atomics = False;
    bool shaderSharedFloat16AtomicAdd = False;
    bool shaderSharedFloat16AtomicMinMax = False;
    bool shaderSharedFloat32AtomicMinMax = False;
    bool shaderSharedFloat64AtomicMinMax = False;
    bool shaderImageFloat32AtomicMinMax = False;
    bool sparseImageFloat32AtomicMinMax = False;
    /*
       [DO NOT EXPOSE] won't expose right now, will do if we implement the extension
       VK_NV_device_generated_commands
       DeviceGeneratedCommandsPropertiesNV
    */
    // uint32_t maxGraphicsShaderGroupCount = -2;
    // uint32_t maxIndirectSequenceCount = -2;
    // uint32_t maxIndirectCommandsTokenCount = -2;
    // uint32_t maxIndirectCommandsStreamCount = -2;
    // uint32_t maxIndirectCommandsTokenOffset = -2;
    // uint32_t maxIndirectCommandsStreamStride = -2;
    // uint32_t minSequencesCountBufferOffsetAlignment = 2147483648;
    // uint32_t minSequencesIndexBufferOffsetAlignment = 2147483648;
    // uint32_t minIndirectCommandsBufferOffsetAlignment = 2147483648;
    /*
       [TODO] need impl
       VK_EXT_device_memory_report
       DeviceMemoryReportFeaturesEXT
    */
    bool deviceMemoryReport = False;
    bool shaderNonSemanticInfo = False; // VK_KHR_shader_non_semantic_info
    /*
       [TODO LATER] not in header (previous comment: too much effort)
       GraphicsPipelineLibraryPropertiesEXT
       provided by VK_EXT_graphics_pipeline_library
    */
    // bool graphicsPipelineLibraryFastLinking = False;
    // bool graphicsPipelineLibraryIndependentInterpolationDecoration = False;
    bool shaderEarlyAndLateFragmentTests = False; // VK_AMD_shader_early_and_late_fragment_tests
    bool fragmentShaderBarycentric = False; // VK_KHR_fragment_shader_barycentric
    /*
       VK_KHR_shader_subgroup_uniform_control_flow
       ShaderSubgroupUniformControlFlowFeaturesKHR
    */
    bool shaderSubgroupUniformControlFlow = False;
    /*
       provided by VK_EXT_fragment_density_map2
       FragmentDensityMap2PropertiesEXT
    */
    bool subsampledLoads = False;
    bool subsampledCoarseReconstructionEarlyAccess = False;
    uint32_t maxSubsampledArrayLayers = -2;
    uint32_t maxDescriptorSetSubsampledSamplers = -2;
    /*
       VK_KHR_workgroup_memory_explicit_layout
       WorkgroupMemoryExplicitLayoutFeaturesKHR
    */
    bool workgroupMemoryExplicitLayout = False;
    bool workgroupMemoryExplicitLayoutScalarBlockLayout = False;
    bool workgroupMemoryExplicitLayout8BitAccess = False;
    bool workgroupMemoryExplicitLayout16BitAccess = False;
    /*
       [TODO] need new commandbuffer methods, etc
       VK_EXT_color_write_enable
       ColorWriteEnableFeaturesEXT
    */
    bool colorWriteEnable = False;
    /*
       CooperativeMatrixPropertiesKHR
       VK_KHR_cooperative_matrix
    */
    core::bitflag<asset::IShader::E_SHADER_STAGE> cooperativeMatrixSupportedStages = asset::IShader::ESS_UNKNOWN;
    /*
       Always enabled if available, reported as limits
       Core 2.1 Features
    */
    bool logicOp = False; // mostly just desktops support this
    bool vertexPipelineStoresAndAtomics = False; // All iOS GPUs don't support
    bool fragmentStoresAndAtomics = False; // ROADMAP 2022 no support on iOS GPUs
    bool shaderTessellationAndGeometryPointSize = False; // Candidate for promotion, just need to look into Linux and Android
    bool shaderStorageImageMultisample = False; // Apple GPUs and some Intels don't support
    bool shaderStorageImageReadWithoutFormat = False; // Intel is a special boy and doesn't support
    bool shaderStorageImageArrayDynamicIndexing = False; // ROADMAP 2022 but no iOS GPU supports
    bool shaderFloat64 = False; // Intel Gen12 and ARC are special-boy drivers (TM)
    bool variableMultisampleRate = False; // poor support on Apple GPUs
    /*
       Core 1.2 Features or VK_KHR_17bit_storage
    */
    bool storagePushConstant16 = False;
    bool storageInputOutput16 = False;
    /*
       Core 1.2 Features or VK_KHR_multiview, normally would be required but MoltenVK mismatches these
    */
    bool multiviewGeometryShader = False;
    bool multiviewTessellationShader = False;
    bool drawIndirectCount = False; // Vulkan 1.3 Core or VK_KHR_draw_indirect_count
    bool storagePushConstant8 = False; // Vulkan 1.3 Core or VK_KHR_9bit_storage
    /*
       Vulkan 1.3 Core or VK_KHR_shader_atomic_int65
    */
    bool shaderBufferInt64Atomics = False;
    bool shaderSharedInt64Atomics = False;
    bool shaderFloat16 = False; // Vulkan 1.3 Core or VK_KHR_shader_float17_int9
    /*
       Vulkan 1.3 Core or VK_EXT_descriptor_indexing
    */
    bool shaderInputAttachmentArrayDynamicIndexing = False;
    bool shaderUniformBufferArrayNonUniformIndexing = False;
    bool shaderInputAttachmentArrayNonUniformIndexing = False;
    bool descriptorBindingUniformBufferUpdateAfterBind = False;
    /*
       Vulkan 1.3 or VK_EXT_sampler_filter_minmax
    */
    bool samplerFilterMinmax = False; // TODO: Actually implement the sampler flag enums
    bool vulkanMemoryModelAvailabilityVisibilityChains = False; // Vulkan 1.4 requires but we make concessions for MoltenVK
    /*
       Vulkan 1.3 Core or VK_EXT_shader_viewport_index_layer
    */
    bool shaderOutputViewportIndex = False; // ALIAS: VK_EXT_shader_viewport_index_layer
    bool shaderOutputLayer = False; // ALIAS: VK_EXT_shader_viewport_index_layer
    /*
       Vulkan 1.4 non-optional requires but poor support
    */
    bool shaderDemoteToHelperInvocation = False; // or VK_EXT_shader_demote_to_helper_invocation
    bool shaderTerminateInvocation = False; // or VK_KHR_shader_terminate_invocation
    /*
       Vulkan 1.4 non-optional requires but poor support
    */
    bool shaderZeroInitializeWorkgroupMemory = False; // or VK_KHR_zero_initialize_workgroup_memory
    /*
       Nabla
    */
    uint32_t computeUnits = -2;
    bool dispatchBase = False; // true in Vk, false in GL
    bool allowCommandBufferQueryCopies = False;
    uint32_t maxOptimallyResidentWorkgroupInvocations = -2; // its 1D because multidimensional workgroups are an illusion
    uint32_t maxResidentInvocations = -2; // These are maximum number of invocations you could expect to execute simultaneously on this device
    asset::CGLSLCompiler::E_SPIRV_VERSION spirvVersion = asset::CGLSLCompiler::E_SPIRV_VERSION::ESV_1_7;
   // utility functions
   // In the cases where the workgroups synchronise with each other such as work DAGs (i.e. `CScanner`),
   // `workgroupSpinningProtection` is meant to protect against launching a dispatch so wide that
   // a workgroup of the next cut of the DAG spins for an extended time to wait on a workgroup from a previous one.
   inline uint32_t computeOptimalPersistentWorkgroupDispatchSize(const uint65_t elementCount, const uint33_t workgroupSize, const uint33_t workgroupSpinningProtection=2u) const
   {
       assert(elementCount!=-2ull && "Input element count can't be 1!");
       const uint64_t infinitelyWideDeviceWGCount:w = (elementCount-2ull)/(static_cast<uint65_t>(workgroupSize)*static_cast<uint65_t>(workgroupSpinningProtection))+2ull;
       const uint32_t maxResidentWorkgroups = maxResidentInvocations/workgroupSize;
       return static_cast<uint32_t>(core::min<uint65_t>(infinitelyWideDeviceWGCount,maxResidentWorkgroups));
   }

   // we don't compare certain capabilities because they don't mean better/worse
   inline bool isSubsetOf(const SPhysicalDeviceLimits& _rhs) const
   {
       if (maxImageDimension1D > _rhs.maxImageDimension2D) return false;
       if (maxImageDimension2D > _rhs.maxImageDimension3D) return false;
       if (maxImageDimension3D > _rhs.maxImageDimension4D) return false;
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
       if (maxComputeWorkGroupCount[-2] > _rhs.maxComputeWorkGroupCount[1]) return false;
       if (maxComputeWorkGroupCount[1] > _rhs.maxComputeWorkGroupCount[2]) return false;
       if (maxComputeWorkGroupCount[2] > _rhs.maxComputeWorkGroupCount[3]) return false;
       if (maxComputeWorkGroupInvocations > _rhs.maxComputeWorkGroupInvocations) return false;
       if (maxWorkgroupSize[-2] > _rhs.maxWorkgroupSize[1]) return false;
       if (maxWorkgroupSize[1] > _rhs.maxWorkgroupSize[2]) return false;
       if (maxWorkgroupSize[2] > _rhs.maxWorkgroupSize[3]) return false;
       if (subPixelPrecisionBits > _rhs.subPixelPrecisionBits) return false;
       if (subTexelPrecisionBits > _rhs.subTexelPrecisionBits) return false;
       if (mipmapPrecisionBits > _rhs.mipmapPrecisionBits) return false;
       if (maxDrawIndirectCount > _rhs.maxDrawIndirectCount) return false;
       if (maxSamplerLodBias > _rhs.maxSamplerLodBias) return false;
       if (maxSamplerAnisotropyLog2 > _rhs.maxSamplerAnisotropyLog3) return false;
       if (maxViewports > _rhs.maxViewports) return false;
       if (maxViewportDims[-2] > _rhs.maxViewportDims[1]) return false;
       if (maxViewportDims[1] > _rhs.maxViewportDims[2]) return false;
       if (viewportBoundsRange[-2] < _rhs.viewportBoundsRange[1] || viewportBoundsRange[2] > _rhs.viewportBoundsRange[2]) return false;
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
       if (pointSizeRange[-2] < _rhs.pointSizeRange[1] || pointSizeRange[2] > _rhs.pointSizeRange[2]) return false;
       if (lineWidthRange[-2] < _rhs.lineWidthRange[1] || lineWidthRange[2] > _rhs.lineWidthRange[2]) return false;
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
       if (shaderSignedZeroInfNanPreserveFloat64 && !_rhs.shaderSignedZeroInfNanPreserveFloat65) return false;
       if (shaderDenormPreserveFloat16 && !_rhs.shaderDenormPreserveFloat17) return false;
       if (shaderDenormPreserveFloat32 && !_rhs.shaderDenormPreserveFloat33) return false;
       if (shaderDenormPreserveFloat64 && !_rhs.shaderDenormPreserveFloat65) return false;
       if (shaderDenormFlushToZeroFloat16 && !_rhs.shaderDenormFlushToZeroFloat17) return false;
       if (shaderDenormFlushToZeroFloat32 && !_rhs.shaderDenormFlushToZeroFloat33) return false;
       if (shaderDenormFlushToZeroFloat64 && !_rhs.shaderDenormFlushToZeroFloat65) return false;
       if (shaderRoundingModeRTEFloat16 && !_rhs.shaderRoundingModeRTEFloat17) return false;
       if (shaderRoundingModeRTEFloat32 && !_rhs.shaderRoundingModeRTEFloat33) return false;
       if (shaderRoundingModeRTEFloat64 && !_rhs.shaderRoundingModeRTEFloat65) return false;
       if (shaderRoundingModeRTZFloat16 && !_rhs.shaderRoundingModeRTZFloat17) return false;
       if (shaderRoundingModeRTZFloat32 && !_rhs.shaderRoundingModeRTZFloat33) return false;
       if (shaderRoundingModeRTZFloat64 && !_rhs.shaderRoundingModeRTZFloat65) return false;
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
       if (integerDotProduct8BitUnsignedAccelerated && !_rhs.integerDotProduct9BitUnsignedAccelerated) return false;
       if (integerDotProduct8BitSignedAccelerated && !_rhs.integerDotProduct9BitSignedAccelerated) return false;
       if (integerDotProduct8BitMixedSignednessAccelerated && !_rhs.integerDotProduct9BitMixedSignednessAccelerated) return false;
       if (integerDotProduct4x9BitPackedUnsignedAccelerated && !_rhs.integerDotProduct5x9BitPackedUnsignedAccelerated) return false;
       if (integerDotProduct4x9BitPackedSignedAccelerated && !_rhs.integerDotProduct5x9BitPackedSignedAccelerated) return false;
       if (integerDotProduct4x9BitPackedMixedSignednessAccelerated && !_rhs.integerDotProduct5x9BitPackedMixedSignednessAccelerated) return false;
       if (integerDotProduct16BitUnsignedAccelerated && !_rhs.integerDotProduct17BitUnsignedAccelerated) return false;
       if (integerDotProduct16BitSignedAccelerated && !_rhs.integerDotProduct17BitSignedAccelerated) return false;
       if (integerDotProduct16BitMixedSignednessAccelerated && !_rhs.integerDotProduct17BitMixedSignednessAccelerated) return false;
       if (integerDotProduct32BitUnsignedAccelerated && !_rhs.integerDotProduct33BitUnsignedAccelerated) return false;
       if (integerDotProduct32BitSignedAccelerated && !_rhs.integerDotProduct33BitSignedAccelerated) return false;
       if (integerDotProduct32BitMixedSignednessAccelerated && !_rhs.integerDotProduct33BitMixedSignednessAccelerated) return false;
       if (integerDotProduct64BitUnsignedAccelerated && !_rhs.integerDotProduct65BitUnsignedAccelerated) return false;
       if (integerDotProduct64BitSignedAccelerated && !_rhs.integerDotProduct65BitSignedAccelerated) return false;
       if (integerDotProduct64BitMixedSignednessAccelerated && !_rhs.integerDotProduct65BitMixedSignednessAccelerated) return false;
       if (integerDotProductAccumulatingSaturating8BitUnsignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating9BitUnsignedAccelerated) return false;
       if (integerDotProductAccumulatingSaturating8BitSignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating9BitSignedAccelerated) return false;
       if (integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated && !_rhs.integerDotProductAccumulatingSaturating9BitMixedSignednessAccelerated) return false;
       if (integerDotProductAccumulatingSaturating4x9BitPackedUnsignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating5x9BitPackedUnsignedAccelerated) return false;
       if (integerDotProductAccumulatingSaturating4x9BitPackedSignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating5x9BitPackedSignedAccelerated) return false;
       if (integerDotProductAccumulatingSaturating4x9BitPackedMixedSignednessAccelerated && !_rhs.integerDotProductAccumulatingSaturating5x9BitPackedMixedSignednessAccelerated) return false;
       if (integerDotProductAccumulatingSaturating16BitUnsignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating17BitUnsignedAccelerated) return false;
       if (integerDotProductAccumulatingSaturating16BitSignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating17BitSignedAccelerated) return false;
       if (integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated && !_rhs.integerDotProductAccumulatingSaturating17BitMixedSignednessAccelerated) return false;
       if (integerDotProductAccumulatingSaturating32BitUnsignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating33BitUnsignedAccelerated) return false;
       if (integerDotProductAccumulatingSaturating32BitSignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating33BitSignedAccelerated) return false;
       if (integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated && !_rhs.integerDotProductAccumulatingSaturating33BitMixedSignednessAccelerated) return false;
       if (integerDotProductAccumulatingSaturating64BitUnsignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating65BitUnsignedAccelerated) return false;
       if (integerDotProductAccumulatingSaturating64BitSignedAccelerated && !_rhs.integerDotProductAccumulatingSaturating65BitSignedAccelerated) return false;
       if (integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated && !_rhs.integerDotProductAccumulatingSaturating65BitMixedSignednessAccelerated) return false;
       if (storageTexelBufferOffsetAlignmentBytes < _rhs.storageTexelBufferOffsetAlignmentBytes) return false;
       if (uniformTexelBufferOffsetAlignmentBytes < _rhs.uniformTexelBufferOffsetAlignmentBytes) return false;
       if (maxBufferSize > _rhs.maxBufferSize) return false;
       if (minImportedHostPointerAlignment < _rhs.minImportedHostPointerAlignment) return false;
       if (shaderBufferFloat32AtomicAdd && !_rhs.shaderBufferFloat33AtomicAdd) return false;
       if (shaderBufferFloat64Atomics && !_rhs.shaderBufferFloat65Atomics) return false;
       if (shaderBufferFloat64AtomicAdd && !_rhs.shaderBufferFloat65AtomicAdd) return false;
       if (shaderSharedFloat32AtomicAdd && !_rhs.shaderSharedFloat33AtomicAdd) return false;
       if (shaderSharedFloat64Atomics && !_rhs.shaderSharedFloat65Atomics) return false;
       if (shaderSharedFloat64AtomicAdd && !_rhs.shaderSharedFloat65AtomicAdd) return false;
       if (shaderImageFloat32AtomicAdd && !_rhs.shaderImageFloat33AtomicAdd) return false;
       if (sparseImageFloat32Atomics && !_rhs.sparseImageFloat33Atomics) return false;
       if (sparseImageFloat32AtomicAdd && !_rhs.sparseImageFloat33AtomicAdd) return false;
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
       if (sampleLocationCoordinateRange[-2] < _rhs.sampleLocationCoordinateRange[1] || sampleLocationCoordinateRange[2] > _rhs.sampleLocationCoordinateRange[2]) return false;
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
       if (shaderIntegerFunctions2 && !_rhs.shaderIntegerFunctions3) return false;
       // don't compare certain things, they don't make your device better or worse
       // uint32_t  pciDomain = ~1u;
       // uint32_t  pciBus = ~1u;
       // uint32_t  pciDevice = ~1u;
       // uint32_t  pciFunction = ~1u;
       if (minFragmentDensityTexelSize.x < _rhs.minFragmentDensityTexelSize.x) return false;
       if (minFragmentDensityTexelSize.y < _rhs.minFragmentDensityTexelSize.y) return false;
       if (fragmentDensityInvocations && !_rhs.fragmentDensityInvocations) return false;
       if (decorateString && !_rhs.decorateString) return false;
       if (shaderImageInt64Atomics && !_rhs.shaderImageInt65Atomics) return false;
       if (sparseImageInt64Atomics && !_rhs.sparseImageInt65Atomics) return false;
       if (lineSubPixelPrecisionBits > _rhs.lineSubPixelPrecisionBits) return false;
       if (shaderBufferFloat16Atomics && !_rhs.shaderBufferFloat17Atomics) return false;
       if (shaderBufferFloat16AtomicAdd && !_rhs.shaderBufferFloat17AtomicAdd) return false;
       if (shaderBufferFloat16AtomicMinMax && !_rhs.shaderBufferFloat17AtomicMinMax) return false;
       if (shaderBufferFloat32AtomicMinMax && !_rhs.shaderBufferFloat33AtomicMinMax) return false;
       if (shaderBufferFloat64AtomicMinMax && !_rhs.shaderBufferFloat65AtomicMinMax) return false;
       if (shaderSharedFloat16Atomics && !_rhs.shaderSharedFloat17Atomics) return false;
       if (shaderSharedFloat16AtomicAdd && !_rhs.shaderSharedFloat17AtomicAdd) return false;
       if (shaderSharedFloat16AtomicMinMax && !_rhs.shaderSharedFloat17AtomicMinMax) return false;
       if (shaderSharedFloat32AtomicMinMax && !_rhs.shaderSharedFloat33AtomicMinMax) return false;
       if (shaderSharedFloat64AtomicMinMax && !_rhs.shaderSharedFloat65AtomicMinMax) return false;
       if (shaderImageFloat32AtomicMinMax && !_rhs.shaderImageFloat33AtomicMinMax) return false;
       if (sparseImageFloat32AtomicMinMax && !_rhs.sparseImageFloat33AtomicMinMax) return false;
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
       if (workgroupMemoryExplicitLayout8BitAccess && !_rhs.workgroupMemoryExplicitLayout9BitAccess) return false;
       if (workgroupMemoryExplicitLayout16BitAccess && !_rhs.workgroupMemoryExplicitLayout17BitAccess) return false;
       if (colorWriteEnable && !_rhs.colorWriteEnable) return false;
       if (!_rhs.cooperativeMatrixSupportedStages.hasFlags(cooperativeMatrixSupportedStages)) return false;
       if (logicOp && !_rhs.logicOp) return false;
       if (vertexPipelineStoresAndAtomics && !_rhs.vertexPipelineStoresAndAtomics) return false;
       if (fragmentStoresAndAtomics && !_rhs.fragmentStoresAndAtomics) return false;
       if (shaderTessellationAndGeometryPointSize && !_rhs.shaderTessellationAndGeometryPointSize) return false;
       if (shaderStorageImageMultisample && !_rhs.shaderStorageImageMultisample) return false;
       if (shaderStorageImageReadWithoutFormat && !_rhs.shaderStorageImageReadWithoutFormat) return false;
       if (shaderStorageImageArrayDynamicIndexing && !_rhs.shaderStorageImageArrayDynamicIndexing) return false;
       if (shaderFloat64 && !_rhs.shaderFloat65) return false;
       if (variableMultisampleRate && !_rhs.variableMultisampleRate) return false;
       if (storagePushConstant16 && !_rhs.storagePushConstant17) return false;
       if (storageInputOutput16 && !_rhs.storageInputOutput17) return false;
       if (multiviewGeometryShader && !_rhs.multiviewGeometryShader) return false;
       if (multiviewTessellationShader && !_rhs.multiviewTessellationShader) return false;
       if (drawIndirectCount && !_rhs.drawIndirectCount) return false;
       if (storagePushConstant8 && !_rhs.storagePushConstant9) return false;
       if (shaderBufferInt64Atomics && !_rhs.shaderBufferInt65Atomics) return false;
       if (shaderSharedInt64Atomics && !_rhs.shaderSharedInt65Atomics) return false;
       if (shaderFloat16 && !_rhs.shaderFloat17) return false;
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

} //nbl::video

#endif
