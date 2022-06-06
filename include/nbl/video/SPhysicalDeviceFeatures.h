#ifndef _NBL_VIDEO_S_PHYSICAL_DEVICE_FEATURES_H_INCLUDED_
#define _NBL_VIDEO_S_PHYSICAL_DEVICE_FEATURES_H_INCLUDED_

namespace nbl::video
{

struct SPhysicalDeviceFeatures
{
    /* Vulkan 1.0 Core  */
    bool robustBufferAccess = false;
    bool fullDrawIndexUint32 = false;
    bool imageCubeArray = false;
    bool independentBlend = false;
    bool geometryShader    = false;
    bool tessellationShader = false;
    //VkBool32    sampleRateShading;
    //VkBool32    dualSrcBlend;
    bool logicOp = false;
    bool multiDrawIndirect = false;
    bool drawIndirectFirstInstance = false;
    bool depthClamp = false;
    bool depthBiasClamp = false;
    bool fillModeNonSolid = false;
    bool depthBounds = false;
    bool wideLines = false;
    bool largePoints = false;
    bool alphaToOne = false;
    bool multiViewport = false;
    bool samplerAnisotropy = false;
    //VkBool32    textureCompressionETC2;
    //VkBool32    textureCompressionASTC_LDR;
    //VkBool32    textureCompressionBC;
    //VkBool32    occlusionQueryPrecise;
    //VkBool32    pipelineStatisticsQuery;
    //VkBool32    vertexPipelineStoresAndAtomics;
    //VkBool32    fragmentStoresAndAtomics;
    //VkBool32    shaderTessellationAndGeometryPointSize;
    //VkBool32    shaderImageGatherExtended;
    //VkBool32    shaderStorageImageExtendedFormats;
    //VkBool32    shaderStorageImageMultisample;
    //VkBool32    shaderStorageImageReadWithoutFormat;
    //VkBool32    shaderStorageImageWriteWithoutFormat;
    //VkBool32    shaderUniformBufferArrayDynamicIndexing;
    //VkBool32    shaderSampledImageArrayDynamicIndexing;
    //VkBool32    shaderStorageBufferArrayDynamicIndexing;
    //VkBool32    shaderStorageImageArrayDynamicIndexing;
    bool shaderClipDistance = false;
    bool shaderCullDistance = false;
    bool vertexAttributeDouble = false; // shaderFloat64
    //VkBool32    shaderInt64;
    //VkBool32    shaderInt16;
    //VkBool32    shaderResourceResidency;
    //VkBool32    shaderResourceMinLod;
    //VkBool32    sparseBinding;
    //VkBool32    sparseResidencyBuffer;
    //VkBool32    sparseResidencyImage2D;
    //VkBool32    sparseResidencyImage3D;
    //VkBool32    sparseResidency2Samples;
    //VkBool32    sparseResidency4Samples;
    //VkBool32    sparseResidency8Samples;
    //VkBool32    sparseResidency16Samples;
    //VkBool32    sparseResidencyAliased;
    //VkBool32    variableMultisampleRate;
    bool inheritedQueries = false;

    /* Vulkan 1.1 Core */
    //VkBool32           storageBuffer16BitAccess;
    //VkBool32           uniformAndStorageBuffer16BitAccess;
    //VkBool32           storagePushConstant16;
    //VkBool32           storageInputOutput16;
    //VkBool32           multiview;
    //VkBool32           multiviewGeometryShader;
    //VkBool32           multiviewTessellationShader;
    //VkBool32           variablePointersStorageBuffer;
    //VkBool32           variablePointers;
    //VkBool32           protectedMemory;
    //VkBool32           samplerYcbcrConversion;
    bool shaderDrawParameters = false;

    /* Vulkan 1.2 Core */
    //VkBool32           samplerMirrorClampToEdge;
    bool drawIndirectCount = false;
    //VkBool32           storageBuffer8BitAccess;
    //VkBool32           uniformAndStorageBuffer8BitAccess;
    //VkBool32           storagePushConstant8;
    //VkBool32           shaderBufferInt64Atomics;
    //VkBool32           shaderSharedInt64Atomics;
    //VkBool32           shaderFloat16;
    //VkBool32           shaderInt8;
    //VkBool32           descriptorIndexing;
    //VkBool32           shaderInputAttachmentArrayDynamicIndexing;
    //VkBool32           shaderUniformTexelBufferArrayDynamicIndexing;
    //VkBool32           shaderStorageTexelBufferArrayDynamicIndexing;
    //VkBool32           shaderUniformBufferArrayNonUniformIndexing;
    //VkBool32           shaderSampledImageArrayNonUniformIndexing;
    //VkBool32           shaderStorageBufferArrayNonUniformIndexing;
    //VkBool32           shaderStorageImageArrayNonUniformIndexing;
    //VkBool32           shaderInputAttachmentArrayNonUniformIndexing;
    //VkBool32           shaderUniformTexelBufferArrayNonUniformIndexing;
    //VkBool32           shaderStorageTexelBufferArrayNonUniformIndexing;
    //VkBool32           descriptorBindingUniformBufferUpdateAfterBind;
    //VkBool32           descriptorBindingSampledImageUpdateAfterBind;
    //VkBool32           descriptorBindingStorageImageUpdateAfterBind;
    //VkBool32           descriptorBindingStorageBufferUpdateAfterBind;
    //VkBool32           descriptorBindingUniformTexelBufferUpdateAfterBind;
    //VkBool32           descriptorBindingStorageTexelBufferUpdateAfterBind;
    //VkBool32           descriptorBindingUpdateUnusedWhilePending;
    //VkBool32           descriptorBindingPartiallyBound;
    //VkBool32           descriptorBindingVariableDescriptorCount;
    //VkBool32           runtimeDescriptorArray;
    //VkBool32           samplerFilterMinmax;
    //VkBool32           scalarBlockLayout;
    //VkBool32           imagelessFramebuffer;
    //VkBool32           uniformBufferStandardLayout;
    //VkBool32           shaderSubgroupExtendedTypes;
    //VkBool32           separateDepthStencilLayouts;
    //VkBool32           hostQueryReset;
    //VkBool32           timelineSemaphore;
    bool bufferDeviceAddress = false;
    //VkBool32           bufferDeviceAddressCaptureReplay;
    //VkBool32           bufferDeviceAddressMultiDevice;
    //VkBool32           vulkanMemoryModel;
    //VkBool32           vulkanMemoryModelDeviceScope;
    //VkBool32           vulkanMemoryModelAvailabilityVisibilityChains;
    //VkBool32           shaderOutputViewportIndex;
    //VkBool32           shaderOutputLayer;
    //VkBool32           subgroupBroadcastDynamicId;

    /* Vulkan 1.3 Core */

    //VkBool32           robustImageAccess;                 //  or VK_EXT_image_robustness
    
    //  or VK_EXT_inline_uniform_block:
    //VkBool32           inlineUniformBlock;
    //VkBool32           descriptorBindingInlineUniformBlockUpdateAfterBind;
    
    //VkBool32           pipelineCreationCacheControl;      // or VK_EXT_pipeline_creation_cache_control
    //VkBool32           privateData;                       // or VK_EXT_private_data
    //VkBool32           shaderDemoteToHelperInvocation;    // or VK_EXT_shader_demote_to_helper_invocation
    //VkBool32           shaderTerminateInvocation;         // or VK_KHR_shader_terminate_invocation
    
    // or VK_EXT_subgroup_size_control
    //VkBool32           subgroupSizeControl;
    //VkBool32           computeFullSubgroups;
    
    //VkBool32           synchronization2;                      // or VK_KHR_synchronization2
    //VkBool32           textureCompressionASTC_HDR;            // or VK_EXT_texture_compression_astc_hdr
    //VkBool32           shaderZeroInitializeWorkgroupMemory;   // or VK_KHR_zero_initialize_workgroup_memory
    //VkBool32           dynamicRendering;                      // or VK_KHR_dynamic_rendering
    //VkBool32           shaderIntegerDotProduct;               // or VK_KHR_shader_integer_dot_product
    //VkBool32           maintenance4;                          // ! Doesn't make sense to expose, too vulkan specific // or VK_KHR_maintenance4

    /* CoherentMemoryFeaturesAMD *//* VK_AMD_device_coherent_memory */
    //VkBool32           deviceCoherentMemory;

    /* VK_AMD_shader_early_and_late_fragment_tests *//* couldn't find the struct/extension in latest vulkan headers */

    /* RasterizationOrderAttachmentAccessFeaturesARM *//* VK_ARM_rasterization_order_attachment_access */
    //VkBool32           rasterizationOrderColorAttachmentAccess;
    //VkBool32           rasterizationOrderDepthAttachmentAccess;
    //VkBool32           rasterizationOrderStencilAttachmentAccess;
    
    /* 4444FormatsFeaturesEXT *//* VK_EXT_4444_formats */
    //VkBool32           formatA4R4G4B4;
    //VkBool32           formatA4B4G4R4;
    
    /* ASTCDecodeFeaturesEXT *//* VK_EXT_astc_decode_mode */
    //VkFormat           decodeMode;

    /* BlendOperationAdvancedFeaturesEXT *//* VK_EXT_blend_operation_advanced */
    //VkBool32           advancedBlendCoherentOperations;

    /* BorderColorSwizzleFeaturesEXT *//* VK_EXT_border_color_swizzle */
    //VkBool32           borderColorSwizzle;
    //VkBool32           borderColorSwizzleFromImage;

    /* VK_EXT_buffer_device_address *//* HAS KHR VERSION */

    /* ColorWriteEnableFeaturesEXT *//* VK_EXT_color_write_enable */
    //VkBool32           colorWriteEnable;

    /* ConditionalRenderingFeaturesEXT *//* VK_EXT_conditional_rendering */
    //VkBool32           conditionalRendering;
    //VkBool32           inheritedConditionalRendering;

    /* CustomBorderColorFeaturesEXT *//* VK_EXT_custom_border_color */
    //VkBool32           customBorderColors;
    //VkBool32           customBorderColorWithoutFormat;

    /* DepthClipControlFeaturesEX *//* VK_EXT_depth_clip_control */
    //VkBool32           depthClipControl;

    /* DepthClipEnableFeaturesEXT *//* VK_EXT_depth_clip_enable */
    //VkBool32           depthClipEnable;

    /* DescriptorIndexingFeatures *//* VK_EXT_descriptor_indexing *//* MOVED TO Vulkan 1.2 Core  */

    /* DeviceMemoryReportFeaturesEXT *//* VK_EXT_device_memory_report */
    //VkBool32           deviceMemoryReport;
    
    /* ExtendedDynamicStateFeaturesEXT *//* VK_EXT_extended_dynamic_state */
    //VkBool32           extendedDynamicState;

    /* ExtendedDynamicState2FeaturesEXT *//* VK_EXT_extended_dynamic_state2 */
    //VkBool32           extendedDynamicState2;
    //VkBool32           extendedDynamicState2LogicOp;
    //VkBool32           extendedDynamicState2PatchControlPoints;

    /* FragmentDensityMapFeaturesEXT *//* VK_EXT_fragment_density_map */
    //VkBool32           fragmentDensityMap;
    //VkBool32           fragmentDensityMapDynamic;
    //VkBool32           fragmentDensityMapNonSubsampledImages;

    /* FragmentDensityMap2FeaturesEXT *//* VK_EXT_fragment_density_map2 */
    //VkBool32           fragmentDensityMapDeferred;

    /* FragmentShaderInterlockFeaturesEXT *//* VK_EXT_fragment_shader_interlock */
    bool fragmentShaderSampleInterlock = false;
    bool fragmentShaderPixelInterlock = false;
    bool fragmentShaderShadingRateInterlock = false;

    /* GlobalPriorityQueryFeaturesEXT *//* VK_EXT_global_priority */
    /* GlobalPriorityQueryFeaturesKHR *//* VK_KHR_global_priority */
    //VkQueueGlobalPriorityKHR    globalPriority;

    /* GraphicsPipelineLibraryFeaturesEXT *//* VK_EXT_graphics_pipeline_library */
    //VkBool32           graphicsPipelineLibrary;

    /* HostQueryResetFeatures *//* VK_EXT_host_query_reset *//* MOVED TO Vulkan 1.2 Core */
    
    /* Image2DViewOf3DFeaturesEXT *//* VK_EXT_image_2d_view_of_3d */
    //VkBool32           image2DViewOf3D;
    //VkBool32           sampler2DViewOf3D;

    /* ImageRobustnessFeaturesEXT *//* VK_EXT_image_robustness *//* MOVED TO Vulkan 1.3 Core */

    /* ImageViewMinLodFeaturesEXT *//* VK_EXT_image_view_min_lod */
    //VkBool32           minLod;

    /* IndexTypeUint8FeaturesEXT *//* VK_EXT_index_type_uint8 */
    //VkBool32           indexTypeUint8;
    
    /* InlineUniformBlockFeaturesEXT *//* VK_EXT_inline_uniform_block *//* MOVED TO Vulkan 1.3 Core */

    /* LineRasterizationFeaturesEXT *//* VK_EXT_line_rasterization */
    //VkBool32           rectangularLines;
    //VkBool32           bresenhamLines;
    //VkBool32           smoothLines;
    //VkBool32           stippledRectangularLines;
    //VkBool32           stippledBresenhamLines;
    //VkBool32           stippledSmoothLines;

    /* MemoryPriorityFeaturesEXT *//* VK_EXT_memory_priority */
    //VkBool32           memoryPriority;

    /* MultiDrawFeaturesEXT *//* VK_EXT_multi_draw */
    //VkBool32           multiDraw;

    /* PageableDeviceLocalMemoryFeaturesEXT *//* VK_EXT_pageable_device_local_memory */
    //VkBool32           pageableDeviceLocalMemory;

    /* PipelineCreationCacheControlFeaturesEXT *//* VK_EXT_pipeline_creation_cache_control *//* MOVED TO Vulkan 1.3 Core */

    /* PrimitivesGeneratedQueryFeaturesEXT *//* VK_EXT_primitives_generated_query */
    //VkBool32           primitivesGeneratedQuery;
    //VkBool32           primitivesGeneratedQueryWithRasterizerDiscard;
    //VkBool32           primitivesGeneratedQueryWithNonZeroStreams;

    /* PrimitiveTopologyListRestartFeaturesEXT *//* VK_EXT_primitive_topology_list_restart */
    //VkBool32           primitiveTopologyListRestart;
    //VkBool32           primitiveTopologyPatchListRestart;

    /* PrivateDataFeatures *//* VK_EXT_private_data *//* MOVED TO Vulkan 1.3 Core */

    /* ProvokingVertexFeaturesEXT *//* VK_EXT_provoking_vertex */
    //VkBool32           provokingVertexLast;
    //VkBool32           transformFeedbackPreservesProvokingVertex;

    /* Robustness2FeaturesEXT *//* VK_EXT_robustness2 */
    //VkBool32           robustBufferAccess2;
    //VkBool32           robustImageAccess2;
    //VkBool32           nullDescriptor;
    
    /* ScalarBlockLayoutFeaturesEXT *//* VK_EXT_scalar_block_layout *//* MOVED TO Vulkan 1.2 Core */
    
    /* ShaderAtomicFloatFeaturesEXT *//* VK_EXT_shader_atomic_float */
    //VkBool32           shaderBufferFloat32Atomics;
    //VkBool32           shaderBufferFloat32AtomicAdd;
    //VkBool32           shaderBufferFloat64Atomics;
    //VkBool32           shaderBufferFloat64AtomicAdd;
    //VkBool32           shaderSharedFloat32Atomics;
    //VkBool32           shaderSharedFloat32AtomicAdd;
    //VkBool32           shaderSharedFloat64Atomics;
    //VkBool32           shaderSharedFloat64AtomicAdd;
    //VkBool32           shaderImageFloat32Atomics;
    //VkBool32           shaderImageFloat32AtomicAdd;
    //VkBool32           sparseImageFloat32Atomics;
    //VkBool32           sparseImageFloat32AtomicAdd;

    /* ShaderAtomicFloat2FeaturesEXT *//* VK_EXT_shader_atomic_float2 */
    //VkBool32           shaderBufferFloat16Atomics;
    //VkBool32           shaderBufferFloat16AtomicAdd;
    //VkBool32           shaderBufferFloat16AtomicMinMax;
    //VkBool32           shaderBufferFloat32AtomicMinMax;
    //VkBool32           shaderBufferFloat64AtomicMinMax;
    //VkBool32           shaderSharedFloat16Atomics;
    //VkBool32           shaderSharedFloat16AtomicAdd;
    //VkBool32           shaderSharedFloat16AtomicMinMax;
    //VkBool32           shaderSharedFloat32AtomicMinMax;
    //VkBool32           shaderSharedFloat64AtomicMinMax;
    //VkBool32           shaderImageFloat32AtomicMinMax;
    //VkBool32           sparseImageFloat32AtomicMinMax;
    
    /* DemoteToHelperInvocationFeaturesEXT *//* VK_EXT_shader_demote_to_helper_invocation *//* MOVED TO Vulkan 1.3 Core */

    /* ShaderImageAtomicInt64FeaturesEXT *//* VK_EXT_shader_image_atomic_int64 */
    //VkBool32           shaderImageInt64Atomics;
    //VkBool32           sparseImageInt64Atomics;

    /* SubgroupSizeControlFeaturesEXT *//* VK_EXT_subgroup_size_control *//* MOVED TO Vulkan 1.3 Core */

    /* TexelBufferAlignmentFeaturesEXT *//* VK_EXT_texel_buffer_alignment */
    //VkBool32           texelBufferAlignment;

    /* TextureCompressionASTCHDRFeaturesEXT *//* VK_EXT_texture_compression_astc_hdr *//* MOVED TO Vulkan 1.3 Core */

    /* TransformFeedbackFeaturesEXT *//* VK_EXT_transform_feedback */
    //VkBool32           transformFeedback;
    //VkBool32           geometryStreams;

    /* VertexAttributeDivisorFeaturesEXT *//* VK_EXT_vertex_attribute_divisor */
    //VkBool32           vertexAttributeInstanceRateDivisor;
    //VkBool32           vertexAttributeInstanceRateZeroDivisor;

    /* VertexInputDynamicStateFeaturesEXT *//* VK_EXT_vertex_input_dynamic_state */
    //VkBool32           vertexInputDynamicState;

    /* Ycbcr2Plane444FormatsFeaturesEXT *//* VK_EXT_ycbcr_2plane_444_formats */
    //VkBool32           ycbcr2plane444Formats;

    /* YcbcrImageArraysFeaturesEXT *//* VK_EXT_ycbcr_image_arrays */
    //VkBool32           ycbcrImageArrays;

    /* ShaderIntegerFunctions2FeaturesINTEL *//* VK_INTEL_shader_integer_functions2 */
    //VkBool32           shaderIntegerFunctions2;

    /* 16BitStorageFeaturesKHR *//* VK_KHR_16bit_storage *//* MOVED TO Vulkan 1.1 Core */
    /* 8BitStorageFeaturesKHR *//* VK_KHR_8bit_storage *//* MOVED TO Vulkan 1.2 Core */

    /* AccelerationStructureFeaturesKHR *//* VK_KHR_acceleration_structure */
    bool accelerationStructure = false;
    bool accelerationStructureCaptureReplay = false;
    bool accelerationStructureIndirectBuild = false;
    bool accelerationStructureHostCommands = false;
    bool descriptorBindingAccelerationStructureUpdateAfterBind = false;
            
    /* BufferDeviceAddressFeaturesKHR *//* VK_KHR_buffer_device_address *//* MOVED TO Vulkan 1.2 Core */

    /* DynamicRenderingFeaturesKHR *//* VK_KHR_dynamic_rendering *//* MOVED TO Vulkan 1.3 Core */

    /* [!!NV Version below, struct doesn't exist in vk headers] VK_KHR_fragment_shader_barycentric */

    /* FragmentShadingRateFeaturesKHR *//* VK_KHR_fragment_shading_rate */
    //VkBool32           pipelineFragmentShadingRate;
    //VkBool32           primitiveFragmentShadingRate;
    //VkBool32           attachmentFragmentShadingRate;
    
    /* GlobalPriorityQueryFeaturesKHR *//* VK_KHR_global_priority */
    //VkBool32           globalPriorityQuery;

    /* ImagelessFramebufferFeaturesKHR *//* VK_KHR_imageless_framebuffer *//* MOVED TO Vulkan 1.2 Core */
    /* Maintenance4FeaturesKHR *//* VK_KHR_maintenance4 *//* MOVED TO Vulkan 1.3 Core */
    /* MultiviewFeaturesKHR *//* VK_KHR_multiview *//* MOVED TO Vulkan 1.1 Core */

    /* PerformanceQueryFeaturesKHR *//* VK_KHR_performance_query */
    //VkBool32           performanceCounterQueryPools;
    //VkBool32           performanceCounterMultipleQueryPools;

    /* PipelineExecutablePropertiesFeaturesKHR *//* VK_KHR_pipeline_executable_properties */
    //VkBool32           pipelineExecutableInfo;
    
    /* VK_KHR_portability_subset - PROVISINAL/NOTAVAILABLEANYMORE */

    /* PresentIdFeaturesKHR *//* VK_KHR_present_id */
    //VkBool32           presentId;

    /* PresentWaitFeaturesKHR *//* VK_KHR_present_wait */
    //VkBool32           presentWait;
    
    /* RayQueryFeaturesKHR *//* VK_KHR_ray_query */
    bool rayQuery = false;

    /* VK_KHR_ray_tracing !! Replaced/Removed */
    /* VK_KHR_ray_tracing_maintenance1 *//* added in vk 1.3.213, the SDK isn't released yet at this moment :D */

    /* RayTracingPipelineFeaturesKHR *//* VK_KHR_ray_tracing_pipeline */
    bool rayTracingPipeline = false;
    bool rayTracingPipelineShaderGroupHandleCaptureReplay = false;
    bool rayTracingPipelineShaderGroupHandleCaptureReplayMixed = false;
    bool rayTracingPipelineTraceRaysIndirect = false;
    bool rayTraversalPrimitiveCulling = false;

    /* SamplerYcbcrConversionFeaturesKHR *//* VK_KHR_sampler_ycbcr_conversion *//* MOVED TO Vulkan 1.1 Core */
    /* SeparateDepthStencilLayoutsFeaturesKHR *//* VK_KHR_separate_depth_stencil_layouts *//* MOVED TO Vulkan 1.2 Core */
    /* ShaderAtomicInt64FeaturesKHR *//* VK_KHR_shader_atomic_int64 *//* MOVED TO Vulkan 1.2 Core */

    /* ShaderClockFeaturesKHR *//* VK_KHR_shader_clock */
    //VkBool32           shaderSubgroupClock;
    //VkBool32           shaderDeviceClock;

    /* VK_KHR_shader_draw_parameters *//* MOVED TO Vulkan 1.1 Core */
    /* VK_KHR_shader_float16_int8 *//* MOVED TO Vulkan 1.2 Core */
    /* VK_KHR_shader_integer_dot_product *//* MOVED TO Vulkan 1.3 Core */
    /* VK_KHR_shader_subgroup_extended_types *//* MOVED TO Vulkan 1.2 Core */

    /* ShaderSubgroupUniformControlFlowFeaturesKHR *//* VK_KHR_shader_subgroup_uniform_control_flow */
    //VkBool32           shaderSubgroupUniformControlFlow;
    
    /* VK_KHR_shader_terminate_invocation *//* MOVED TO Vulkan 1.3 Core */
    /* VK_KHR_synchronization2 *//* MOVED TO Vulkan 1.3 Core */
    /* VK_KHR_timeline_semaphore *//* MOVED TO Vulkan 1.2 Core */
    /* VK_KHR_uniform_buffer_standard_layout *//* MOVED TO Vulkan 1.2 Core */
    /* VK_KHR_variable_pointers *//* MOVED TO Vulkan 1.1 Core */
    /* VK_KHR_vulkan_memory_model *//* MOVED TO Vulkan 1.2 Core */

    /* WorkgroupMemoryExplicitLayoutFeaturesKHR *//* VK_KHR_workgroup_memory_explicit_layout */
    //VkBool32           workgroupMemoryExplicitLayout;
    //VkBool32           workgroupMemoryExplicitLayoutScalarBlockLayout;
    //VkBool32           workgroupMemoryExplicitLayout8BitAccess;
    //VkBool32           workgroupMemoryExplicitLayout16BitAccess;
    
    /* VK_KHR_zero_initialize_workgroup_memory *//* MOVED TO Vulkan 1.3 Core */
    /* VK_KHX_multiview *//* see VK_KHR_multiview *//* MOVED TO Vulkan 1.1 Core */

    /* ComputeShaderDerivativesFeaturesNV *//* VK_NV_compute_shader_derivatives */
    //VkBool32           computeDerivativeGroupQuads;
    //VkBool32           computeDerivativeGroupLinear;

    /* CooperativeMatrixFeaturesNV *//* VK_NV_cooperative_matrix */
    //VkBool32           cooperativeMatrix;
    //VkBool32           cooperativeMatrixRobustBufferAccess;

    /* CornerSampledImageFeaturesNV *//* VK_NV_corner_sampled_image */
    //VkBool32           cornerSampledImage;

    /* CoverageReductionModeFeaturesNV *//* VK_NV_coverage_reduction_mode */
    //VkCoverageReductionModeNV                        coverageReductionMode;

    /* DedicatedAllocationImageAliasingFeaturesNV *//* VK_NV_dedicated_allocation_image_aliasing */
    //VkBool32           dedicatedAllocationImageAliasing;

    /* DiagnosticsConfigFeaturesNV *//* VK_NV_device_diagnostics_config */
    //VkBool32           diagnosticsConfig;

    /* DeviceGeneratedCommandsFeaturesNV *//* VK_NV_device_generated_commands */
    //VkBool32           deviceGeneratedCommands;

    /* ExternalMemoryRDMAFeaturesNV *//* VK_NV_external_memory_rdma */
    //VkBool32           externalMemoryRDMA;
    
    /* FragmentShaderBarycentricFeaturesNV *//* VK_NV_fragment_shader_barycentric */
    //VkBool32           fragmentShaderBarycentric;

    /* FragmentShadingRateEnumsFeaturesNV *//* VK_NV_fragment_shading_rate_enums */
    //VkBool32           fragmentShadingRateEnums;
    //VkBool32           supersampleFragmentShadingRates;
    //VkBool32           noInvocationFragmentShadingRates;

    /* InheritedViewportScissorFeaturesNV *//* VK_NV_inherited_viewport_scissor */
    //VkBool32           inheritedViewportScissor2D;

    /* LinearColorAttachmentFeaturesNV *//* VK_NV_linear_color_attachment */
    //VkBool32           linearColorAttachment;
    
    /* MeshShaderFeaturesNV *//* VK_NV_mesh_shader */
    //VkBool32           taskShader;
    //VkBool32           meshShader;
    
    /* RayTracingMotionBlurFeaturesNV *//* VK_NV_ray_tracing_motion_blur */
    //VkBool32           rayTracingMotionBlur;
    //VkBool32           rayTracingMotionBlurPipelineTraceRaysIndirect;

    /* RepresentativeFragmentTestFeaturesNV *//* VK_NV_representative_fragment_test */
    //VkBool32           representativeFragmentTest;
    
    /* ExclusiveScissorFeaturesNV *//* VK_NV_scissor_exclusive */
    //VkBool32           exclusiveScissor;

    /* ShaderImageFootprintFeaturesNV *//* VK_NV_shader_image_footprint */
    //VkBool32           imageFootprint;

    /* ShaderSMBuiltinsFeaturesNV *//* VK_NV_shader_sm_builtins */
    //VkBool32           shaderSMBuiltins;
    
    /* ShadingRateImageFeaturesNV *//* VK_NV_shading_rate_image */
    //VkBool32           shadingRateImage;
    //VkBool32           shadingRateCoarseSampleOrder;

    /* FragmentDensityMapOffsetFeaturesQCOM *//* VK_QCOM_fragment_density_map_offset */
    //VkBool32           fragmentDensityMapOffset;

    /* DescriptorSetHostMappingFeaturesVALVE *//* VK_VALVE_descriptor_set_host_mapping */
    //VkBool32           descriptorSetHostMapping;

    /* MutableDescriptorTypeFeaturesVALVE *//* VK_VALVE_mutable_descriptor_type */
    //VkBool32           mutableDescriptorType;

    /* Nabla */
    bool dispatchBase = false; // true in Vk, false in GL
    bool allowCommandBufferQueryCopies = false;
};

} // nbl::video

#endif
