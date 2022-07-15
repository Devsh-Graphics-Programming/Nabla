#ifndef _NBL_VIDEO_S_PHYSICAL_DEVICE_FEATURES_H_INCLUDED_
#define _NBL_VIDEO_S_PHYSICAL_DEVICE_FEATURES_H_INCLUDED_

namespace nbl::video
{
    
enum E_SWAPCHAIN_MODE : uint32_t
{
    ESM_NONE = 0,
    ESM_SURFACE = 0x01,
    // ESM_DISPLAY = 0x02 TODO, as we won't write the API interfaces to deal with direct-to-display swapchains yet.,
    /* TODO: KHR_swapchain if SURFACE or DISPLAY flag present & KHR_display_swapchain if DISPLAY flag present */
};

struct SPhysicalDeviceFeatures
{
    /* Vulkan 1.0 Core  */
    bool robustBufferAccess = false;
    bool fullDrawIndexUint32 = false;
    bool imageCubeArray = false;
    bool independentBlend = false;
    bool geometryShader    = false;
    bool tessellationShader = false;
    bool sampleRateShading = false;
    bool dualSrcBlend = false;
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

    // [DO NOT EXPOSE] these 3 don't make a difference, just shortcut from Querying support from PhysicalDevice
    //bool    textureCompressionETC2;
    //bool    textureCompressionASTC_LDR;
    //bool    textureCompressionBC;
    
    bool occlusionQueryPrecise = false;
    //bool    pipelineStatisticsQuery; [TODO]

    // [TODO] Always enable ones below, report as limit
    bool vertexPipelineStoresAndAtomics = false;
    bool fragmentStoresAndAtomics = false;
    bool shaderTessellationAndGeometryPointSize = false;
    bool shaderImageGatherExtended = false;

    bool shaderStorageImageExtendedFormats = false;
    bool shaderStorageImageMultisample = false;
    bool shaderStorageImageReadWithoutFormat = false;
    bool shaderStorageImageWriteWithoutFormat = false;
    bool shaderUniformBufferArrayDynamicIndexing = false;
    bool shaderSampledImageArrayDynamicIndexing = false;
    bool shaderStorageBufferArrayDynamicIndexing = false;
    bool shaderStorageImageArrayDynamicIndexing = false;
    bool shaderClipDistance = false;
    bool shaderCullDistance = false;
    bool vertexAttributeDouble = false; // shaderFloat64

    // [TODO] Always enable ones below, report as limit
    bool shaderInt64 = false;
    bool shaderInt16 = false;

    bool shaderResourceResidency = false;
    bool shaderResourceMinLod = false;
    
    // [TODO] cause we haven't implemented sparse resources yet
    //bool    sparseBinding;
    //bool    sparseResidencyBuffer;
    //bool    sparseResidencyImage2D;
    //bool    sparseResidencyImage3D;
    //bool    sparseResidency2Samples;
    //bool    sparseResidency4Samples;
    //bool    sparseResidency8Samples;
    //bool    sparseResidency16Samples;
    //bool    sparseResidencyAliased;
    
    bool variableMultisampleRate = false;
    bool inheritedQueries = false;

    /* Vulkan 1.1 Core */

    // [TODO] Always enable ones below, report as limit
    bool storageBuffer16BitAccess = false;
    bool uniformAndStorageBuffer16BitAccess = false;
    bool storagePushConstant16 = false;
    bool storageInputOutput16 = false;
    
    // [TODO] do not expose multiview yet
    //bool           multiview;
    //bool           multiviewGeometryShader;
    //bool           multiviewTessellationShader;
    
    // [Future TODO]:
    //bool           variablePointersStorageBuffer;
    //bool           variablePointers;
    
    //bool           protectedMemory; // [DO NOT EXPOSE] not gonna expose until we have a need to
 
    // [DO NOT EXPOSE] Enables certain formats in Vulkan, we just enable them if available or else we need to make format support query functions in LogicalDevice as well
    //bool           samplerYcbcrConversion;
    bool shaderDrawParameters = false;




    /* Vulkan 1.2 Core */

    bool samplerMirrorClampToEdge = false;          // ALIAS: VK_KHR_sampler_mirror_clamp_to_edge
    bool drawIndirectCount = false;                 // ALIAS: VK_KHR_draw_indirect_count

    // [TODO] Always enable VK_KHR_8bit_storage, VK_KHR_shader_atomic_int64 & VK_KHR_shader_float16_int8; report as limit
    // or VK_KHR_8bit_storage:
    bool storageBuffer8BitAccess = false;
    bool uniformAndStorageBuffer8BitAccess = false;
    bool storagePushConstant8 = false;
    
    // or VK_KHR_shader_atomic_int64:
    bool shaderBufferInt64Atomics = false;
    bool shaderSharedInt64Atomics = false;
   
    // or VK_KHR_shader_float16_int8:
    bool shaderFloat16 = false;
    bool shaderInt8 = false;
    
    // or VK_EXT_descriptor_indexing
    bool descriptorIndexing = false;
    bool shaderInputAttachmentArrayDynamicIndexing = false;
    bool shaderUniformTexelBufferArrayDynamicIndexing = false;
    bool shaderStorageTexelBufferArrayDynamicIndexing = false;
    bool shaderUniformBufferArrayNonUniformIndexing = false;
    bool shaderSampledImageArrayNonUniformIndexing = false;
    bool shaderStorageBufferArrayNonUniformIndexing = false;
    bool shaderStorageImageArrayNonUniformIndexing = false;
    bool shaderInputAttachmentArrayNonUniformIndexing = false;
    bool shaderUniformTexelBufferArrayNonUniformIndexing = false;
    bool shaderStorageTexelBufferArrayNonUniformIndexing = false;
    bool descriptorBindingUniformBufferUpdateAfterBind = false;
    bool descriptorBindingSampledImageUpdateAfterBind = false;
    bool descriptorBindingStorageImageUpdateAfterBind = false;
    bool descriptorBindingStorageBufferUpdateAfterBind = false;
    bool descriptorBindingUniformTexelBufferUpdateAfterBind = false;
    bool descriptorBindingStorageTexelBufferUpdateAfterBind = false;
    bool descriptorBindingUpdateUnusedWhilePending = false;
    bool descriptorBindingPartiallyBound = false;
    bool descriptorBindingVariableDescriptorCount = false;
    bool runtimeDescriptorArray = false;
    
    bool samplerFilterMinmax = false;   // ALIAS: VK_EXT_sampler_filter_minmax
    
    bool scalarBlockLayout = false;     // or VK_EXT_scalar_block_layout
    
    //bool           imagelessFramebuffer;  // or VK_KHR_imageless_framebuffer // [FUTURE TODO]
    
    bool           uniformBufferStandardLayout = false;   // or VK_KHR_uniform_buffer_standard_layout
    
    bool shaderSubgroupExtendedTypes = false;   // or VK_KHR_shader_subgroup_extended_types
    
    bool           separateDepthStencilLayouts = false;   // or VK_KHR_separate_depth_stencil_layouts
    
    // [TODO] And add implementation to engine
    //bool           hostQueryReset;                // or VK_EXT_host_query_reset
    
    //bool           timelineSemaphore;             // or VK_KHR_timeline_semaphore // [FUTURE TODO] won't expose for a long time
    
    // or VK_KHR_buffer_device_address:
    bool bufferDeviceAddress = false;
    // bool           bufferDeviceAddressCaptureReplay; // [DO NOT EXPOSE] for capture tools not engines
    bool           bufferDeviceAddressMultiDevice = false;
    
    // or VK_KHR_vulkan_memory_model
    bool           vulkanMemoryModel = false;
    bool           vulkanMemoryModelDeviceScope = false;
    bool           vulkanMemoryModelAvailabilityVisibilityChains = false;
   
    bool           subgroupBroadcastDynamicId = false;    // if Vulkan 1.2 is supported




    /* Vulkan 1.3 Core */
    
    // [TODO] robustness stuff
    //bool           robustImageAccess;                 //  or VK_EXT_image_robustness
    
    // [DO NOT EXPOSE]
    //  or VK_EXT_inline_uniform_block:
    //bool           inlineUniformBlock;
    //bool           descriptorBindingInlineUniformBlockUpdateAfterBind;
    
    // [TODO]
    //bool           pipelineCreationCacheControl;      // or VK_EXT_pipeline_creation_cache_control
   
    // [DO NOT EXPOSE] ever
    //bool           privateData;                       // or VK_EXT_private_data
    
    bool           shaderDemoteToHelperInvocation = false;    // or VK_EXT_shader_demote_to_helper_invocation
    bool           shaderTerminateInvocation = false;         // or VK_KHR_shader_terminate_invocation
    
    // or VK_EXT_subgroup_size_control
    bool subgroupSizeControl  = false;
    bool computeFullSubgroups = false;
    
    // [DO NOT EXPOSE] and false because we havent rewritten our frontend API for that: https://github.com/Devsh-Graphics-Programming/Nabla/issues/384
    //bool           synchronization2;                      // or VK_KHR_synchronization2
    
    // [DO NOT EXPOSE] Doesn't make a difference, just shortcut from Querying support from PhysicalDevice
    //bool           textureCompressionASTC_HDR;            // or VK_EXT_texture_compression_astc_hdr
    
    // [DO NOT EXPOSE] would require doing research into the GL/GLES robustness extensions
    //bool           shaderZeroInitializeWorkgroupMemory;   // or VK_KHR_zero_initialize_workgroup_memory
    
    // [DO NOT EXPOSE] EVIL
    //bool           dynamicRendering;                      // or VK_KHR_dynamic_rendering
    
    bool           shaderIntegerDotProduct = false;               // or VK_KHR_shader_integer_dot_product
    //bool           maintenance4;                          // [DO NOT EXPOSE] doesn't make sense




    /* Vulkan Extensions */

    // [TODO] no such thing on GL, but trivial to implement.
    /* CoherentMemoryFeaturesAMD *//* VK_AMD_device_coherent_memory */
    //bool           deviceCoherentMemory;

    // [TODO] this one isn't in the headers?
    /* VK_AMD_shader_early_and_late_fragment_tests */
    //bool shaderEarlyAndLateFragmentTests;

    /* RasterizationOrderAttachmentAccessFeaturesARM *//* VK_ARM_rasterization_order_attachment_access */
    bool rasterizationOrderColorAttachmentAccess;
    bool rasterizationOrderDepthAttachmentAccess;
    bool rasterizationOrderStencilAttachmentAccess;
    
    // [DO NOT EXPOSE] Enables certain formats in Vulkan, we just enable them if available or else we need to make format support query functions in LogicalDevice as well
    /* 4444FormatsFeaturesEXT *//* VK_EXT_4444_formats */
    
    // [TODO]
    /* ASTCDecodeFeaturesEXT *//* VK_EXT_astc_decode_mode */
    //VkFormat           decodeMode;

    // [DO NOT EXPOSE] right now, no idea if we'll ever expose and implement those but they'd all be false for OpenGL
    /* BlendOperationAdvancedFeaturesEXT *//* VK_EXT_blend_operation_advanced */
    //bool           advancedBlendCoherentOperations;
    
    // [DO NOT EXPOSE] not going to expose custom border colors for now
    /* BorderColorSwizzleFeaturesEXT *//* VK_EXT_border_color_swizzle */
    //bool           borderColorSwizzle;
    //bool           borderColorSwizzleFromImage;

    /* VK_EXT_buffer_device_address *//* HAS KHR VERSION */

    // [TODO] would need new commandbuffer methods, etc
    /* ColorWriteEnableFeaturesEXT *//* VK_EXT_color_write_enable */
    //bool           colorWriteEnable;

    // [TODO] would need API to deal with queries and begin/end conditional blocks
    /* ConditionalRenderingFeaturesEXT *//* VK_EXT_conditional_rendering */
    //bool           conditionalRendering;
    //bool           inheritedConditionalRendering;
    
    // [DO NOT EXPOSE] not going to expose custom border colors for now
    /* CustomBorderColorFeaturesEXT *//* VK_EXT_custom_border_color */
    //bool           customBorderColors;
    //bool           customBorderColorWithoutFormat;

    // [DO NOT EXPOSE] EVER, VULKAN DEPTH RANGE ONLY!
    /* DepthClipControlFeaturesEX *//* VK_EXT_depth_clip_control */
    //bool           depthClipControl;

    // [DO NOT EXPOSE] only useful for D3D emulators
    /* DepthClipEnableFeaturesEXT *//* VK_EXT_depth_clip_enable */
    //bool           depthClipEnable;

    /* DescriptorIndexingFeatures *//* VK_EXT_descriptor_indexing *//* MOVED TO Vulkan 1.2 Core  */

    // [TODO]
    /* DeviceMemoryReportFeaturesEXT *//* VK_EXT_device_memory_report */
    //bool           deviceMemoryReport;
    
    // [DO NOT EXPOSE] we're fully GPU driven anyway with sorted pipelines.
    /* ExtendedDynamicStateFeaturesEXT *//* VK_EXT_extended_dynamic_state */
    //bool           extendedDynamicState;
    
    // [DO NOT EXPOSE] we're fully GPU driven anyway with sorted pipelines.
    /* ExtendedDynamicState2FeaturesEXT *//* VK_EXT_extended_dynamic_state2 */
    //bool           extendedDynamicState2;
    //bool           extendedDynamicState2LogicOp;
    //bool           extendedDynamicState2PatchControlPoints;

    // [TODO]
    /* FragmentDensityMapFeaturesEXT *//* VK_EXT_fragment_density_map */
    //bool           fragmentDensityMap;
    //bool           fragmentDensityMapDynamic;
    //bool           fragmentDensityMapNonSubsampledImages;
    
    // [TODO]
    /* FragmentDensityMap2FeaturesEXT *//* VK_EXT_fragment_density_map2 */
    //bool           fragmentDensityMapDeferred;

    /* FragmentShaderInterlockFeaturesEXT *//* VK_EXT_fragment_shader_interlock */
    bool fragmentShaderSampleInterlock = false;
    bool fragmentShaderPixelInterlock = false;
    bool fragmentShaderShadingRateInterlock = false;

    // [TODO]
    /* GlobalPriorityQueryFeaturesEXT *//* VK_EXT_global_priority */
    /* GlobalPriorityQueryFeaturesKHR *//* VK_KHR_global_priority */
    //VkQueueGlobalPriorityKHR    globalPriority;

    // [TODO] too much effort
    /* GraphicsPipelineLibraryFeaturesEXT *//* VK_EXT_graphics_pipeline_library */
    //bool           graphicsPipelineLibrary;

    /* HostQueryResetFeatures *//* VK_EXT_host_query_reset *//* MOVED TO Vulkan 1.2 Core */
    
    // [TODO] Investigate later
    /* Image2DViewOf3DFeaturesEXT *//* VK_EXT_image_2d_view_of_3d */
    //bool           image2DViewOf3D;
    //bool           sampler2DViewOf3D;

    /* ImageRobustnessFeaturesEXT *//* VK_EXT_image_robustness *//* MOVED TO Vulkan 1.3 Core */
    
    // [DO NOT EXPOSE] pointless to implement currently
    /* ImageViewMinLodFeaturesEXT *//* VK_EXT_image_view_min_lod */
    //bool           minLod;

    /* IndexTypeUint8FeaturesEXT *//* VK_EXT_index_type_uint8 */
    bool           indexTypeUint8 = false;
    
    /* InlineUniformBlockFeaturesEXT *//* VK_EXT_inline_uniform_block *//* MOVED TO Vulkan 1.3 Core */

    // [TODO] this feature introduces new/more pipeline state with VkPipelineRasterizationLineStateCreateInfoEXT
    /* LineRasterizationFeaturesEXT *//* VK_EXT_line_rasterization */
    // GL HINT (remove when implemented): MULTI_SAMPLE_LINE_WIDTH_RANGE (which is necessary for this) is guarded by !IsGLES || Version>=320 no idea is something enables this or not
    //bool           rectangularLines;
    //bool           bresenhamLines;
    //bool           smoothLines;
    // GL HINT (remove when implemented): !IsGLES for all stipples
    //bool           stippledRectangularLines;
    //bool           stippledBresenhamLines;
    //bool           stippledSmoothLines;

    // [TODO] trivial to implement later, false on GL
    /* MemoryPriorityFeaturesEXT *//* VK_EXT_memory_priority */
    //bool           memoryPriority;

    // [DO NOT EXPOSE] this extension is dumb, if we're recording that many draws we will be using Multi Draw INDIRECT which is better supported
    /* MultiDrawFeaturesEXT *//* VK_EXT_multi_draw */
    //bool           multiDraw;

    // [DO NOT EXPOSE] pointless to expose without exposing VK_EXT_memory_priority and the memory query feature first
    /* PageableDeviceLocalMemoryFeaturesEXT *//* VK_EXT_pageable_device_local_memory */
    //bool           pageableDeviceLocalMemory;

    /* PipelineCreationCacheControlFeaturesEXT *//* VK_EXT_pipeline_creation_cache_control *//* MOVED TO Vulkan 1.3 Core */

    // [DO NOT EXPOSE] requires and relates to EXT_transform_feedback which we'll never expose
    /* PrimitivesGeneratedQueryFeaturesEXT *//* VK_EXT_primitives_generated_query */
    //bool           primitivesGeneratedQuery;
    //bool           primitivesGeneratedQueryWithRasterizerDiscard;
    //bool           primitivesGeneratedQueryWithNonZeroStreams;
    
    // [DO NOT EXPOSE]
    /* PrimitiveTopologyListRestartFeaturesEXT *//* VK_EXT_primitive_topology_list_restart */
    //bool           primitiveTopologyListRestart;
    //bool           primitiveTopologyPatchListRestart;

    /* PrivateDataFeatures *//* VK_EXT_private_data *//* MOVED TO Vulkan 1.3 Core */

    // [DO NOT EXPOSE] provokingVertexLast will not expose (we always use First Vertex Vulkan-like convention), anything to do with XForm-feedback we don't expose
    /* ProvokingVertexFeaturesEXT *//* VK_EXT_provoking_vertex */
    //bool           provokingVertexLast;
    //bool           transformFeedbackPreservesProvokingVertex;

    // [TODO]
    /* Robustness2FeaturesEXT *//* VK_EXT_robustness2 */
    //bool           robustBufferAccess2;
    //bool           robustImageAccess2;
    //bool           nullDescriptor;
    
    /* ScalarBlockLayoutFeaturesEXT *//* VK_EXT_scalar_block_layout *//* MOVED TO Vulkan 1.2 Core */
    
    /* ShaderAtomicFloatFeaturesEXT *//* VK_EXT_shader_atomic_float */
    bool           shaderBufferFloat32Atomics = false;
    bool           shaderBufferFloat32AtomicAdd = false;
    bool           shaderBufferFloat64Atomics = false;
    bool           shaderBufferFloat64AtomicAdd = false;
    bool           shaderSharedFloat32Atomics = false;
    bool           shaderSharedFloat32AtomicAdd = false;
    bool           shaderSharedFloat64Atomics = false;
    bool           shaderSharedFloat64AtomicAdd = false;
    bool           shaderImageFloat32Atomics = false;
    bool           shaderImageFloat32AtomicAdd = false;
    bool           sparseImageFloat32Atomics = false;
    bool           sparseImageFloat32AtomicAdd = false;

    /* ShaderAtomicFloat2FeaturesEXT *//* VK_EXT_shader_atomic_float2 */
    bool           shaderBufferFloat16Atomics = false;
    bool           shaderBufferFloat16AtomicAdd = false;
    bool           shaderBufferFloat16AtomicMinMax = false;
    bool           shaderBufferFloat32AtomicMinMax = false;
    bool           shaderBufferFloat64AtomicMinMax = false;
    bool           shaderSharedFloat16Atomics = false;
    bool           shaderSharedFloat16AtomicAdd = false;
    bool           shaderSharedFloat16AtomicMinMax = false;
    bool           shaderSharedFloat32AtomicMinMax = false;
    bool           shaderSharedFloat64AtomicMinMax = false;
    bool           shaderImageFloat32AtomicMinMax = false;
    bool           sparseImageFloat32AtomicMinMax = false;
    
    /* DemoteToHelperInvocationFeaturesEXT *//* VK_EXT_shader_demote_to_helper_invocation *//* MOVED TO Vulkan 1.3 Core */

    /* ShaderImageAtomicInt64FeaturesEXT *//* VK_EXT_shader_image_atomic_int64 */
    bool           shaderImageInt64Atomics = false;
    bool           sparseImageInt64Atomics = false;

    /* SubgroupSizeControlFeaturesEXT *//* VK_EXT_subgroup_size_control *//* MOVED TO Vulkan 1.3 Core */

    // [DO NOT EXPOSE] always enable if we can
    /* TexelBufferAlignmentFeaturesEXT *//* VK_EXT_texel_buffer_alignment */
    //bool           texelBufferAlignment;

    /* TextureCompressionASTCHDRFeaturesEXT *//* VK_EXT_texture_compression_astc_hdr *//* MOVED TO Vulkan 1.3 Core */

    // [DO NOT EXPOSE] ever because of our disdain for XForm feedback
    /* TransformFeedbackFeaturesEXT *//* VK_EXT_transform_feedback */
    //bool           transformFeedback;
    //bool           geometryStreams;

    // [TODO] we would have to change the API
    /* VertexAttributeDivisorFeaturesEXT *//* VK_EXT_vertex_attribute_divisor */
    //bool           vertexAttributeInstanceRateDivisor;
    //bool           vertexAttributeInstanceRateZeroDivisor;

    // [DO NOT EXPOSE] too much API Fudjery
    /* VertexInputDynamicStateFeaturesEXT *//* VK_EXT_vertex_input_dynamic_state */
    //bool           vertexInputDynamicState;

    // [DO NOT EXPOSE] Enables certain formats in Vulkan, we just enable them if available or else we need to make format support query functions in LogicalDevice as well
    /* Ycbcr2Plane444FormatsFeaturesEXT *//* VK_EXT_ycbcr_2plane_444_formats */

    // [DO NOT EXPOSE] Expose nothing to do with video atm
    /* YcbcrImageArraysFeaturesEXT *//* VK_EXT_ycbcr_image_arrays */
    //bool           ycbcrImageArrays;

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
    
    // [DO NOT EXPOSE] not implementing or exposing VRS in near or far future
    /* FragmentShadingRateFeaturesKHR *//* VK_KHR_fragment_shading_rate */
    //bool           pipelineFragmentShadingRate;
    //bool           primitiveFragmentShadingRate;
    //bool           attachmentFragmentShadingRate;

    /* ImagelessFramebufferFeaturesKHR *//* VK_KHR_imageless_framebuffer *//* MOVED TO Vulkan 1.2 Core */
    /* Maintenance4FeaturesKHR *//* VK_KHR_maintenance4 *//* MOVED TO Vulkan 1.3 Core */
    /* MultiviewFeaturesKHR *//* VK_KHR_multiview *//* MOVED TO Vulkan 1.1 Core */
    
    // [TODO]
    /* PerformanceQueryFeaturesKHR *//* VK_KHR_performance_query */
    //bool           performanceCounterQueryPools;
    //bool           performanceCounterMultipleQueryPools;

    // [TODO]
    /* PipelineExecutablePropertiesFeaturesKHR *//* VK_KHR_pipeline_executable_properties */
    //bool           pipelineExecutableInfo;
    
    /* VK_KHR_portability_subset - PROVISINAL/NOTAVAILABLEANYMORE */

    // [DO NOT EXPOSE] no point exposing until an extension more useful than VK_KHR_present_wait arrives
    /* PresentIdFeaturesKHR *//* VK_KHR_present_id */
    //bool           presentId;
    
    // [DO NOT EXPOSE] won't expose, this extension is poop, I should have a Fence-andQuery-like object to query the presentation timestamp, not a blocking call that may unblock after an arbitrary delay from the present
    /* PresentWaitFeaturesKHR *//* VK_KHR_present_wait */
    //bool           presentWait;
    
    /* RayQueryFeaturesKHR *//* VK_KHR_ray_query */
    bool rayQuery = false;

    /* VK_KHR_ray_tracing !! Replaced/Removed */
    /* VK_KHR_ray_tracing_maintenance1 *//* added in vk 1.3.213, the SDK isn't released yet at this moment :D */

    /* RayTracingPipelineFeaturesKHR *//* VK_KHR_ray_tracing_pipeline */
    bool rayTracingPipeline = false;
    // bool rayTracingPipelineShaderGroupHandleCaptureReplay; // [DO NOT EXPOSE] for capture tools
    // bool rayTracingPipelineShaderGroupHandleCaptureReplayMixed; // [DO NOT EXPOSE] for capture tools
    bool rayTracingPipelineTraceRaysIndirect = false;
    bool rayTraversalPrimitiveCulling = false;

    /* SamplerYcbcrConversionFeaturesKHR *//* VK_KHR_sampler_ycbcr_conversion *//* MOVED TO Vulkan 1.1 Core */
    /* SeparateDepthStencilLayoutsFeaturesKHR *//* VK_KHR_separate_depth_stencil_layouts *//* MOVED TO Vulkan 1.2 Core */
    /* ShaderAtomicInt64FeaturesKHR *//* VK_KHR_shader_atomic_int64 *//* MOVED TO Vulkan 1.2 Core */

    /* ShaderClockFeaturesKHR *//* VK_KHR_shader_clock */
    bool           shaderDeviceClock = false;

    /* VK_KHR_shader_draw_parameters *//* MOVED TO Vulkan 1.1 Core */
    /* VK_KHR_shader_float16_int8 *//* MOVED TO Vulkan 1.2 Core */
    /* VK_KHR_shader_integer_dot_product *//* MOVED TO Vulkan 1.3 Core */
    /* VK_KHR_shader_subgroup_extended_types *//* MOVED TO Vulkan 1.2 Core */

    /* ShaderSubgroupUniformControlFlowFeaturesKHR *//* VK_KHR_shader_subgroup_uniform_control_flow */
    bool           shaderSubgroupUniformControlFlow = false;
    
    /* VK_KHR_shader_terminate_invocation *//* MOVED TO Vulkan 1.3 Core */
    /* VK_KHR_synchronization2 *//* MOVED TO Vulkan 1.3 Core */
    /* VK_KHR_timeline_semaphore *//* MOVED TO Vulkan 1.2 Core */
    /* VK_KHR_uniform_buffer_standard_layout *//* MOVED TO Vulkan 1.2 Core */
    /* VK_KHR_variable_pointers *//* MOVED TO Vulkan 1.1 Core */
    /* VK_KHR_vulkan_memory_model *//* MOVED TO Vulkan 1.2 Core */

    /* WorkgroupMemoryExplicitLayoutFeaturesKHR *//* VK_KHR_workgroup_memory_explicit_layout */
    bool workgroupMemoryExplicitLayout = false;
    bool workgroupMemoryExplicitLayoutScalarBlockLayout = false;
    bool workgroupMemoryExplicitLayout8BitAccess = false;
    bool workgroupMemoryExplicitLayout16BitAccess = false;

    /* VK_KHR_zero_initialize_workgroup_memory *//* MOVED TO Vulkan 1.3 Core */
    /* VK_KHX_multiview *//* see VK_KHR_multiview *//* MOVED TO Vulkan 1.1 Core */

    /* ComputeShaderDerivativesFeaturesNV *//* VK_NV_compute_shader_derivatives */
    bool           computeDerivativeGroupQuads = false;
    bool           computeDerivativeGroupLinear = false;

    /* CooperativeMatrixFeaturesNV *//* VK_NV_cooperative_matrix */
    bool cooperativeMatrix = false;
    bool cooperativeMatrixRobustBufferAccess = false;

    /* RayTracingMotionBlurFeaturesNV *//* VK_NV_ray_tracing_motion_blur */
    bool           rayTracingMotionBlur = false;
    bool           rayTracingMotionBlurPipelineTraceRaysIndirect = false;

    /* CoverageReductionModeFeaturesNV *//* VK_NV_coverage_reduction_mode */
    bool                        coverageReductionMode;

    /* DeviceGeneratedCommandsFeaturesNV *//* VK_NV_device_generated_commands */
    bool           deviceGeneratedCommands = false;

    /* MeshShaderFeaturesNV *//* VK_NV_mesh_shader */
    bool           taskShader = false;
    bool           meshShader = false;

    /* RepresentativeFragmentTestFeaturesNV *//* VK_NV_representative_fragment_test */
    bool           representativeFragmentTest = false;

    /* VK_AMD_mixed_attachment_samples *//* OR *//* VK_NV_framebuffer_mixed_samples */
    bool mixedAttachmentSamples = false;

    /* VK_EXT_hdr_metadata */
    bool hdrMetadata = false;

    /* VK_GOOGLE_display_timing */
    bool displayTiming = false;

    /* VK_AMD_rasterization_order */
    bool rasterizationOrder = false;

    /* VK_AMD_shader_explicit_vertex_parameter */
    bool shaderExplicitVertexParameter = false;

    /* VK_AMD_shader_info */
    bool shaderInfoAMD = false;

    // [TODO] Promoted to VK1.1 core, haven't updated API to match
    /* VK_KHR_descriptor_update_template */

    // [TODO] Always enable, expose as limit
    /* VK_NV_sample_mask_override_coverage */

    // [TODO] Always enable, have it contribute to shaderSubgroup reporting & report as limit
    /* VK_NV_shader_subgroup_partitioned */

    // [TODO] Always enable, expose as limit
    /* VK_AMD_gcn_shader */

    // [TODO] Always enable, expose as limit (Note: Promoted to VK_KHR_shader_float16_int8)
    /* VK_AMD_gpu_shader_half_float */

    // [TODO] Always enable, expose as limit (Note: Promoted to VK_AMD_gpu_shader_int16)
    /* VK_AMD_gpu_shader_int16 */

    // [TODO] Always enable, have it contribute to shaderSubgroup reporting
    /* VK_AMD_shader_ballot */

    // [TODO] Always enable, expose as limit
    /* VK_AMD_shader_image_load_store_lod */

    // [TODO] Enable when available, report as limit
    /* VK_AMD_shader_trinary_minmax */

    // [TODO] needs to figure out how extending our LOAD_OP enum would affect the GL backend
    /* VK_EXT_load_store_op_none */

    // [TODO] Always enable, expose as limit
    /* VK_EXT_post_depth_coverage */

    // [TODO] Always enable, expose as limit
    /* VK_EXT_shader_stencil_export */

    // [TODO] Always enable, expose as limit
    /* VK_GOOGLE_decorate_string */

    // [TODO] Always enable, expose as limit
    /* VK_KHR_external_fence_fd */

    // [TODO] Always enable, expose as limit
    /* VK_KHR_external_fence_win32 */

    // [TODO] Always enable, expose as limit
    /* VK_KHR_external_memory_fd */

    // [TODO] Always enable, expose as limit
    /* VK_KHR_external_memory_win32 */

    // [TODO] Always enable, expose as limit
    /* VK_KHR_external_semaphore_fd */

    // [TODO] Always enable, expose as limit
    /* VK_KHR_external_semaphore_win32 */

    // [TODO] Shader extension, always enable, expose as limit
    /* VK_KHR_shader_non_semantic_info */

    // [TODO] Always enable, expose as limit
    /* VK_NV_geometry_shader_passthrough */

    // [TODO] Always enable, expose as limit
    /* VK_NV_viewport_swizzle */

    // Enabled by Default, Moved to Limits 
    //bool           shaderOutputViewportIndex;     // ALIAS: VK_EXT_shader_viewport_index_layer
    //bool           shaderOutputLayer;             // ALIAS: VK_EXT_shader_viewport_index_layer

    // Enabled by Default, Moved to Limits 
    /* ShaderIntegerFunctions2FeaturesINTEL *//* VK_INTEL_shader_integer_functions2 */
    //bool           shaderIntegerFunctions2 = false;

    // Enabled by Default, Moved to Limits 
    /* ShaderClockFeaturesKHR *//* VK_KHR_shader_clock */
    //bool           shaderSubgroupClock;

    // Enabled by Default, Moved to Limits 
    /* ShaderImageFootprintFeaturesNV *//* VK_NV_shader_image_footprint */
    //bool           imageFootprint;

    // [TODO LATER] Won't expose for now, API changes necessary
    /* VK_AMD_texture_gather_bias_lod */

    // [TODO LATER] requires extra API work to use
    // GL Hint: in GL/GLES this is NV_scissor_exclusive
    /* ExclusiveScissorFeaturesNV *//* VK_NV_scissor_exclusive */
    //bool           exclusiveScissor;

    // [TODO LATER] when we do multi-gpu
    /* ExternalMemoryRDMAFeaturesNV *//* VK_NV_external_memory_rdma */
    //bool           externalMemoryRDMA;

    // [DO NOT EXPOSE] for a very long time
    /* CornerSampledImageFeaturesNV *//* VK_NV_corner_sampled_image */
    //bool           cornerSampledImage;

    // [DO NOT EXPOSE] insane oxymoron, dedicated means dedicated, not aliased, won't expose
    /* DedicatedAllocationImageAliasingFeaturesNV *//* VK_NV_dedicated_allocation_image_aliasing */
    //bool           dedicatedAllocationImageAliasing;

    // [DO NOT EXPOSE]
    /* DiagnosticsConfigFeaturesNV *//* VK_NV_device_diagnostics_config */
    //bool           diagnosticsConfig;

    // [DEPRECATED]
    /* FragmentShaderBarycentricFeaturesNV *//* VK_NV_fragment_shader_barycentric */
    //bool           fragmentShaderBarycentric;
    
    // [DO NOT EXPOSE] would first need to expose VK_KHR_fragment_shading_rate before
    /* FragmentShadingRateEnumsFeaturesNV *//* VK_NV_fragment_shading_rate_enums */
    //bool           fragmentShadingRateEnums;
    //bool           supersampleFragmentShadingRates;
    //bool           noInvocationFragmentShadingRates;

    // [DO NOT EXPOSE] won't expose, the existing inheritance of state is enough
    /* InheritedViewportScissorFeaturesNV *//* VK_NV_inherited_viewport_scissor */
    //bool           inheritedViewportScissor2D;

    // [DO NOT EXPOSE] no idea what real-world beneficial use case would be
    /* LinearColorAttachmentFeaturesNV *//* VK_NV_linear_color_attachment */
    //bool           linearColorAttachment;
    
    // [ENABLE BY DEFAULT]
    /* ShaderSMBuiltinsFeaturesNV *//* VK_NV_shader_sm_builtins */
    //bool           shaderSMBuiltins;
    
    // [DO NOT EXPOSE] not implementing or exposing VRS in near or far future
    /* ShadingRateImageFeaturesNV *//* VK_NV_shading_rate_image */
    //bool           shadingRateImage;
    //bool           shadingRateCoarseSampleOrder;

    // [DO NOT EXPOSE] not implementing or exposing VRS in near or far future
    /* FragmentDensityMapOffsetFeaturesQCOM *//* VK_QCOM_fragment_density_map_offset */
    //bool           fragmentDensityMapOffset;

    // [DO NOT EXPOSE] This extension is only intended for use in specific embedded environments with known implementation details, and is therefore undocumented.
    /* DescriptorSetHostMappingFeaturesVALVE *//* VK_VALVE_descriptor_set_host_mapping */
    //bool           descriptorSetHostMapping;

    // [DO NOT EXPOSE] its a D3D special use extension, shouldn't expose
    /* MutableDescriptorTypeFeaturesVALVE *//* VK_VALVE_mutable_descriptor_type */
    //bool           mutableDescriptorType;

    // [DO NOT EXPOSE]
    /* SubpassShadingFeaturesHUAWEI *//* VK_HUAWEI_subpass_shading */
    // bool           subpassShading;


    /* VK_AMD_buffer_marker */
    bool bufferMarkerAMD = false;
    

    /* Extensions Exposed as Features: */

    // [DO NOT EXPOSE] Waiting for cross platform
    /* VK_AMD_display_native_hdr */

    // [DO NOT EXPOSE] Promoted to KHR version already exposed
    /* VK_AMD_draw_indirect_count */
    
    // [DO NOT EXPOSE] 0 documentation
    /* VK_AMD_gpa_interface */

    // [TODO LATER] (When it has documentation): Always enable, expose as limit
    /* VK_AMD_gpu_shader_half_float_fetch */

    // [DO NOT EXPOSE] 0 documentation
    /* VK_AMD_image_layout_resolve */

    // [DO NOT EXPOSE]
    /* VK_AMD_memory_overallocation_behavior */

    // [DO NOT EXPOSE] Promoted to VK_KHR_maintenance1, core VK 1.1
    /* VK_AMD_negative_viewport_height */

    // [DO NOT EXPOSE] Too vendor specific
    /* VK_AMD_pipeline_compiler_control */

    // [DO NOT EXPOSE] Promoted to VK_EXT_sample_locations 
    /* VK_AMD_programmable_sample_locations */

    // [DO NOT EXPOSE] 0 documentation
    /* VK_AMD_wave_limits */

    // [TODO LATER] Requires exposing external memory first
    /* VK_ANDROID_external_memory_android_hardware_buffer */

    // [DO NOT EXPOSE] supported=disabled
    /* VK_ANDROID_native_buffer */

    // [TODO LATER] Requires changes to API
    /* VK_EXT_calibrated_timestamps */

    // [DO NOT EXPOSE] Promoted to VK_EXT_debug_utils (instance ext)
    /* VK_EXT_debug_marker */

    // [TODO LATER] Will expose some day
    /* VK_EXT_depth_range_unrestricted */
    
    // [TODO LATER] Requires handling display swapchain stuff
    /* VK_EXT_display_control */

    // [TODO LATER] Requires exposing external memory first
    /* VK_EXT_external_memory_dma_buf */

    // [TODO LATER] limited utility and availability, might expose if feel like wasting time
    /* VK_EXT_filter_cubic */

    // [TODO LATER] Requires API changes
    /* VK_EXT_full_screen_exclusive */

    // [DO NOT EXPOSE] absorbed into KHR_global_priority
    /* VK_EXT_global_priority_query */

    // [DO NOT EXPOSE] Too "intrinsically linux"
    /* VK_EXT_image_drm_format_modifier */

    // [TODO LATER] Expose when we support MoltenVK
    /* VK_EXT_metal_objects */

    // [DO NOT EXPOSE] Never expose this, it was a mistake for that GL quirk to exist in the first place
    /* VK_EXT_non_seamless_cube_map */

    // [DO NOT EXPOSE] Too "intrinsically linux"
    /* VK_EXT_physical_device_drm */

    // [TODO LATER] would like to expose, but too much API to change
    /* VK_EXT_pipeline_creation_feedback */

    /* VK_EXT_queue_family_foreign */

    // [DO NOT EXPOSE] wont expose yet (or ever), requires VK_KHR_sampler_ycbcr_conversion
    /* VK_EXT_rgba10x6_formats */

    /* VK_EXT_separate_stencil_usage */

    // [DO NOT EXPOSE] stupid to expose, it would be extremely dumb to want to provide some identifiers instead of VkShaderModule outside of some emulator which has no control over pipeline combo explosion
    /* VK_EXT_shader_module_identifier */

    // [DO NOT EXPOSE] we dont need to care or know about it
    /* VK_EXT_tooling_info */

    // [TODO LATER] Expose when we start to experience slowdowns from validation
    /* VK_EXT_validation_cache */

    // [DO NOT EXPOSE] Provisional
    /* VK_EXT_video_decode_h264 */

    // [DO NOT EXPOSE] Provisional
    /* VK_EXT_video_decode_h265 */

    // [DO NOT EXPOSE] Provisional
    /* VK_EXT_video_encode_h264 */

    // [DO NOT EXPOSE] Provisional
    /* VK_EXTX_portability_subset */

    // [DO NOT EXPOSE]
    /* VK_GOOGLE_hlsl_functionality1 */

    // [DO NOT EXPOSE] 0 documentation
    /* VK_GOOGLE_sampler_filtering_precision */

    // [DO NOT EXPOSE] 0 documentation
    /* VK_GOOGLE_user_type */

    // [DO NOT EXPOSE] 0 documentation
    /* VK_HUAWEI_prerotation */

    // [DO NOT EXPOSE] 0 documentation
    /* VK_HUAWEI_smart_cache */

    // [DO NOT EXPOSE] Vendor specific, superceeded by VK_EXT_filter_cubic, won't expose for a long time
    /* VK_IMG_filter_cubic */

    // [DO NOT EXPOSE] Enables certain formats in Vulkan, we just enable them if available or else we need to make format support query functions in LogicalDevice as well
    /* VK_IMG_format_pvrtc */

    // [DO NOT EXPOSE] Promoted to VK_KHR_performance_query, VK1.1 core
    /* VK_INTEL_performance_query */
    /* VK_KHR_bind_memory2 */

    // [DO NOT EXPOSE] Promoted to VK1.3 core, migrate to it when/if we need it
    /* VK_KHR_copy_commands2 */

    /* VK_KHR_create_renderpass2 */

    // [DO NOT EXPOSE] required for acceleration_structure and only that extension, do not expose until another comes that actually makes use of it
    /* VK_KHR_deferred_host_operations */

    // [DO NOT EXPOSE] Promoted to core VK 1.1
    /* VK_KHR_device_group */

    /* VK_KHR_display_swapchain */

    // [DO NOT EXPOSE] Promoted to core VK 1.1
    /* VK_KHR_external_fence */

    // [DO NOT EXPOSE] Promoted to core VK 1.1
    /* VK_KHR_external_memory */

    // [DO NOT EXPOSE] Promoted to core VK 1.1
    /* VK_KHR_external_semaphore */

    /* VK_KHR_format_feature_flags2 */

    // [DO NOT EXPOSE] Promoted to core VK 1.1
    /* VK_KHR_get_memory_requirements2 */

    // [DO NOT EXPOSE] Promoted to core VK 1.1
    /* VK_KHR_get_physical_device_properties2 */

    // [TODO LATER] Requires API changes and asset converter upgrades; default to it and forbid old API use
    /* VK_KHR_image_format_list */

    // [DO NOT EXPOSE] this is "swap with damange" known from EGL, cant be arsed to support
    /* VK_KHR_incremental_present */

    // [DO NOT EXPOSE] Promoted to core VK 1.1
    /* VK_KHR_relaxed_block_layout */

    // [DO NOT EXPOSE] Leave for later consideration
    /* VK_KHR_shared_presentable_image */

    // [DO NOT EXPOSE] Promoted to core VK 1.1
    /* VK_KHR_storage_buffer_storage_class */

    // [DO NOT EXPOSE] Instance extension & should enable implicitly if swapchain is enabled
    /* VK_KHR_surface */

    /* VK_KHR_swapchain */ // we want to expose this extension as feature
    core::bitflag<E_SWAPCHAIN_MODE> swapchainMode = E_SWAPCHAIN_MODE::ESM_NONE;
    
    // [TODO LATER] Requires VK_KHR_image_format_list to be enabled for any device-level functionality
    /* VK_KHR_swapchain_mutable_format */

    // [DO NOT EXPOSE] Provisional
    /* VK_KHR_video_decode_queue */

    // [DO NOT EXPOSE] Provisional
    /* VK_KHR_video_encode_queue */
    
    // [TODO LATER] Used for dx11 interop
    /* VK_KHR_win32_keyed_mutex */

    // [DO NOT EXPOSE] 0 documentation
    /* VK_MESA_query_timestamp */

    // [TODO LATER] won't decide yet, requires VK_EXT_direct_mode_display anyway
    /* VK_NV_acquire_winrt_display */
    
    // [TODO LATER] Don't expose VR features for now
    /* VK_NV_clip_space_w_scaling */

    // [DO NOT EXPOSE] 0 documentation
    /* VK_NV_cuda_kernel_launch */
    
    // [DO NOT EXPOSE] Promoted to KHR_dedicated_allocation, core VK 1.1
    /* VK_NV_dedicated_allocation */

    // [DO NOT EXPOSE] Promoted to VK_KHR_external_memory_win32 
    /* VK_NV_external_memory_win32 */

    // [DO NOT EXPOSE] For now. For 2D ui
    /* VK_NV_fill_rectangle */

    // [TODO LATER] Requires API changes
    /* VK_NV_fragment_coverage_to_color */

    // [DO NOT EXPOSE]
    /* VK_NV_glsl_shader */

    // [DO NOT EXPOSE] 0 documentation
    /* VK_NV_low_latency */

    /* VK_NV_present_barrier */

    // [DO NOT EXPOSE] 0 documentation
    /* VK_NV_rdma_memory */

    // [DO NOT EXPOSE] 0 documentation
    /* VK_NV_texture_dirty_tile_map */
    
    // [DO NOT EXPOSE] Will be promoted to KHR_video_queue.
    /* VK_NV_video_queue */

    /* VK_NV_viewport_array2 */

    // [DO NOT EXPOSE] Promoted to VK_KHR_win32_keyed_mutex 
    /* VK_NV_win32_keyed_mutex */

    // [TODO LATER] Wait until VK_AMD_shader_fragment_mask & VK_QCOM_render_pass_shader_resolve converge to something useful
    /* VK_AMD_shader_fragment_mask */
    // [TODO LATER] Wait until VK_AMD_shader_fragment_mask & VK_QCOM_render_pass_shader_resolve converge to something useful
    /* VK_QCOM_render_pass_shader_resolve */

    // [DO NOT EXPOSE] absorbed into VK_EXT_load_store_op_none
    /* VK_QCOM_render_pass_store_ops */

    // [DO NOT EXPOSE] Too vendor specific
    /* VK_QCOM_render_pass_transform */

    // [DO NOT EXPOSE] Too vendor specific
    /* VK_QCOM_rotated_copy_commands */
    
    /* VK_KHR_spirv_1_4 */ // We do not expose because we always enable
    
    /* VK_EXT_image_compression_control */
    /* VK_EXT_image_compression_control_swapchain */
    /* VK_EXT_multisampled_render_to_single_sampled */
    /* VK_EXT_pipeline_properties */

    /* Nabla */
    // No Nabla Specific Features for now
};

} // nbl::video

#endif
