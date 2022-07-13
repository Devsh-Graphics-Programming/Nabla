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
    VkBool32 sampleRateShading = false;
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
    //VkBool32    textureCompressionETC2;
    //VkBool32    textureCompressionASTC_LDR;
    //VkBool32    textureCompressionBC;
    
    bool occlusionQueryPrecise = false;
    //VkBool32    pipelineStatisticsQuery; [TODO]
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
    bool shaderInt64 = false;
    bool shaderInt16 = false;
    bool shaderResourceResidency = false;
    bool shaderResourceMinLod = false;
    
    // [TODO] cause we haven't implemented sparse resources yet
    //VkBool32    sparseBinding;
    //VkBool32    sparseResidencyBuffer;
    //VkBool32    sparseResidencyImage2D;
    //VkBool32    sparseResidencyImage3D;
    //VkBool32    sparseResidency2Samples;
    //VkBool32    sparseResidency4Samples;
    //VkBool32    sparseResidency8Samples;
    //VkBool32    sparseResidency16Samples;
    //VkBool32    sparseResidencyAliased;
    
    bool variableMultisampleRate = false;
    bool inheritedQueries = false;

    /* Vulkan 1.1 Core */
    bool storageBuffer16BitAccess = false;
    bool uniformAndStorageBuffer16BitAccess = false;
    bool storagePushConstant16 = false;
    bool storageInputOutput16 = false;
    
    // [TODO] do not expose multiview yet
    //VkBool32           multiview;
    //VkBool32           multiviewGeometryShader;
    //VkBool32           multiviewTessellationShader;
    
    // [Future TODO]:
    //VkBool32           variablePointersStorageBuffer;
    //VkBool32           variablePointers;
    
    //VkBool32           protectedMemory; // [DO NOT EXPOSE] not gonna expose until we have a need to
 
    // [DO NOT EXPOSE] Enables certain formats in Vulkan, we just enable them if available or else we need to make format support query functions in LogicalDevice as well
    //VkBool32           samplerYcbcrConversion;
    bool shaderDrawParameters = false;




    /* Vulkan 1.2 Core */

    bool samplerMirrorClampToEdge = false;          // ALIAS: VK_KHR_sampler_mirror_clamp_to_edge
    bool drawIndirectCount = false;                 // ALIAS: VK_KHR_draw_indirect_count

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
    VkBool32 runtimeDescriptorArray = false;
    
    bool samplerFilterMinmax = false;   // ALIAS: VK_EXT_sampler_filter_minmax
    
    VkBool32 scalarBlockLayout = false;     // or VK_EXT_scalar_block_layout
    
    //VkBool32           imagelessFramebuffer;  // or VK_KHR_imageless_framebuffer // [FUTURE TODO]
    
    VkBool32           uniformBufferStandardLayout = false;   // or VK_KHR_uniform_buffer_standard_layout
    
    bool shaderSubgroupExtendedTypes;   // or VK_KHR_shader_subgroup_extended_types
    
    VkBool32           separateDepthStencilLayouts;   // or VK_KHR_separate_depth_stencil_layouts
    
    // [TODO] And add implementation to engine
    //VkBool32           hostQueryReset;                // or VK_EXT_host_query_reset
    
    //VkBool32           timelineSemaphore;             // or VK_KHR_timeline_semaphore // [FUTURE TODO] won't expose for a long time
    
    // or VK_KHR_buffer_device_address:
    bool bufferDeviceAddress = false;
    // VkBool32           bufferDeviceAddressCaptureReplay; // [DO NOT EXPOSE] for capture tools not engines
    VkBool32           bufferDeviceAddressMultiDevice;
    
    // or VK_KHR_vulkan_memory_model
    VkBool32           vulkanMemoryModel;
    VkBool32           vulkanMemoryModelDeviceScope;
    VkBool32           vulkanMemoryModelAvailabilityVisibilityChains;
   
    VkBool32           subgroupBroadcastDynamicId = false;    // if Vulkan 1.2 is supported




    /* Vulkan 1.3 Core */
    
    // [TODO] robustness stuff
    //VkBool32           robustImageAccess;                 //  or VK_EXT_image_robustness
    
    // [DO NOT EXPOSE]
    //  or VK_EXT_inline_uniform_block:
    //VkBool32           inlineUniformBlock;
    //VkBool32           descriptorBindingInlineUniformBlockUpdateAfterBind;
    
    // [TODO]
    //VkBool32           pipelineCreationCacheControl;      // or VK_EXT_pipeline_creation_cache_control
   
    // [DO NOT EXPOSE] ever
    //VkBool32           privateData;                       // or VK_EXT_private_data
    
    VkBool32           shaderDemoteToHelperInvocation;    // or VK_EXT_shader_demote_to_helper_invocation
    VkBool32           shaderTerminateInvocation;         // or VK_KHR_shader_terminate_invocation
    
    // or VK_EXT_subgroup_size_control
    bool subgroupSizeControl  = false;
    bool computeFullSubgroups = false;
    
    // [DO NOT EXPOSE] and false because we havent rewritten our frontend API for that: https://github.com/Devsh-Graphics-Programming/Nabla/issues/384
    //VkBool32           synchronization2;                      // or VK_KHR_synchronization2
    
    // [DO NOT EXPOSE] Doesn't make a difference, just shortcut from Querying support from PhysicalDevice
    //VkBool32           textureCompressionASTC_HDR;            // or VK_EXT_texture_compression_astc_hdr
    
    // [DO NOT EXPOSE] would require doing research into the GL/GLES robustness extensions
    //VkBool32           shaderZeroInitializeWorkgroupMemory;   // or VK_KHR_zero_initialize_workgroup_memory
    
    // [DO NOT EXPOSE] EVIL
    //VkBool32           dynamicRendering;                      // or VK_KHR_dynamic_rendering
    
    VkBool32           shaderIntegerDotProduct;               // or VK_KHR_shader_integer_dot_product
    //VkBool32           maintenance4;                          // [DO NOT EXPOSE] doesn't make sense




    /* Vulkan Extensions */

    // [TODO] no such thing on GL, but trivial to implement.
    /* CoherentMemoryFeaturesAMD *//* VK_AMD_device_coherent_memory */
    //VkBool32           deviceCoherentMemory;

    // [TODO] this one isn't in the headers?
    /* VK_AMD_shader_early_and_late_fragment_tests */
    //VkBool32 shaderEarlyAndLateFragmentTests;

    /* RasterizationOrderAttachmentAccessFeaturesARM *//* VK_ARM_rasterization_order_attachment_access */
    VkBool32 rasterizationOrderColorAttachmentAccess;
    VkBool32 rasterizationOrderDepthAttachmentAccess;
    VkBool32 rasterizationOrderStencilAttachmentAccess;
    
    // [DO NOT EXPOSE] Enables certain formats in Vulkan, we just enable them if available or else we need to make format support query functions in LogicalDevice as well
    /* 4444FormatsFeaturesEXT *//* VK_EXT_4444_formats */
    
    // [TODO]
    /* ASTCDecodeFeaturesEXT *//* VK_EXT_astc_decode_mode */
    //VkFormat           decodeMode;

    // [DO NOT EXPOSE] right now, no idea if we'll ever expose and implement those but they'd all be false for OpenGL
    /* BlendOperationAdvancedFeaturesEXT *//* VK_EXT_blend_operation_advanced */
    //VkBool32           advancedBlendCoherentOperations;
    
    // [DO NOT EXPOSE] not going to expose custom border colors for now
    /* BorderColorSwizzleFeaturesEXT *//* VK_EXT_border_color_swizzle */
    //VkBool32           borderColorSwizzle;
    //VkBool32           borderColorSwizzleFromImage;

    /* VK_EXT_buffer_device_address *//* HAS KHR VERSION */

    // [TODO] would need new commandbuffer methods, etc
    /* ColorWriteEnableFeaturesEXT *//* VK_EXT_color_write_enable */
    //VkBool32           colorWriteEnable;

    // [TODO] would need API to deal with queries and begin/end conditional blocks
    /* ConditionalRenderingFeaturesEXT *//* VK_EXT_conditional_rendering */
    //VkBool32           conditionalRendering;
    //VkBool32           inheritedConditionalRendering;
    
    // [DO NOT EXPOSE] not going to expose custom border colors for now
    /* CustomBorderColorFeaturesEXT *//* VK_EXT_custom_border_color */
    //VkBool32           customBorderColors;
    //VkBool32           customBorderColorWithoutFormat;

    // [DO NOT EXPOSE] EVER, VULKAN DEPTH RANGE ONLY!
    /* DepthClipControlFeaturesEX *//* VK_EXT_depth_clip_control */
    //VkBool32           depthClipControl;

    // [DO NOT EXPOSE] only useful for D3D emulators
    /* DepthClipEnableFeaturesEXT *//* VK_EXT_depth_clip_enable */
    //VkBool32           depthClipEnable;

    /* DescriptorIndexingFeatures *//* VK_EXT_descriptor_indexing *//* MOVED TO Vulkan 1.2 Core  */

    // [TODO]
    /* DeviceMemoryReportFeaturesEXT *//* VK_EXT_device_memory_report */
    //VkBool32           deviceMemoryReport;
    
    // [DO NOT EXPOSE] we're fully GPU driven anyway with sorted pipelines.
    /* ExtendedDynamicStateFeaturesEXT *//* VK_EXT_extended_dynamic_state */
    //VkBool32           extendedDynamicState;
    
    // [DO NOT EXPOSE] we're fully GPU driven anyway with sorted pipelines.
    /* ExtendedDynamicState2FeaturesEXT *//* VK_EXT_extended_dynamic_state2 */
    //VkBool32           extendedDynamicState2;
    //VkBool32           extendedDynamicState2LogicOp;
    //VkBool32           extendedDynamicState2PatchControlPoints;

    // [TODO]
    /* FragmentDensityMapFeaturesEXT *//* VK_EXT_fragment_density_map */
    //VkBool32           fragmentDensityMap;
    //VkBool32           fragmentDensityMapDynamic;
    //VkBool32           fragmentDensityMapNonSubsampledImages;
    
    // [TODO]
    /* FragmentDensityMap2FeaturesEXT *//* VK_EXT_fragment_density_map2 */
    //VkBool32           fragmentDensityMapDeferred;

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
    //VkBool32           graphicsPipelineLibrary;

    /* HostQueryResetFeatures *//* VK_EXT_host_query_reset *//* MOVED TO Vulkan 1.2 Core */
    
    // [TODO] Investigate later
    /* Image2DViewOf3DFeaturesEXT *//* VK_EXT_image_2d_view_of_3d */
    //VkBool32           image2DViewOf3D;
    //VkBool32           sampler2DViewOf3D;

    /* ImageRobustnessFeaturesEXT *//* VK_EXT_image_robustness *//* MOVED TO Vulkan 1.3 Core */
    
    // [DO NOT EXPOSE] pointless to implement currently
    /* ImageViewMinLodFeaturesEXT *//* VK_EXT_image_view_min_lod */
    //VkBool32           minLod;

    /* IndexTypeUint8FeaturesEXT *//* VK_EXT_index_type_uint8 */
    VkBool32           indexTypeUint8 = false;
    
    /* InlineUniformBlockFeaturesEXT *//* VK_EXT_inline_uniform_block *//* MOVED TO Vulkan 1.3 Core */

    // [TODO] this feature introduces new/more pipeline state with VkPipelineRasterizationLineStateCreateInfoEXT
    /* LineRasterizationFeaturesEXT *//* VK_EXT_line_rasterization */
    // GL HINT (remove when implemented): MULTI_SAMPLE_LINE_WIDTH_RANGE (which is necessary for this) is guarded by !IsGLES || Version>=320 no idea is something enables this or not
    //VkBool32           rectangularLines;
    //VkBool32           bresenhamLines;
    //VkBool32           smoothLines;
    // GL HINT (remove when implemented): !IsGLES for all stipples
    //VkBool32           stippledRectangularLines;
    //VkBool32           stippledBresenhamLines;
    //VkBool32           stippledSmoothLines;

    // [TODO] trivial to implement later, false on GL
    /* MemoryPriorityFeaturesEXT *//* VK_EXT_memory_priority */
    //VkBool32           memoryPriority;

    // [DO NOT EXPOSE] this extension is dumb, if we're recording that many draws we will be using Multi Draw INDIRECT which is better supported
    /* MultiDrawFeaturesEXT *//* VK_EXT_multi_draw */
    //VkBool32           multiDraw;

    // [DO NOT EXPOSE] pointless to expose without exposing VK_EXT_memory_priority and the memory query feature first
    /* PageableDeviceLocalMemoryFeaturesEXT *//* VK_EXT_pageable_device_local_memory */
    //VkBool32           pageableDeviceLocalMemory;

    /* PipelineCreationCacheControlFeaturesEXT *//* VK_EXT_pipeline_creation_cache_control *//* MOVED TO Vulkan 1.3 Core */

    // [DO NOT EXPOSE] requires and relates to EXT_transform_feedback which we'll never expose
    /* PrimitivesGeneratedQueryFeaturesEXT *//* VK_EXT_primitives_generated_query */
    //VkBool32           primitivesGeneratedQuery;
    //VkBool32           primitivesGeneratedQueryWithRasterizerDiscard;
    //VkBool32           primitivesGeneratedQueryWithNonZeroStreams;
    
    // [DO NOT EXPOSE]
    /* PrimitiveTopologyListRestartFeaturesEXT *//* VK_EXT_primitive_topology_list_restart */
    //VkBool32           primitiveTopologyListRestart;
    //VkBool32           primitiveTopologyPatchListRestart;

    /* PrivateDataFeatures *//* VK_EXT_private_data *//* MOVED TO Vulkan 1.3 Core */

    // [DO NOT EXPOSE] provokingVertexLast will not expose (we always use First Vertex Vulkan-like convention), anything to do with XForm-feedback we don't expose
    /* ProvokingVertexFeaturesEXT *//* VK_EXT_provoking_vertex */
    //VkBool32           provokingVertexLast;
    //VkBool32           transformFeedbackPreservesProvokingVertex;

    // [TODO]
    /* Robustness2FeaturesEXT *//* VK_EXT_robustness2 */
    //VkBool32           robustBufferAccess2;
    //VkBool32           robustImageAccess2;
    //VkBool32           nullDescriptor;
    
    /* ScalarBlockLayoutFeaturesEXT *//* VK_EXT_scalar_block_layout *//* MOVED TO Vulkan 1.2 Core */
    
    /* ShaderAtomicFloatFeaturesEXT *//* VK_EXT_shader_atomic_float */
    VkBool32           shaderBufferFloat32Atomics;
    VkBool32           shaderBufferFloat32AtomicAdd;
    VkBool32           shaderBufferFloat64Atomics;
    VkBool32           shaderBufferFloat64AtomicAdd;
    VkBool32           shaderSharedFloat32Atomics;
    VkBool32           shaderSharedFloat32AtomicAdd;
    VkBool32           shaderSharedFloat64Atomics;
    VkBool32           shaderSharedFloat64AtomicAdd;
    VkBool32           shaderImageFloat32Atomics;
    VkBool32           shaderImageFloat32AtomicAdd;
    VkBool32           sparseImageFloat32Atomics;
    VkBool32           sparseImageFloat32AtomicAdd;

    /* ShaderAtomicFloat2FeaturesEXT *//* VK_EXT_shader_atomic_float2 */
    VkBool32           shaderBufferFloat16Atomics;
    VkBool32           shaderBufferFloat16AtomicAdd;
    VkBool32           shaderBufferFloat16AtomicMinMax;
    VkBool32           shaderBufferFloat32AtomicMinMax;
    VkBool32           shaderBufferFloat64AtomicMinMax;
    VkBool32           shaderSharedFloat16Atomics;
    VkBool32           shaderSharedFloat16AtomicAdd;
    VkBool32           shaderSharedFloat16AtomicMinMax;
    VkBool32           shaderSharedFloat32AtomicMinMax;
    VkBool32           shaderSharedFloat64AtomicMinMax;
    VkBool32           shaderImageFloat32AtomicMinMax;
    VkBool32           sparseImageFloat32AtomicMinMax;
    
    /* DemoteToHelperInvocationFeaturesEXT *//* VK_EXT_shader_demote_to_helper_invocation *//* MOVED TO Vulkan 1.3 Core */

    /* ShaderImageAtomicInt64FeaturesEXT *//* VK_EXT_shader_image_atomic_int64 */
    VkBool32           shaderImageInt64Atomics;
    VkBool32           sparseImageInt64Atomics;

    /* SubgroupSizeControlFeaturesEXT *//* VK_EXT_subgroup_size_control *//* MOVED TO Vulkan 1.3 Core */

    // [DO NOT EXPOSE] always enable if we can
    /* TexelBufferAlignmentFeaturesEXT *//* VK_EXT_texel_buffer_alignment */
    //VkBool32           texelBufferAlignment;

    /* TextureCompressionASTCHDRFeaturesEXT *//* VK_EXT_texture_compression_astc_hdr *//* MOVED TO Vulkan 1.3 Core */

    // [DO NOT EXPOSE] ever because of our disdain for XForm feedback
    /* TransformFeedbackFeaturesEXT *//* VK_EXT_transform_feedback */
    //VkBool32           transformFeedback;
    //VkBool32           geometryStreams;

    // [TODO] we would have to change the API
    /* VertexAttributeDivisorFeaturesEXT *//* VK_EXT_vertex_attribute_divisor */
    //VkBool32           vertexAttributeInstanceRateDivisor;
    //VkBool32           vertexAttributeInstanceRateZeroDivisor;

    // [DO NOT EXPOSE] too much API Fudjery
    /* VertexInputDynamicStateFeaturesEXT *//* VK_EXT_vertex_input_dynamic_state */
    //VkBool32           vertexInputDynamicState;

    // [DO NOT EXPOSE] Enables certain formats in Vulkan, we just enable them if available or else we need to make format support query functions in LogicalDevice as well
    /* Ycbcr2Plane444FormatsFeaturesEXT *//* VK_EXT_ycbcr_2plane_444_formats */

    // [DO NOT EXPOSE] Expose nothing to do with video atm
    /* YcbcrImageArraysFeaturesEXT *//* VK_EXT_ycbcr_image_arrays */
    //VkBool32           ycbcrImageArrays;

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
    //VkBool32           pipelineFragmentShadingRate;
    //VkBool32           primitiveFragmentShadingRate;
    //VkBool32           attachmentFragmentShadingRate;

    /* ImagelessFramebufferFeaturesKHR *//* VK_KHR_imageless_framebuffer *//* MOVED TO Vulkan 1.2 Core */
    /* Maintenance4FeaturesKHR *//* VK_KHR_maintenance4 *//* MOVED TO Vulkan 1.3 Core */
    /* MultiviewFeaturesKHR *//* VK_KHR_multiview *//* MOVED TO Vulkan 1.1 Core */
    
    // [TODO]
    /* PerformanceQueryFeaturesKHR *//* VK_KHR_performance_query */
    //VkBool32           performanceCounterQueryPools;
    //VkBool32           performanceCounterMultipleQueryPools;

    // [TODO]
    /* PipelineExecutablePropertiesFeaturesKHR *//* VK_KHR_pipeline_executable_properties */
    //VkBool32           pipelineExecutableInfo;
    
    /* VK_KHR_portability_subset - PROVISINAL/NOTAVAILABLEANYMORE */

    // [DO NOT EXPOSE] no point exposing until an extension more useful than VK_KHR_present_wait arrives
    /* PresentIdFeaturesKHR *//* VK_KHR_present_id */
    //VkBool32           presentId;
    
    // [DO NOT EXPOSE] won't expose, this extension is poop, I should have a Fence-andQuery-like object to query the presentation timestamp, not a blocking call that may unblock after an arbitrary delay from the present
    /* PresentWaitFeaturesKHR *//* VK_KHR_present_wait */
    //VkBool32           presentWait;
    
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
    VkBool32           shaderDeviceClock;

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
    bool cooperativeMatrix;
    bool cooperativeMatrixRobustBufferAccess;

    // [DO NOT EXPOSE] for a very long time
    /* CornerSampledImageFeaturesNV *//* VK_NV_corner_sampled_image */
    //VkBool32           cornerSampledImage;

    // [TODO]
    /* CoverageReductionModeFeaturesNV *//* VK_NV_coverage_reduction_mode */
    //VkCoverageReductionModeNV                        coverageReductionMode;

    // [DO NOT EXPOSE] insane oxymoron, dedicated means dedicated, not aliased, won't expose
    /* DedicatedAllocationImageAliasingFeaturesNV *//* VK_NV_dedicated_allocation_image_aliasing */
    //VkBool32           dedicatedAllocationImageAliasing;

    // [DO NOT EXPOSE]
    /* DiagnosticsConfigFeaturesNV *//* VK_NV_device_diagnostics_config */
    //VkBool32           diagnosticsConfig;

    // [TODO]
    /* DeviceGeneratedCommandsFeaturesNV *//* VK_NV_device_generated_commands */
    //VkBool32           deviceGeneratedCommands;

    // [TODO] when we do multi-gpu
    /* ExternalMemoryRDMAFeaturesNV *//* VK_NV_external_memory_rdma */
    //VkBool32           externalMemoryRDMA;
    
    // [DEPRECATED]
    /* FragmentShaderBarycentricFeaturesNV *//* VK_NV_fragment_shader_barycentric */
    //VkBool32           fragmentShaderBarycentric;
    
    // [DO NOT EXPOSE] would first need to expose VK_KHR_fragment_shading_rate before
    /* FragmentShadingRateEnumsFeaturesNV *//* VK_NV_fragment_shading_rate_enums */
    //VkBool32           fragmentShadingRateEnums;
    //VkBool32           supersampleFragmentShadingRates;
    //VkBool32           noInvocationFragmentShadingRates;

    // [DO NOT EXPOSE] won't expose, the existing inheritance of state is enough
    /* InheritedViewportScissorFeaturesNV *//* VK_NV_inherited_viewport_scissor */
    //VkBool32           inheritedViewportScissor2D;

    // [DO NOT EXPOSE] no idea what real-world beneficial use case would be
    /* LinearColorAttachmentFeaturesNV *//* VK_NV_linear_color_attachment */
    //VkBool32           linearColorAttachment;
    
    // [TODO]
    /* MeshShaderFeaturesNV *//* VK_NV_mesh_shader */
    //VkBool32           taskShader;
    //VkBool32           meshShader;
    
    /* RayTracingMotionBlurFeaturesNV *//* VK_NV_ray_tracing_motion_blur */
    VkBool32           rayTracingMotionBlur;
    VkBool32           rayTracingMotionBlurPipelineTraceRaysIndirect;

    // [TODO] Expose Soon, Useful
    /* RepresentativeFragmentTestFeaturesNV *//* VK_NV_representative_fragment_test */
    //VkBool32           representativeFragmentTest;
 
    // [TODO] requires extra API work to use
    // GL Hint: in GL/GLES this is NV_scissor_exclusive
    /* ExclusiveScissorFeaturesNV *//* VK_NV_scissor_exclusive */
    //VkBool32           exclusiveScissor;

    /* ShaderImageFootprintFeaturesNV *//* VK_NV_shader_image_footprint */
    //VkBool32           imageFootprint;

    // [ENABLE BY DEFAULT]
    /* ShaderSMBuiltinsFeaturesNV *//* VK_NV_shader_sm_builtins */
    //VkBool32           shaderSMBuiltins;
    
    // [DO NOT EXPOSE] not implementing or exposing VRS in near or far future
    /* ShadingRateImageFeaturesNV *//* VK_NV_shading_rate_image */
    //VkBool32           shadingRateImage;
    //VkBool32           shadingRateCoarseSampleOrder;

    // [DO NOT EXPOSE] not implementing or exposing VRS in near or far future
    /* FragmentDensityMapOffsetFeaturesQCOM *//* VK_QCOM_fragment_density_map_offset */
    //VkBool32           fragmentDensityMapOffset;

    // [DO NOT EXPOSE] This extension is only intended for use in specific embedded environments with known implementation details, and is therefore undocumented.
    /* DescriptorSetHostMappingFeaturesVALVE *//* VK_VALVE_descriptor_set_host_mapping */
    //VkBool32           descriptorSetHostMapping;

    // [DO NOT EXPOSE] its a D3D special use extension, shouldn't expose
    /* MutableDescriptorTypeFeaturesVALVE *//* VK_VALVE_mutable_descriptor_type */
    //VkBool32           mutableDescriptorType;

    // [DO NOT EXPOSE]
    /* SubpassShadingFeaturesHUAWEI *//* VK_HUAWEI_subpass_shading */
    // VkBool32           subpassShading;

    
    /* Extensions Exposed as Features: */

    /* VK_AMD_buffer_marker */
    /* VK_AMD_calibrated_timestamps */
    /* VK_AMD_display_native_hdr */
    /* VK_AMD_draw_indirect_count */
    /* VK_AMD_gcn_shader */
    /* VK_AMD_gpa_interface */
    /* VK_AMD_gpu_shader_half_float */
    /* VK_AMD_gpu_shader_half_float_fetch */
    /* VK_AMD_gpu_shader_int16 */
    /* VK_AMD_image_layout_resolve */
    /* VK_AMD_memory_overallocation_behavior */
    /* VK_AMD_mixed_attachment_samples */
    /* VK_AMD_negative_viewport_height */
    /* VK_AMD_pipeline_compiler_control */
    /* VK_AMD_programmable_sample_locations */
    /* VK_AMD_rasterization_order */
    /* VK_AMD_shader_ballot */
    /* VK_AMD_shader_explicit_vertex_parameter */
    /* VK_AMD_shader_fragment_mask */
    /* VK_AMD_shader_image_load_store_lod */
    /* VK_AMD_shader_info */
    /* VK_AMD_shader_trinary_minmax */
    /* VK_AMD_texture_gather_bias_lod */
    /* VK_AMD_wave_limits */
    /* VK_ANDROID_external_memory_android_hardware_buffer */
    /* VK_ANDROID_native_buffer */
    /* VK_EXT_2d_imageview_3d_image */
    /* VK_EXT_calibrated_timestamps */
    /* VK_EXT_debug_marker */
    /* VK_EXT_debug_report */
    /* VK_EXT_debug_utils */
    /* VK_EXT_depth_clip_control */
    /* VK_EXT_depth_clip_enable */
    /* VK_EXT_depth_range_unrestricted */
    /* VK_EXT_device_memory_report */
    /* VK_EXT_display_control */
    /* VK_EXT_external_memory_dma_buf */
    /* VK_EXT_filter_cubic */
    /* VK_EXT_full_screen_exclusive */
    /* VK_EXT_global_priority_query */
    /* VK_EXT_hdr_metadata */
    /* VK_EXT_image_drm_format_modifier */
    /* VK_EXT_load_store_op_none */
    /* VK_EXT_memory_priority */
    /* VK_EXT_metal_objects */
    /* VK_EXT_non_seamless_cube_map */
    /* VK_EXT_physical_device_drm */
    /* VK_EXT_pipeline_creation_feedback */
    /* VK_EXT_post_depth_coverage */
    /* VK_EXT_queue_family_foreign */
    /* VK_EXT_rgba10x6_formats */
    /* VK_EXT_separate_stencil_usage */
    /* VK_EXT_shader_module_identifier */
    /* VK_EXT_shader_stencil_export */
    /* VK_EXT_swapchain_colorspace */
    /* VK_EXT_tooling_info */
    /* VK_EXT_validation_cache */
    /* VK_EXT_vertex_input_dynamic_state */
    /* VK_EXT_video_decode_h264 */
    /* VK_EXT_video_decode_h265 */
    /* VK_EXT_video_encode_h264 */
    /* VK_EXTX_portability_subset */
    /* VK_GOOGLE_decorate_string */
    /* VK_GOOGLE_display_timing */
    /* VK_GOOGLE_hlsl_functionality1 */
    /* VK_GOOGLE_sampler_filtering_precision */
    /* VK_GOOGLE_user_type */
    /* VK_HUAWEI_prerotation */
    /* VK_HUAWEI_smart_cache */
    /* VK_IMG_filter_cubic */

    // [DO NOT EXPOSE] Enables certain formats in Vulkan, we just enable them if available or else we need to make format support query functions in LogicalDevice as well
    /* VK_IMG_format_pvrtc */

    /* VK_INTEL_performance_query */
    /* VK_KHR_bind_memory2 */
    /* VK_KHR_copy_commands2 */
    /* VK_KHR_create_renderpass2 */
    /* VK_KHR_deferred_host_operations */
    /* VK_KHR_descriptor_update_template */
    /* VK_KHR_device_group */
    /* VK_KHR_display_swapchain */
    /* VK_KHR_external_fence */
    /* VK_KHR_external_fence_fd */
    /* VK_KHR_external_fence_win32 */
    /* VK_KHR_external_memory */
    /* VK_KHR_external_memory_fd */
    /* VK_KHR_external_memory_win32 */
    /* VK_KHR_external_semaphore */
    /* VK_KHR_external_semaphore_fd */
    /* VK_KHR_external_semaphore_win32 */
    /* VK_KHR_format_feature_flags2 */
    /* VK_KHR_get_memory_requirements2 */
    /* VK_KHR_get_physical_device_properties2 */
    /* VK_KHR_image_format_list */
    /* VK_KHR_incremental_present */
    /* VK_KHR_pipeline_library */
    /* VK_KHR_relaxed_block_layout */
    /* VK_KHR_shader_clock */
    /* VK_KHR_shader_non_semantic_info */
    /* VK_KHR_shared_presentable_image */
    /* VK_KHR_storage_buffer_storage_class */
    /* VK_KHR_surface */

    /* VK_KHR_swapchain */ // we want to expose this extension as feature
    core::bitflag<E_SWAPCHAIN_MODE> swapchainMode = E_SWAPCHAIN_MODE::ESM_NONE;

    /* VK_KHR_swapchain_mutable_format */
    /* VK_KHR_video_decode_queue */
    /* VK_KHR_video_encode_queue */
    /* VK_KHR_win32_keyed_mutex */
    /* VK_KHR_win32_surface */
    /* VK_MESA_query_timestamp */
    /* VK_MVK_ios_surface */
    /* VK_MVK_macos_surface */
    /* VK_MVK_moltenvk */
    /* VK_NV_acquire_winrt_display */
    /* VK_NV_clip_space_w_scaling */
    /* VK_NV_cuda_kernel_launch */
    /* VK_NV_dedicated_allocation */
    /* VK_NV_device_debug_checkpoints */
    /* VK_NV_external_memory_win32 */
    /* VK_NV_fill_rectangle */
    /* VK_NV_fragment_coverage_to_color */
    /* VK_NV_framebuffer_mixed_samples */
    /* VK_NV_geometry_shader_passthrough */
    /* VK_NV_glsl_shader */
    /* VK_NV_low_latency */
    /* VK_NV_present_barrier */
    /* VK_NV_rdma_memory */
    /* VK_NV_sample_mask_override_coverage */
    /* VK_NV_shader_subgroup_partitioned */
    /* VK_NV_texture_dirty_tile_map */
    /* VK_NV_video_queue */
    /* VK_NV_viewport_array2 */
    /* VK_NV_viewport_swizzle */
    /* VK_NV_win32_keyed_mutex */
    /* VK_QCOM_render_pass_shader_resolve */
    /* VK_QCOM_render_pass_store_ops */
    /* VK_QCOM_render_pass_transform */
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
