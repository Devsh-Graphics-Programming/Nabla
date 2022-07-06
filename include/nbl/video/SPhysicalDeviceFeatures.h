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

    // [DO NOT EXPOSE] these 3 don't make a difference, just shortcut from Querying support from PhysicalDevice
    //VkBool32    textureCompressionETC2;
    //VkBool32    textureCompressionASTC_LDR;
    //VkBool32    textureCompressionBC;
    
    //VkBool32    occlusionQueryPrecise;
    //VkBool32    pipelineStatisticsQuery; [TODO]
    //VkBool32    vertexPipelineStoresAndAtomics;
    //VkBool32    fragmentStoresAndAtomics;
    //VkBool32    shaderTessellationAndGeometryPointSize;
    //VkBool32    shaderImageGatherExtended;
    //VkBool32    shaderStorageImageExtendedFormats;
    bool shaderStorageImageMultisample = false;
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
    
    //VkBool32    variableMultisampleRate;
    bool inheritedQueries = false;

    /* Vulkan 1.1 Core */
    //VkBool32           storageBuffer16BitAccess;
    //VkBool32           uniformAndStorageBuffer16BitAccess;
    //VkBool32           storagePushConstant16;
    //VkBool32           storageInputOutput16;
    
    // [TODO] do not expose multiview yet
    //VkBool32           multiview;
    //VkBool32           multiviewGeometryShader;
    //VkBool32           multiviewTessellationShader;
    
    // [Future TODO]:
    //VkBool32           variablePointersStorageBuffer;
    //VkBool32           variablePointers;
    
    //VkBool32           protectedMemory;
    
    // [DO NOT EXPOSE] Enables certain formats in Vulkan, we just enable them if available or else we need to make format support query functions in LogicalDevice as well
    //VkBool32           samplerYcbcrConversion;
    bool shaderDrawParameters = false;




    /* Vulkan 1.2 Core */

    bool samplerMirrorClampToEdge = false;          // ALIAS: VK_KHR_sampler_mirror_clamp_to_edge
    bool drawIndirectCount = false;                 // ALIAS: VK_KHR_draw_indirect_count

    // or VK_KHR_8bit_storage:
    //VkBool32           storageBuffer8BitAccess;
    //VkBool32           uniformAndStorageBuffer8BitAccess;
    //VkBool32           storagePushConstant8;
    
    // or VK_KHR_shader_atomic_int64:
    //VkBool32           shaderBufferInt64Atomics;
    //VkBool32           shaderSharedInt64Atomics;
   
    // or VK_KHR_shader_float16_int8:
    //VkBool32           shaderFloat16;
    //VkBool32           shaderInt8;
    
    // or VK_EXT_descriptor_indexing
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
    //VkBool32           runtimeDescriptorArray; // [FUTURE TODO]
    
    bool                 samplerFilterMinmax = false;   // ALIAS: VK_EXT_sampler_filter_minmax
    
    //VkBool32           scalarBlockLayout;     // or VK_EXT_scalar_block_layout // [FUTURE TODO]
    
    //VkBool32           imagelessFramebuffer;  // or VK_KHR_imageless_framebuffer 
    
    //VkBool32           uniformBufferStandardLayout;   // or VK_KHR_uniform_buffer_standard_layout
    
    // [DO NOT EXPOSE]
    //VkBool32           shaderSubgroupExtendedTypes;   // or VK_KHR_shader_subgroup_extended_types
    
    //VkBool32           separateDepthStencilLayouts;   // or VK_KHR_separate_depth_stencil_layouts
    
    // [TODO] And add implementation to engine
    //VkBool32           hostQueryReset;                // or VK_EXT_host_query_reset
    
    //VkBool32           timelineSemaphore;             // or VK_KHR_timeline_semaphore
    
    // or VK_KHR_timeline_semaphore:
    bool bufferDeviceAddress = false;
    //VkBool32           bufferDeviceAddressCaptureReplay; // [DO NOT EXPOSE] for capture tools not engines
    //VkBool32           bufferDeviceAddressMultiDevice;
    
    // or VK_KHR_vulkan_memory_model
    //VkBool32           vulkanMemoryModel;
    //VkBool32           vulkanMemoryModelDeviceScope;
    //VkBool32           vulkanMemoryModelAvailabilityVisibilityChains;
    
    //VkBool32           shaderOutputViewportIndex;     // ALIAS: VK_EXT_shader_viewport_index_layer
    //VkBool32           shaderOutputLayer;             // ALIAS: VK_EXT_shader_viewport_index_layer
    //VkBool32           subgroupBroadcastDynamicId;    // if Vulkan 1.2 is supported




    /* Vulkan 1.3 Core */
    
    // [TODO] robustness stuff
    //VkBool32           robustImageAccess;                 //  or VK_EXT_image_robustness
    
    //  or VK_EXT_inline_uniform_block:
    //VkBool32           inlineUniformBlock;
    //VkBool32           descriptorBindingInlineUniformBlockUpdateAfterBind;
    
    // [TODO]
    //VkBool32           pipelineCreationCacheControl;      // or VK_EXT_pipeline_creation_cache_control
   
    // [DO NOT EXPOSE] ever
    //VkBool32           privateData;                       // or VK_EXT_private_data
    
    //VkBool32           shaderDemoteToHelperInvocation;    // or VK_EXT_shader_demote_to_helper_invocation
    //VkBool32           shaderTerminateInvocation;         // or VK_KHR_shader_terminate_invocation
    
    // or VK_EXT_subgroup_size_control
    bool subgroupSizeControl  = false;
    bool computeFullSubgroups = false;
    
    // [DO NOT EXPOSE] and false because we havent rewritten our frontend API for that
    //VkBool32           synchronization2;                      // or VK_KHR_synchronization2
    
    // [DO NOT EXPOSE] Doesn't make a difference, just shortcut from Querying support from PhysicalDevice
    //VkBool32           textureCompressionASTC_HDR;            // or VK_EXT_texture_compression_astc_hdr
    
    // [DO NOT EXPOSE] would require doing research into the GL/GLES robustness extensions
    //VkBool32           shaderZeroInitializeWorkgroupMemory;   // or VK_KHR_zero_initialize_workgroup_memory
    
    // [DO NOT EXPOSE] EVIL
    //VkBool32           dynamicRendering;                      // or VK_KHR_dynamic_rendering
    
    //VkBool32           shaderIntegerDotProduct;               // or VK_KHR_shader_integer_dot_product
    //VkBool32           maintenance4;                          // [DO NOT EXPOSE] doesn't make sense




    /* Vulkan Extensions */

    /* CoherentMemoryFeaturesAMD *//* VK_AMD_device_coherent_memory */
    //VkBool32           deviceCoherentMemory;

    /* VK_AMD_shader_early_and_late_fragment_tests *//* couldn't find the struct/extension in latest vulkan headers */

    /* RasterizationOrderAttachmentAccessFeaturesARM *//* VK_ARM_rasterization_order_attachment_access */
    //VkBool32           rasterizationOrderColorAttachmentAccess;
    //VkBool32           rasterizationOrderDepthAttachmentAccess;
    //VkBool32           rasterizationOrderStencilAttachmentAccess;
    
    // [DO NOT EXPOSE] Enables certain formats in Vulkan, we just enable them if available or else we need to make format support query functions in LogicalDevice as well
    /* 4444FormatsFeaturesEXT *//* VK_EXT_4444_formats */
    
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
    //VkBool32           indexTypeUint8;
    
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
    //VkBool32           rayTracingMotionBlur;
    //VkBool32           rayTracingMotionBlurPipelineTraceRaysIndirect;

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

    // [DO NOT EXPOSE] Enables certain formats in Vulkan, we just enable them if available or else we need to make format support query functions in LogicalDevice as well
    /* VK_IMG_format_pvrtc */
   
    /* Nabla */
    // No Nabla Specific Features for now
};

} // nbl::video

#endif
