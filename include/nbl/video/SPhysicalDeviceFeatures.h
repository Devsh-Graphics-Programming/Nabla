#ifndef _NBL_VIDEO_S_PHYSICAL_DEVICE_FEATURES_H_INCLUDED_
#define _NBL_VIDEO_S_PHYSICAL_DEVICE_FEATURES_H_INCLUDED_

namespace nbl::video
{

struct SPhysicalDeviceFeatures
{
    /* Vulkan 1.0 Core  */
    bool robustBufferAccess = false;
    //VkBool32    fullDrawIndexUint32;
    bool imageCubeArray = false;
    //VkBool32    independentBlend;
    bool geometryShader    = false;
    //VkBool32    tessellationShader;
    //VkBool32    sampleRateShading;
    //VkBool32    dualSrcBlend;
    bool logicOp = false;
    bool multiDrawIndirect = false;
    //VkBool32    drawIndirectFirstInstance;
    //VkBool32    depthClamp;
    //VkBool32    depthBiasClamp;
    //VkBool32    fillModeNonSolid;
    //VkBool32    depthBounds;
    //VkBool32    wideLines;
    //VkBool32    largePoints;
    //VkBool32    alphaToOne;
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
    //VkBool32    shaderClipDistance;
    //VkBool32    shaderCullDistance;
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
    //VkBool32           shaderDrawParameters;

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
    //VkBool32           bufferDeviceAddress;
    //VkBool32           bufferDeviceAddressCaptureReplay;
    //VkBool32           bufferDeviceAddressMultiDevice;
    //VkBool32           vulkanMemoryModel;
    //VkBool32           vulkanMemoryModelDeviceScope;
    //VkBool32           vulkanMemoryModelAvailabilityVisibilityChains;
    //VkBool32           shaderOutputViewportIndex;
    //VkBool32           shaderOutputLayer;
    //VkBool32           subgroupBroadcastDynamicId;

    /* Vulkan 1.3 Core */
    //VkBool32           robustImageAccess;
    //VkBool32           inlineUniformBlock;
    //VkBool32           descriptorBindingInlineUniformBlockUpdateAfterBind;
    //VkBool32           pipelineCreationCacheControl;
    //VkBool32           privateData;
    //VkBool32           shaderDemoteToHelperInvocation;
    //VkBool32           shaderTerminateInvocation;
    //VkBool32           subgroupSizeControl;
    //VkBool32           computeFullSubgroups;
    //VkBool32           synchronization2;
    //VkBool32           textureCompressionASTC_HDR;
    //VkBool32           shaderZeroInitializeWorkgroupMemory;
    //VkBool32           dynamicRendering;
    //VkBool32           shaderIntegerDotProduct;
    //VkBool32           maintenance4; -> Doesn't make sense to expose, too vulkan specific

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

    /* BufferDeviceAddressFeaturesKHR *//* VK_EXT_buffer_device_address */
    bool bufferDeviceAddress = false;
    //VkBool32           bufferDeviceAddressCaptureReplay;
    //VkBool32           bufferDeviceAddressMultiDevice;

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

    /* RayQueryFeaturesKHR */
    bool rayQuery = false;
            
    /* AccelerationStructureFeaturesKHR */
    bool accelerationStructure = false;
    bool accelerationStructureCaptureReplay = false;
    bool accelerationStructureIndirectBuild = false;
    bool accelerationStructureHostCommands = false;
    bool descriptorBindingAccelerationStructureUpdateAfterBind = false;
            
    /* RayTracingPipelineFeaturesKHR */
    bool rayTracingPipeline = false;
    bool rayTracingPipelineShaderGroupHandleCaptureReplay = false;
    bool rayTracingPipelineShaderGroupHandleCaptureReplayMixed = false;
    bool rayTracingPipelineTraceRaysIndirect = false;
    bool rayTraversalPrimitiveCulling = false;

    /* Nabla */
    bool dispatchBase = false; // true in Vk, false in GL
    bool allowCommandBufferQueryCopies = false;
};

} // nbl::video

#endif
