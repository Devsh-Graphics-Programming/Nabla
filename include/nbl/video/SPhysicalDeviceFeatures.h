#ifndef _NBL_VIDEO_S_PHYSICAL_DEVICE_FEATURES_H_INCLUDED_
#define _NBL_VIDEO_S_PHYSICAL_DEVICE_FEATURES_H_INCLUDED_

#include "nbl/video/ECommonEnums.h"

namespace nbl::video
{
//! Usage of feature API
//! ## LogicalDevice creation enabled features shouldn't necessarily equal the ones it reports as enabled (superset)
//! **RARE: Creating a physical device with all advertised features/extensions:**
//! auto features = physicalDevice->getFeatures();
//! 
//! ILogicalDevice::SCreationParams params = {};
//! params.queueParamsCount = ; // set queue stuff
//! params.queueParams = ; // set queue stuff
//! params.enabledFeatures = features;
//! auto device = physicalDevice->createLogicalDevice(params);
//! **FREQUENT: Choosing a physical device with the features**
//! IPhysicalDevice::SRequiredProperties props = {}; // default initializes to apiVersion=1.1, deviceType = ET_UNKNOWN, pipelineCacheUUID = '\0', device UUID=`\0`, driverUUID=`\0`, deviceLUID=`\0`, deviceNodeMask= ~0u, driverID=UNKNOWN
//! // example of particular config
//! props.apiVersion = 1.2;
//! props.deviceTypeMask = ~IPhysicalDevice::ET_CPU;
//! props.driverIDMask = ~(EDI_AMD_PROPRIETARY|EDI_INTEL_PROPRIETARY_WINDOWS); // would be goot to turn the enum into a mask
//! props.conformanceVersion = 1.2;
//! 
//! SDeviceFeatures requiredFeatures = {};
//! requiredFeatures.rayQuery = true;
//! 
//! SDeviceLimits minimumLimits = {}; // would default initialize to worst possible values (small values for maximum sizes, large values for alignments, etc.)

struct SPhysicalDeviceFeatures
{
    /* Vulkan 1.0 Core  */

    bool robustBufferAccess = false;

    // [DO NOT EXPOSE] Roadmap 2022 requires support for these, device support is ubiquitous and enablement is unlikely to harm performance
    //bool fullDrawIndexUint32 = true; // TODO: expose the `maxDrawIndexedIndexValue` limit instead
 
    // [DO NOT EXPOSE] ROADMAP 2022 and good device support
    //bool imageCubeArray = true;
    //bool independentBlend = true;

    bool geometryShader    = false;
    bool tessellationShader = false;

    // [DO NOT EXPOSE] ROADMAP 2022 and good device support
    //bool sampleRateShading = true;

    bool dualSrcBlend = false;
    bool logicOp = false;

    // [DO NOT EXPOSE] Roadmap 2022 requires support for these, device support is ubiquitous and enablement is unlikely to harm performance
    //bool multiDrawIndirect = true;
    //bool drawIndirectFirstInstance = true;
    //bool depthClamp = true;
    //bool depthBiasClamp = true;

    bool fillModeNonSolid = false;
    bool depthBounds = false;
    bool wideLines = false;
    bool largePoints = false;
    bool alphaToOne = false;
    bool multiViewport = false;

    // [DO NOT EXPOSE] Roadmap 2022 requires support for these, device support is ubiquitous
    // bool samplerAnisotropy = true;

    // [DO NOT EXPOSE] these 3 don't make a difference, just shortcut from Querying support from PhysicalDevice
    //bool    textureCompressionETC2;
    //bool    textureCompressionASTC_LDR;
    //bool    textureCompressionBC;
 
    // [DO NOT EXPOSE] ROADMAP 2022 and good device support
    //bool occlusionQueryPrecise = true;

    bool pipelineStatisticsQuery = false;

    // Enabled by Default, Moved to Limits
    //bool vertexPipelineStoresAndAtomics;
    //bool fragmentStoresAndAtomics;
    //bool shaderTessellationAndGeometryPointSize;
    //bool shaderImageGatherExtended;

    // VK SPEC: "shaderStorageImageExtendedFormats feature only adds a guarantee of format support, which is specified for the whole physical device.
    // Therefore enabling or disabling the feature via vkCreateDevice has no practical effect."
    //bool shaderStorageImageExtendedFormats = true;

    // [EXPOSE AS A LIMIT] Cannot be always enabled cause Intel ARC is handicapped
    //bool shaderStorageImageMultisample;

    bool shaderStorageImageReadWithoutFormat = false;
    bool shaderStorageImageWriteWithoutFormat = false;

    // [DO NOT EXPOSE] ROADMAP 2022 and good device support
    //bool shaderUniformBufferArrayDynamicIndexing = true;
    //bool shaderSampledImageArrayDynamicIndexing = true;
    //bool shaderStorageBufferArrayDynamicIndexing = true;
 
    // [EXPOSE AS A LIMIT] ROADMAP 2022 but Apple GPUs have poor support
    //bool shaderStorageImageArrayDynamicIndexing;

    bool shaderClipDistance = false;
    bool shaderCullDistance = false;

    // [EXPOSE AS A LIMIT] Cannot be always enabled cause Intel ARC is handicapped
    //bool shaderFloat64;

    // Enabled by Default, Moved to Limits
    //bool shaderInt64 = false;
    //bool shaderInt16 = false;

    bool shaderResourceResidency = false;
    bool shaderResourceMinLod = false;
    
    // [TODO LATER] once we implemented sparse resources
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

    // [EXPOSE AS A LIMIT] Enabled by Default, Moved to Limits : ALIAS VK_KHR_16bit_storage
    //bool storageBuffer16BitAccess = false;
    //bool uniformAndStorageBuffer16BitAccess = false;
    //bool storagePushConstant16 = false;
    //bool storageInputOutput16 = false;

    // [DO NOT EXPOSE] Required to be present when Vulkan 1.1 is supported
    //bool multiview;
    // 
    // [TODO LATER] do not expose this part of multiview yet
    /* VK_KHR_multiview */ 
    //bool multiviewGeometryShader;
    //bool multiviewTessellationShader;

    // [EXPOSE AS A LIMIT] Its just a shader capability
    //bool variablePointers;
    // [DO NOT EXPOSE] Under Vulkan 1.1 if `variablePointers` is present it implies `variablePointersStorageBuffer`
    //bool variablePointersStorageBuffer = variablePointers;
    
    /* or via VkPhysicalDeviceProtectedMemoryProperties provided by Vulkan 1.1 */
    //bool           protectedMemory; // [DO NOT EXPOSE] not gonna expose until we have a need to

    // [DO NOT EXPOSE] Enables certain formats in Vulkan, we just enable them if available or else we need to make format support query functions in LogicalDevice as well
    //bool           samplerYcbcrConversion;

    bool shaderDrawParameters = false;  /* ALIAS: VK_KHR_shader_draw_parameters */


    /* Vulkan 1.2 Core */

    // [DO NOT EXPOSE] Device support ubiquitous
    //bool samplerMirrorClampToEdge = true;          // ALIAS: VK_KHR_sampler_mirror_clamp_to_edge
 
    // [EXPOSE AS A LIMIT] ROADMAP 2022 requires support but mobile GPUs don't support well, exposed as a limit `maxDrawIndirectCount`
    //bool drawIndirectCount; // ALIAS: VK_KHR_draw_indirect_count

    // Enabled by Default, Moved to Limits
    // or VK_KHR_8bit_storage:
    //bool storageBuffer8BitAccess = false;
    //bool uniformAndStorageBuffer8BitAccess = false;
    //bool storagePushConstant8 = false;
 
    // Enabled by Default, Moved to Limits
    // or VK_KHR_shader_atomic_int64:
    //bool shaderBufferInt64Atomics = false;
    //bool shaderSharedInt64Atomics = false;

    // Enabled by Default, Moved to Limits
    // or VK_KHR_shader_float16_int8:
    //bool shaderFloat16 = false;
    //bool shaderInt8 = false;
    
    // or VK_EXT_descriptor_indexing
    bool descriptorIndexing = false;  // always enable ?
    bool shaderInputAttachmentArrayDynamicIndexing = false;

    // on ROADMAP 2022 but need to check stuff first
    bool shaderUniformTexelBufferArrayDynamicIndexing = false;
    bool shaderStorageTexelBufferArrayDynamicIndexing = false;
    bool shaderUniformBufferArrayNonUniformIndexing = false;
    bool shaderSampledImageArrayNonUniformIndexing = false;
    bool shaderStorageBufferArrayNonUniformIndexing = false;
    bool shaderStorageImageArrayNonUniformIndexing = false;

    bool shaderInputAttachmentArrayNonUniformIndexing = false;

    // on ROADMAP 2022 but need to check stuff first
    bool shaderUniformTexelBufferArrayNonUniformIndexing = false;
    bool shaderStorageTexelBufferArrayNonUniformIndexing = false;

    bool descriptorBindingUniformBufferUpdateAfterBind = false;

    // on ROADMAP 2022 but need to check stuff first
    bool descriptorBindingSampledImageUpdateAfterBind = false;
    bool descriptorBindingStorageImageUpdateAfterBind = false;
    bool descriptorBindingStorageBufferUpdateAfterBind = false;
    bool descriptorBindingUniformTexelBufferUpdateAfterBind = false;
    bool descriptorBindingStorageTexelBufferUpdateAfterBind = false;

    // supported by ROADMAP 2022 but might be expensive to enable for the implementation
    bool descriptorBindingUpdateUnusedWhilePending = false;
    bool descriptorBindingPartiallyBound = false;
    bool descriptorBindingVariableDescriptorCount = false;
    bool runtimeDescriptorArray = false;
    
    bool samplerFilterMinmax = false;   // ALIAS: VK_EXT_sampler_filter_minmax

    // [DO NOT EXPOSE] Roadmap 2022 requires support for these we always enable and they're unlikely to harm performance
    //bool scalarBlockLayout = true;     // or VK_EXT_scalar_block_layout
    
    // [DO NOT EXPOSE] Vulkan 1.2 requires these
    //bool imagelessFramebuffer = true;  // or VK_KHR_imageless_framebuffer // [FUTURE TODO] https://github.com/Devsh-Graphics-Programming/Nabla/issues/378
    //bool uniformBufferStandardLayout = true; // or VK_KHR_uniform_buffer_standard_layout
    //bool shaderSubgroupExtendedTypes = true;   // or VK_KHR_shader_subgroup_extended_types
    //bool separateDepthStencilLayouts = true;   // or VK_KHR_separate_depth_stencil_layoutse [TODO Implement]
    //bool hostQueryReset = true;                // or VK_EXT_host_query_reset [TODO Implement]
    //bool timelineSemaphore = true;             // or VK_KHR_timeline_semaphore [TODO Implement]
    
    // or VK_KHR_buffer_device_address:
    bool bufferDeviceAddress = false;
    // bool           bufferDeviceAddressCaptureReplay; // [DO NOT EXPOSE] for capture tools not engines
    bool bufferDeviceAddressMultiDevice = false;
    
    // [EXPOSE AS A LIMIT] ROADMAP2022 wants them. ALIAS VK_KHR_vulkan_memory_model
    //bool vulkanMemoryModel;
    //bool vulkanMemoryModelDeviceScope;
    //bool vulkanMemoryModelAvailabilityVisibilityChains;


    /* Vulkan 1.3 Core */
    
    // [DO NOT EXPOSE] EVIL regressive step back into OpenGL/Dx10 times
    //  or VK_EXT_inline_uniform_block:
    //bool           inlineUniformBlock;
    //bool           descriptorBindingInlineUniformBlockUpdateAfterBind;

    // [DO NOT EXPOSE] ever we have our own mechanism
    //bool           privateData;                       // or VK_EXT_private_data
    
    bool shaderDemoteToHelperInvocation = false;    // or VK_EXT_shader_demote_to_helper_invocation
    bool shaderTerminateInvocation = false;         // or VK_KHR_shader_terminate_invocation
    
    // or VK_EXT_subgroup_size_control
    bool subgroupSizeControl  = false;
    bool computeFullSubgroups = false;
    
    // [DO NOT EXPOSE] REQUIRE but we need to rewrite our frontend API for that: https://github.com/Devsh-Graphics-Programming/Nabla/issues/384
    //bool           synchronization2;                      // or VK_KHR_synchronization2
    
    // [DO NOT EXPOSE] Doesn't make a difference, just shortcut from Querying support from PhysicalDevice
    //bool           textureCompressionASTC_HDR;            // or VK_EXT_texture_compression_astc_hdr
    
    // [TODO] Expose
    //bool           shaderZeroInitializeWorkgroupMemory;   // or VK_KHR_zero_initialize_workgroup_memory
    
    // [DO NOT EXPOSE] EVIL
    //bool           dynamicRendering;                      // or VK_KHR_dynamic_rendering
    
    bool shaderIntegerDotProduct = false;               // or VK_KHR_shader_integer_dot_product



    /* Vulkan Extensions */

    /* RasterizationOrderAttachmentAccessFeaturesARM *//* VK_ARM_rasterization_order_attachment_access */
    bool rasterizationOrderColorAttachmentAccess = false;
    bool rasterizationOrderDepthAttachmentAccess = false;
    bool rasterizationOrderStencilAttachmentAccess = false;
    
    // [DO NOT EXPOSE] Enables certain formats in Vulkan, we just enable them if available or else we need to make format support query functions in LogicalDevice as well
    /* 4444FormatsFeaturesEXT *//* VK_EXT_4444_formats */

    // [DO NOT EXPOSE] right now, no idea if we'll ever expose and implement those but they'd all be false for OpenGL
    /* BlendOperationAdvancedFeaturesEXT *//* VK_EXT_blend_operation_advanced */
    //bool           advancedBlendCoherentOperations;
    
    // [DO NOT EXPOSE] not going to expose custom border colors for now
    /* BorderColorSwizzleFeaturesEXT *//* VK_EXT_border_color_swizzle */
    //bool           borderColorSwizzle;
    //bool           borderColorSwizzleFromImage;

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

    // [DO NOT EXPOSE] we're fully GPU driven anyway with sorted pipelines.
    /* ExtendedDynamicStateFeaturesEXT *//* VK_EXT_extended_dynamic_state */
    //bool           extendedDynamicState;
    
    // [DO NOT EXPOSE] we're fully GPU driven anyway with sorted pipelines.
    /* ExtendedDynamicState2FeaturesEXT *//* VK_EXT_extended_dynamic_state2 */
    //bool           extendedDynamicState2;
    //bool           extendedDynamicState2LogicOp;
    //bool           extendedDynamicState2PatchControlPoints;

    /* FragmentShaderInterlockFeaturesEXT *//* VK_EXT_fragment_shader_interlock */
    bool fragmentShaderSampleInterlock = false;
    bool fragmentShaderPixelInterlock = false;
    bool fragmentShaderShadingRateInterlock = false;

    // [DO NOT EXPOSE] pointless to implement currently
    /* ImageViewMinLodFeaturesEXT *//* VK_EXT_image_view_min_lod */
    //bool           minLod;

    /* IndexTypeUint8FeaturesEXT *//* VK_EXT_index_type_uint8 */
    bool indexTypeUint8 = false;
    
    // [DO NOT EXPOSE] this extension is dumb, if we're recording that many draws we will be using Multi Draw INDIRECT which is better supported
    /* MultiDrawFeaturesEXT *//* VK_EXT_multi_draw */
    //bool           multiDraw;

    // [DO NOT EXPOSE] pointless to expose without exposing VK_EXT_memory_priority and the memory query feature first
    /* PageableDeviceLocalMemoryFeaturesEXT *//* VK_EXT_pageable_device_local_memory */
    //bool           pageableDeviceLocalMemory;

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

    /* ShaderAtomicFloatFeaturesEXT *//* VK_EXT_shader_atomic_float */
    bool shaderBufferFloat32Atomics = false;
    bool shaderBufferFloat32AtomicAdd = false;
    bool shaderBufferFloat64Atomics = false;
    bool shaderBufferFloat64AtomicAdd = false;
    bool shaderSharedFloat32Atomics = false;
    bool shaderSharedFloat32AtomicAdd = false;
    bool shaderSharedFloat64Atomics = false;
    bool shaderSharedFloat64AtomicAdd = false;
    bool shaderImageFloat32Atomics = false;
    bool shaderImageFloat32AtomicAdd = false;
    bool sparseImageFloat32Atomics = false;
    bool sparseImageFloat32AtomicAdd = false;

    /* ShaderAtomicFloat2FeaturesEXT *//* VK_EXT_shader_atomic_float2 */
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
    
    /* ShaderImageAtomicInt64FeaturesEXT *//* VK_EXT_shader_image_atomic_int64 */
    bool shaderImageInt64Atomics = false;
    bool sparseImageInt64Atomics = false;

    // [DO NOT EXPOSE] ever because of our disdain for XForm feedback
    /* TransformFeedbackFeaturesEXT *//* VK_EXT_transform_feedback */
    //bool           transformFeedback;
    //bool           geometryStreams;

    // [TODO LATER] we would have to change the API
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

    /* AccelerationStructureFeaturesKHR *//* VK_KHR_acceleration_structure */
    bool accelerationStructure = false;
    bool accelerationStructureIndirectBuild = false;
    bool accelerationStructureHostCommands = false;
    bool descriptorBindingAccelerationStructureUpdateAfterBind = false;
            
    // [DO NOT EXPOSE] not implementing or exposing VRS in near or far future
    /* FragmentShadingRateFeaturesKHR *//* VK_KHR_fragment_shading_rate */
    //bool           pipelineFragmentShadingRate;
    //bool           primitiveFragmentShadingRate;
    //bool           attachmentFragmentShadingRate;

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
    bool shaderDeviceClock = false;
    //bool shaderSubgroupClock; // Enabled by Default, Moved to Limits 

    /* VK_KHR_shader_draw_parameters *//* MOVED TO Vulkan 1.1 Core */
    /* VK_KHR_shader_float16_int8 *//* MOVED TO Vulkan 1.2 Core */
    /* VK_KHR_shader_integer_dot_product *//* MOVED TO Vulkan 1.3 Core */
    /* VK_KHR_shader_subgroup_extended_types *//* MOVED TO Vulkan 1.2 Core */

    /* ShaderSubgroupUniformControlFlowFeaturesKHR *//* VK_KHR_shader_subgroup_uniform_control_flow */
    bool shaderSubgroupUniformControlFlow = false;
    
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
    bool computeDerivativeGroupQuads = false;
    bool computeDerivativeGroupLinear = false;

    /* CooperativeMatrixFeaturesNV *//* VK_NV_cooperative_matrix */
    bool cooperativeMatrix = false;
    bool cooperativeMatrixRobustBufferAccess = false;

    /* RayTracingMotionBlurFeaturesNV *//* VK_NV_ray_tracing_motion_blur */
    bool rayTracingMotionBlur = false;
    bool rayTracingMotionBlurPipelineTraceRaysIndirect = false;

    /* CoverageReductionModeFeaturesNV *//* VK_NV_coverage_reduction_mode */
    bool coverageReductionMode = false;

    /* DeviceGeneratedCommandsFeaturesNV *//* VK_NV_device_generated_commands */
    bool deviceGeneratedCommands = false;

    /* MeshShaderFeaturesNV *//* VK_NV_mesh_shader */
    bool taskShader = false;
    bool meshShader = false;

    /* RepresentativeFragmentTestFeaturesNV *//* VK_NV_representative_fragment_test */
    bool representativeFragmentTest = false;

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

    bool pipelineCreationCacheControl = false;      // or VK_EXT_pipeline_creation_cache_control

    // [TODO] need new commandbuffer methods, etc
    /* ColorWriteEnableFeaturesEXT *//* VK_EXT_color_write_enable */
    bool colorWriteEnable = false;

    // [TODO] now we need API to deal with queries and begin/end conditional blocks
    /* ConditionalRenderingFeaturesEXT *//* VK_EXT_conditional_rendering */
    bool conditionalRendering = false;
    bool inheritedConditionalRendering = false;

    /* DeviceMemoryReportFeaturesEXT *//* VK_EXT_device_memory_report */
    bool deviceMemoryReport = false;

    /* FragmentDensityMapFeaturesEXT *//* VK_EXT_fragment_density_map */
    bool fragmentDensityMap = false;
    bool fragmentDensityMapDynamic = false;
    bool fragmentDensityMapNonSubsampledImages = false;

    /* FragmentDensityMap2FeaturesEXT *//* VK_EXT_fragment_density_map2 */
    bool fragmentDensityMapDeferred = false;

    // [TODO] Investigate later
    /* Image2DViewOf3DFeaturesEXT *//* VK_EXT_image_2d_view_of_3d */
    //bool           image2DViewOf3D;
    //bool           sampler2DViewOf3D;

    /*
        This feature adds stricter requirements for how out of bounds reads from images are handled. 
        Rather than returning undefined values,
        most out of bounds reads return R, G, and B values of zero and alpha values of either zero or one.
        Components not present in the image format may be set to zero 
        or to values based on the format as described in Conversion to RGBA in vulkan specification.
    */
    bool robustImageAccess = false;                 //  or VK_EXT_image_robustness

    /* InlineUniformBlockFeaturesEXT *//* VK_EXT_inline_uniform_block *//* MOVED TO Vulkan 1.3 Core */
    bool inlineUniformBlock = false;
    bool descriptorBindingInlineUniformBlockUpdateAfterBind = false;

    // [TODO] this feature introduces new/more pipeline state with VkPipelineRasterizationLineStateCreateInfoEXT
    /* LineRasterizationFeaturesEXT *//* VK_EXT_line_rasterization */
    // GL HINT (remove when implemented): MULTI_SAMPLE_LINE_WIDTH_RANGE (which is necessary for this) is guarded by !IsGLES || Version>=320 no idea is something enables this or not
    bool rectangularLines = false;
    bool bresenhamLines = false;
    bool smoothLines = false;
    // end of hint
    bool stippledRectangularLines = false;
    bool stippledBresenhamLines = false;
    bool stippledSmoothLines = false;

    /* MemoryPriorityFeaturesEXT *//* VK_EXT_memory_priority */
    bool memoryPriority = false;

    /* Robustness2FeaturesEXT *//* VK_EXT_robustness2 */
    // [TODO] Better descriptive name for the Vulkan robustBufferAccess2, robustImageAccess2 features
    bool robustBufferAccess2 = false;
    bool robustImageAccess2 = false;
    /*
        ! nullDescriptor: you can use `nullptr` for writing descriptors to sets and Accesses to null descriptors have well-defined behavior.
        [TODO] Handle `nullDescriptor` feature in the engine.
    */
    bool nullDescriptor = false;

    /* PerformanceQueryFeaturesKHR *//* VK_KHR_performance_query */
    bool performanceCounterQueryPools = false;
    bool performanceCounterMultipleQueryPools = false;

    /* PipelineExecutablePropertiesFeaturesKHR *//* VK_KHR_pipeline_executable_properties */
    bool pipelineExecutableInfo = false;

    /* CoherentMemoryFeaturesAMD *//* VK_AMD_device_coherent_memory */
    bool deviceCoherentMemory = false;

    /* VK_AMD_buffer_marker */
    bool bufferMarkerAMD = false;

    // [TODO]
    /* ASTCDecodeFeaturesEXT *//* VK_EXT_astc_decode_mode */
    //VkFormat           decodeMode;

    // [TODO] Promoted to VK1.1 core, haven't updated API to match
    /* VK_KHR_descriptor_update_template */

    // Enabled by Default, Moved to Limits
    /* TexelBufferAlignmentFeaturesEXT *//* VK_EXT_texel_buffer_alignment */

    // Enabled by Default, Moved to Limits
    /* VK_NV_sample_mask_override_coverage */

    // Enabled by Default, Moved to Limits
    /* VK_NV_shader_subgroup_partitioned */ // [TODO] have it contribute to shaderSubgroup reporting

    // Enabled by Default, Moved to Limits
    /* VK_AMD_gcn_shader */

    // Enabled by Default, Moved to Limits
    /* VK_AMD_gpu_shader_half_float */

    // Enabled by Default, Moved to Limits
    /* VK_AMD_gpu_shader_int16 */

    // Enabled by Default, Moved to Limits
    /* VK_AMD_shader_ballot */
    
    // Enabled by Default, Moved to Limits
    /* VK_AMD_shader_image_load_store_lod */

    // Enabled by Default, Moved to Limits
    /* VK_AMD_shader_trinary_minmax */

    // Enabled by Default, Moved to Limits
    /* VK_EXT_post_depth_coverage */

    // Enabled by Default, Moved to Limits
    /* VK_EXT_shader_stencil_export */

    // Enabled by Default, Moved to Limits
    /* VK_GOOGLE_decorate_string */

    // Enabled by Default, Moved to Limits
    /* VK_KHR_external_fence_fd */

    // Enabled by Default, Moved to Limits
    /* VK_KHR_external_fence_win32 */

    // Enabled by Default, Moved to Limits
    /* VK_KHR_external_memory_fd */

    // Enabled by Default, Moved to Limits
    /* VK_KHR_external_memory_win32 */

    // Enabled by Default, Moved to Limits
    /* VK_KHR_external_semaphore_fd */

    // Enabled by Default, Moved to Limits
    /* VK_KHR_external_semaphore_win32 */

    // Enabled by Default, Moved to Limits
    /* VK_KHR_shader_non_semantic_info */

    // Enabled by Default, Moved to Limits
    /* ShaderSMBuiltinsFeaturesNV *//* VK_NV_shader_sm_builtins */

    // Enabled by Default, Moved to Limits
    /* VK_KHR_fragment_shader_barycentric */

    // Enabled by Default, Moved to Limits
    /* VK_NV_geometry_shader_passthrough */

    // Enabled by Default, Moved to Limits
    /* VK_NV_viewport_swizzle */

    // [TODO] this one isn't in the headers
    /* GlobalPriorityQueryFeaturesKHR *//* VK_KHR_global_priority */
    //VkQueueGlobalPriorityKHR    globalPriority;

    // [TODO] this one isn't in the headers
    /* GraphicsPipelineLibraryFeaturesEXT *//* VK_EXT_graphics_pipeline_library */
    //bool           graphicsPipelineLibrary;

    // [TODO]
    // lots of support for anything that isn't mobile (list of unsupported devices since the extension was published: https://pastebin.com/skZAbL4F)
    /* VK_KHR_format_feature_flags2 */ // Promoted to core 1.3;

    // [TODO] this one isn't in the headers
    /* HostQueryResetFeatures *//* VK_EXT_host_query_reset *//* MOVED TO Vulkan 1.2 Core */

    // [TODO] this one isn't in the headers // Always enable, expose as limit
    /* VK_AMD_shader_early_and_late_fragment_tests */
    //bool shaderEarlyAndLateFragmentTests;

    // Enabled by Default, Moved to Limits 
    //bool           shaderOutputViewportIndex;     // ALIAS: VK_EXT_shader_viewport_index_layer
    //bool           shaderOutputLayer;             // ALIAS: VK_EXT_shader_viewport_index_layer

    // Enabled by Default, Moved to Limits 
    /* ShaderIntegerFunctions2FeaturesINTEL *//* VK_INTEL_shader_integer_functions2 */
    //bool           shaderIntegerFunctions2 = false;

    // Enabled by Default, Moved to Limits 
    /* ShaderImageFootprintFeaturesNV *//* VK_NV_shader_image_footprint */
    //bool           imageFootprint;

    // [TODO LATER] needs to figure out how extending our LOAD_OP enum would affect the GL backend
    /* VK_EXT_load_store_op_none */

    // [TODO LATER] Won't expose for now, API changes necessary
    /* VK_AMD_texture_gather_bias_lod */

    // [TODO LATER] when released in the SDK:
    // -Support for `GLSL_EXT_ray_cull_mask`, lets call it `rayCullMask`
    // - new pipeline stage and access masks but only in `KHR_synchronization2` which we don't use
    // - two new acceleration structure query parameters
    // - `rayTracingPipelineTraceRaysIndirect2` feature, same as `rayTracingPipelineTraceRaysIndirect` but with indirect SBTand dispatch dimensions
    // 
    // Lets have
    // ```cpp
    // bool accelerationStructureSizeAndBLASPointersQuery = false;
    // 
    // // Do not expose, we don't use KHR_synchronization2 yet
    // //bool accelerationStructureCopyStageAndSBTAccessType;
    // 
    // bool rayCullMask = false;
    // 
    // bool rayTracingPipelineTraceRaysIndirectDimensionsAndSBT = false;
    // ```
    // 
    // Lets enable `rayTracingMaintenance1`and `rayTracingPipelineTraceRaysIndirect2` whenever required by the above.
    /* VK_KHR_ray_tracing_maintenance1 *//* added in vk 1.3.213, the SDK isn't released yet at this moment :D */

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

    // [DO NOT EXPOSE] Deprecated
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

    /* Extensions Exposed as Features: */

    /* VK_KHR_swapchain */
    /* Dependant on `IAPIConnection::SFeatures::swapchainMode` enabled on apiConnection Creation */
    core::bitflag<E_SWAPCHAIN_MODE> swapchainMode = E_SWAPCHAIN_MODE::ESM_NONE;

    // Enabled when possible, exposed as limit in as spirvVersion
    /* VK_KHR_spirv_1_4 */

    // [TODO] handle with a single num
    /* VK_KHR_display_swapchain */

    // [TODO LATER] (When it has documentation): Always enable, expose as limit
    /* VK_AMD_gpu_shader_half_float_fetch */

    // [TODO LATER] Requires exposing external memory first
    /* VK_ANDROID_external_memory_android_hardware_buffer */

    // [TODO LATER] Requires changes to API
    /* VK_EXT_calibrated_timestamps */

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

    // [TODO LATER] Expose when we support MoltenVK
    /* VK_EXT_metal_objects */

    // [TODO LATER] would like to expose, but too much API to change
    /* VK_EXT_pipeline_creation_feedback */

    // [TODO LATER] Expose when we start to experience slowdowns from validation
    /* VK_EXT_validation_cache */

    // [TODO LATER] Requires API changes and asset converter upgrades; default to it and forbid old API use
    /* VK_KHR_image_format_list */

    // [TODO LATER] Requires VK_KHR_image_format_list to be enabled for any device-level functionality
    /* VK_KHR_swapchain_mutable_format */

    // [TODO LATER] Used for dx11 interop
    /* VK_KHR_win32_keyed_mutex */

    // [TODO LATER] won't decide yet, requires VK_EXT_direct_mode_display anyway
    /* VK_NV_acquire_winrt_display */

    // [TODO LATER] Don't expose VR features for now
    /* VK_NV_clip_space_w_scaling */

    // [TODO LATER] Requires API changes
    /* VK_NV_fragment_coverage_to_color */

    // [TODO LATER] Wait until VK_AMD_shader_fragment_mask & VK_QCOM_render_pass_shader_resolve converge to something useful
    /* VK_AMD_shader_fragment_mask */

    // [TODO LATER] Wait until VK_AMD_shader_fragment_mask & VK_QCOM_render_pass_shader_resolve converge to something useful
    /* VK_QCOM_render_pass_shader_resolve */

    // [DO NOT EXPOSE] Waiting for cross platform
    /* VK_AMD_display_native_hdr */

    // [DO NOT EXPOSE] Promoted to KHR version already exposed
    /* VK_AMD_draw_indirect_count */
    
    // [DO NOT EXPOSE] 0 documentation
    /* VK_AMD_gpa_interface */

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

    // [DO NOT EXPOSE] supported=disabled
    /* VK_ANDROID_native_buffer */

    // [DO NOT EXPOSE] Promoted to VK_EXT_debug_utils (instance ext)
    /* VK_EXT_debug_marker */

    // [DO NOT EXPOSE] absorbed into KHR_global_priority
    /* VK_EXT_global_priority_query */

    // [DO NOT EXPOSE] Too "intrinsically linux"
    /* VK_EXT_image_drm_format_modifier */

    // [DO NOT EXPOSE] Never expose this, it was a mistake for that GL quirk to exist in the first place
    /* VK_EXT_non_seamless_cube_map */

    // [DO NOT EXPOSE] Too "intrinsically linux"
    /* VK_EXT_physical_device_drm */

    // [DO NOT EXPOSE] wont expose yet (or ever), requires VK_KHR_sampler_ycbcr_conversion
    /* VK_EXT_rgba10x6_formats */

    // [DO NOT EXPOSE] stupid to expose, it would be extremely dumb to want to provide some identifiers instead of VkShaderModule outside of some emulator which has no control over pipeline combo explosion
    /* VK_EXT_shader_module_identifier */

    // [DO NOT EXPOSE] we dont need to care or know about it
    /* VK_EXT_tooling_info */

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

    // [DO NOT EXPOSE] Promoted to VK1.3 core but serves no purpose other than providing a pNext chain for the usage of a single QCOM extension
    /* VK_KHR_copy_commands2 */

    /* VK_KHR_deferred_host_operations */
    bool deferredHostOperations = false;

    // [DO NOT EXPOSE YET] Promoted to core VK 1.1
    /* VK_KHR_device_group */
    /* VK_KHR_device_group_creation */

    // [DO NOT EXPOSE] Promoted to core VK 1.1
    /* VK_KHR_external_fence */

    // [DO NOT EXPOSE] Promoted to core VK 1.1
    /* VK_KHR_external_memory */

    // [DO NOT EXPOSE] Promoted to core VK 1.1
    /* VK_KHR_external_semaphore */

    // [DO NOT EXPOSE] Promoted to core VK 1.1
    /* VK_KHR_get_memory_requirements2 */

    // [DO NOT EXPOSE] Promoted to core VK 1.1
    /* VK_KHR_get_physical_device_properties2 */

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

    // [DO NOT EXPOSE] Provisional
    /* VK_KHR_video_decode_queue */

    // [DO NOT EXPOSE] Provisional
    /* VK_KHR_video_encode_queue */

    // [DO NOT EXPOSE] 0 documentation
    /* VK_MESA_query_timestamp */

    // [DO NOT EXPOSE] 0 documentation
    /* VK_NV_cuda_kernel_launch */
    
    // [DO NOT EXPOSE] Promoted to KHR_dedicated_allocation, core VK 1.1
    /* VK_NV_dedicated_allocation */

    // [DO NOT EXPOSE] Promoted to VK_KHR_external_memory_win32 
    /* VK_NV_external_memory_win32 */

    // [DO NOT EXPOSE] For now. For 2D ui
    /* VK_NV_fill_rectangle */

    // [DO NOT EXPOSE]
    /* VK_NV_glsl_shader */

    // [DO NOT EXPOSE] 0 documentation
    /* VK_NV_low_latency */

    // [DO NOT EXPOSE] 0 documentation
    /* VK_NV_rdma_memory */

    // [DO NOT EXPOSE] 0 documentation
    /* VK_NV_texture_dirty_tile_map */
    
    // [DO NOT EXPOSE] Will be promoted to KHR_video_queue.
    /* VK_NV_video_queue */

    // [DO NOT EXPOSE] Promoted to VK_KHR_win32_keyed_mutex 
    /* VK_NV_win32_keyed_mutex */

    // [DO NOT EXPOSE] absorbed into VK_EXT_load_store_op_none
    /* VK_QCOM_render_pass_store_ops */

    // [DO NOT EXPOSE] Too vendor specific
    /* VK_QCOM_render_pass_transform */

    // [DO NOT EXPOSE] Too vendor specific
    /* VK_QCOM_rotated_copy_commands */

    // [TODO] Triage leftover extensions below    

    /* VK_NV_present_barrier */
    /* VK_EXT_queue_family_foreign */
    /* VK_EXT_separate_stencil_usage */
    /* VK_KHR_create_renderpass2 */
    /* VK_KHR_bind_memory2 */
    /* VK_NV_viewport_array2 */
    /* VK_EXT_image_compression_control */
    /* VK_EXT_image_compression_control_swapchain */
    /* VK_EXT_multisampled_render_to_single_sampled */
    /* VK_EXT_pipeline_properties */
    
    /* Nabla */
    // No Nabla Specific Features for now
				
    inline bool isSubsetOf(const SPhysicalDeviceFeatures& _rhs) const
    {
        if (robustBufferAccess && !_rhs.robustBufferAccess) return false;
        if (geometryShader && !_rhs.geometryShader) return false;
        if (tessellationShader && !_rhs.tessellationShader) return false;
        if (dualSrcBlend && !_rhs.dualSrcBlend) return false;
        if (logicOp && !_rhs.logicOp) return false;
        if (fillModeNonSolid && !_rhs.fillModeNonSolid) return false;
        if (depthBounds && !_rhs.depthBounds) return false;
        if (wideLines && !_rhs.wideLines) return false;
        if (largePoints && !_rhs.largePoints) return false;
        if (alphaToOne && !_rhs.alphaToOne) return false;
        if (multiViewport && !_rhs.multiViewport) return false;
        if (pipelineStatisticsQuery && !_rhs.pipelineStatisticsQuery) return false;
        if (shaderStorageImageReadWithoutFormat && !_rhs.shaderStorageImageReadWithoutFormat) return false;
        if (shaderStorageImageWriteWithoutFormat && !_rhs.shaderStorageImageWriteWithoutFormat) return false;
        if (shaderClipDistance && !_rhs.shaderClipDistance) return false;
        if (shaderCullDistance && !_rhs.shaderCullDistance) return false;
        if (shaderResourceResidency && !_rhs.shaderResourceResidency) return false;
        if (shaderResourceMinLod && !_rhs.shaderResourceMinLod) return false;
        if (variableMultisampleRate && !_rhs.variableMultisampleRate) return false;
        if (inheritedQueries && !_rhs.inheritedQueries) return false;
        if (shaderDrawParameters && !_rhs.shaderDrawParameters) return false;
        if (descriptorIndexing && !_rhs.descriptorIndexing) return false;
        if (shaderInputAttachmentArrayDynamicIndexing && !_rhs.shaderInputAttachmentArrayDynamicIndexing) return false;
        if (shaderUniformTexelBufferArrayDynamicIndexing && !_rhs.shaderUniformTexelBufferArrayDynamicIndexing) return false;
        if (shaderStorageTexelBufferArrayDynamicIndexing && !_rhs.shaderStorageTexelBufferArrayDynamicIndexing) return false;
        if (shaderUniformBufferArrayNonUniformIndexing && !_rhs.shaderUniformBufferArrayNonUniformIndexing) return false;
        if (shaderSampledImageArrayNonUniformIndexing && !_rhs.shaderSampledImageArrayNonUniformIndexing) return false;
        if (shaderStorageBufferArrayNonUniformIndexing && !_rhs.shaderStorageBufferArrayNonUniformIndexing) return false;
        if (shaderStorageImageArrayNonUniformIndexing && !_rhs.shaderStorageImageArrayNonUniformIndexing) return false;
        if (shaderInputAttachmentArrayNonUniformIndexing && !_rhs.shaderInputAttachmentArrayNonUniformIndexing) return false;
        if (shaderUniformTexelBufferArrayNonUniformIndexing && !_rhs.shaderUniformTexelBufferArrayNonUniformIndexing) return false;
        if (shaderStorageTexelBufferArrayNonUniformIndexing && !_rhs.shaderStorageTexelBufferArrayNonUniformIndexing) return false;
        if (descriptorBindingUniformBufferUpdateAfterBind && !_rhs.descriptorBindingUniformBufferUpdateAfterBind) return false;
        if (descriptorBindingSampledImageUpdateAfterBind && !_rhs.descriptorBindingSampledImageUpdateAfterBind) return false;
        if (descriptorBindingStorageImageUpdateAfterBind && !_rhs.descriptorBindingStorageImageUpdateAfterBind) return false;
        if (descriptorBindingStorageBufferUpdateAfterBind && !_rhs.descriptorBindingStorageBufferUpdateAfterBind) return false;
        if (descriptorBindingUniformTexelBufferUpdateAfterBind && !_rhs.descriptorBindingUniformTexelBufferUpdateAfterBind) return false;
        if (descriptorBindingStorageTexelBufferUpdateAfterBind && !_rhs.descriptorBindingStorageTexelBufferUpdateAfterBind) return false;
        if (descriptorBindingUpdateUnusedWhilePending && !_rhs.descriptorBindingUpdateUnusedWhilePending) return false;
        if (descriptorBindingPartiallyBound && !_rhs.descriptorBindingPartiallyBound) return false;
        if (descriptorBindingVariableDescriptorCount && !_rhs.descriptorBindingVariableDescriptorCount) return false;
        if (runtimeDescriptorArray && !_rhs.runtimeDescriptorArray) return false;
        if (samplerFilterMinmax && !_rhs.samplerFilterMinmax) return false;
        if (bufferDeviceAddress && !_rhs.bufferDeviceAddress) return false;
        if (bufferDeviceAddressMultiDevice && !_rhs.bufferDeviceAddressMultiDevice) return false;
        if (shaderDemoteToHelperInvocation && !_rhs.shaderDemoteToHelperInvocation) return false;
        if (shaderTerminateInvocation && !_rhs.shaderTerminateInvocation) return false;
        if (subgroupSizeControl && !_rhs.subgroupSizeControl) return false;
        if (computeFullSubgroups && !_rhs.computeFullSubgroups) return false;
        if (shaderIntegerDotProduct && !_rhs.shaderIntegerDotProduct) return false;
        if (rasterizationOrderColorAttachmentAccess && !_rhs.rasterizationOrderColorAttachmentAccess) return false;
        if (rasterizationOrderDepthAttachmentAccess && !_rhs.rasterizationOrderDepthAttachmentAccess) return false;
        if (rasterizationOrderStencilAttachmentAccess && !_rhs.rasterizationOrderStencilAttachmentAccess) return false;
        if (fragmentShaderSampleInterlock && !_rhs.fragmentShaderSampleInterlock) return false;
        if (fragmentShaderPixelInterlock && !_rhs.fragmentShaderPixelInterlock) return false;
        if (fragmentShaderShadingRateInterlock && !_rhs.fragmentShaderShadingRateInterlock) return false;
        if (indexTypeUint8 && !_rhs.indexTypeUint8) return false;
        if (shaderBufferFloat32Atomics && !_rhs.shaderBufferFloat32Atomics) return false;
        if (shaderBufferFloat32AtomicAdd && !_rhs.shaderBufferFloat32AtomicAdd) return false;
        if (shaderBufferFloat64Atomics && !_rhs.shaderBufferFloat64Atomics) return false;
        if (shaderBufferFloat64AtomicAdd && !_rhs.shaderBufferFloat64AtomicAdd) return false;
        if (shaderSharedFloat32Atomics && !_rhs.shaderSharedFloat32Atomics) return false;
        if (shaderSharedFloat32AtomicAdd && !_rhs.shaderSharedFloat32AtomicAdd) return false;
        if (shaderSharedFloat64Atomics && !_rhs.shaderSharedFloat64Atomics) return false;
        if (shaderSharedFloat64AtomicAdd && !_rhs.shaderSharedFloat64AtomicAdd) return false;
        if (shaderImageFloat32Atomics && !_rhs.shaderImageFloat32Atomics) return false;
        if (shaderImageFloat32AtomicAdd && !_rhs.shaderImageFloat32AtomicAdd) return false;
        if (sparseImageFloat32Atomics && !_rhs.sparseImageFloat32Atomics) return false;
        if (sparseImageFloat32AtomicAdd && !_rhs.sparseImageFloat32AtomicAdd) return false;
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
        if (shaderImageInt64Atomics && !_rhs.shaderImageInt64Atomics) return false;
        if (sparseImageInt64Atomics && !_rhs.sparseImageInt64Atomics) return false;
        if (accelerationStructure && !_rhs.accelerationStructure) return false;
        if (accelerationStructureIndirectBuild && !_rhs.accelerationStructureIndirectBuild) return false;
        if (accelerationStructureHostCommands && !_rhs.accelerationStructureHostCommands) return false;
        if (descriptorBindingAccelerationStructureUpdateAfterBind && !_rhs.descriptorBindingAccelerationStructureUpdateAfterBind) return false;
        if (rayQuery && !_rhs.rayQuery) return false;
        if (rayTracingPipeline && !_rhs.rayTracingPipeline) return false;
        if (rayTracingPipelineTraceRaysIndirect && !_rhs.rayTracingPipelineTraceRaysIndirect) return false;
        if (rayTraversalPrimitiveCulling && !_rhs.rayTraversalPrimitiveCulling) return false;
        if (shaderDeviceClock && !_rhs.shaderDeviceClock) return false;
        if (shaderSubgroupUniformControlFlow && !_rhs.shaderSubgroupUniformControlFlow) return false;
        if (workgroupMemoryExplicitLayout && !_rhs.workgroupMemoryExplicitLayout) return false;
        if (workgroupMemoryExplicitLayoutScalarBlockLayout && !_rhs.workgroupMemoryExplicitLayoutScalarBlockLayout) return false;
        if (workgroupMemoryExplicitLayout8BitAccess && !_rhs.workgroupMemoryExplicitLayout8BitAccess) return false;
        if (workgroupMemoryExplicitLayout16BitAccess && !_rhs.workgroupMemoryExplicitLayout16BitAccess) return false;
        if (computeDerivativeGroupQuads && !_rhs.computeDerivativeGroupQuads) return false;
        if (computeDerivativeGroupLinear && !_rhs.computeDerivativeGroupLinear) return false;
        if (cooperativeMatrix && !_rhs.cooperativeMatrix) return false;
        if (cooperativeMatrixRobustBufferAccess && !_rhs.cooperativeMatrixRobustBufferAccess) return false;
        if (rayTracingMotionBlur && !_rhs.rayTracingMotionBlur) return false;
        if (rayTracingMotionBlurPipelineTraceRaysIndirect && !_rhs.rayTracingMotionBlurPipelineTraceRaysIndirect) return false;
        if (coverageReductionMode && !_rhs.coverageReductionMode) return false;
        if (deviceGeneratedCommands && !_rhs.deviceGeneratedCommands) return false;
        if (taskShader && !_rhs.taskShader) return false;
        if (meshShader && !_rhs.meshShader) return false;
        if (representativeFragmentTest && !_rhs.representativeFragmentTest) return false;
        if (mixedAttachmentSamples && !_rhs.mixedAttachmentSamples) return false;
        if (hdrMetadata && !_rhs.hdrMetadata) return false;
        if (displayTiming && !_rhs.displayTiming) return false;
        if (rasterizationOrder && !_rhs.rasterizationOrder) return false;
        if (shaderExplicitVertexParameter && !_rhs.shaderExplicitVertexParameter) return false;
        if (shaderInfoAMD && !_rhs.shaderInfoAMD) return false;
        if (pipelineCreationCacheControl && !_rhs.pipelineCreationCacheControl) return false;
        if (colorWriteEnable && !_rhs.colorWriteEnable) return false;
        if (conditionalRendering && !_rhs.conditionalRendering) return false;
        if (inheritedConditionalRendering && !_rhs.inheritedConditionalRendering) return false;
        if (deviceMemoryReport && !_rhs.deviceMemoryReport) return false;
        if (fragmentDensityMap && !_rhs.fragmentDensityMap) return false;
        if (fragmentDensityMapDynamic && !_rhs.fragmentDensityMapDynamic) return false;
        if (fragmentDensityMapNonSubsampledImages && !_rhs.fragmentDensityMapNonSubsampledImages) return false;
        if (fragmentDensityMapDeferred && !_rhs.fragmentDensityMapDeferred) return false;
        if (robustImageAccess && !_rhs.robustImageAccess) return false;
        if (inlineUniformBlock && !_rhs.inlineUniformBlock) return false;
        if (descriptorBindingInlineUniformBlockUpdateAfterBind && !_rhs.descriptorBindingInlineUniformBlockUpdateAfterBind) return false;
        if (rectangularLines && !_rhs.rectangularLines) return false;
        if (bresenhamLines && !_rhs.bresenhamLines) return false;
        if (smoothLines && !_rhs.smoothLines) return false;
        if (stippledRectangularLines && !_rhs.stippledRectangularLines) return false;
        if (stippledBresenhamLines && !_rhs.stippledBresenhamLines) return false;
        if (stippledSmoothLines && !_rhs.stippledSmoothLines) return false;
        if (stippledSmoothLines && !_rhs.stippledSmoothLines) return false;
        if (memoryPriority && !_rhs.memoryPriority) return false;
        if (robustBufferAccess2 && !_rhs.robustBufferAccess2) return false;
        if (robustImageAccess2 && !_rhs.robustImageAccess2) return false;
        if (nullDescriptor && !_rhs.nullDescriptor) return false;
        if (performanceCounterQueryPools && !_rhs.performanceCounterQueryPools) return false;
        if (performanceCounterMultipleQueryPools && !_rhs.performanceCounterMultipleQueryPools) return false;
        if (pipelineExecutableInfo && !_rhs.pipelineExecutableInfo) return false;
        if (deviceCoherentMemory && !_rhs.deviceCoherentMemory) return false;
        if (bufferMarkerAMD && !_rhs.bufferMarkerAMD) return false;
        if (!_rhs.swapchainMode.hasFlags(swapchainMode)) return false;
        if (deferredHostOperations && !_rhs.deferredHostOperations) return false;
        return true;
    }
};

template<typename T>
concept DeviceFeatureDependantClass = requires(const SPhysicalDeviceFeatures& availableFeatures, SPhysicalDeviceFeatures& features) { 
    T::enableRequiredFeautres(features);
    T::enablePreferredFeatures(availableFeatures, features);
};

} // nbl::video
#endif