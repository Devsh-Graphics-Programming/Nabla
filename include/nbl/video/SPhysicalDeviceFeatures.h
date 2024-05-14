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

    // widely supported but has performance overhead, so remains an optional feature to enable
    bool robustBufferAccess = false;

    // [REQUIRE] Roadmap 2022 requires support for these, device support is ubiquitous and enablement is unlikely to harm performance
    //bool fullDrawIndexUint32 = true;
    //bool imageCubeArray = true;
    //bool independentBlend = true;

    // I have no clue if these cause overheads from simply being enabled
    bool geometryShader    = false;
    bool tessellationShader = false;

    // [REQUIRE] ROADMAP 2022 and good device support
    //bool sampleRateShading = true;
 
    // [REQUIRE] good device support
    //bool dualSrcBlend = true;

    // [EXPOSE AS A LIMIT] Somewhat legacy feature
    //bool logicOp;

    // [REQUIRE] Roadmap 2022 requires support for these, device support is ubiquitous and enablement is unlikely to harm performance
    //bool multiDrawIndirect = true;
    //bool drawIndirectFirstInstance = true;
    //bool depthClamp = true;
    //bool depthBiasClamp = true;

    // [REQUIRE] good device support
    //bool fillModeNonSolid = true;

    // this is kinda like a weird clip-plane that doesn't count towards clip plane count
    bool depthBounds = false;

    // good device support, but a somewhat useless feature (constant screenspace width with limits on width)
    bool wideLines = false;
    // good device support, but a somewhat useless feature (axis aligned screenspace boxes with limits on size)
    bool largePoints = false;

    // Some AMD don't support
    bool alphaToOne = true;

    // [REQUIRE] good device support
    //bool multiViewport = true;

    // [REQUIRE] Roadmap 2022 requires support for these, device support is ubiquitous
    // bool samplerAnisotropy = true;

    // [DO NOT EXPOSE] these 3 don't make a difference, just shortcut from Querying support from PhysicalDevice
    //bool    textureCompressionETC2;
    //bool    textureCompressionASTC_LDR;
    //bool    textureCompressionBC;
 
    // [REQUIRE] ROADMAP 2022 and good device support
    //bool occlusionQueryPrecise = true;

    bool pipelineStatisticsQuery = false;

    // [EXPOSE AS LIMIT] Enabled by Default, Moved to Limits
    //bool vertexPipelineStoresAndAtomics;
    //bool fragmentStoresAndAtomics;
    //bool shaderTessellationAndGeometryPointSize;

    // [REQUIRE] ROADMAP 2024 good device support
    //bool shaderImageGatherExtended = true;

    // [REQUIRE] ROADMAP 2022 and good device support
    //bool shaderStorageImageExtendedFormats = true;

    // [EXPOSE AS LIMIT] Cannot be always enabled cause Intel ARC is handicapped
    //bool shaderStorageImageMultisample;

    // TODO: format feature reporting unimplemented yet for both of the below! (should we move to usage reporting?)
    // [EXPOSE AS LIMIT] always enable, shouldn't cause overhead by just being enabled
    //bool shaderStorageImageReadWithoutFormat;
 
    // [REQUIRE] good device support
    //bool shaderStorageImageWriteWithoutFormat = true;

    // [REQUIRE] ROADMAP 2022 and good device support
    //bool shaderUniformBufferArrayDynamicIndexing = true;
    //bool shaderSampledImageArrayDynamicIndexing = true;
    //bool shaderStorageBufferArrayDynamicIndexing = true;
 
    // [EXPOSE AS A LIMIT] ROADMAP 2022 but Apple GPUs have poor support
    //bool shaderStorageImageArrayDynamicIndexing;

    // [REQUIRE] good device support
    // bool shaderClipDistance = true;

    bool shaderCullDistance = false;

    // [EXPOSE AS A LIMIT] Cannot be always enabled cause Intel ARC is handicapped
    //bool shaderFloat64;

    // [REQUIRE]
    //bool shaderInt64 = true;
    // [REQUIRE] ROADMAP 2024
    //bool shaderInt16 = true;

    // poor support on Apple GPUs
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
    
    // [EXPOSE AS LIMIT] poor support on Apple GPUs
    //bool variableMultisampleRate;

    // [REQUIRE] Always enabled, good device support.
    // bool inheritedQueries = true;


    /* Vulkan 1.1 Core */

    // [REQUIRED] ROADMAP 2024
    //bool storageBuffer16BitAccess = true;
    // [REQUIRED] Force Enabled : ALIAS VK_KHR_16bit_storage
    //bool uniformAndStorageBuffer16BitAccess = true;
  
    // [EXPOSE AS A LIMIT] Enabled by Default, Moved to Limits : ALIAS VK_KHR_16bit_storage
    //bool storagePushConstant16;
    //bool storageInputOutput16;

    // [REQUIRE] Required to be present when Vulkan 1.1 is supported
    //bool multiview = true;

    // [EXPOSE AS A LIMIT] VK_KHR_multiview required but these depend on pipelines and MoltenVK mismatches these
    //bool multiviewGeometryShader;
    //bool multiviewTessellationShader;

    // [REQUIRE] Will eventually be required by HLSL202x if it implements references or pointers (even the non-generic type) 
    //bool variablePointers = true;
    //bool variablePointersStorageBuffer = true;
    
    // [DO NOT EXPOSE] not gonna expose until we have a need to
    /* or via VkPhysicalDeviceProtectedMemoryProperties provided by Vulkan 1.1 */
    //bool           protectedMemory = false;

    // [DO NOT EXPOSE] ROADMAP 2022 Enables certain formats in Vulkan, we just enable them if available or else we need to make format support query functions in LogicalDevice as well
    //bool           samplerYcbcrConversion = false;

    // [REQUIRE] ROADMAP2024 and Force Enabled : VK_KHR_shader_draw_parameters
    //bool shaderDrawParameters = true;


    /* Vulkan 1.2 Core */

    // [REQUIRE] ROADMAP 2022 and device support ubiquitous
    //bool samplerMirrorClampToEdge = true;          // ALIAS: VK_KHR_sampler_mirror_clamp_to_edge
 
    // [EXPOSE AS A LIMIT] ROADMAP 2022 requires support but MoltenVK doesn't support, exposed as a limit `drawIndirectCount`
    //bool drawIndirectCount; // ALIAS: VK_KHR_draw_indirect_count

    // or VK_KHR_8bit_storage:
    // [REQUIRE] ROADMAP 2024 and good device coverage
    //bool storageBuffer8BitAccess = true;
    // [REQUIRE] good device coverage
    //bool uniformAndStorageBuffer8BitAccess = true;
    // [EXPOSE AS LIMIT] not great support yet
    //bool storagePushConstant8;
 
    // [EXPOSE AS LIMIT] or VK_KHR_shader_atomic_int64:
    //bool shaderBufferInt64Atomics;
    //bool shaderSharedInt64Atomics;

    // or VK_KHR_shader_float16_int8:
    // [EXPOSE AS LIMIT] ROADMAP 2024 but not great support yet
    //bool shaderFloat16;
    // [REQUIRE] ROADMAP 2024 good device coverage
    //bool shaderInt8 = true;
    
    // [REQUIRE] ROADMAP 2022
    //bool descriptorIndexing = true;
    // [EXPOSE AS A LIMIT] This is for a SPIR-V capability, the overhead should only be incurred if the pipeline uses this capability
    //bool shaderInputAttachmentArrayDynamicIndexing;
    // [REQUIRE] because we also require `descriptorIndexing`
    //bool shaderUniformTexelBufferArrayDynamicIndexing = true;
    //bool shaderStorageTexelBufferArrayDynamicIndexing = true;
    // [EXPOSE AS A LIMIT] ROADMAP 2022 mandates but poor device support
    //bool shaderUniformBufferArrayNonUniformIndexing;
    // [REQUIRE] because we also require `descriptorIndexing`
    //bool shaderSampledImageArrayNonUniformIndexing = true;
    //bool shaderStorageBufferArrayNonUniformIndexing = true;
    // [REQUIRE] ROADMAP 2022
    //bool shaderStorageImageArrayNonUniformIndexing = true;
    // [EXPOSE AS A LIMIT] This is for a SPIR-V capability, the overhead should only be incurred if the pipeline uses this capability
    //bool shaderInputAttachmentArrayNonUniformIndexing;
    // [REQUIRE] because we also require `descriptorIndexing`
    //bool shaderUniformTexelBufferArrayNonUniformIndexing = true;
    // [REQUIRE] ROADMAP 2022 and good device support
    //bool shaderStorageTexelBufferArrayNonUniformIndexing = true;
    // We have special bits on the Descriptor Layout Bindings and those should decide the overhead, not the enablement of a feature like the following
    // [EXPOSE AS A LIMIT] not great coverage but still can enable when available
    //bool descriptorBindingUniformBufferUpdateAfterBind;
    // [REQUIRE] because we also require `descriptorIndexing`
    //bool descriptorBindingSampledImageUpdateAfterBind = true;
    //bool descriptorBindingStorageImageUpdateAfterBind = true;
    //bool descriptorBindingStorageBufferUpdateAfterBind = true;
    //bool descriptorBindingUniformTexelBufferUpdateAfterBind = true;
    //bool descriptorBindingStorageTexelBufferUpdateAfterBind = true;
    //bool descriptorBindingUpdateUnusedWhilePending = true;
    //bool descriptorBindingPartiallyBound = true;
    // [REQUIRE] ROADMAP 2022 and good device support
    //bool descriptorBindingVariableDescriptorCount = true;
    //bool runtimeDescriptorArray = true;
    
    // [EXPOSE AS A LIMIT]
    //bool samplerFilterMinmax;   // ALIAS: VK_EXT_sampler_filter_minmax

    // [REQUIRE] Roadmap 2022 requires support for these we always enable and they're unlikely to harm performance
    //bool scalarBlockLayout = true;     // or VK_EXT_scalar_block_layout

    // [DO NOT EXPOSE] Decided against exposing, API is braindead, for details see: https://github.com/Devsh-Graphics-Programming/Nabla/issues/378
    //bool imagelessFramebuffer = false;  // or VK_KHR_imageless_framebuffer
    
    // [REQUIRE] Vulkan 1.2 requires these
    //bool uniformBufferStandardLayout = true; // or VK_KHR_uniform_buffer_standard_layout
    //bool shaderSubgroupExtendedTypes = true;   // or VK_KHR_shader_subgroup_extended_types
    //bool separateDepthStencilLayouts = true;   // or VK_KHR_separate_depth_stencil_layouts
    //bool hostQueryReset = true;                // or VK_EXT_host_query_reset [TODO Implement]
    //bool timelineSemaphore = true;             // or VK_KHR_timeline_semaphore [TODO Implement]
    
    // or VK_KHR_buffer_device_address:
    // [REQUIRE] Vulkan 1.3 requires
    //bool bufferDeviceAddress = true;
    // [DO NOT EXPOSE] for capture tools not engines
    //bool bufferDeviceAddressCaptureReplay;
    bool bufferDeviceAddressMultiDevice = false;
    
    // [REQUIRE] ROADMAP2022 wants them. ALIAS VK_KHR_vulkan_memory_model
    //bool vulkanMemoryModel = true;
    //bool vulkanMemoryModelDeviceScope = true;
    // [EXPOSE AS A LIMIT] ROADMAP2022 wants them, but device support low
    //bool vulkanMemoryModelAvailabilityVisibilityChains;

    // [EXPOSE AS A LIMIT]
    //bool shaderOutputViewportIndex;     // ALIAS: VK_EXT_shader_viewport_index_layer
    //bool shaderOutputLayer;             // ALIAS: VK_EXT_shader_viewport_index_layer

    // [REQUIRED] ubiquitous device support
    //bool subgroupBroadcastDynamicId = true;

    /* Vulkan 1.3 Core */

    /*
        This feature adds stricter requirements for how out of bounds reads from images are handled.
        Rather than returning undefined values,
        most out of bounds reads return R, G, and B values of zero and alpha values of either zero or one.
        Components not present in the image format may be set to zero
        or to values based on the format as described in Conversion to RGBA in vulkan specification.
    */
    // widely supported but has performance overhead, so remains an optional feature to enable
    bool robustImageAccess = false;                 //  or VK_EXT_image_robustness
    
    // [DO NOT EXPOSE] VK_EXT_inline_uniform_block EVIL regressive step back into OpenGL/Dx10 times? Or an intermediate step between PC and UBO?
    // Vulkan 1.3, Nabla Core Profile:
    //bool           inlineUniformBlock = false;
    // ROADMAP 2022, Nabla Core Profile:
    //bool           descriptorBindingInlineUniformBlockUpdateAfterBind = false;

    // [REQUIRE] Vulkan 1.3 non-optional and Nabla Core Profile but TODO: need impl
    //bool pipelineCreationCacheControl = true;      // or VK_EXT_pipeline_creation_cache_control

    // [DO NOT EXPOSE] ever we have our own mechanism, unless we can somehow get the data out of `VkObject`?
    //bool           privateData = false;                       // or VK_EXT_private_data
    
    // [EXPOSE AS LIMIT] Vulkan 1.3 non-optional requires but poor support
    //bool shaderDemoteToHelperInvocation;    // or VK_EXT_shader_demote_to_helper_invocation
    //bool shaderTerminateInvocation;         // or VK_KHR_shader_terminate_invocation
    
    // [REQUIRE] Nabla Core Profile, Vulkan 1.3 or VK_EXT_subgroup_size_control
    //bool subgroupSizeControl  = true;
    //bool computeFullSubgroups = true;
    
    // [REQUIRE] REQUIRE 
    //bool           synchronization2 = true;                      // or VK_KHR_synchronization2
    
    // [DO NOT EXPOSE] Doesn't make a difference, just shortcut from Querying support from PhysicalDevice
    //bool           textureCompressionASTC_HDR;            // or VK_EXT_texture_compression_astc_hdr

    // [EXPOSE AS LIMIT] Vulkan 1.3 non-optional requires but poor support
    //bool           shaderZeroInitializeWorkgroupMemory;   // or VK_KHR_zero_initialize_workgroup_memory
    
    // [DO NOT EXPOSE] EVIL
    //bool           dynamicRendering = false;                      // or VK_KHR_dynamic_rendering

    // [REQUIRE] Vulkan 1.3 non-optional requires, you probably want to look at the individual limits anyway
    //bool shaderIntegerDotProduct = true;               // or VK_KHR_shader_integer_dot_product


    /* Nabla Core Profile Extensions */
    // [TODO] Better descriptive name for the Vulkan robustBufferAccess2, robustImageAccess2 features
    bool robustBufferAccess2 = false;
    // Nabla Core Profile but still a feature because enabling has overhead
    bool robustImageAccess2 = false;
    /*
    ! nullDescriptor: you can use `nullptr` for writing descriptors to sets and Accesses to null descriptors have well-defined behavior.
    [TODO] Handle `nullDescriptor` feature in the engine.
    */
    bool nullDescriptor = false;


    /* Vulkan Extensions */

    // [DO NOT EXPOSE] Instance extension & should enable implicitly if swapchain is enabled
    /* VK_KHR_surface */

    /* VK_KHR_swapchain */
    /* Dependant on `IAPIConnection::SFeatures::swapchainMode` enabled on apiConnection Creation */
    core::bitflag<E_SWAPCHAIN_MODE> swapchainMode = E_SWAPCHAIN_MODE::ESM_NONE;

    // [DO NOT EXPOSE] We don't support yet
    // VK_KHR_display

    // [TODO] handle with a single num
    /* VK_KHR_display_swapchain */

    //[DO NOT EXPOSE] OS-specific INSTANCE extensions we enable implicitly as we detect the platform
    // VK_KHR_xlib_surface
    // VK_KHR_xcb_surface
    // VK_KHR_wayland_surface
    // VK_KHR_mir_surface
    // VK_KHR_android_surface
    // VK_KHR_win32_surface

    // [DO NOT EXPOSE] supported=disabled
    /* VK_ANDROID_native_buffer */

    // [DEPRECATED] by VK_EXT_debug_utils
    // VK_EXT_debug_report

    // [DO NOT EXPOSE] EVER
    /* VK_NV_glsl_shader */

    // [TODO LATER] Will expose some day
    /* VK_EXT_depth_range_unrestricted */

    // [DEPRECATED] deprecated by Vulkan 1.2
    // VK_KHR_sampler_mirror_clamp_to_edge

    // [DO NOT EXPOSE] Vendor specific, superceeded by VK_EXT_filter_cubic, won't expose for a long time
    /* VK_IMG_filter_cubic */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_AMD_extension_17
    // VK_AMD_extension_18

    // [DO NOT EXPOSE] Meme extension
    /* VK_AMD_rasterization_order */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_AMD_extension_20

    // [EXPOSE AS LIMIT] Enabled by Default, Moved to Limits
    /* VK_AMD_shader_trinary_minmax */

    // [EXPOSE AS LIMIT]
    /* VK_AMD_shader_explicit_vertex_parameter */

    // [DEPRECATED] Promoted to VK_EXT_debug_utils (instance ext)
    /* VK_EXT_debug_marker */

    // [DO NOT EXPOSE] We don't support yet
    // VK_KHR_video_queue

    // [TODO] Provisional
    /* VK_KHR_video_decode_queue */

    // [DO NOT EXPOSE] Core features, KHR, EXT and smaller AMD features supersede everything in here
    /* VK_AMD_gcn_shader */

    // [DEPRECATED] Promoted to KHR_dedicated_allocation, non-optional core VK 1.1
    /* VK_NV_dedicated_allocation */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_28

    // [DO NOT EXPOSE] ever because of our disdain for XForm feedback
    /* TransformFeedbackFeaturesEXT *//* VK_EXT_transform_feedback */

    // [DO NOT EXPOSE] We don't support yet
    // VK_NVX_binary_import

    // [DO NOT EXPOSE] We don't support yet
    // VK_NVX_image_view_handle

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_AMD_extension_32
    // VK_AMD_extension_33

    // [DEPRECATED] Vulkan core 1.2
    /* VK_AMD_draw_indirect_count */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_AMD_extension_35

    // [DEPRECATED] Promoted to VK_KHR_maintenance1, non-optional core VK 1.1
    /* VK_AMD_negative_viewport_height */

    // [EXPOSE AS LIMIT] The only reason we still keep it around is because it provides FP16 trig and special functions
    /* VK_AMD_gpu_shader_half_float */

    // [DEPRECATED] Superseded by KHR_shader_subgroup_ballot
    /* VK_AMD_shader_ballot */

    // [DO NOT EXPOSE] Don't want to be on the hook for the MPEG-LA useless Patent Trolls
    /* VK_EXT_video_encode_h264 */

    // [DO NOT EXPOSE] Don't want to be on the hook for the MPEG-LA useless Patent Trolls
    // VK_EXT_video_encode_h265

    // [DO NOT EXPOSE] Don't want to be on the hook for the MPEG-LA useless Patent Trolls
    // VK_KHR_video_decode_h264

    // [TODO LATER] Won't expose for now, API changes necessary
    /* VK_AMD_texture_gather_bias_lod */

    // [TODO] need impl, also expose as a limit?
    /* VK_AMD_shader_info */
    bool shaderInfoAMD = false;

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_AMD_extension_44

    // [DO NOT EXPOSE] We don't support yet
    // VK_KHR_dynamic_rendering

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_AMD_extension_46

    // [EXPOSE AS LIMIT]
    /* VK_AMD_shader_image_load_store_lod */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_NVX_extension_48
    // VK_GOOGLE_extension_49

    // [DO NOT EXPOSE] This used to be for Stadia, Stadia is dead now
    // VK_GGP_stream_descriptor_surface

    // [DO NOT EXPOSE] for a very long time
    /* CornerSampledImageFeaturesNV *//* VK_NV_corner_sampled_image */

    // [DO NOT EXPOSE] We don't support yet
    // VK_NV_private_vendor_info

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_NV_extension_53

    // [DO NOT EXPOSE] We don't support yet
    // VK_KHR_multiview

    // [DO NOT EXPOSE] Enables certain formats in Vulkan, we just enable them if available or else we need to make format support query functions in LogicalDevice as well
    /* VK_IMG_format_pvrtc */

    // [DEPRECATED] by VK_KHR_external_memory_capabilities
    // VK_NV_external_memory_capabilities

    // [DEPRECATED] by VK_KHR_external_memory
    // VK_NV_external_memory

    // [DEPRECATED] Promoted to VK_KHR_external_memory_win32 
    /* VK_NV_external_memory_win32 */

    // [DEPRECATED] Promoted to VK_KHR_win32_keyed_mutex 
    /* VK_NV_win32_keyed_mutex */

    // [DEPRECATED] Promoted to non-optional core VK 1.1
    /* VK_KHR_get_physical_device_properties2 */

    // [DEPRECATED] Promoted to non-optional core VK 1.1
    /* VK_KHR_device_group */

    // [DEPRECATED] by VK_EXT_validation_features
    // VK_EXT_validation_flags

    // [DO NOT EXPOSE] We don't support yet
    // VK_NN_vi_surface

    // [DEPRECATED] Vulkan 1.1 Core
    /* VK_KHR_shader_draw_parameters */

    // [DEPRECATED] by VK_VERSION_1_2
    // VK_EXT_shader_subgroup_ballot

    // [DEPRECATED] by VK_VERSION_1_1
    // VK_EXT_shader_subgroup_vote

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_texture_compression_astc_hdr

    // [TODO]
    /* ASTCDecodeFeaturesEXT *//* VK_EXT_astc_decode_mode */
    //VkFormat           decodeMode;

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_pipeline_robustness

    // [DEPRECATED] Promoted to non-optional core
    // VK_KHR_maintenance1

    // [DEPRECATED] Promoted to non-optional core VK 1.1
    // VK_KHR_device_group_creation

    // [DEPRECATED] Promoted to non-optional core Vk 1.1
    // VK_KHR_external_memory_capabilities

    // [DEPRECATED] Promoted to non-optional core VK 1.1
    /* VK_KHR_external_memory */

    // [DEPRECATED] Always Enabled but requires Instance Extensions during API Connection Creation!
    /* VK_KHR_external_memory_win32 */

    // [DEPRECATED] Always Enabled but requires Instance Extensions during API Connection Creation!
    /* VK_KHR_external_memory_fd */

    // [DO NOT EXPOSE] Always enabled, used for dx11 interop
    /* VK_KHR_win32_keyed_mutex */

    // [DEPRECATED] Promoted to non-optional core VK 1.1
    // VK_KHR_external_semaphore_capabilities
    /* VK_KHR_external_semaphore */

    // [DEPRECATED] Always Enabled but requires Instance Extensions during API Connection Creation!
    /* VK_KHR_external_semaphore_win32 */

    // [DEPRECATED] Always Enabled but requires Instance Extensions during API Connection Creation!
    /* VK_KHR_external_semaphore_fd */

    // [DO NOT EXPOSE] We don't support yet
    // VK_KHR_push_descriptor

    // [TODO] now we need API to deal with queries and begin/end conditional blocks
    /* ConditionalRenderingFeaturesEXT *//* VK_EXT_conditional_rendering */
    bool conditionalRendering = false;
    bool inheritedConditionalRendering = false;

    // [DEPRECATED] Vulkan 1.2 Core
    /* VK_KHR_shader_float16_int8 */

    // [DEPRECATED] Vulkan 1.2 Core and Required
    // VK_KHR_16bit_storage

    // [DO NOT EXPOSE] this is "swap with damange" known from EGL, cant be arsed to support
    /* VK_KHR_incremental_present */

    // [TODO] Promoted to VK1.1 non-optional core, haven't updated API to match
    /* VK_KHR_descriptor_update_template */

    // [DEPRECATED] now VK_NV_device_generated_commands
    // VK_NVX_device_generated_commands

    // [TODO LATER] Don't expose VR features for now
    /* VK_NV_clip_space_w_scaling */

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_direct_mode_display

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_acquire_xlib_display

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_display_surface_counter

    // [TODO LATER] Requires handling display swapchain stuff
    /* VK_EXT_display_control */

    // [EXPOSE AS LIMIT]
    /* VK_GOOGLE_display_timing */

    // [DO NOT EXPOSE] We don't support yet
    // VK_RESERVED_do_not_use_94

    // [TODO] Investigate
    /* VK_NV_sample_mask_override_coverage */

    /* VK_NV_geometry_shader_passthrough */
    // The only reason its not a limit is because it needs geometryShader to be enabled
    bool geometryShaderPassthrough = false;

    // [DO NOT EXPOSE] We don't support yet
    // VK_NVX_multiview_per_view_attributes

    // [DO NOT EXPOSE] A silly Nvidia extension thats specific to singlepass cubemap rendering and voxelization with geometry shader
    /* VK_NV_viewport_swizzle */

    // [EXPOSE AS A LIMIT]
    // VK_EXT_discard_rectangles

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_NV_extension_101

    // [EXPOSE AS A LIMIT]
    // VK_EXT_conservative_rasterization

    // [DO NOT EXPOSE] only useful for D3D emulators
    /* DepthClipEnableFeaturesEXT *//* VK_EXT_depth_clip_enable */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_NV_extension_104

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_swapchain_colorspace

    // [TODO] need impl
    /* VK_EXT_hdr_metadata */
    bool hdrMetadata = false;

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_IMG_extension_107
    // VK_IMG_extension_108

    // [DO NOT EXPOSE] We don't support yet
    // VK_KHR_imageless_framebuffer

    // [DEPRECATED] Core 1.2 implemented on default path and there's no choice in not using it
    // VK_KHR_create_renderpass2

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_IMG_extension_111

    // [DO NOT EXPOSE] Leave for later consideration
    /* VK_KHR_shared_presentable_image */

    // [DEPRECATED] Promoted to non-optional core VK 1.1
    // VK_KHR_external_fence_capabilities

    // [DEPRECATED] Promoted to non-optional core VK 1.1
    /* VK_KHR_external_fence */

    // [DEPRECATED] Always Enabled but requires Instance Extensions during API Connection Creation!
    /* VK_KHR_external_fence_win32 */

    // [DEPRECATED] Always Enabled but requires Instance Extensions during API Connection Creation!
    /* VK_KHR_external_fence_fd */

    /* PerformanceQueryFeaturesKHR *//* VK_KHR_performance_query */
    bool performanceCounterQueryPools = false;
    bool performanceCounterMultipleQueryPools = false;

    // [DEPRECATED] Core in Vulkan 1.x
    // VK_KHR_maintenance2

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_KHR_extension_119

    // [DO NOT EXPOSE] We don't support yet
    // VK_KHR_get_surface_capabilities2

    // [DEPRECATED] Vulkan 1.1 Core
    /* VK_KHR_variable_pointers */

    // [DO NOT EXPOSE] We don't support yet
    // VK_KHR_get_display_properties2

    // [DEPRECATED] by VK_EXT_metal_surface
    // VK_MVK_ios_surface

    // [DEPRECATED] by VK_EXT_metal_surface
    // VK_MVK_macos_surface

    // [DO NOT EXPOSE] We don't support yet
    // VK_MVK_moltenvk

    // [TODO LATER] Requires exposing external memory first
    /* VK_EXT_external_memory_dma_buf */

    // [EXPOSE AS A LIMIT] 
    // VK_EXT_queue_family_foreign

    // [DEPRECATED] Vulkan 1.1 core now
    // VK_KHR_dedicated_allocation

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_debug_utils

    // [TODO LATER] Requires exposing external memory first
    /* VK_ANDROID_external_memory_android_hardware_buffer */

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_sampler_filter_minmax

    // [DEPRECATED] Promoted to non-optional core VK 1.1
    /* VK_KHR_storage_buffer_storage_class */

    // [DEPRECATED] Just check for `shaderInt16` and related `16BitAccess` limits
    /* VK_AMD_gpu_shader_int16 */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_AMD_extension_134

    // [DO NOT EXPOSE] We don't support yet
    // VK_AMDX_shader_enqueue

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_AMD_extension_136

    // [TODO] need impl
    /* VK_AMD_mixed_attachment_samples *//* OR *//* VK_NV_framebuffer_mixed_samples */
    bool mixedAttachmentSamples = false;

    // [TODO LATER] Wait until VK_AMD_shader_fragment_mask & VK_QCOM_render_pass_shader_resolve converge to something useful
    /* VK_AMD_shader_fragment_mask */

    // [DEPRECATED] Required wholly by ROADMAP 2022 and Nabla Core Profile
    /* InlineUniformBlockFeaturesEXT *//* VK_EXT_inline_uniform_block */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_AMD_extension_140

    // Enabled by Default, Moved to Limits
    /* VK_EXT_shader_stencil_export */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_AMD_extension_142
    // VK_AMD_extension_143

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_sample_locations

    // [DEPRECATED] Promoted to non-optional core VK 1.1
    /* VK_KHR_relaxed_block_layout */

    // [DO NOT EXPOSE] We don't support yet
    // VK_RESERVED_do_not_use_146

    // [DEPRECATED] Promoted to core non-optional VK 1.1
    /* VK_KHR_get_memory_requirements2 */

    // [DEPRECATED] Vulkan 1.2 core non-optional
    /* VK_KHR_image_format_list */

    // [DO NOT EXPOSE] This is dumb, you can implement whatever blend equation you want with `EXT_fragment_shader_interlock` and EXT_shader_tile_image
    /* BlendOperationAdvancedFeaturesEXT *//* VK_EXT_blend_operation_advanced */

    // [TODO LATER] Requires API changes
    /* VK_NV_fragment_coverage_to_color */

    /* AccelerationStructureFeaturesKHR *//* VK_KHR_acceleration_structure */
    bool accelerationStructure = false;
    bool accelerationStructureIndirectBuild = false;
    bool accelerationStructureHostCommands = false;
    // [DO NOT EXPOSE] implied by `accelerationStructure`
    //bool descriptorBindingAccelerationStructureUpdateAfterBind = accelerationStructure;

    /* RayTracingPipelineFeaturesKHR *//* VK_KHR_ray_tracing_pipeline */
    bool rayTracingPipeline = false;
    // bool rayTracingPipelineShaderGroupHandleCaptureReplay; // [DO NOT EXPOSE] for capture tools
    // bool rayTracingPipelineShaderGroupHandleCaptureReplayMixed; // [DO NOT EXPOSE] for capture tools
    // [DO NOT EXPOSE] Vulkan feature requirements
    //bool rayTracingPipelineTraceRaysIndirect = rayTracingPipeline;
    bool rayTraversalPrimitiveCulling = false;

    /* RayQueryFeaturesKHR *//* VK_KHR_ray_query */
    bool rayQuery = false;

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_NV_extension_152

    // [ALIASED TO] VK_AMD_mixed_attachment_samples
    // VK_NV_framebuffer_mixed_samples

    // [DO NOT EXPOSE] For now. For 2D ui
    /* VK_NV_fill_rectangle */

    // [EXPOSE AS A LIMIT] Enabled by Default, Moved to Limits
    /* ShaderSMBuiltinsFeaturesNV *//* VK_NV_shader_sm_builtins */

    // Enabled by Default, Moved to Limits
    /* VK_EXT_post_depth_coverage */

    // [DEPRECATED] Vulkan 1.1 Core and ROADMAP 2022
    /* SamplerYcbcrConversionFeaturesKHR *//* VK_KHR_sampler_ycbcr_conversion */

    // [DEPRECATED] Core 1.1 implemented on default path and there's no choice in not using it
    /* VK_KHR_bind_memory2 */

    // [DO NOT EXPOSE] Too "intrinsically linux"
    /* VK_EXT_image_drm_format_modifier */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_160

    // [TODO LATER] Expose when we start to experience slowdowns from validation
    /* VK_EXT_validation_cache */

    // [DEPRECATED] KHR extension supersedes and then Vulkan 1.2
    // VK_EXT_descriptor_indexing

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_shader_viewport_index_layer

    /* VK_KHR_portability_subset - PROVISINAL/NOTAVAILABLEANYMORE */

    // [DO NOT EXPOSE] not implementing or exposing VRS in near or far future, also has interactions with fragment density maps
    /* ShadingRateImageFeaturesNV *//* VK_NV_shading_rate_image */

    // [DEPRECATED] Superseded by KHR
    // VK_NV_ray_tracing

    /* RepresentativeFragmentTestFeaturesNV *//* VK_NV_representative_fragment_test */
    bool representativeFragmentTest = false;

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_NV_extension_168

    // [DO NOT EXPOSE] We don't support yet
    // VK_KHR_maintenance3

    // [DEPRECATED] Core in Vulkan 1.x
    // VK_KHR_draw_indirect_count

    // [TODO LATER] limited utility and availability, might expose if feel like wasting time
    /* VK_EXT_filter_cubic */

    // [TODO LATER] Wait until VK_AMD_shader_fragment_mask & VK_QCOM_render_pass_shader_resolve converge to something useful
    /* VK_QCOM_render_pass_shader_resolve */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_QCOM_extension_173
    // VK_QCOM_extension_174

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_global_priority

    // [DEPRECATED] Vulkan 1.2 Core non-optional
    /* VK_KHR_shader_subgroup_extended_types */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_177

    // [DEPRECATED] Promoted to VK Core
    // VK_KHR_8bit_storage

    // [DO NOT EXPOSE] TODO: support in the CUDA PR
    // VK_EXT_external_memory_host

    // [TODO] need impl/more research
    /* VK_AMD_buffer_marker */
    bool bufferMarkerAMD = false;

    // [DEPRECATED] Vulkan 1.2 Core
    /* VK_KHR_shader_atomic_int64 */

    // [EXPOSE AS LIMIT]
    /* ShaderClockFeaturesKHR *//* VK_KHR_shader_clock */
    //bool shaderDeviceClock;
    //bool shaderSubgroupClock = shaderDeviceClock;

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_AMD_extension_183

    // [DO NOT EXPOSE] Too vendor specific
    /* VK_AMD_pipeline_compiler_control */

    // [TODO LATER] Requires changes to API
    /* VK_EXT_calibrated_timestamps */

    // [DEPRECATED] Superseded by VK_AMD_shader_core_properties
    // VK_AMD_shader_core_properties

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_AMD_extension_187

    // [DO NOT EXPOSE] We don't want to be on the hook with the MPEG-LA patent trolls
    // VK_KHR_video_decode_h265

    // [TODO] this one isn't in the headers yet
    /* GlobalPriorityQueryFeaturesKHR *//* VK_KHR_global_priority */
    //VkQueueGlobalPriorityKHR    globalPriority;

    // [DO NOT EXPOSE]
    /* VK_AMD_memory_overallocation_behavior */

    // [DO NOT EXPOSE] we would have to change the API
    /* VertexAttributeDivisorFeaturesEXT *//* VK_EXT_vertex_attribute_divisor */

    // [DEPRECATED] Stadia is dead
    // VK_GGP_frame_token

    // [TODO LATER] would like to expose, but too much API to change
    /* VK_EXT_pipeline_creation_feedback */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_GOOGLE_extension_194
    // VK_GOOGLE_extension_195
    // VK_GOOGLE_extension_196

    // [DEPRECATED] Promoted to VK Core 1.x
    // VK_KHR_driver_properties

    // [DEPRECATED] Promoted to VK Core 1.x
    // VK_KHR_shader_float_controls

    // [DEPRECATED] Superseded by `clustered` subgroup ops
    /* VK_NV_shader_subgroup_partitioned */

    // [DEPRECATED] Promoted to VK Core 1.x
    // VK_KHR_depth_stencil_resolve

    // [DEPRECATED] Vulkan 1.2 core non-optional
    /* VK_KHR_swapchain_mutable_format */

    // [EXPOSE AS LIMIT]
    /* ComputeShaderDerivativesFeaturesNV *//* VK_NV_compute_shader_derivatives */

    // [DEPRECATED] Expose the KHR extension instead
    /* MeshShaderFeaturesNV *//* VK_NV_mesh_shader */

    // [DO NOT EXPOSE] Deprecated by KHR version
    /* FragmentShaderBarycentricFeaturesNV *//* VK_NV_fragment_shader_barycentric */

    // [EXPOSE AS LIMIT] Enabled by Default, Moved to Limits 
    /* ShaderImageFootprintFeaturesNV *//* VK_NV_shader_image_footprint */
    //bool           imageFootprint;

    // [TODO LATER] requires extra API work to use
    // GL Hint: in GL/GLES this is NV_scissor_exclusive
    /* ExclusiveScissorFeaturesNV *//* VK_NV_scissor_exclusive */
    //bool           exclusiveScissor;

    // [DO NOT EXPOSE] We don't support yet
    // VK_NV_device_diagnostic_checkpoints

    // [DEPRECATED] Vulkan 1.2 Core non-optional
    /* VK_KHR_timeline_semaphore */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_KHR_extension_209

    // [EXPOSE AS LIMIT] Enabled by Default, Moved to Limits 
    /* ShaderIntegerFunctions2FeaturesINTEL *//* VK_INTEL_shader_integer_functions2 */
    //bool           shaderIntegerFunctions2 = false;

    // [DEPRECATED] Promoted to VK_KHR_performance_query, VK1.1 core
    /* VK_INTEL_performance_query */

    // [DEPRECATED] Vulkan 1.2 Core but 1.3 non-optional and we require it
    /* VK_KHR_vulkan_memory_model */

    // [LIMIT] We just report it
    // VK_EXT_pci_bus_info

    // [DO NOT EXPOSE] Waiting for cross platform
    /* VK_AMD_display_native_hdr */

    // [DO NOT EXPOSE] We don't support yet
    // VK_FUCHSIA_imagepipe_surface

    // [DEPRECATED] Vulkan 1.3 Core
    /* VK_KHR_shader_terminate_invocation */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_GOOGLE_extension_217

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_metal_surface

    // [TODO] need impl
    /* FragmentDensityMapFeaturesEXT *//* VK_EXT_fragment_density_map */
    bool fragmentDensityMap = false;
    bool fragmentDensityMapDynamic = false;
    bool fragmentDensityMapNonSubsampledImages = false;

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_220
    // VK_KHR_extension_221

    // [DEPRECATED] Vulkan 1.3 core and Nabla core profile required
    // VK_EXT_scalar_block_layout

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_223

    // [DO NOT EXPOSE] We compile our own SPIR-V like real men
    /* VK_GOOGLE_hlsl_functionality1 */

    // Enabled by Default, Moved to Limits
    /* VK_GOOGLE_decorate_string */

    // [DO NOT EXPOSE] We don't support yet, but TODO
    // VK_EXT_subgroup_size_control

    // [DO NOT EXPOSE] not implementing or exposing VRS in near or far future
    /* FragmentShadingRateFeaturesKHR *//* VK_KHR_fragment_shading_rate */

    // [DO NOT EXPOSE] We don't support yet
    // VK_AMD_shader_core_properties2

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_AMD_extension_229

    // [TODO] need impl/more research
    /* CoherentMemoryFeaturesAMD *//* VK_AMD_device_coherent_memory */
    bool deviceCoherentMemory = false;

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_AMD_extension_231
    // VK_AMD_extension_232
    // VK_AMD_extension_233
    // VK_AMD_extension_234

    // [EXPOSE AS A LIMIT]
    // VK_EXT_shader_image_atomic_int64

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_AMD_extension_236

    // [DEPRECATED] We now require it with Vulkan 1.2 and its non-optional
    /* VK_KHR_spirv_1_4 */

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_memory_budget

    // [TODO] need impl
    /* MemoryPriorityFeaturesEXT *//* VK_EXT_memory_priority */
    bool memoryPriority = false;

    // [DO NOT EXPOSE] We don't support yet
    // VK_KHR_surface_protected_capabilities

    // [DO NOT EXPOSE] insane oxymoron, dedicated means dedicated, not aliased, won't expose
    /* DedicatedAllocationImageAliasingFeaturesNV *//* VK_NV_dedicated_allocation_image_aliasing */

    // [DEPRECATED] Vulkan 1.2 Core non-optional
    /* VK_KHR_separate_depth_stencil_layouts */
    /* SeparateDepthStencilLayoutsFeaturesKHR */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_INTEL_extension_243
    // VK_MESA_extension_244

    // [DEPRECATED] by VK_KHR_buffer_device_address
    // VK_EXT_buffer_device_address

    // [DO NOT EXPOSE] we dont need to care or know about it, unless for BDA/RT Replays?
    /* VK_EXT_tooling_info */

    // [DEPRECATED] Core 1.2 implemented on default path and there's no choice in not using it
    /* VK_EXT_separate_stencil_usage */

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_validation_features

    // [DO NOT EXPOSE] won't expose, this extension is poop, I should have a Fence-andQuery-like object to query the presentation timestamp, not a blocking call that may unblock after an arbitrary delay from the present
    /* PresentWaitFeaturesKHR *//* VK_KHR_present_wait */

    // [DEPRECATED] replaced by VK_KHR_cooperative_matrix
    /* CooperativeMatrixFeaturesNV *//* VK_NV_cooperative_matrix */

    // [TODO] need impl or waaay too vendor specific?
    /* CoverageReductionModeFeaturesNV *//* VK_NV_coverage_reduction_mode */
    //bool coverageReductionMode = false;

    /* FragmentShaderInterlockFeaturesEXT *//* VK_EXT_fragment_shader_interlock */
    bool fragmentShaderSampleInterlock = false;
    bool fragmentShaderPixelInterlock = false;
    bool fragmentShaderShadingRateInterlock = false;

    // [DO NOT EXPOSE] Expose nothing to do with video atm
    /* YcbcrImageArraysFeaturesEXT *//* VK_EXT_ycbcr_image_arrays */

    // [DEPRECATED] Vulkan 1.2 Core non-optional
    // VK_KHR_uniform_buffer_standard_layout

    // [DO NOT EXPOSE] provokingVertexLast will not expose (we always use First Vertex Vulkan-like convention), anything to do with XForm-feedback we don't expose
    /* ProvokingVertexFeaturesEXT *//* VK_EXT_provoking_vertex */

    // [TODO LATER] Requires API changes
    /* VK_EXT_full_screen_exclusive */

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_headless_surface

    // [DEPRECATED] Core in Vulkan 1.3 and NAbla Core Profile
    // VK_KHR_buffer_device_address

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_extension_259

    // [TODO] need impl, this feature introduces new/more pipeline state with VkPipelineRasterizationLineStateCreateInfoEXT
    /* LineRasterizationFeaturesEXT *//* VK_EXT_line_rasterization */
    bool rectangularLines = false;
    bool bresenhamLines = false;
    bool smoothLines = false;
    bool stippledRectangularLines = false;
    bool stippledBresenhamLines = false;
    bool stippledSmoothLines = false;

    // [NAbla core Profile LIMIT] 
    /* ShaderAtomicFloatFeaturesEXT *//* VK_EXT_shader_atomic_float */

    // [DEPRECATED] MOVED TO Vulkan 1.2 Core
    /* HostQueryResetFeatures *//* VK_EXT_host_query_reset */
    
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_GGP_extension_263
    // VK_BRCM_extension_264
    // VK_BRCM_extension_265

    // [TODO] need impl
    /* IndexTypeUint8FeaturesEXT *//* VK_EXT_index_type_uint8 */
    bool indexTypeUint8 = false;

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_267

    // [DO NOT EXPOSE] we're fully GPU driven anyway with sorted pipelines.
    /* ExtendedDynamicStateFeaturesEXT *//* VK_EXT_extended_dynamic_state */

    /* VK_KHR_deferred_host_operations */
    bool deferredHostOperations = false;

    // [TODO] need impl
    /* PipelineExecutablePropertiesFeaturesKHR *//* VK_KHR_pipeline_executable_properties */
    bool pipelineExecutableInfo = false;

    // [DO NOT EXPOSE] We don't support yet, but should when ubiquitous
    // VK_EXT_host_image_copy

    // [DO NOT EXPOSE] We don't support yet
    // VK_KHR_map_memory2

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_INTEL_extension_273
    
    // [EXPOSE AS LIMIT]
    /* ShaderAtomicFloat2FeaturesEXT *//* VK_EXT_shader_atomic_float2 */

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_surface_maintenance1

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_swapchain_maintenance1

    // [DEPRECATED] Core in Vulkan 1.3
    // VK_EXT_shader_demote_to_helper_invocation

    // [TODO] need impl
    /* DeviceGeneratedCommandsFeaturesNV *//* VK_NV_device_generated_commands */
    bool deviceGeneratedCommands = false;

    // [DO NOT EXPOSE] won't expose, the existing inheritance of state is enough
    /* InheritedViewportScissorFeaturesNV *//* VK_NV_inherited_viewport_scissor */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_KHR_extension_280

    // [DEPRECATED] Vulkan 1.3 Core non-optional
    /* VK_KHR_shader_integer_dot_product */

    // [DEPRECATED] Vulkan 1.3 non-optional and Nabla Core Profile.
    /* TexelBufferAlignmentFeaturesEXT *//* VK_EXT_texel_buffer_alignment */

    // [DO NOT EXPOSE] Too vendor specific
    /* VK_QCOM_render_pass_transform */

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_depth_bias_control

    // [EXPOSE AS LIMIT]
    /* DeviceMemoryReportFeaturesEXT *//* VK_EXT_device_memory_report */

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_acquire_drm_display

    // [Nabla CORE PROFILE]
    /* Robustness2FeaturesEXT *//* VK_EXT_robustness2 */

    // [DO NOT EXPOSE] not going to expose custom border colors for now
    /* CustomBorderColorFeaturesEXT *//* VK_EXT_custom_border_color */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_289

    // [DO NOT EXPOSE] 0 documentation
    /* VK_GOOGLE_user_type */

    // [DO NOT EXPOSE] We don't support yet
    // VK_KHR_pipeline_library

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_NV_extension_292

    // [DO NOT EXPOSE] Triage leftover extensions below    
    /* VK_NV_present_barrier */

    // [EXPOSE AS A LIMIT] Enabled by Default, Moved to Limits
    /* VK_KHR_shader_non_semantic_info */

    // [DO NOT EXPOSE] no point exposing until an extension more useful than VK_KHR_present_wait arrives
    /* PresentIdFeaturesKHR *//* VK_KHR_present_id */

    // [DEPRECATED] Vulkan 1.3 core non-optional
    /* PrivateDataFeatures *//* VK_EXT_private_data */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_KHR_extension_297

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_pipeline_creation_cache_control

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_KHR_extension_299

    // [TODO] Provisional
    /* VK_KHR_video_encode_queue */

    // [DO NOT EXPOSE]
    /* DiagnosticsConfigFeaturesNV *//* VK_NV_device_diagnostics_config */

    // [DO NOT EXPOSE] absorbed into VK_EXT_load_store_op_none
    /* VK_QCOM_render_pass_store_ops */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_QCOM_extension_303
    // VK_QCOM_extension_304
    // VK_QCOM_extension_305
    // VK_QCOM_extension_306
    // VK_QCOM_extension_307
    // VK_NV_extension_308

    // [DO NOT EXPOSE] We don't support yet
    // VK_KHR_object_refresh

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_QCOM_extension_310

    // [DO NOT EXPOSE] 0 documentation
    /* VK_NV_low_latency */

    // [TODO LATER] Expose when we support MoltenVK
    /* VK_EXT_metal_objects */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_313
    // VK_AMD_extension_314

    // [DEPRECATED] Vulkan 1.3 Core
    // VK_KHR_synchronization2

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_AMD_extension_316

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_descriptor_buffer

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_AMD_extension_318
    // VK_AMD_extension_319
    // VK_AMD_extension_320

    // [TODO] this one isn't in the headers yet
    /* GraphicsPipelineLibraryFeaturesEXT *//* VK_EXT_graphics_pipeline_library */
    //bool           graphicsPipelineLibrary;

    // [EXPOSE AS A LIMIT]
    /* VK_AMD_shader_early_and_late_fragment_tests */
    //bool shaderEarlyAndLateFragmentTests;

    // [EXPOSE AS A LIMIT] Enabled by Default, Moved to Limits
    /* VK_KHR_fragment_shader_barycentric */

    // [EXPOSE AS LIMIT]
    /* ShaderSubgroupUniformControlFlowFeaturesKHR *//* VK_KHR_shader_subgroup_uniform_control_flow */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_KHR_extension_325

    // [DEPRECATED] Vulkan 1.3 Core
    /* VK_KHR_zero_initialize_workgroup_memory */

    // [DO NOT EXPOSE] would first need to expose VK_KHR_fragment_shading_rate before
    /* VK_NV_fragment_shading_rate_enums */

    /* RayTracingMotionBlurFeaturesNV *//* VK_NV_ray_tracing_motion_blur */
    bool rayTracingMotionBlur = false;
    bool rayTracingMotionBlurPipelineTraceRaysIndirect = false;

    // [DEPRECATED] By KHR_mesh_shader
    // VK_EXT_mesh_shader

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_NV_extension_330

    // [DO NOT EXPOSE] Enables certain formats in Vulkan, we just enable them if available or else we need to make format support query functions in LogicalDevice as well
    /* Ycbcr2Plane444FormatsFeaturesEXT *//* VK_EXT_ycbcr_2plane_444_formats */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_NV_extension_332

    /* FragmentDensityMap2FeaturesEXT *//* VK_EXT_fragment_density_map2 */
    bool fragmentDensityMapDeferred = false;

    // [DO NOT EXPOSE] Too vendor specific
    /* VK_QCOM_rotated_copy_commands */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_KHR_extension_335

    // [DEPRECATED] Vulkan 1.3 core non-optional
    /* VK_EXT_image_robustness */

    // [EXPOSE AS LIMIT]
    /* WorkgroupMemoryExplicitLayoutFeaturesKHR *//* VK_KHR_workgroup_memory_explicit_layout */

    // [DO NOT EXPOSE] Promoted to VK 1.3 non-optional core and present in Nabla Core Profile, but serves no purpose other than providing a pNext chain for the usage of a single QCOM extension
    /* VK_KHR_copy_commands2 */

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_image_compression_control

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_attachment_feedback_loop_layout

    // [DO NOT EXPOSE] Vulkan 1.3 non-optional, we just enable them if available or else we need to make format support query functions in LogicalDevice as well
    /* 4444FormatsFeaturesEXT *//* VK_EXT_4444_formats */

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_device_fault

    // [TODO] need impl or just expose `shader_tile_image` instead?
    /* RasterizationOrderAttachmentAccessFeaturesARM *//* VK_ARM_rasterization_order_attachment_access */
    bool rasterizationOrderColorAttachmentAccess = false;
    bool rasterizationOrderDepthAttachmentAccess = false;
    bool rasterizationOrderStencilAttachmentAccess = false;

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_ARM_extension_344

    // [DO NOT EXPOSE] wont expose yet (or ever), requires VK_KHR_sampler_ycbcr_conversion
    /* VK_EXT_rgba10x6_formats */

    // [TODO LATER] won't decide yet, requires VK_EXT_direct_mode_display anyway
    /* VK_NV_acquire_winrt_display */

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_directfb_surface

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_KHR_extension_350
    // VK_NV_extension_351

    // [DO NOT EXPOSE] its a D3D special use extension, shouldn't expose
    /* VK_VALVE_mutable_descriptor_type */

    // [DO NOT EXPOSE] too much API Fudgery
    /* VK_EXT_vertex_input_dynamic_state */

    // [DO NOT EXPOSE] Too "intrinsically linux"
    /* VK_EXT_physical_device_drm */

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_device_address_binding_report

    // [DO NOT EXPOSE] EVER, VULKAN DEPTH RANGE ONLY!
    /* VK_EXT_depth_clip_control */

    // [DO NOT EXPOSE]
    /* VK_EXT_primitive_topology_list_restart */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_KHR_extension_358
    // VK_EXT_extension_359
    // VK_EXT_extension_360

    // [DEPRECATED] Promoted to non - optional Core 1.3, we always use it!
    /* VK_KHR_format_feature_flags2 */ // Promoted to core 1.3;
    // [TODO]
    // lots of support for anything that isn't mobile (list of unsupported devices since the extension was published: https://pastebin.com/skZAbL4F)

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_362
    // VK_EXT_extension_363
    // VK_FUCHSIA_extension_364

    // [DO NOT EXPOSE] We don't support yet
    // VK_FUCHSIA_external_memory

    // [DO NOT EXPOSE] We don't support yet
    // VK_FUCHSIA_external_semaphore

    // [DO NOT EXPOSE] We don't support yet
    // VK_FUCHSIA_buffer_collection

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_FUCHSIA_extension_368
    // VK_QCOM_extension_369

    // [DO NOT EXPOSE]
    /* SubpassShadingFeaturesHUAWEI *//* VK_HUAWEI_subpass_shading */

    // [DO NOT EXPOSE] We don't support yet
    // VK_HUAWEI_invocation_mask

    // [TODO LATER] when we do multi-gpu
    /* ExternalMemoryRDMAFeaturesNV *//* VK_NV_external_memory_rdma */
    //bool           externalMemoryRDMA;

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_pipeline_properties

    // [DEPRECATED] by VK_NV_external_sci_sync2
    // VK_NV_external_sci_sync

    // [DO NOT EXPOSE] We don't support yet
    // VK_NV_external_memory_sci_buf

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_frame_boundary

    // [DO NOT EXPOSE] We don't support yet, probably only useful for stencil-K-routed OIT
    // VK_EXT_multisampled_render_to_single_sampled 

    // [DO NOT EXPOSE] we're fully GPU driven anyway with sorted pipelines.
    /* ExtendedDynamicState2FeaturesEXT *//* VK_EXT_extended_dynamic_state2 */

    // [DO NOT EXPOSE] We don't support yet
    // VK_QNX_screen_surface

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_KHR_extension_380
    // VK_KHR_extension_381

    // [EXPOSE AS LIMIT]
    /* ColorWriteEnableFeaturesEXT *//* VK_EXT_color_write_enable */

    // [DO NOT EXPOSE] requires and relates to EXT_transform_feedback which we'll never expose
    /* PrimitivesGeneratedQueryFeaturesEXT *//* VK_EXT_primitives_generated_query */
    //bool           primitivesGeneratedQuery;
    //bool           primitivesGeneratedQueryWithRasterizerDiscard;
    //bool           primitivesGeneratedQueryWithNonZeroStreams;

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_384
    // VK_MESA_extension_385
    // VK_GOOGLE_extension_386

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

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_388

    // [DEPRECATED] absorbed into KHR_global_priority
    /* VK_EXT_global_priority_query */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_390
    // VK_EXT_extension_391

    // [DO NOT EXPOSE] pointless to implement currently
    /* ImageViewMinLodFeaturesEXT *//* VK_EXT_image_view_min_lod */

    // [DO NOT EXPOSE] this extension is dumb, if we're recording that many draws we will be using Multi Draw INDIRECT which is better supported
    /* MultiDrawFeaturesEXT *//* VK_EXT_multi_draw */

    // [TODO] Investigate later
    /* Image2DViewOf3DFeaturesEXT *//* VK_EXT_image_2d_view_of_3d */
    //bool           image2DViewOf3D;
    //bool           sampler2DViewOf3D;

    // [DO NOT EXPOSE] We don't support yet
    // VK_KHR_portability_enumeration

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_shader_tile_image

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_opacity_micromap

    // [DO NOT EXPOSE] We don't support yet
    // VK_NV_displacement_micromap

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_JUICE_extension_399
    // VK_JUICE_extension_400

    // [TODO LATER] ROADMAP 2024 but need to figure out how extending our LOAD_OP enum would affect us
    /* VK_EXT_load_store_op_none */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_FB_extension_402
    // VK_FB_extension_403
    // VK_FB_extension_404

    // [DO NOT EXPOSE] We don't support yet
    // VK_HUAWEI_cluster_culling_shader

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_HUAWEI_extension_406
    // VK_GGP_extension_407
    // VK_GGP_extension_408
    // VK_GGP_extension_409
    // VK_GGP_extension_410
    // VK_GGP_extension_411

    // [DO NOT EXPOSE] not going to expose custom border colors for now
    /* BorderColorSwizzleFeaturesEXT *//* VK_EXT_border_color_swizzle */

    // [DO NOT EXPOSE] pointless to expose without exposing VK_EXT_memory_priority and the memory query feature first
    /* PageableDeviceLocalMemoryFeaturesEXT *//* VK_EXT_pageable_device_local_memory */

    // [DO NOT EXPOSE] We don't support yet
    // VK_KHR_maintenance4

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_HUAWEI_extension_415

    // [DO NOT EXPOSE] We don't support yet
    // VK_ARM_shader_core_properties

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_KHR_extension_417
    // VK_ARM_extension_418

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_image_sliced_view_of_3d

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_420

    // [DO NOT EXPOSE] This extension is only intended for use in specific embedded environments with known implementation details, and is therefore undocumented.
    /* DescriptorSetHostMappingFeaturesVALVE *//* VK_VALVE_descriptor_set_host_mapping */

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_depth_clamp_zero_one

    // [DO NOT EXPOSE] Never expose this, it was a mistake for that GL quirk to exist in the first place
    /* VK_EXT_non_seamless_cube_map */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_ARM_extension_424
    // VK_ARM_extension_425

    // [DO NOT EXPOSE] not implementing or exposing VRS in near or far future
    /* FragmentDensityMapOffsetFeaturesQCOM *//* VK_QCOM_fragment_density_map_offset */

    // [DO NOT EXPOSE] We don't support yet
    // VK_NV_copy_memory_indirect

    // [DO NOT EXPOSE] We don't support yet
    // VK_NV_memory_decompression

    // [DO NOT EXPOSE] We don't support yet
    // VK_NV_device_generated_commands_compute

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_NV_extension_430

    // [DO NOT EXPOSE] no idea what real-world beneficial use case would be
    /* LinearColorAttachmentFeaturesNV *//* VK_NV_linear_color_attachment */

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_NV_extension_432
    // VK_NV_extension_433

    // [DO NOT EXPOSE] We don't support yet
    // VK_GOOGLE_surfaceless_query

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_KHR_extension_435

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_application_parameters

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_437

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_image_compression_control_swapchain

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_SEC_extension_439
    // VK_QCOM_extension_440

    // [DO NOT EXPOSE] We don't support yet
    // VK_QCOM_image_processing

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_COREAVI_extension_442
    // VK_COREAVI_extension_443
    // VK_COREAVI_extension_444
    // VK_COREAVI_extension_445
    // VK_COREAVI_extension_446
    // VK_COREAVI_extension_447
    // VK_SEC_extension_448
    // VK_SEC_extension_449
    // VK_SEC_extension_450
    // VK_SEC_extension_451
    // VK_NV_extension_452
    // VK_ARM_extension_453

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_external_memory_acquire_unmodified

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_GOOGLE_extension_455

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_extended_dynamic_state3

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_457
    // VK_EXT_extension_458

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_subpass_merge_feedback

    // [DO NOT EXPOSE] We don't support yet
    // VK_LUNARG_direct_driver_loading

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_461
    // VK_EXT_extension_462

    // [TODO LATER] Basically a custom hash/ID for which you can use instead of SPIR-V contents to look up IGPUShader in the cache
    /* VK_EXT_shader_module_identifier */

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_rasterization_order_attachment_access

    // TODO: implement
    /* VK_NV_optical_flow */

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_legacy_dithering

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_pipeline_protected_access

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_468
    // VK_ANDROID_extension_469
    // VK_AMD_extension_470

    // [DO NOT EXPOSE] We don't support yet
    // VK_KHR_maintenance5

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_AMD_extension_472
    // VK_AMD_extension_473
    // VK_AMD_extension_474
    // VK_AMD_extension_475
    // VK_AMD_extension_476
    // VK_AMD_extension_477
    // VK_AMD_extension_478
    // VK_AMD_extension_479
    // VK_EXT_extension_480
    // VK_EXT_extension_481

    // TODO: implement
    // VK_KHR_ray_tracing_position_fetch

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_shader_object

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_extension_484

    // [DO NOT EXPOSE] We don't support yet
    // VK_QCOM_tile_properties

    // [DO NOT EXPOSE] We don't support yet
    // VK_SEC_amigo_profiling

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_487
    // VK_EXT_extension_488

    // [DO NOT EXPOSE] We don't support yet
    // VK_QCOM_multiview_per_view_viewports

    // [DO NOT EXPOSE] We don't support yet
    // VK_NV_external_sci_sync2

    // [DO NOT EXPOSE] We don't support yet, but a TODO
    // VK_NV_ray_tracing_invocation_reorder

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_NV_extension_492
    // VK_NV_extension_493
    // VK_NV_extension_494

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_mutable_descriptor_type

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_496
    // VK_EXT_extension_497

    // [DO NOT EXPOSE] We don't support yet
    // VK_ARM_shader_core_builtins

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_pipeline_library_group_handles

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_dynamic_rendering_unused_attachments

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_501
    // VK_EXT_extension_502
    // VK_EXT_extension_503
    // VK_NV_extension_504
    // VK_EXT_extension_505
    // VK_NV_extension_506

    /* CooperativeMatrixFeaturesKHR *//* VK_KHR_cooperative_matrix */
    // [EXPOSE AS LIMIT] redundant
    //bool cooperativeMatrix = limits.cooperativeMatrixSupportedStages.any();
    // leaving as a feature because of overhead
    bool cooperativeMatrixRobustBufferAccess = false;

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_508
    // VK_EXT_extension_509
    // VK_MESA_extension_510
    // 
    // [DO NOT EXPOSE] We don't support yet
    // VK_QCOM_multiview_per_view_render_areas

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_512
    // VK_KHR_extension_513
    // VK_KHR_extension_514
    // VK_KHR_extension_515
    // VK_KHR_extension_516
    // VK_EXT_extension_517
    // VK_MESA_extension_518

    // [DO NOT EXPOSE] We don't support yet
    // VK_QCOM_image_processing2

    // [DO NOT EXPOSE] We don't support yet
    // VK_QCOM_filter_cubic_weights

    // [DO NOT EXPOSE] We don't support yet
    // VK_QCOM_ycbcr_degamma

    // [DO NOT EXPOSE] We don't support yet
    // VK_QCOM_filter_cubic_clamp

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_523
    // VK_EXT_extension_524

    // [DO NOT EXPOSE] We don't support yet
    // VK_EXT_attachment_feedback_loop_dynamic_state

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_EXT_extension_526
    // VK_EXT_extension_527
    // VK_EXT_extension_528
    // VK_KHR_extension_529

    // [DO NOT EXPOSE] We don't support yet
    // VK_QNX_external_memory_screen_buffer

    // [DO NOT EXPOSE] We don't support yet
    // VK_MSFT_layered_driver

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_KHR_extension_532
    // VK_EXT_extension_533
    // VK_KHR_extension_534
    // VK_KHR_extension_535
    // VK_QCOM_extension_536
    // VK_EXT_extension_537
    // VK_EXT_extension_538
    // VK_EXT_extension_539
    // VK_EXT_extension_540
    // VK_EXT_extension_541
    // VK_EXT_extension_542
    // VK_EXT_extension_543
    // VK_KHR_extension_544
    // VK_KHR_extension_545
    // VK_KHR_extension_546

    // [DO NOT EXPOSE] We don't support yet
    // VK_NV_descriptor_pool_overallocation

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_QCOM_extension_548


    // Eternal TODO: how many new extensions since we last looked? (We're up to ext number 548)
    
    /* Nabla */
    // No Nabla Specific Features for now
    
    inline bool operator==(const SPhysicalDeviceFeatures& _rhs) const
    {
        return memcmp(this, &_rhs, sizeof(SPhysicalDeviceFeatures)) == 0u;
    }

    inline bool isSubsetOf(const SPhysicalDeviceFeatures& _rhs) const
    {
        const auto& intersection = intersectWith(_rhs);
        return intersection == *this;
    }

    inline SPhysicalDeviceFeatures unionWith(const SPhysicalDeviceFeatures& _rhs) const
    {
        SPhysicalDeviceFeatures res = *this;
             
        // VK 1.0 core
        res.robustBufferAccess |= _rhs.robustBufferAccess;

        res.geometryShader |= _rhs.geometryShader;
        res.tessellationShader |= _rhs.tessellationShader;

        res.depthBounds |= _rhs.depthBounds;
        res.wideLines |= _rhs.wideLines;
        res.largePoints |= _rhs.largePoints;

        res.alphaToOne |= _rhs.alphaToOne;

        res.pipelineStatisticsQuery |= _rhs.pipelineStatisticsQuery;

        res.shaderCullDistance |= _rhs.shaderCullDistance;

        res.shaderResourceResidency |= _rhs.shaderResourceResidency;
        res.shaderResourceMinLod |= _rhs.shaderResourceMinLod;

        // Vk 1.1 everything is either a Limit or Required

        // Vk 1.2
        res.bufferDeviceAddressMultiDevice |= _rhs.bufferDeviceAddressMultiDevice;

        // Vk 1.3
        res.robustImageAccess |= _rhs.robustImageAccess;

        // Nabla Core Extensions
        res.robustBufferAccess2 |= _rhs.robustBufferAccess2;
        res.robustImageAccess2 |= _rhs.robustImageAccess2;

        res.nullDescriptor |= _rhs.nullDescriptor;

        // Extensions
        res.swapchainMode |= _rhs.swapchainMode;

        res.shaderInfoAMD |= _rhs.shaderInfoAMD;

        res.conditionalRendering |= _rhs.conditionalRendering;
        res.inheritedConditionalRendering |= _rhs.inheritedConditionalRendering;

        res.geometryShaderPassthrough |= _rhs.geometryShaderPassthrough;

        res.hdrMetadata |= _rhs.hdrMetadata;

        res.performanceCounterQueryPools |= _rhs.performanceCounterQueryPools;
        res.performanceCounterMultipleQueryPools |= _rhs.performanceCounterMultipleQueryPools;

        res.mixedAttachmentSamples |= _rhs.mixedAttachmentSamples;


        res.accelerationStructure |= _rhs.accelerationStructure;
        res.accelerationStructureIndirectBuild |= _rhs.accelerationStructureIndirectBuild;
        res.accelerationStructureHostCommands |= _rhs.accelerationStructureHostCommands;

        res.rayTracingPipeline |= _rhs.rayTracingPipeline;

        res.rayTraversalPrimitiveCulling |= _rhs.rayTraversalPrimitiveCulling;

        res.rayQuery |= _rhs.rayQuery;

        res.representativeFragmentTest |= _rhs.representativeFragmentTest;

        res.bufferMarkerAMD |= _rhs.bufferMarkerAMD;

        res.fragmentDensityMap |= _rhs.fragmentDensityMap;
        res.fragmentDensityMapDynamic |= _rhs.fragmentDensityMapDynamic;
        res.fragmentDensityMapNonSubsampledImages |= _rhs.fragmentDensityMapNonSubsampledImages;

        res.deviceCoherentMemory |= _rhs.deviceCoherentMemory;

        res.memoryPriority |= _rhs.memoryPriority;

        res.fragmentShaderSampleInterlock |= _rhs.fragmentShaderSampleInterlock;
        res.fragmentShaderPixelInterlock |= _rhs.fragmentShaderPixelInterlock;
        res.fragmentShaderShadingRateInterlock |= _rhs.fragmentShaderShadingRateInterlock;

        res.rectangularLines |= _rhs.rectangularLines;
        res.bresenhamLines |= _rhs.bresenhamLines;
        res.smoothLines |= _rhs.smoothLines;
        res.stippledRectangularLines |= _rhs.stippledRectangularLines;
        res.stippledBresenhamLines |= _rhs.stippledBresenhamLines;
        res.stippledSmoothLines |= _rhs.stippledSmoothLines;

        res.indexTypeUint8 |= _rhs.indexTypeUint8;

        res.deferredHostOperations |= _rhs.deferredHostOperations;

        res.pipelineExecutableInfo |= _rhs.pipelineExecutableInfo;

        res.deviceGeneratedCommands |= _rhs.deviceGeneratedCommands;

        res.rayTracingMotionBlur |= _rhs.rayTracingMotionBlur;
        res.rayTracingMotionBlurPipelineTraceRaysIndirect |= _rhs.rayTracingMotionBlurPipelineTraceRaysIndirect;

        res.fragmentDensityMapDeferred |= _rhs.fragmentDensityMapDeferred;

        res.rasterizationOrderColorAttachmentAccess |= _rhs.rasterizationOrderColorAttachmentAccess;
        res.rasterizationOrderDepthAttachmentAccess |= _rhs.rasterizationOrderDepthAttachmentAccess;
        res.rasterizationOrderStencilAttachmentAccess |= _rhs.rasterizationOrderStencilAttachmentAccess;

        res.cooperativeMatrixRobustBufferAccess |= _rhs.cooperativeMatrixRobustBufferAccess;

        return res;
    }

    inline SPhysicalDeviceFeatures intersectWith(const SPhysicalDeviceFeatures& _rhs) const
    {
        SPhysicalDeviceFeatures res = *this;

        // VK 1.0 core
        res.robustBufferAccess &= _rhs.robustBufferAccess;

        res.geometryShader &= _rhs.geometryShader;
        res.tessellationShader &= _rhs.tessellationShader;

        res.depthBounds &= _rhs.depthBounds;
        res.wideLines &= _rhs.wideLines;
        res.largePoints &= _rhs.largePoints;

        res.alphaToOne &= _rhs.alphaToOne;

        res.pipelineStatisticsQuery &= _rhs.pipelineStatisticsQuery;

        res.shaderCullDistance &= _rhs.shaderCullDistance;

        res.shaderResourceResidency &= _rhs.shaderResourceResidency;
        res.shaderResourceMinLod &= _rhs.shaderResourceMinLod;

        // Vk 1.1 everything is either a Limit or Required

        // Vk 1.2
        res.bufferDeviceAddressMultiDevice &= _rhs.bufferDeviceAddressMultiDevice;

        // Vk 1.3
        res.robustImageAccess &= _rhs.robustImageAccess;

        // Nabla Core Extensions
        res.robustBufferAccess2 &= _rhs.robustBufferAccess2;
        res.robustImageAccess2 &= _rhs.robustImageAccess2;

        res.nullDescriptor &= _rhs.nullDescriptor;

        // Extensions
        res.swapchainMode &= _rhs.swapchainMode;

        res.shaderInfoAMD &= _rhs.shaderInfoAMD;

        res.conditionalRendering &= _rhs.conditionalRendering;
        res.inheritedConditionalRendering &= _rhs.inheritedConditionalRendering;

        res.geometryShaderPassthrough &= _rhs.geometryShaderPassthrough;

        res.hdrMetadata &= _rhs.hdrMetadata;

        res.performanceCounterQueryPools &= _rhs.performanceCounterQueryPools;
        res.performanceCounterMultipleQueryPools &= _rhs.performanceCounterMultipleQueryPools;

        res.mixedAttachmentSamples &= _rhs.mixedAttachmentSamples;


        res.accelerationStructure &= _rhs.accelerationStructure;
        res.accelerationStructureIndirectBuild &= _rhs.accelerationStructureIndirectBuild;
        res.accelerationStructureHostCommands &= _rhs.accelerationStructureHostCommands;

        res.rayTracingPipeline &= _rhs.rayTracingPipeline;

        res.rayTraversalPrimitiveCulling &= _rhs.rayTraversalPrimitiveCulling;

        res.rayQuery &= _rhs.rayQuery;

        res.representativeFragmentTest &= _rhs.representativeFragmentTest;

        res.bufferMarkerAMD &= _rhs.bufferMarkerAMD;

        res.fragmentDensityMap &= _rhs.fragmentDensityMap;
        res.fragmentDensityMapDynamic &= _rhs.fragmentDensityMapDynamic;
        res.fragmentDensityMapNonSubsampledImages &= _rhs.fragmentDensityMapNonSubsampledImages;

        res.deviceCoherentMemory &= _rhs.deviceCoherentMemory;

        res.memoryPriority &= _rhs.memoryPriority;

        res.fragmentShaderSampleInterlock &= _rhs.fragmentShaderSampleInterlock;
        res.fragmentShaderPixelInterlock &= _rhs.fragmentShaderPixelInterlock;
        res.fragmentShaderShadingRateInterlock &= _rhs.fragmentShaderShadingRateInterlock;

        res.rectangularLines &= _rhs.rectangularLines;
        res.bresenhamLines &= _rhs.bresenhamLines;
        res.smoothLines &= _rhs.smoothLines;
        res.stippledRectangularLines &= _rhs.stippledRectangularLines;
        res.stippledBresenhamLines &= _rhs.stippledBresenhamLines;
        res.stippledSmoothLines &= _rhs.stippledSmoothLines;

        res.indexTypeUint8 &= _rhs.indexTypeUint8;

        res.deferredHostOperations &= _rhs.deferredHostOperations;

        res.pipelineExecutableInfo &= _rhs.pipelineExecutableInfo;

        res.deviceGeneratedCommands &= _rhs.deviceGeneratedCommands;

        res.rayTracingMotionBlur &= _rhs.rayTracingMotionBlur;
        res.rayTracingMotionBlurPipelineTraceRaysIndirect &= _rhs.rayTracingMotionBlurPipelineTraceRaysIndirect;

        res.fragmentDensityMapDeferred &= _rhs.fragmentDensityMapDeferred;

        res.rasterizationOrderColorAttachmentAccess &= _rhs.rasterizationOrderColorAttachmentAccess;
        res.rasterizationOrderDepthAttachmentAccess &= _rhs.rasterizationOrderDepthAttachmentAccess;
        res.rasterizationOrderStencilAttachmentAccess &= _rhs.rasterizationOrderStencilAttachmentAccess;

        res.cooperativeMatrixRobustBufferAccess &= _rhs.cooperativeMatrixRobustBufferAccess;

        return res;
    }
};

template<typename T>
concept DeviceFeatureDependantClass = requires(const SPhysicalDeviceFeatures& availableFeatures, SPhysicalDeviceFeatures& features) { 
    T::enableRequiredFeautres(features);
    T::enablePreferredFeatures(availableFeatures, features);
};

} // nbl::video
#endif