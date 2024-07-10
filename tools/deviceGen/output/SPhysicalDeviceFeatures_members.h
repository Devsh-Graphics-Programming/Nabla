    // Vulkan 1.0 Core

    // widely supported but has performance overhead, so remains an optional feature to enable
    bool robustBufferAccess = false;

    // [REQUIRE]Roadmap 2022 requires support for these, device support is ubiquitous and enablement is unlikely to harm performance
    // bool fullDrawIndexUint32 = true;
    // bool imageCubeArray = true;
    // bool independentBlend = true;

    // I have no clue if these cause overheads from simply being enabled
    bool geometryShader = false;
    bool tessellationShader = false;

    // [REQUIRE]
    // ROADMAP 2022 and good device support
    // bool sampleRateShading = true;

    // [REQUIRE]
    // good device support
    // bool dualSrcBlend = true;

    // [EXPOSE AS A LIMIT] somewhat legacy features
    // mostly just desktops support this
    // bool logicOp = false;

    // [REQUIRE]Roadmap 2022 requires support for these, device support is ubiquitous and enablement is unlikely to harm performance
    // bool multiDrawIndirect = true;
    // bool drawIndirectFirstInstance = true;
    // bool depthClamp = true;
    // bool depthBiasClamp = true;

    // [REQUIRE]
    // good device support
    // bool fillModeNonSolid = true;

    // this is kinda like a weird clip-plane that doesn't count towards clip plane count
    bool depthBounds = false;

    // good device support, but a somewhat useless feature (constant screenspace width with limits on width)
    bool wideLines = false;
    // good device support, but a somewhat useless feature (axis aligned screenspace boxes with limits on size)
    bool largePoints = false;

    // Some AMD don't support
    bool alphaToOne = true;

    // [REQUIRE]
    // good device support
    // bool multiViewport = true;

    // [REQUIRE]
    // Roadmap 2022 requires support for these, device support is ubiquitous
    // bool samplerAnisotropy = true;

    // [DO NOT EXPOSE] these 3 don't make a difference, just shortcut from Querying support from PhysicalDevice
    // bool textureCompressionETC2;
    // bool textureCompressionASTC_LDR;
    // bool textureCompressionBC;

    // [REQUIRE]
    // ROADMAP 2022 and good device support
    // bool occlusionQueryPrecise = true;

    bool pipelineStatisticsQuery = false;

    // [EXPOSE AS LIMIT] Enabled by Default, Moved to Limits
    // All iOS GPUs don't support
    // bool vertexPipelineStoresAndAtomics = false;
    // ROADMAP 2022 no supporton iOS GPUs
    // bool fragmentStoresAndAtomics = false;
    // Candidate for promotion, just need to look into Linux and Android
    // bool shaderTessellationAndGeometryPointSize = false;

    // [REQUIRE]
    // ROADMAP 2024 good device support
    // bool shaderImageGatherExtended = true;

    // [REQUIRE]
    // ROADMAP 2024 good device support
    // bool shaderStorageImageExtendedFormats = true;

    // [EXPOSE AS LIMIT] Cannot be always enabled cause Intel ARC is handicapped
    // Apple GPUs and some Intels don't support
    // bool shaderStorageImageMultisample = false;

    // TODO: format feature reporting unimplemented yet for both of the below! (should we move to usage reporting?)
    // [EXPOSE AS LIMIT] always enable, shouldn't cause overhead by just being enabled
    // bool shaderStorageImageReadWithoutFormat = false;

    // [REQUIRE]
    // good device support
    // bool shaderStorageImageWriteWithoutFormat = true;

    // [REQUIRE]ROADMAP 2022 and good device support
    // bool shaderUniformBufferArrayDynamicIndexing = true;
    // bool shaderSampledImageArrayDynamicIndexing = true;
    // bool shaderStorageBufferArrayDynamicIndexing = true;

    // [EXPOSE AS A LIMIT] ROADMAP 2022 but Apple GPUs have poor support
    // ROADMAP 2022 but no iOS GPU supports
    // bool shaderStorageImageArrayDynamicIndexing = false;

    // [REQUIRE]
    // good device support
    // bool shaderClipDistance = true;

    bool shaderCullDistance = false;

    // [EXPOSE AS A LIMIT] Cannot be always enabled cause Intel ARC is handicapped
    // Intel Gen12 and ARC are special-boy drivers (TM)
    // bool shaderFloat64 = false;

    // 
    // bool shaderInt64 = true;
    // ROADMAP 2024
    // bool shaderInt16 = true;

    // poor support on Apple GPUs
    bool shaderResourceResidency = false;
    bool shaderResourceMinLod = false;

    // [TODO LATER] once we implemented sparse resources
    // bool sparseBinding;
    // bool sparseResidencyBuffer;
    // bool sparseResidencyImage2D;
    // bool sparseResidencyImage3D;
    // bool sparseResidency2Samples;
    // bool sparseResidency4Samples;
    // bool sparseResidency8Samples;
    // bool sparseResidency16Samples;
    // bool sparseResidencyAliased;

    // [EXPOSE AS LIMIT] poor support on Apple GPUs
    // bool variableMultisampleRate = false;

    // [REQUIRE]
    // Always enabled, good device support.
    // bool inheritedQueries = true;

    // Vulkan 1.1 Core

    // ROADMAP 2024
    // bool storageBuffer16BitAccess = true;
    // Force Enabled : ALIAS VK_KHR_16bit_storage
    // bool uniformAndStorageBuffer16BitAccess = true;

    // Core 1.1 Features or VK_KHR_16bit_storage
    // [EXPOSE AS A LIMIT] Enabled by Default, Moved to Limits : ALIAS VK_KHR_16bit_storage
    // bool storagePushConstant16 = false;
    // bool storageInputOutput16 = false;

    // [REQUIRE]
    // Required to be present when Vulkan 1.1 is supported
    // bool multiview = true;

    // Core 1.1 Features or VK_KHR_multiview, normally would be required but MoltenVK mismatches these
    // [EXPOSE AS A LIMIT] VK_KHR_multiview required but these depend on pipelines and MoltenVK mismatches these
    // bool multiviewGeometryShader = false;
    // bool multiviewTessellationShader = false;

    // [REQUIRE]Will eventually be required by HLSL202x if it implements references or pointers (even the non-generic type)
    // bool variablePointers = true;
    // bool variablePointersStorageBuffer = true;

    // [DO NOT EXPOSE] not gonna expose until we have a need to
    // or via VkPhysicalDeviceProtectedMemoryProperties provided by Vulkan 1.1
    // bool protectedMemory = false;

    // [DO NOT EXPOSE] ROADMAP 2022 Enables certain formats in Vulkan
    // we just enable them if available or else we need to make format support query functions in LogicalDevice as well
    // bool samplerYcbcrConversion = false;

    // [REQUIRE]
    // ROADMAP2024 and Force Enabled : VK_KHR_shader_draw_parameters
    // bool shaderDrawParameters = true;

    // Vulkan 1.2 Core

    // [REQUIRE]
    // ROADMAP 2022 and device support ubiquitous
    // ALIAS: VK_KHR_sampler_mirror_clamp_to_edge
    // bool samplerMirrorClampToEdge = true;

    // [EXPOSE AS A LIMIT] ROADMAP 2022 requires support but MoltenVK doesn't support
    // exposed as a limit `drawIndirectCount`
    // ALIAS: VK_KHR_draw_indirect_count
    // Vulkan 1.2 Core or VK_KHR_draw_indirect_count
    // bool drawIndirectCount = false;

    // [REQUIRE]or VK_KHR_8bit_storage:
    // ROADMAP 2022 and device support ubiquitous
    // bool storageBuffer8BitAccess = true;
    // good device coverage
    // bool uniformAndStorageBuffer8BitAccess = true;
    // [EXPOSE AS LIMIT] not great support yet
    // Vulkan 1.2 Core or VK_KHR_9bit_storage
    // bool storagePushConstant8 = false;

    // Vulkan 1.2 Core or VK_KHR_shader_atomic_int64
    // [EXPOSE AS LIMIT] or VK_KHR_shader_atomic_int64
    // bool shaderBufferInt64Atomics = false;
    // bool shaderSharedInt64Atomics = false;

    // [REQUIRE]or VK_KHR_shader_float16_int8:
    // [EXPOSE AS LIMIT] ROADMAP 2024 but not great support yet
    // Vulkan 1.2 Core or VK_KHR_shader_float16_int8
    // bool shaderFloat16 = false;
    // ROADMAP 2024 good device coverage
    // bool shaderInt8 = true;

    // ROADMAP 2022
    // bool descriptorIndexing = true;
    // [EXPOSE AS A LIMIT] This is for a SPIR-V capability, the overhead should only be incurred if the pipeline uses this capability
    // Vulkan 1.2 Core or VK_EXT_descriptor_indexing
    // bool shaderInputAttachmentArrayDynamicIndexing = false;
    // because we also require `descriptorIndexing`
    // bool shaderUniformTexelBufferArrayDynamicIndexing = true;
    // bool shaderStorageTexelBufferArrayDynamicIndexing = true;
    // [EXPOSE AS A LIMIT] ROADMAP 2022 mandates but poor device support
    // bool shaderUniformBufferArrayNonUniformIndexing = false;
    // because we also require `descriptorIndexing`
    // bool shaderSampledImageArrayNonUniformIndexing = true;
    // bool shaderStorageBufferArrayNonUniformIndexing = true;
    // ROADMAP 2022
    // bool shaderStorageImageArrayNonUniformIndexing = true;
    // [EXPOSE AS A LIMIT] This is for a SPIR-V capability, the overhead should only be incurred if the pipeline uses this capability
    // bool shaderInputAttachmentArrayNonUniformIndexing = false;
    // because we also require `descriptorIndexing`
    // bool shaderUniformTexelBufferArrayNonUniformIndexing = true;
    // ROADMAP 2022 and good device support
    // bool shaderStorageTexelBufferArrayNonUniformIndexing = true;
    // We have special bits on the Descriptor Layout Bindings and those should decide the overhead, not the enablement of a feature like the following
    // [EXPOSE AS A LIMIT] not great coverage but still can enable when available
    // bool descriptorBindingUniformBufferUpdateAfterBind = false;
    // because we also require `descriptorIndexing`
    // bool descriptorBindingSampledImageUpdateAfterBind = true;
    // bool descriptorBindingStorageImageUpdateAfterBind = true;
    // bool descriptorBindingStorageBufferUpdateAfterBind = true;
    // bool descriptorBindingUniformTexelBufferUpdateAfterBind = true;
    // bool descriptorBindingStorageTexelBufferUpdateAfterBind = true;
    // bool descriptorBindingUpdateUnusedWhilePending = true;
    // bool descriptorBindingPartiallyBound = true;
    // ROADMAP 2022 and good device support
    // bool descriptorBindingVariableDescriptorCount = true;
    // bool runtimeDescriptorArray = true;

    // [EXPOSE AS A LIMIT]
    // ALIAS: VK_EXT_sampler_filter_minmax
    // Vulkan 1.2 or VK_EXT_sampler_filter_minmax
    // TODO: Actually implement the sampler flag enums
    // bool samplerFilterMinmax = false;

    // [REQUIRE]
    // Roadmap 2022 requires support for these we always enable and they're unlikely to harm performance
    // or VK_EXT_scalar_block_layout
    // bool scalarBlockLayout = true;

    // [DO NOT EXPOSE] Decided against exposing, API is braindead, for details see: https://github.com/Devsh-Graphics-Programming/Nabla/issues/378
    // or VK_KHR_imageless_framebuffer
    // bool imagelessFramebuffer = false;

    // [REQUIRE]Vulkan 1.2 requires these
    // or VK_KHR_uniform_buffer_standard_layout
    // bool uniformBufferStandardLayout = true;
    // or VK_KHR_shader_subgroup_extended_types
    // bool shaderSubgroupExtendedTypes = true;
    // or VK_KHR_separate_depth_stencil_layouts
    // bool separateDepthStencilLayouts = true;
    // or VK_EXT_host_query_reset [TODO Implement]
    // bool hostQueryReset = true;
    // or VK_KHR_timeline_semaphore [TODO Implement]
    // bool timelineSemaphore = true;

    // or VK_KHR_buffer_device_address:
    // Vulkan 1.3 requires
    // bool bufferDeviceAddress = true;
    // [DO NOT EXPOSE] for capture tools not engines
    // bool bufferDeviceAddressCaptureReplay;
    bool bufferDeviceAddressMultiDevice = false;

    // [REQUIRE]ROADMAP2022 wants them. ALIAS VK_KHR_vulkan_memory_model
    // bool vulkanMemoryModel = true;
    // bool vulkanMemoryModelDeviceScope = true;

    // [EXPOSE AS A LIMIT] ROADMAP2022 wants them, but device support low
    // Vulkan 1.3 requires but we make concessions for MoltenVK
    // bool vulkanMemoryModelAvailabilityVisibilityChains = false;

    // Vulkan 1.2 Core or VK_EXT_shader_viewport_index_layer
    // [EXPOSE AS A LIMIT]
    // ALIAS: VK_EXT_shader_viewport_index_layer
    // bool shaderOutputViewportIndex = false;
    // ALIAS: VK_EXT_shader_viewport_index_layer
    // bool shaderOutputLayer = false;

    // [REQUIRE]
    // ubiquitous device support
    // bool subgroupBroadcastDynamicId = true;

    // Vulkan 1.3 Core

    // This feature adds stricter requirements for how out of bounds reads from images are handled.
    // Rather than returning undefined values,
    // most out of bounds reads return R, G, and B values of zero and alpha values of either zero or one.
    // Components not present in the image format may be set to zero
    // or to values based on the format as described in Conversion to RGBA in vulkan specification.
    // widely supported but has performance overhead, so remains an optional feature to enable
    // or VK_EXT_image_robustness
    bool robustImageAccess = false;

    // [DO NOT EXPOSE] VK_EXT_inline_uniform_block EVIL regressive step back into OpenGL/Dx10 times? Or an intermediate step between PC and UBO?
    // Vulkan 1.3, Nabla Core Profile:
    // bool inlineUniformBlock = false;
    // ROADMAP 2022, Nabla Core Profile:
    // bool descriptorBindingInlineUniformBlockUpdateAfterBind = false;

    // [REQUIRE]
    // Vulkan 1.3 non-optional and Nabla Core Profile but TODO: need impl
    // or VK_EXT_pipeline_creation_cache_control
    // bool pipelineCreationCacheControl = true;

    // [DO NOT EXPOSE] ever we have our own mechanism, unless we can somehow get the data out of `VkObject`?
    // or VK_EXT_private_data
    // bool privateData = false;

    // Vulkan 1.3 non-optional requires but poor support
    // [EXPOSE AS LIMIT] Vulkan 1.3 non-optional requires but poor support
    // or VK_EXT_shader_demote_to_helper_invocation
    // bool shaderDemoteToHelperInvocation = false;
    // or VK_KHR_shader_terminate_invocation
    // bool shaderTerminateInvocation = false;

    // [REQUIRE]Nabla Core Profile, Vulkan 1.3 or VK_EXT_subgroup_size_control
    // bool subgroupSizeControl = true;
    // bool computeFullSubgroups = true;

    // [REQUIRE]
    // REQUIRE
    // or VK_KHR_synchronization2
    // bool synchronization2 = true;

    // [DO NOT EXPOSE] Doesn't make a difference, just shortcut from Querying support from PhysicalDevice
    // or VK_EXT_texture_compression_astc_hdr
    // bool textureCompressionASTC_HDR;

    // [EXPOSE AS LIMIT] Vulkan 1.3 non-optional requires but poor support
    // or VK_KHR_zero_initialize_workgroup_memory
    // bool shaderZeroInitializeWorkgroupMemory = false;

    // [DO NOT EXPOSE] EVIL
    // or VK_KHR_dynamic_rendering
    // bool dynamicRendering = false;

    // [REQUIRE]
    // Vulkan 1.3 non-optional requires, you probably want to look at the individual limits anyway
    // or VK_KHR_shader_integer_dot_product
    // bool shaderIntegerDotProduct = true;

    // Nabla Core Profile

    // [TODO] Better descriptive name for the Vulkan robustBufferAccess2, robustImageAccess2 features
    bool robustBufferAccess2 = false;
    // Nabla Core Profile but still a feature because enabling has overhead
    bool robustImageAccess2 = false;
    // ! nullDescriptor: you can use `nullptr` for writing descriptors to sets and Accesses to null descriptors have well-defined behavior.
    // [TODO] Handle `nullDescriptor` feature in the engine.
    bool nullDescriptor = false;

    // Vulkan Extensions

    // VK_KHR_surface
    // [DO NOT EXPOSE] Instance extension & should enable implicitly if swapchain is enabled

    // VK_KHR_swapchain
    // Dependant on `IAPIConnection::SFeatures::swapchainMode` enabled on apiConnection Creation
    core::bitflag<E_SWAPCHAIN_MODE> swapchainMode = E_SWAPCHAIN_MODE::ESM_NONE;

    // VK_KHR_display
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_display_swapchain
    // [TODO] handle with a single num

    // VK_KHR_win32_surface
    // VK_KHR_android_surface
    // VK_KHR_mir_surface
    // VK_KHR_wayland_surface
    // VK_KHR_xcb_surface
    // VK_KHR_xlib_surface
    // [DO NOT EXPOSE] OS-specific INSTANCE extensions we enable implicitly as we detect the platform

    // VK_ANDROID_native_buffer
    // [DO NOT EXPOSE] supported=disabled

    // VK_EXT_debug_report
    // [DEPRECATED] by VK_EXT_debug_utils

    // VK_NV_glsl_shader
    // [DO NOT EXPOSE] EVER

    // VK_EXT_depth_range_unrestricted
    // [TODO LATER] Will expose some day

    // VK_KHR_sampler_mirror_clamp_to_edge
    // [DEPRECATED] deprecated by Vulkan 1.2

    // VK_IMG_filter_cubic
    // [DO NOT EXPOSE] Vendor specific, superceeded by VK_EXT_filter_cubic, won't expose for a long time

    // VK_AMD_extension_18
    // VK_AMD_extension_17
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_AMD_rasterization_order
    // [DO NOT EXPOSE] Meme extension

    // VK_AMD_extension_20
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_AMD_shader_trinary_minmax
    // [EXPOSE AS LIMIT] Enabled by Default, Moved to Limits

    // VK_AMD_shader_explicit_vertex_parameter
    // [EXPOSE AS LIMIT]

    // VK_EXT_debug_marker
    // [DEPRECATED] Promoted to VK_EXT_debug_utils (instance ext)

    // VK_KHR_video_queue
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_video_decode_queue
    // [TODO] Provisional

    // VK_AMD_gcn_shader
    // [DO NOT EXPOSE] Core features, KHR, EXT and smaller AMD features supersede everything in here

    // VK_NV_dedicated_allocation
    // [DEPRECATED] Promoted to KHR_dedicated_allocation, non-optional core VK 1.1

    // VK_EXT_extension_28
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // TransformFeedbackFeaturesEXT ** VK_EXT_transform_feedback
    // [DO NOT EXPOSE] ever because of our disdain for XForm feedback

    // VK_NVX_binary_import
    // [DO NOT EXPOSE] We don't support yet

    // VK_NVX_image_view_handle
    // [DO NOT EXPOSE] We don't support yet

    // VK_AMD_extension_33
    // VK_AMD_extension_32
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_AMD_draw_indirect_count
    // [DEPRECATED] Vulkan core 1.2

    // VK_AMD_extension_35
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_AMD_negative_viewport_height
    // [DEPRECATED] Promoted to VK_KHR_maintenance1, non-optional core VK 1.1

    // VK_AMD_gpu_shader_half_float
    // [EXPOSE AS LIMIT] The only reason we still keep it around is because it provides FP16 trig and special functions

    // VK_AMD_shader_ballot
    // [DEPRECATED] Superseded by KHR_shader_subgroup_ballot

    // VK_EXT_video_encode_h264
    // [DO NOT EXPOSE] Don't want to be on the hook for the MPEG-LA useless Patent Trolls

    // VK_EXT_video_encode_h265
    // [DO NOT EXPOSE] Don't want to be on the hook for the MPEG-LA useless Patent Trolls

    // VK_KHR_video_decode_h264
    // [DO NOT EXPOSE] Don't want to be on the hook for the MPEG-LA useless Patent Trolls

    // VK_AMD_texture_gather_bias_lod
    // [TODO LATER] Won't expose for now, API changes necessary

    // [TODO] need impl, also expose as a limit?
    // VK_AMD_shader_info
    bool shaderInfoAMD = false;

    // VK_AMD_extension_44
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_KHR_dynamic_rendering
    // [DO NOT EXPOSE] We don't support yet

    // VK_AMD_extension_46
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_AMD_shader_image_load_store_lod
    // [EXPOSE AS LIMIT]

    // VK_GOOGLE_extension_49
    // VK_NVX_extension_48
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_GGP_stream_descriptor_surface
    // [DO NOT EXPOSE] This used to be for Stadia, Stadia is dead now

    // VK_NV_corner_sampled_image
    // CornerSampledImageFeaturesNV
    // [DO NOT EXPOSE] for a very long time

    // VK_NV_private_vendor_info
    // [DO NOT EXPOSE] We don't support yet

    // VK_NV_extension_53
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_KHR_multiview
    // [DO NOT EXPOSE] We don't support yet

    // VK_IMG_format_pvrtc
    // [DO NOT EXPOSE] Enables certain formats in Vulkan, we just enable them if available or else we need to make format support query functions in LogicalDevice as well

    // VK_NV_external_memory_capabilities
    // [DEPRECATED] by VK_KHR_external_memory_capabilities

    // VK_NV_external_memory
    // [DEPRECATED] by VK_KHR_external_memory

    // VK_NV_external_memory_win32
    // [DEPRECATED] Promoted to VK_KHR_external_memory_win32 

    // VK_NV_win32_keyed_mutex
    // [DEPRECATED] Promoted to VK_KHR_win32_keyed_mutex 

    // VK_KHR_get_physical_device_properties2
    // [DEPRECATED] Promoted to non-optional core VK 1.1

    // VK_KHR_device_group
    // [DEPRECATED] Promoted to non-optional core VK 1.1

    // VK_EXT_validation_flags
    // [DEPRECATED] by VK_EXT_validation_features

    // VK_NN_vi_surface
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_shader_draw_parameters
    // [DEPRECATED] Vulkan 1.1 Core

    // VK_EXT_shader_subgroup_ballot
    // [DEPRECATED] by VK_VERSION_1_2

    // VK_EXT_shader_subgroup_vote
    // [DEPRECATED] by VK_VERSION_1_1

    // VK_EXT_texture_compression_astc_hdr
    // [DO NOT EXPOSE] We don't support yet

    // [TODO]
    // ASTCDecodeFeaturesEXT
    // VK_EXT_astc_decode_mode
    // VkFormat decodeMode;

    // VK_EXT_pipeline_robustness
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_maintenance1
    // [DEPRECATED] Promoted to non-optional core

    // VK_KHR_device_group_creation
    // [DEPRECATED] Promoted to non-optional core VK 1.1

    // VK_KHR_external_memory_capabilities
    // [DEPRECATED] Promoted to non-optional core Vk 1.1

    // VK_KHR_external_memory
    // [DEPRECATED] Promoted to non-optional core VK 1.1

    // VK_KHR_external_memory_win32
    // [DEPRECATED] Always Enabled but requires Instance Extensions during API Connection Creation!

    // VK_KHR_external_memory_fd
    // [DEPRECATED] Always Enabled but requires Instance Extensions during API Connection Creation!

    // VK_KHR_win32_keyed_mutex
    // [DO NOT EXPOSE] Always enabled, used for dx11 interop

    // VK_KHR_external_semaphore
    // VK_KHR_external_semaphore_capabilities
    // [DEPRECATED] Promoted to non-optional core VK 1.1

    // VK_KHR_external_semaphore_win32
    // [DEPRECATED] Always Enabled but requires Instance Extensions during API Connection Creation!

    // VK_KHR_external_semaphore_fd
    // [DEPRECATED] Always Enabled but requires Instance Extensions during API Connection Creation!

    // VK_KHR_push_descriptor
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_conditional_rendering
    // ConditionalRenderingFeaturesEXT
    // [TODO] now we need API to deal with queries and begin/end conditional blocks
    bool conditionalRendering = false;
    bool inheritedConditionalRendering = false;

    // VK_KHR_shader_float16_int8
    // [DEPRECATED] Vulkan 1.2 Core

    // VK_KHR_16bit_storage
    // [DEPRECATED] Vulkan 1.2 Core and Required

    // VK_KHR_incremental_present
    // [DO NOT EXPOSE] this is `swap with damage` known from EGL, cant be arsed to support

    // VK_KHR_descriptor_update_template
    // [TODO] Promoted to VK1.1 non-optional core, haven't updated API to match

    // VK_NVX_device_generated_commands
    // [DEPRECATED] now VK_NV_device_generated_commands

    // VK_NV_clip_space_w_scaling
    // [TODO LATER] Don't expose VR features for now

    // VK_EXT_direct_mode_display
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_acquire_xlib_display
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_display_surface_counter
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_display_control
    // [TODO LATER] Requires handling display swapchain stuff

    // VK_GOOGLE_display_timing
    // [EXPOSE AS LIMIT]

    // VK_RESERVED_do_not_use_94
    // [DO NOT EXPOSE] We don't support yet

    // [TODO] Investigate
    // VK_NV_sample_mask_override_coverage
    // VK_NV_geometry_shader_passthrough
    // The only reason its not a limit is because it needs geometryShader to be enabled
    bool geometryShaderPassthrough = false;

    // VK_NVX_multiview_per_view_attributes
    // [DO NOT EXPOSE] We don't support yet

    // VK_NV_viewport_swizzle
    // [DO NOT EXPOSE] A silly Nvidia extension thats specific to singlepass cubemap rendering and voxelization with geometry shader

    // VK_EXT_discard_rectangles
    // [EXPOSE AS A LIMIT]

    // VK_NV_extension_101
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_conservative_rasterization
    // [EXPOSE AS A LIMIT]

    // VK_EXT_depth_clip_enable
    // DepthClipEnableFeaturesEXT
    // [DO NOT EXPOSE] only useful for D3D emulators

    // VK_NV_extension_104
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_swapchain_colorspace
    // [DO NOT EXPOSE] We don't support yet

    // [TODO] need impl
    // VK_EXT_hdr_metadata
    bool hdrMetadata = false;

    // VK_IMG_extension_108
    // VK_IMG_extension_107
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_KHR_imageless_framebuffer
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_create_renderpass2
    // [DEPRECATED] Core 1.2 implemented on default path and there's no choice in not using it

    // VK_IMG_extension_111
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_KHR_shared_presentable_image
    // [DO NOT EXPOSE] Leave for later consideration

    // VK_KHR_external_fence_capabilities
    // [DEPRECATED] Promoted to non-optional core VK 1.1

    // VK_KHR_external_fence
    // [DEPRECATED] Promoted to non-optional core VK 1.1

    // VK_KHR_external_fence_win32
    // [DEPRECATED] Always Enabled but requires Instance Extensions during API Connection Creation!

    // VK_EXT_hdr_metadataHR_performance_query
    // PerformanceQueryFeaturesKHR
    // VK_KHR_external_fence_fd
    // [DEPRECATED] Always Enabled but requires Instance Extensions during API Connection Creation!
    bool performanceCounterQueryPools = false;
    bool performanceCounterMultipleQueryPools = false;

    // VK_KHR_maintenance2
    // [DEPRECATED] Core in Vulkan 1.x

    // VK_KHR_extension_119
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_KHR_get_surface_capabilities2
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_variable_pointers
    // [DEPRECATED] Vulkan 1.1 Core

    // VK_KHR_get_display_properties2
    // [DO NOT EXPOSE] We don't support yet

    // VK_MVK_ios_surface
    // [DEPRECATED] by VK_EXT_metal_surface

    // VK_MVK_macos_surface
    // [DEPRECATED] by VK_EXT_metal_surface

    // VK_MVK_moltenvk
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_external_memory_dma_buf
    // [TODO LATER] Requires exposing external memory first

    // VK_EXT_queue_family_foreign
    // [EXPOSE AS A LIMIT] 

    // VK_KHR_dedicated_allocation
    // [DEPRECATED] Vulkan 1.1 core now

    // VK_EXT_debug_utils
    // [DO NOT EXPOSE] We don't support yet

    // VK_ANDROID_external_memory_android_hardware_buffer
    // [TODO LATER] Requires exposing external memory first

    // VK_EXT_sampler_filter_minmax
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_storage_buffer_storage_class
    // [DEPRECATED] Promoted to non-optional core VK 1.1

    // VK_AMD_gpu_shader_int16
    // [DEPRECATED] Just check for `shaderInt16` and related `16BitAccess` limits

    // VK_AMD_extension_134
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_AMDX_shader_enqueue
    // [DO NOT EXPOSE] We don't support yet

    // VK_AMD_extension_136
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // [TODO] need impl
    // VK_AMD_mixed_attachment_samples
    // OR
    // VK_NV_framebuffer_mixed_samples
    bool mixedAttachmentSamples = false;

    // VK_AMD_shader_fragment_mask
    // [TODO LATER] Wait until VK_AMD_shader_fragment_mask & VK_QCOM_render_pass_shader_resolve converge to something useful

    // VK_EXT_inline_uniform_block
    // InlineUniformBlockFeaturesEXT
    // [DEPRECATED] Required wholly by ROADMAP 2022 and Nabla Core Profile

    // VK_EXT_shader_stencil_export
    // Enabled by Default, Moved to Limits
    // VK_AMD_extension_140
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_AMD_extension_143
    // VK_AMD_extension_142
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_sample_locations
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_relaxed_block_layout
    // [DEPRECATED] Promoted to non-optional core VK 1.1

    // VK_RESERVED_do_not_use_146
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_get_memory_requirements2
    // [DEPRECATED] Promoted to core non-optional VK 1.1

    // VK_KHR_image_format_list
    // [DEPRECATED] Vulkan 1.2 core non-optional

    // VK_EXT_blend_operation_advanced
    // BlendOperationAdvancedFeaturesEXT
    // [DO NOT EXPOSE] This is dumb, you can implement whatever blend equation you want with `EXT_fragment_shader_interlock` and EXT_shader_tile_image

    // VK_KHR_acceleration_structure
    // AccelerationStructureFeaturesKHR
    // VK_NV_fragment_coverage_to_color
    // [TODO LATER] Requires API changes
    bool accelerationStructure = false;
    bool accelerationStructureIndirectBuild = false;
    bool accelerationStructureHostCommands = false;

    // [DO NOT EXPOSE] implied by `accelerationStructure`
    // bool descriptorBindingAccelerationStructureUpdateAfterBind = accelerationStructure;

    // VK_KHR_ray_tracing_pipeline
    // RayTracingPipelineFeaturesKHR
    bool rayTracingPipeline = false;
    // [DO NOT EXPOSE] for capture tools
    // bool rayTracingPipelineShaderGroupHandleCaptureReplay;
    // [DO NOT EXPOSE] for capture tools
    // bool rayTracingPipelineShaderGroupHandleCaptureReplayMixed;
    // [DO NOT EXPOSE] Vulkan feature requirements
    // bool rayTracingPipelineTraceRaysIndirect = rayTracingPipeline;
    bool rayTraversalPrimitiveCulling = false;

    // RayQueryFeaturesKHR
    // VK_KHR_ray_query
    bool rayQuery = false;

    // VK_NV_extension_152
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_NV_framebuffer_mixed_samples
    // [ALIASED TO] VK_AMD_mixed_attachment_samples

    // VK_NV_fill_rectangle
    // [DO NOT EXPOSE] For now. For 2D ui

    // VK_EXT_post_depth_coverage
    // Enabled by Default, Moved to Limits
    // VK_NV_shader_sm_builtins
    // ShaderSMBuiltinsFeaturesNV
    // [EXPOSE AS A LIMIT] Enabled by Default, Moved to Limits

    // VK_KHR_sampler_ycbcr_conversion
    // SamplerYcbcrConversionFeaturesKHR
    // [DEPRECATED] Vulkan 1.1 Core and ROADMAP 2022

    // VK_KHR_bind_memory2
    // [DEPRECATED] Core 1.1 implemented on default path and there's no choice in not using it

    // linux""VK_EXT_image_drm_format_modifier
    // intrinsically
    // [DO NOT EXPOSE] Too 

    // VK_EXT_extension_160
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_validation_cache
    // [TODO LATER] Expose when we start to experience slowdowns from validation

    // VK_EXT_descriptor_indexing
    // [DEPRECATED] KHR extension supersedes and then Vulkan 1.2

    // VK_KHR_portability_subset - PROVISINAL/NOTAVAILABLEANYMORE
    // VK_EXT_shader_viewport_index_layer
    // [DO NOT EXPOSE] We don't support yet

    // VK_NV_shading_rate_image
    // ShadingRateImageFeaturesNV
    // [DO NOT EXPOSE] not implementing or exposing VRS in near or far future, also has interactions with fragment density maps

    // [DEPRECATED] Superseded by KHR
    // VK_NV_ray_tracing
    // RepresentativeFragmentTestFeaturesNV
    // VK_NV_representative_fragment_test
    bool representativeFragmentTest = false;

    // VK_NV_extension_168
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_KHR_maintenance3
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_draw_indirect_count
    // [DEPRECATED] Core in Vulkan 1.x

    // VK_EXT_filter_cubic
    // [TODO LATER] limited utility and availability, might expose if feel like wasting time

    // VK_QCOM_render_pass_shader_resolve
    // [TODO LATER] Wait until VK_AMD_shader_fragment_mask & VK_QCOM_render_pass_shader_resolve converge to something useful

    // VK_QCOM_extension_174
    // VK_QCOM_extension_173
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_global_priority
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_shader_subgroup_extended_types
    // [DEPRECATED] Vulkan 1.2 Core non-optional

    // VK_EXT_extension_177
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_KHR_8bit_storage
    // [DEPRECATED] Promoted to VK Core

    // VK_EXT_external_memory_host
    // [DO NOT EXPOSE] TODO: support in the CUDA PR

    // [TODO] need impl/more research
    // VK_AMD_buffer_marker
    bool bufferMarkerAMD = false;

    // VK_KHR_shader_atomic_int64
    // [DEPRECATED] Vulkan 1.2 Core

    // VK_KHR_shader_clock
    // ShaderClockFeaturesKHR
    // [EXPOSE AS LIMIT]
    // bool shaderDeviceClock = false;
    // bool shaderSubgroupClock = false;

    // VK_AMD_extension_183
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_AMD_pipeline_compiler_control
    // [DO NOT EXPOSE] Too vendor specific

    // VK_EXT_calibrated_timestamps
    // [TODO LATER] Requires changes to API

    // VK_AMD_shader_core_properties
    // [DEPRECATED] Superseded by VK_AMD_shader_core_properties

    // VK_AMD_extension_187
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_KHR_video_decode_h265
    // [DO NOT EXPOSE] We don't want to be on the hook with the MPEG-LA patent trolls

    // VK_KHR_global_priority
    // GlobalPriorityQueryFeaturesKHR
    // [TODO] this one isn't in the headers yet

    // VK_AMD_memory_overallocation_behavior
    // [DO NOT EXPOSE]

    // VK_EXT_vertex_attribute_divisor
    // VertexAttributeDivisorFeaturesEXT
    // [DO NOT EXPOSE] we would have to change the API

    // VK_GGP_frame_token
    // [DEPRECATED] Stadia is dead

    // VK_EXT_pipeline_creation_feedback
    // [TODO LATER] would like to expose, but too much API to change

    // VK_GOOGLE_extension_196
    // VK_GOOGLE_extension_195
    // VK_GOOGLE_extension_194
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_KHR_driver_properties
    // [DEPRECATED] Promoted to VK Core 1.x

    // VK_KHR_shader_float_controls
    // [DEPRECATED] Promoted to VK Core 1.x

    // VK_NV_shader_subgroup_partitioned
    // [DEPRECATED] Superseded by `clustered` subgroup ops

    // VK_KHR_depth_stencil_resolve
    // [DEPRECATED] Promoted to VK Core 1.x

    // VK_KHR_swapchain_mutable_format
    // [DEPRECATED] Vulkan 1.2 core non-optional

    // VK_NV_compute_shader_derivatives
    // ComputeShaderDerivativesFeaturesNV
    // [EXPOSE AS LIMIT]

    // VK_NV_mesh_shader
    // MeshShaderFeaturesNV
    // [DEPRECATED] Expose the KHR extension instead

    // VK_NV_fragment_shader_barycentric
    // FragmentShaderBarycentricFeaturesNV
    // [DO NOT EXPOSE] Deprecated by KHR version

    // [EXPOSE AS LIMIT] Enabled by Default, Moved to Limits
    // ShaderImageFootprintFeaturesNV
    // VK_NV_shader_image_footprint
    // bool imageFootPrint = false;

    // [TODO LATER] requires extra API work to use
    // GL Hint: in GL/GLES this is NV_scissor_exclusive
    // ExclusiveScissorFeaturesNV
    // VK_NV_scissor_exclusive
    // bool exclusiveScissor;

    // VK_NV_device_diagnostic_checkpoints
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_timeline_semaphore
    // [DEPRECATED] Vulkan 1.2 Core non-optional

    // VK_KHR_extension_209
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // [EXPOSE AS LIMIT] Enabled by Default, Moved to Limits 
    // ShaderIntegerFunctions2FeaturesINTEL
    // VK_INTEL_shader_integer_functions2
    // bool shaderIntegerFunctions2 = false;

    // VK_INTEL_performance_query
    // [DEPRECATED] Promoted to VK_KHR_performance_query, VK1.1 core

    // VK_KHR_vulkan_memory_model
    // [DEPRECATED] Vulkan 1.2 Core but 1.3 non-optional and we require it

    // VK_EXT_pci_bus_info
    // [LIMIT] We just report it

    // VK_AMD_display_native_hdr
    // [DO NOT EXPOSE] Waiting for cross platform

    // VK_FUCHSIA_imagepipe_surface
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_shader_terminate_invocation
    // [DEPRECATED] Vulkan 1.3 Core

    // VK_GOOGLE_extension_217
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_metal_surface
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_fragment_density_map
    // FragmentDensityMapFeaturesEXT
    // [TODO] need impl
    bool fragmentDensityMap = false;
    bool fragmentDensityMapDynamic = false;
    bool fragmentDensityMapNonSubsampledImages = false;

    // VK_KHR_extension_221
    // VK_EXT_extension_220
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_scalar_block_layout
    // [DEPRECATED] Vulkan 1.3 core and Nabla core profile required

    // VK_EXT_extension_223
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_GOOGLE_decorate_string
    // Enabled by Default, Moved to Limits
    // VK_GOOGLE_hlsl_functionality1
    // [DO NOT EXPOSE] We compile our own SPIR-V like real men

    // VK_EXT_subgroup_size_control
    // [DO NOT EXPOSE] We don't support yet, but TODO

    // VK_KHR_fragment_shading_rate
    // FragmentShadingRateFeaturesKHR
    // [DO NOT EXPOSE] not implementing or exposing VRS in near or far future

    // VK_AMD_shader_core_properties2
    // [DO NOT EXPOSE] We don't support yet

    // VK_AMD_extension_229
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // [TODO] need impl/more research
    // CoherentMemoryFeaturesAMD
    // VK_AMD_device_coherent_memory
    bool deviceCoherentMemory = false;

    // VK_AMD_extension_234
    // VK_AMD_extension_233
    // VK_AMD_extension_232
    // VK_AMD_extension_231
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_shader_image_atomic_int64
    // [EXPOSE AS A LIMIT]

    // VK_AMD_extension_236
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_KHR_spirv_1_4
    // [DEPRECATED] We now require it with Vulkan 1.2 and its non-optional

    // VK_EXT_memory_budget
    // [DO NOT EXPOSE] We don't support yet

    // [TODO] need impl
    // MemoryPriorityFeaturesEXT
    // VK_EXT_memory_priority
    bool memoryPriority = false;

    // VK_KHR_surface_protected_capabilities
    // [DO NOT EXPOSE] We don't support yet

    // VK_NV_dedicated_allocation_image_aliasing
    // DedicatedAllocationImageAliasingFeaturesNV
    // [DO NOT EXPOSE] insane oxymoron, dedicated means dedicated, not aliased, won't expose

    // SeparateDepthStencilLayoutsFeaturesKHR
    // VK_KHR_separate_depth_stencil_layouts
    // [DEPRECATED] Vulkan 1.2 Core non-optional

    // VK_MESA_extension_244
    // VK_INTEL_extension_243
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_buffer_device_address
    // [DEPRECATED] by VK_KHR_buffer_device_address

    // VK_EXT_tooling_info
    // [DO NOT EXPOSE] we dont need to care or know about it, unless for BDA/RT Replays?

    // VK_EXT_separate_stencil_usage
    // [DEPRECATED] Core 1.2 implemented on default path and there's no choice in not using it

    // VK_EXT_validation_features
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_present_wait
    // PresentWaitFeaturesKHR
    // [DO NOT EXPOSE] won't expose, this extension is poop, I should have a Fence-andQuery-like object to query the presentation timestamp, not a blocking call that may unblock after an arbitrary delay from the present

    // VK_NV_cooperative_matrix
    // CooperativeMatrixFeaturesNV
    // [DEPRECATED] replaced by VK_KHR_cooperative_matrix

    // [TODO] need impl or waaay too vendor specific?
    // CoverageReductionModeFeaturesNV
    // VK_NV_coverage_reduction_mode
    // bool coverageReductionMode = false;

    // VK_EXT_fragment_shader_interlock
    // FragmentShaderInterlockFeaturesEXT
    bool fragmentShaderSampleInterlock = false;
    bool fragmentShaderPixelInterlock = false;
    bool fragmentShaderShadingRateInterlock = false;

    // VK_EXT_ycbcr_image_arrays
    // YcbcrImageArraysFeaturesEXT
    // [DO NOT EXPOSE] Expose nothing to do with video atm

    // VK_KHR_uniform_buffer_standard_layout
    // [DEPRECATED] Vulkan 1.2 Core non-optional

    // VK_EXT_provoking_vertex
    // ProvokingVertexFeaturesEXT
    // [DO NOT EXPOSE] provokingVertexLast will not expose (we always use First Vertex Vulkan-like convention), anything to do with XForm-feedback we don't expose

    // VK_EXT_full_screen_exclusive
    // [TODO LATER] Requires API changes

    // VK_EXT_headless_surface
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_buffer_device_address
    // [DEPRECATED] Core in Vulkan 1.3 and NAbla Core Profile

    // VK_EXT_extension_259
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_line_rasterization
    // LineRasterizationFeaturesEXT
    // [TODO] need impl, this feature introduces new/more pipeline state with VkPipelineRasterizationLineStateCreateInfoEXT
    bool rectangularLines = false;
    bool bresenhamLines = false;
    bool smoothLines = false;
    bool stippledRectangularLines = false;
    bool stippledBresenhamLines = false;
    bool stippledSmoothLines = false;

    // VK_EXT_shader_atomic_float
    // ShaderAtomicFloatFeaturesEXT
    // [NAbla core Profile LIMIT] 

    // VK_EXT_host_query_reset
    // HostQueryResetFeatures
    // [DEPRECATED] MOVED TO Vulkan 1.2 Core

    // VK_BRCM_extension_265
    // VK_BRCM_extension_264
    // VK_GGP_extension_263
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // [TODO] need impl
    // IndexTypeUint8FeaturesEXT
    // VK_EXT_index_type_uint8
    bool indexTypeUint8 = false;

    // VK_EXT_extension_267
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_extended_dynamic_state
    // ExtendedDynamicStateFeaturesEXT
    // [DO NOT EXPOSE] we're fully GPU driven anyway with sorted pipelines.

    // VK_KHR_deferred_host_operations
    bool deferredHostOperations = false;

    // [TODO] need impl
    // PipelineExecutablePropertiesFeaturesKHR
    // VK_KHR_pipeline_executable_properties
    bool pipelineExecutableInfo = false;

    // VK_EXT_host_image_copy
    // [DO NOT EXPOSE] We don't support yet, but should when ubiquitous

    // VK_KHR_map_memory2
    // [DO NOT EXPOSE] We don't support yet

    // VK_INTEL_extension_273
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_shader_atomic_float2
    // ShaderAtomicFloat2FeaturesEXT
    // [EXPOSE AS LIMIT]

    // VK_EXT_surface_maintenance1
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_swapchain_maintenance1
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_shader_demote_to_helper_invocation
    // [DEPRECATED] Core in Vulkan 1.3

    // [TODO] need impl
    // DeviceGeneratedCommandsFeaturesNV
    // VK_NV_device_generated_commands
    bool deviceGeneratedCommands = false;

    // VK_NV_inherited_viewport_scissor
    // InheritedViewportScissorFeaturesNV
    // [DO NOT EXPOSE] won't expose, the existing inheritance of state is enough

    // VK_KHR_extension_280
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_KHR_shader_integer_dot_product
    // [DEPRECATED] Vulkan 1.3 Core non-optional

    // VK_EXT_texel_buffer_alignment
    // TexelBufferAlignmentFeaturesEXT
    // [DEPRECATED] Vulkan 1.3 non-optional and Nabla Core Profile.

    // VK_QCOM_render_pass_transform
    // [DO NOT EXPOSE] Too vendor specific

    // VK_EXT_depth_bias_control
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_device_memory_report
    // DeviceMemoryReportFeaturesEXT
    // [EXPOSE AS LIMIT]

    // VK_EXT_acquire_drm_display
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_robustness2
    // Robustness2FeaturesEXT
    // [Nabla CORE PROFILE]

    // VK_EXT_custom_border_color
    // CustomBorderColorFeaturesEXT
    // [DO NOT EXPOSE] not going to expose custom border colors for now

    // VK_EXT_extension_289
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_GOOGLE_user_type
    // [DO NOT EXPOSE] 0 documentation

    // VK_KHR_pipeline_library
    // [DO NOT EXPOSE] We don't support yet

    // VK_NV_extension_292
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_NV_present_barrier
    // [DO NOT EXPOSE] Triage leftover extensions below    

    // VK_KHR_shader_non_semantic_info
    // [EXPOSE AS A LIMIT] Enabled by Default, Moved to Limits

    // VK_KHR_present_id
    // PresentIdFeaturesKHR
    // [DO NOT EXPOSE] no point exposing until an extension more useful than VK_KHR_present_wait arrives

    // VK_EXT_private_data
    // PrivateDataFeatures
    // [DEPRECATED] Vulkan 1.3 core non-optional

    // VK_KHR_extension_297
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_pipeline_creation_cache_control
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_extension_299
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_KHR_video_encode_queue
    // [TODO] Provisional

    // VK_NV_device_diagnostics_config
    // DiagnosticsConfigFeaturesNV
    // [DO NOT EXPOSE]

    // VK_QCOM_render_pass_store_ops
    // [DO NOT EXPOSE] absorbed into VK_EXT_load_store_op_none

    // VK_NV_extension_308
    // VK_QCOM_extension_307
    // VK_QCOM_extension_306
    // VK_QCOM_extension_305
    // VK_QCOM_extension_304
    // VK_QCOM_extension_303
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_KHR_object_refresh
    // [DO NOT EXPOSE] We don't support yet

    // VK_QCOM_extension_310
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_NV_low_latency
    // [DO NOT EXPOSE] 0 documentation

    // VK_EXT_metal_objects
    // [TODO LATER] Expose when we support MoltenVK

    // VK_AMD_extension_314
    // VK_EXT_extension_313
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_KHR_synchronization2
    // [DEPRECATED] Vulkan 1.3 Core

    // VK_AMD_extension_316
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_descriptor_buffer
    // [DO NOT EXPOSE] We don't support yet

    // VK_AMD_extension_320
    // VK_AMD_extension_319
    // VK_AMD_extension_318
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // [TODO] this one isn't in the headers yet
    // GraphicsPipelineLibraryFeaturesEXT
    // VK_EXT_graphics_pipeline_library
    // bool graphicsPipelineLibrary;

    // [EXPOSE AS A LIMIT]
    // VK_AMD_shader_early_and_late_fragment_tests
    // bool shaderEarlyAndLateFragmentTests = false;

    // VK_KHR_fragment_shader_barycentric
    // [EXPOSE AS A LIMIT] Enabled by Default, Moved to Limits

    // VK_KHR_shader_subgroup_uniform_control_flow
    // ShaderSubgroupUniformControlFlowFeaturesKHR
    // [EXPOSE AS LIMIT]

    // VK_KHR_extension_325
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_KHR_zero_initialize_workgroup_memory
    // [DEPRECATED] Vulkan 1.3 Core

    // VK_NV_ray_tracing_motion_blur
    // RayTracingMotionBlurFeaturesNV
    // VK_NV_fragment_shading_rate_enums
    // [DO NOT EXPOSE] would first need to expose VK_KHR_fragment_shading_rate before
    bool rayTracingMotionBlur = false;
    bool rayTracingMotionBlurPipelineTraceRaysIndirect = false;

    // VK_EXT_mesh_shader
    // [DEPRECATED] By KHR_mesh_shader

    // VK_NV_extension_330
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_ycbcr_2plane_444_formats
    // Ycbcr2Plane444FormatsFeaturesEXT
    // [DO NOT EXPOSE] Enables certain formats in Vulkan, we just enable them if available or else we need to make format support query functions in LogicalDevice as well

    // [DO NOT EXPOSE] just reserved numbers, extension never shipped
    // VK_NV_extension_332
    // FragmentDensityMap2FeaturesEXT
    // VK_EXT_fragment_density_map2
    bool fragmentDensityMapDeferred = false;

    // VK_QCOM_rotated_copy_commands
    // [DO NOT EXPOSE] Too vendor specific

    // VK_KHR_extension_335
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_image_robustness
    // [DEPRECATED] Vulkan 1.3 core non-optional

    // VK_KHR_workgroup_memory_explicit_layout
    // WorkgroupMemoryExplicitLayoutFeaturesKHR
    // [EXPOSE AS LIMIT]

    // VK_KHR_copy_commands2
    // [DO NOT EXPOSE] Promoted to VK 1.3 non-optional core and present in Nabla Core Profile, but serves no purpose other than providing a pNext chain for the usage of a single QCOM extension

    // VK_EXT_image_compression_control
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_attachment_feedback_loop_layout
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_4444_formats
    // 4444FormatsFeaturesEXT
    // [DO NOT EXPOSE] Vulkan 1.3 non-optional, we just enable them if available or else we need to make format support query functions in LogicalDevice as well

    // VK_EXT_device_fault
    // [DO NOT EXPOSE] We don't support yet

    // VK_ARM_rasterization_order_attachment_access
    // RasterizationOrderAttachmentAccessFeaturesARM
    // [TODO] need impl or just expose `shader_tile_image` instead?
    bool rasterizationOrderColorAttachmentAccess = false;
    bool rasterizationOrderDepthAttachmentAccess = false;
    bool rasterizationOrderStencilAttachmentAccess = false;

    // VK_ARM_extension_344
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_rgba10x6_formats
    // [DO NOT EXPOSE] wont expose yet (or ever), requires VK_KHR_sampler_ycbcr_conversion

    // VK_NV_acquire_winrt_display
    // [TODO LATER] won't decide yet, requires VK_EXT_direct_mode_display anyway

    // VK_EXT_directfb_surface
    // [DO NOT EXPOSE] We don't support yet

    // VK_NV_extension_351
    // VK_KHR_extension_350
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_VALVE_mutable_descriptor_type
    // [DO NOT EXPOSE] its a D3D special use extension, shouldn't expose

    // VK_EXT_vertex_input_dynamic_state
    // [DO NOT EXPOSE] too much API Fudgery

    // VK_EXT_physical_device_drm
    // [DO NOT EXPOSE] Too "intrinsically" linux

    // VK_EXT_device_address_binding_report
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_depth_clip_control
    // [DO NOT EXPOSE] EVER, VULKAN DEPTH RANGE ONLY!

    // VK_EXT_primitive_topology_list_restart
    // [DO NOT EXPOSE]

    // VK_EXT_extension_360
    // VK_EXT_extension_359
    // VK_KHR_extension_358
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_KHR_format_feature_flags2Promoted to core 1.3;
    // [DEPRECATED] Promoted to non - optional Core 1.3, we always use it!

    // lots of support for anything that isn't mobile (list of unsupported devices since the extension was published: https://pastebin.com/skZAbL4F)
    // [TODO]

    // VK_FUCHSIA_extension_364
    // VK_EXT_extension_363
    // VK_EXT_extension_362
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_FUCHSIA_external_memory
    // [DO NOT EXPOSE] We don't support yet

    // VK_FUCHSIA_external_semaphore
    // [DO NOT EXPOSE] We don't support yet

    // VK_FUCHSIA_buffer_collection
    // [DO NOT EXPOSE] We don't support yet

    // VK_QCOM_extension_369
    // VK_FUCHSIA_extension_368
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_HUAWEI_subpass_shading
    // SubpassShadingFeaturesHUAWEI
    // [DO NOT EXPOSE]

    // VK_HUAWEI_invocation_mask
    // [DO NOT EXPOSE] We don't support yet

    // [TODO LATER] when we do multi-gpu
    // ExternalMemoryRDMAFeaturesNV
    // VK_NV_external_memory_rdma
    // bool externalMemoryRDMA;

    // VK_EXT_pipeline_properties
    // [DO NOT EXPOSE] We don't support yet

    // VK_NV_external_sci_sync
    // [DEPRECATED] by VK_NV_external_sci_sync2

    // VK_NV_external_memory_sci_buf
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_frame_boundary
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_multisampled_render_to_single_sampled 
    // [DO NOT EXPOSE] We don't support yet, probably only useful for stencil-K-routed OIT

    // VK_EXT_extended_dynamic_state2
    // ExtendedDynamicState2FeaturesEXT
    // [DO NOT EXPOSE] we're fully GPU driven anyway with sorted pipelines.

    // VK_QNX_screen_surface
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_extension_381
    // VK_KHR_extension_380
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_color_write_enable
    // ColorWriteEnableFeaturesEXT
    // [EXPOSE AS LIMIT]

    // VK_EXT_primitives_generated_query
    // PrimitivesGeneratedQueryFeaturesEXT
    // [DO NOT EXPOSE] requires and relates to EXT_transform_feedback which we'll never expose
    // bool primitivesGeneratedQuery;
    // bool primitivesGeneratedQueryWithRasterizerDiscard;
    // bool primitivesGeneratedQueryWithNonZeroStreams;

    // VK_GOOGLE_extension_386
    // VK_MESA_extension_385
    // VK_EXT_extension_384
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // added in vk 1.3.213, the SDK isn't released yet at this moment :D
    // VK_KHR_ray_tracing_maintenance1
    // Lets enable `rayTracingMaintenance1`and `rayTracingPipelineTraceRaysIndirect2` whenever required by the above.
    // ```
    // bool rayTracingPipelineTraceRaysIndirectDimensionsAndSBT = false;
    // bool rayCullMask = false;
    // //bool accelerationStructureCopyStageAndSBTAccessType;
    // // Do not expose, we don't use KHR_synchronization2 yet
    // bool accelerationStructureSizeAndBLASPointersQuery = false;
    // ```cpp
    // Lets have
    // - `rayTracingPipelineTraceRaysIndirect2` feature, same as `rayTracingPipelineTraceRaysIndirect` but with indirect SBTand dispatch dimensions
    // - two new acceleration structure query parameters
    // - new pipeline stage and access masks but only in `KHR_synchronization2` which we don't use
    // -Support for `GLSL_EXT_ray_cull_mask`, lets call it `rayCullMask`
    // [TODO LATER] when released in the SDK:

    // VK_EXT_extension_388
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_global_priority_query
    // [DEPRECATED] absorbed into KHR_global_priority

    // VK_EXT_extension_391
    // VK_EXT_extension_390
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_image_view_min_lod
    // ImageViewMinLodFeaturesEXT
    // [DO NOT EXPOSE] pointless to implement currently

    // VK_EXT_multi_draw
    // MultiDrawFeaturesEXT
    // [DO NOT EXPOSE] this extension is dumb, if we're recording that many draws we will be using Multi Draw INDIRECT which is better supported

    // VK_EXT_image_2d_view_of_3d
    // Image2DViewOf3DFeaturesEXT
    // [TODO] Investigate later
    // bool image2DViewOf3D;
    // bool sampler2DViewOf3D;

    // VK_KHR_portability_enumeration
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_shader_tile_image
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_opacity_micromap
    // [DO NOT EXPOSE] We don't support yet

    // VK_NV_displacement_micromap
    // [DO NOT EXPOSE] We don't support yet

    // VK_JUICE_extension_400
    // VK_JUICE_extension_399
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_load_store_op_none
    // [TODO LATER] ROADMAP 2024 but need to figure out how extending our LOAD_OP enum would affect us

    // VK_FB_extension_404
    // VK_FB_extension_403
    // VK_FB_extension_402
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_HUAWEI_cluster_culling_shader
    // [DO NOT EXPOSE] We don't support yet

    // VK_GGP_extension_411
    // VK_GGP_extension_410
    // VK_GGP_extension_409
    // VK_GGP_extension_408
    // VK_GGP_extension_407
    // VK_HUAWEI_extension_406
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_border_color_swizzle
    // BorderColorSwizzleFeaturesEXT
    // [DO NOT EXPOSE] not going to expose custom border colors for now

    // VK_EXT_pageable_device_local_memory
    // PageableDeviceLocalMemoryFeaturesEXT
    // [DO NOT EXPOSE] pointless to expose without exposing VK_EXT_memory_priority and the memory query feature first

    // VK_KHR_maintenance4
    // [DO NOT EXPOSE] We don't support yet

    // VK_HUAWEI_extension_415
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_ARM_shader_core_properties
    // [DO NOT EXPOSE] We don't support yet

    // VK_ARM_extension_418
    // VK_KHR_extension_417
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_image_sliced_view_of_3d
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_extension_420
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_VALVE_descriptor_set_host_mapping
    // DescriptorSetHostMappingFeaturesVALVE
    // [DO NOT EXPOSE] This extension is only intended for use in specific embedded environments with known implementation details, and is therefore undocumented.

    // VK_EXT_depth_clamp_zero_one
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_non_seamless_cube_map
    // [DO NOT EXPOSE] Never expose this, it was a mistake for that GL quirk to exist in the first place

    // VK_ARM_extension_425
    // VK_ARM_extension_424
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_QCOM_fragment_density_map_offset
    // FragmentDensityMapOffsetFeaturesQCOM
    // [DO NOT EXPOSE] not implementing or exposing VRS in near or far future

    // VK_NV_copy_memory_indirect
    // [DO NOT EXPOSE] We don't support yet

    // VK_NV_memory_decompression
    // [DO NOT EXPOSE] We don't support yet

    // VK_NV_device_generated_commands_compute
    // [DO NOT EXPOSE] We don't support yet

    // VK_NV_extension_430
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_NV_linear_color_attachment
    // LinearColorAttachmentFeaturesNV
    // [DO NOT EXPOSE] no idea what real-world beneficial use case would be

    // VK_NV_extension_433
    // VK_NV_extension_432
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_GOOGLE_surfaceless_query
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_extension_435
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_application_parameters
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_extension_437
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_image_compression_control_swapchain
    // [DO NOT EXPOSE] We don't support yet

    // VK_QCOM_extension_440
    // VK_SEC_extension_439
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_QCOM_image_processing
    // [DO NOT EXPOSE] We don't support yet

    // VK_ARM_extension_453
    // VK_NV_extension_452
    // VK_SEC_extension_451
    // VK_SEC_extension_450
    // VK_SEC_extension_449
    // VK_SEC_extension_448
    // VK_COREAVI_extension_447
    // VK_COREAVI_extension_446
    // VK_COREAVI_extension_445
    // VK_COREAVI_extension_444
    // VK_COREAVI_extension_443
    // VK_COREAVI_extension_442
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_external_memory_acquire_unmodified
    // [DO NOT EXPOSE] We don't support yet

    // VK_GOOGLE_extension_455
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_extended_dynamic_state3
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_extension_458
    // VK_EXT_extension_457
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_subpass_merge_feedback
    // [DO NOT EXPOSE] We don't support yet

    // VK_LUNARG_direct_driver_loading
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_extension_462
    // VK_EXT_extension_461
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_shader_module_identifier
    // [TODO LATER] Basically a custom hash/ID for which you can use instead of SPIR-V contents to look up IGPUShader in the cache

    // VK_NV_optical_flow
    // TODO: implement
    // VK_EXT_rasterization_order_attachment_access
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_legacy_dithering
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_pipeline_protected_access
    // [DO NOT EXPOSE] We don't support yet

    // VK_AMD_extension_470
    // VK_ANDROID_extension_469
    // VK_EXT_extension_468
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_KHR_maintenance5
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_ray_tracing_position_fetch
    // TODO: implement
    // VK_EXT_extension_481
    // VK_EXT_extension_480
    // VK_AMD_extension_479
    // VK_AMD_extension_478
    // VK_AMD_extension_477
    // VK_AMD_extension_476
    // VK_AMD_extension_475
    // VK_AMD_extension_474
    // VK_AMD_extension_473
    // VK_AMD_extension_472
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_shader_object
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_extension_484
    // [DO NOT EXPOSE] We don't support yet

    // VK_QCOM_tile_properties
    // [DO NOT EXPOSE] We don't support yet

    // VK_SEC_amigo_profiling
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_extension_488
    // VK_EXT_extension_487
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_QCOM_multiview_per_view_viewports
    // [DO NOT EXPOSE] We don't support yet

    // VK_NV_external_sci_sync2
    // [DO NOT EXPOSE] We don't support yet

    // VK_NV_ray_tracing_invocation_reorder
    // [DO NOT EXPOSE] We don't support yet, but a TODO

    // VK_NV_extension_494
    // VK_NV_extension_493
    // VK_NV_extension_492
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_mutable_descriptor_type
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_extension_497
    // VK_EXT_extension_496
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_ARM_shader_core_builtins
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_pipeline_library_group_handles
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_dynamic_rendering_unused_attachments
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_cooperative_matrix
    // CooperativeMatrixFeaturesKHR
    // VK_NV_extension_506
    // VK_EXT_extension_505
    // VK_NV_extension_504
    // VK_EXT_extension_503
    // VK_EXT_extension_502
    // VK_EXT_extension_501
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // [EXPOSE AS LIMIT] redundant
    // bool cooperativeMatrix = limits.cooperativeMatrixSupportedStages.any();
    // leaving as a feature because of overhead
    bool cooperativeMatrixRobustBufferAccess = false;

    // VK_MESA_extension_510
    // VK_EXT_extension_509
    // VK_EXT_extension_508
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_QCOM_multiview_per_view_render_areas
    // [DO NOT EXPOSE] We don't support yet

    // VK_MESA_extension_518
    // VK_EXT_extension_517
    // VK_KHR_extension_516
    // VK_KHR_extension_515
    // VK_KHR_extension_514
    // VK_KHR_extension_513
    // VK_EXT_extension_512
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_QCOM_image_processing2
    // [DO NOT EXPOSE] We don't support yet

    // VK_QCOM_filter_cubic_weights
    // [DO NOT EXPOSE] We don't support yet

    // VK_QCOM_ycbcr_degamma
    // [DO NOT EXPOSE] We don't support yet

    // VK_QCOM_filter_cubic_clamp
    // [DO NOT EXPOSE] We don't support yet

    // VK_EXT_extension_524
    // VK_EXT_extension_523
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_EXT_attachment_feedback_loop_dynamic_state
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_extension_529
    // VK_EXT_extension_528
    // VK_EXT_extension_527
    // VK_EXT_extension_526
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_QNX_external_memory_screen_buffer
    // [DO NOT EXPOSE] We don't support yet

    // VK_MSFT_layered_driver
    // [DO NOT EXPOSE] We don't support yet

    // VK_KHR_extension_546
    // VK_KHR_extension_545
    // VK_KHR_extension_544
    // VK_EXT_extension_543
    // VK_EXT_extension_542
    // VK_EXT_extension_541
    // VK_EXT_extension_540
    // VK_EXT_extension_539
    // VK_EXT_extension_538
    // VK_EXT_extension_537
    // VK_QCOM_extension_536
    // VK_KHR_extension_535
    // VK_KHR_extension_534
    // VK_EXT_extension_533
    // VK_KHR_extension_532
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // VK_NV_descriptor_pool_overallocation
    // [DO NOT EXPOSE] We don't support yet

    // VK_QCOM_extension_548
    // [DO NOT EXPOSE] just reserved numbers, extension never shipped

    // Eternal TODO: how many new extensions since we last looked? (We're up to ext number 548)

    // No Nabla Specific Features for now
    // Nabla

