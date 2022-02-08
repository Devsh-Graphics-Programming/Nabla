// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_OPEN_GL_FEATURE_MAP_H_INCLUDED__
#define __NBL_C_OPEN_GL_FEATURE_MAP_H_INCLUDED__

#include "nbl/core/core.h"
#include "nbl/system/compile_config.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_

#include "COpenGLStateManager.h"

#ifdef _NBL_WINDOWS_API_
	// include windows headers for HWND
	#include "../src/3rdparty/GL/wglext.h"
#elif defined(_NBL_COMPILE_WITH_X11_)
    #include "GL/glx.h"
    #include "../src/3rdparty/GL/glxext.h"
#endif

#include "os.h"
#include "nbl/video/IGPUImageView.h"


namespace nbl
{
namespace video
{



struct DrawArraysIndirectCommand
{
    GLuint count;
    GLuint instanceCount;
    GLuint first;
    GLuint reservedMustBeZero;
};

struct DrawElementsIndirectCommand
{
    GLuint count;
    GLuint instanceCount;
    GLuint firstIndex;
    GLuint baseVertex;
    GLuint baseInstance;
};


static const char* const OpenGLFeatureStrings[] = {
	"GL_3DFX_multisample",
	"GL_3DFX_tbuffer",
	"GL_3DFX_texture_compression_FXT1",
	"GL_AMD_blend_minmax_factor",
	"GL_AMD_conservative_depth",
	"GL_AMD_debug_output",
	"GL_AMD_depth_clamp_separate",
	"GL_AMD_draw_buffers_blend",
	"GL_AMD_multi_draw_indirect",
	"GL_AMD_name_gen_delete",
	"GL_AMD_performance_monitor",
	"GL_AMD_sample_positions",
	"GL_AMD_seamless_cubemap_per_texture",
	"GL_AMD_shader_stencil_export",
	"GL_AMD_texture_texture4",
	"GL_AMD_transform_feedback3_lines_triangles",
	"GL_AMD_vertex_shader_tesselator",
    "GL_AMD_gcn_shader",
    "GL_AMD_gpu_shader_half_float_fetch",
    "GL_AMD_shader_explicit_vertex_parameter",
    "GL_AMD_shader_fragment_mask",
    "GL_AMD_shader_image_load_store_lod",
    "GL_AMD_shader_trinary_minmax",
    "GL_AMD_texture_gather_bias_lod",
    "GL_AMD_vertex_shader_viewport_index",
    "GL_AMD_vertex_shader_layer",
    "GL_AMD_sparse_texture",
    "GL_AMD_shader_stencil_value_export",
    "GL_AMD_gpu_shader_int64",
    "GL_AMD_shader_ballot",
	"GL_APPLE_aux_depth_stencil",
	"GL_APPLE_client_storage",
	"GL_APPLE_element_array",
	"GL_APPLE_fence",
	"GL_APPLE_float_pixels",
	"GL_APPLE_flush_buffer_range",
	"GL_APPLE_object_purgeable",
	"GL_APPLE_rgb_422",
	"GL_APPLE_row_bytes",
	"GL_APPLE_specular_vector",
	"GL_APPLE_texture_range",
	"GL_APPLE_transform_hint",
	"GL_APPLE_vertex_array_object",
	"GL_APPLE_vertex_array_range",
	"GL_APPLE_vertex_program_evaluators",
	"GL_APPLE_ycbcr_422",
	"GL_ARB_base_instance",
	"GL_ARB_bindless_texture",
	"GL_ARB_buffer_storage",
	"GL_ARB_blend_func_extended",
	"GL_ARB_cl_event",
	"GL_ARB_clip_control",
	"GL_ARB_color_buffer_float",
	"GL_ARB_compatibility",
	"GL_ARB_compressed_texture_pixel_storage",
	"GL_ARB_compute_shader",
	"GL_ARB_conservative_depth",
	"GL_ARB_copy_buffer",
	"GL_ARB_debug_output",
	"GL_ARB_depth_buffer_float",
	"GL_ARB_depth_clamp",
	"GL_ARB_depth_texture",
	"GL_ARB_direct_state_access",
	"GL_ARB_draw_buffers",
	"GL_ARB_draw_buffers_blend",
	"GL_ARB_draw_elements_base_vertex",
	"GL_ARB_draw_indirect",
	"GL_ARB_draw_instanced",
	"GL_ARB_ES2_compatibility",
	"GL_ARB_explicit_attrib_location",
	"GL_ARB_explicit_uniform_location",
	"GL_ARB_fragment_coord_conventions",
	"GL_ARB_fragment_program",
	"GL_ARB_fragment_program_shadow",
	"GL_ARB_fragment_shader",
	"GL_ARB_fragment_shader_interlock",
	"GL_ARB_framebuffer_object",
	"GL_ARB_framebuffer_sRGB",
	"GL_ARB_geometry_shader4",
	"GL_ARB_get_program_binary",
	"GL_ARB_get_texture_sub_image",
	"GL_ARB_gpu_shader5",
	"GL_ARB_gpu_shader_fp64",
	"GL_ARB_half_float_pixel",
	"GL_ARB_half_float_vertex",
	"GL_ARB_imaging",
	"GL_ARB_indirect_parameters",
	"GL_ARB_instanced_arrays",
	"GL_ARB_internalformat_query",
	"GL_ARB_internalformat_query2",
	"GL_ARB_map_buffer_alignment",
	"GL_ARB_map_buffer_range",
	"GL_ARB_matrix_palette",
	"GL_ARB_multi_bind",
	"GL_ARB_multi_draw_indirect",
	"GL_ARB_multisample",
	"GL_ARB_multitexture",
	"GL_ARB_occlusion_query",
	"GL_ARB_occlusion_query2",
	"GL_ARB_pixel_buffer_object",
	"GL_ARB_point_parameters",
	"GL_ARB_point_sprite",
	"GL_ARB_program_interface_query",
	"GL_ARB_provoking_vertex",
	"GL_ARB_query_buffer_object",
	"GL_ARB_robustness",
	"GL_ARB_sample_shading",
	"GL_ARB_sampler_objects",
	"GL_ARB_seamless_cube_map",
	"GL_ARB_separate_shader_objects",
	"GL_ARB_shader_atomic_counters",
	"GL_ARB_shader_ballot",
	"GL_ARB_shader_bit_encoding",
	"GL_ARB_shader_draw_parameters",
	"GL_ARB_shader_group_vote",
	"GL_ARB_shader_image_load_store",
	"GL_ARB_shader_objects",
	"GL_ARB_shader_precision",
	"GL_ARB_shader_stencil_export",
	"GL_ARB_shader_subroutine",
	"GL_ARB_shader_texture_lod",
	"GL_ARB_shading_language_100",
	"GL_ARB_shading_language_420pack",
	"GL_ARB_shading_language_include",
	"GL_ARB_shading_language_packing",
	"GL_ARB_shadow",
	"GL_ARB_shadow_ambient",
	"GL_ARB_sync",
	"GL_ARB_tessellation_shader",
	"GL_ARB_texture_barrier",
	"GL_ARB_texture_border_clamp",
	"GL_ARB_texture_buffer_object",
	"GL_ARB_texture_buffer_object_rgb32",
	"GL_ARB_texture_buffer_range",
	"GL_ARB_texture_compression",
	"GL_ARB_texture_compression_bptc",
	"GL_ARB_texture_compression_rgtc",
	"GL_ARB_texture_cube_map",
	"GL_ARB_texture_cube_map_array",
	"GL_ARB_texture_env_add",
	"GL_ARB_texture_env_combine",
	"GL_ARB_texture_env_crossbar",
	"GL_ARB_texture_env_dot3",
	"GL_ARB_texture_float",
	"GL_ARB_texture_gather",
	"GL_ARB_texture_mirrored_repeat",
	"GL_ARB_texture_multisample",
	"GL_ARB_texture_non_power_of_two",
	"GL_ARB_texture_query_lod",
	"GL_ARB_texture_rectangle",
	"GL_ARB_texture_rg",
	"GL_ARB_texture_rgb10_a2ui",
	"GL_ARB_texture_stencil8",
	"GL_ARB_texture_storage",
	"GL_ARB_texture_storage_multisample",
	"GL_ARB_texture_swizzle",
	"GL_ARB_texture_view",
	"GL_ARB_timer_query",
	"GL_ARB_transform_feedback2",
	"GL_ARB_transform_feedback3",
	"GL_ARB_transform_feedback_instanced",
	"GL_ARB_transpose_matrix",
	"GL_ARB_uniform_buffer_object",
	"GL_ARB_vertex_array_bgra",
	"GL_ARB_vertex_array_object",
	"GL_ARB_vertex_attrib_64bit",
	"GL_ARB_vertex_attrib_binding",
	"GL_ARB_vertex_blend",
	"GL_ARB_vertex_buffer_object",
	"GL_ARB_vertex_program",
	"GL_ARB_vertex_shader",
	"GL_ARB_vertex_type_2_10_10_10_rev",
	"GL_ARB_viewport_array",
	"GL_ARB_window_pos",
    "GL_ARB_enhanced_layouts",
    "GL_ARB_cull_distance",
    "GL_ARB_derivative_control",
    "GL_ARB_shader_texture_image_samples",
    "GL_ARB_gpu_shader_int64",
    "GL_ARB_post_depth_coverage",
    "GL_ARB_shader_clock",
    "GL_ARB_shader_viewport_layer_array",
    "GL_ARB_sparse_texture2",
    "GL_ARB_sparse_texture_clamp",
    "GL_ARB_gl_spirv",
    "GL_ARB_spirv_extensions",
	"GL_ATI_draw_buffers",
	"GL_ATI_element_array",
	"GL_ATI_envmap_bumpmap",
	"GL_ATI_fragment_shader",
	"GL_ATI_map_object_buffer",
	"GL_ATI_meminfo",
	"GL_ATI_pixel_format_float",
	"GL_ATI_pn_triangles",
	"GL_ATI_separate_stencil",
	"GL_ATI_text_fragment_shader",
	"GL_ATI_texture_env_combine3",
	"GL_ATI_texture_float",
	"GL_ATI_texture_mirror_once",
	"GL_ATI_vertex_array_object",
	"GL_ATI_vertex_attrib_array_object",
	"GL_ATI_vertex_streams",
	"GL_EXT_422_pixels",
	"GL_EXT_abgr",
	"GL_EXT_bgra",
	"GL_EXT_bindable_uniform",
	"GL_EXT_blend_color",
	"GL_EXT_blend_equation_separate",
	"GL_EXT_blend_func_separate",
	"GL_EXT_blend_logic_op",
	"GL_EXT_blend_minmax",
	"GL_EXT_blend_subtract",
	"GL_EXT_clip_volume_hint",
	"GL_EXT_cmyka",
	"GL_EXT_color_subtable",
	"GL_EXT_compiled_vertex_array",
	"GL_EXT_convolution",
	"GL_EXT_coordinate_frame",
	"GL_EXT_copy_texture",
	"GL_EXT_cull_vertex",
	"GL_EXT_depth_bounds_test",
	"GL_EXT_direct_state_access",
	"GL_EXT_draw_buffers2",
	"GL_EXT_draw_instanced",
	"GL_EXT_draw_range_elements",
	"GL_EXT_fog_coord",
	"GL_EXT_framebuffer_blit",
	"GL_EXT_framebuffer_multisample",
	"GL_EXT_framebuffer_multisample_blit_scaled",
	"GL_EXT_framebuffer_object",
	"GL_EXT_framebuffer_sRGB",
	"GL_EXT_geometry_shader4",
	"GL_EXT_gpu_program_parameters",
	"GL_EXT_gpu_shader4",
	"GL_EXT_histogram",
	"GL_EXT_index_array_formats",
	"GL_EXT_index_func",
	"GL_EXT_index_material",
	"GL_EXT_index_texture",
	"GL_EXT_light_texture",
	"GL_EXT_misc_attribute",
	"GL_EXT_multi_draw_arrays",
	"GL_EXT_multisample",
	"GL_EXT_packed_depth_stencil",
	"GL_EXT_packed_float",
	"GL_EXT_packed_pixels",
	"GL_EXT_paletted_texture",
	"GL_EXT_pixel_buffer_object",
	"GL_EXT_pixel_transform",
	"GL_EXT_pixel_transform_color_table",
	"GL_EXT_point_parameters",
	"GL_EXT_polygon_offset",
	"GL_EXT_provoking_vertex",
	"GL_EXT_rescale_normal",
	"GL_EXT_secondary_color",
	"GL_EXT_separate_shader_objects",
	"GL_EXT_separate_specular_color",
	"GL_EXT_shader_image_load_store",
	"GL_EXT_shadow_funcs",
	"GL_EXT_shared_texture_palette",
	"GL_EXT_stencil_clear_tag",
	"GL_EXT_stencil_two_side",
	"GL_EXT_stencil_wrap",
	"GL_EXT_subtexture",
	"GL_EXT_texture",
	"GL_EXT_texture3D",
	"GL_EXT_texture_array",
	"GL_EXT_texture_buffer_object",
	"GL_EXT_texture_compression_latc",
	"GL_EXT_texture_compression_rgtc",
	"GL_EXT_texture_compression_s3tc",
	"GL_EXT_texture_cube_map",
	"GL_EXT_texture_env_add",
	"GL_EXT_texture_env_combine",
	"GL_EXT_texture_env_dot3",
	"GL_EXT_texture_filter_anisotropic",
	"GL_EXT_texture_integer",
	"GL_EXT_texture_lod_bias",
	"GL_EXT_texture_mirror_clamp",
	"GL_EXT_texture_object",
	"GL_EXT_texture_perturb_normal",
	"GL_EXT_texture_shared_exponent",
	"GL_EXT_texture_snorm",
	"GL_EXT_texture_sRGB",
	"GL_EXT_texture_sRGB_decode",
	"GL_EXT_texture_sRGB_R8",
	"GL_EXT_texture_sRGB_RG8",
	"GL_EXT_texture_swizzle",
	"GL_EXT_texture_view",
	"GL_EXT_timer_query",
	"GL_EXT_transform_feedback",
	"GL_EXT_vertex_array",
	"GL_EXT_vertex_array_bgra",
	"GL_EXT_vertex_attrib_64bit",
	"GL_EXT_vertex_shader",
	"GL_EXT_vertex_weighting",
	"GL_EXT_x11_sync_object",
    "GL_EXT_shader_pixel_local_storage",
    "GL_EXT_shader_pixel_local_storage2",
    "GL_EXT_shader_integer_mix",
    "GL_EXT_shader_image_load_formatted",
    "GL_EXT_post_depth_coverage",
    "GL_EXT_sparse_texture2",
    "GL_EXT_shader_framebuffer_fetch",
    "GL_EXT_shader_framebuffer_fetch_non_coherent",
	"GL_FfdMaskSGIX",
	"GL_GREMEDY_frame_terminator",
	"GL_GREMEDY_string_marker",
	"GL_HP_convolution_border_modes",
	"GL_HP_image_transform",
	"GL_HP_occlusion_test",
	"GL_HP_texture_lighting",
	"GL_IBM_cull_vertex",
	"GL_IBM_multimode_draw_arrays",
	"GL_IBM_rasterpos_clip",
	"GL_IBM_texture_mirrored_repeat",
	"GL_IBM_vertex_array_lists",
	"GL_INGR_blend_func_separate",
	"GL_INGR_color_clamp",
	"GL_INGR_interlace_read",
	"GL_INGR_palette_buffer",
	"GL_INTEL_fragment_shader_interlock",
	"GL_INTEL_parallel_arrays",
	"GL_INTEL_texture_scissor",
    "GL_INTEL_conservative_rasterization",
    "GL_INTEL_blackhole_render",
	"GL_KHR_debug",
	"GL_MESA_pack_invert",
	"GL_MESA_resize_buffers",
	"GL_MESA_window_pos",
	"GL_MESAX_texture_stack",
	"GL_MESA_ycbcr_texture",
	"GL_NV_blend_square",
	"GL_NV_conditional_render",
	"GL_NV_copy_depth_to_color",
	"GL_NV_copy_image",
	"GL_NV_depth_buffer_float",
	"GL_NV_depth_clamp",
	"GL_NV_evaluators",
	"GL_NV_explicit_multisample",
	"GL_NV_fence",
	"GL_NV_float_buffer",
	"GL_NV_fog_distance",
	"GL_NV_fragment_program",
	"GL_NV_fragment_program2",
	"GL_NV_fragment_program4",
	"GL_NV_fragment_program_option",
	"GL_NV_fragment_shader_interlock",
	"GL_NV_framebuffer_multisample_coverage",
	"GL_NV_geometry_program4",
	"GL_NV_geometry_shader4",
	"GL_NV_gpu_program4",
	"GL_NV_gpu_program5",
	"GL_NV_gpu_shader5",
	"GL_NV_half_float",
	"GL_NV_light_max_exponent",
	"GL_NV_multisample_coverage",
	"GL_NV_multisample_filter_hint",
	"GL_NV_occlusion_query",
	"GL_NV_packed_depth_stencil",
	"GL_NV_parameter_buffer_object",
	"GL_NV_parameter_buffer_object2",
	"GL_NV_pixel_data_range",
	"GL_NV_point_sprite",
	"GL_NV_present_video",
	"GL_NV_primitive_restart",
	"GL_NV_register_combiners",
	"GL_NV_register_combiners2",
	"GL_NV_shader_buffer_load",
	"GL_NV_shader_buffer_store",
	"GL_NV_shader_thread_group",
	"GL_NV_shader_thread_shuffle",
	"GL_NV_tessellation_program5",
	"GL_NV_texgen_emboss",
	"GL_NV_texgen_reflection",
	"GL_NV_texture_barrier",
	"GL_NV_texture_compression_vtc",
	"GL_NV_texture_env_combine4",
	"GL_NV_texture_expand_normal",
	"GL_NV_texture_multisample",
	"GL_NV_texture_rectangle",
	"GL_NV_texture_shader",
	"GL_NV_texture_shader2",
	"GL_NV_texture_shader3",
	"GL_NV_transform_feedback",
	"GL_NV_transform_feedback2",
	"GL_NV_vdpau_interop",
	"GL_NV_vertex_array_range",
	"GL_NV_vertex_array_range2",
	"GL_NV_vertex_attrib_integer_64bit",
	"GL_NV_vertex_buffer_unified_memory",
	"GL_NV_vertex_program",
	"GL_NV_vertex_program1_1",
	"GL_NV_vertex_program2",
	"GL_NV_vertex_program2_option",
	"GL_NV_vertex_program3",
	"GL_NV_vertex_program4",
	"GL_NV_video_capture",
    "GL_NV_viewport_array2",
    "GL_NV_stereo_view_rendering",
    "GL_NV_sample_mask_override_coverage",
    "GL_NV_geometry_shader_passthrough",
    "GL_NV_shader_subgroup_partitioned",
    "GL_NV_compute_shader_derivatives",
    "GL_NV_fragment_shader_barycentric",
    "GL_NV_mesh_shader",
    "GL_NV_shader_image_footprint",
    "GL_NV_shading_rate_image",
    "GL_NV_bindless_texture",
    "GL_NV_shader_atomic_float",
    "GL_NV_shader_atomic_int64",
    "GL_NV_sample_locations",
    "GL_NV_shader_atomic_fp16_vector",
    "GL_NV_command_list",
    "GL_NV_shader_atomic_float64",
    "GL_NV_conservative_raster_pre_snap",
    "GL_NV_shader_texture_footprint",
	"GL_OES_read_format",
	"GL_OML_interlace",
	"GL_OML_resample",
	"GL_OML_subsample",
    "GL_OVR_multiview",
    "GL_OVR_multiview2",
	"GL_PGI_misc_hints",
	"GL_PGI_vertex_hints",
	"GL_REND_screen_coordinates",
	"GL_S3_s3tc",
	"GL_SGI_color_matrix",
	"GL_SGI_color_table",
	"GL_SGI_depth_pass_instrument",
	"GL_SGIS_detail_texture",
	"GL_SGIS_fog_function",
	"GL_SGIS_generate_mipmap",
	"GL_SGIS_multisample",
	"GL_SGIS_pixel_texture",
	"GL_SGIS_point_line_texgen",
	"GL_SGIS_point_parameters",
	"GL_SGIS_sharpen_texture",
	"GL_SGIS_texture4D",
	"GL_SGIS_texture_border_clamp",
	"GL_SGIS_texture_color_mask",
	"GL_SGIS_texture_edge_clamp",
	"GL_SGIS_texture_filter4",
	"GL_SGIS_texture_lod",
	"GL_SGIS_texture_select",
	"GL_SGI_texture_color_table",
	"GL_SGIX_async",
	"GL_SGIX_async_histogram",
	"GL_SGIX_async_pixel",
	"GL_SGIX_blend_alpha_minmax",
	"GL_SGIX_calligraphic_fragment",
	"GL_SGIX_clipmap",
	"GL_SGIX_convolution_accuracy",
	"GL_SGIX_depth_pass_instrument",
	"GL_SGIX_depth_texture",
	"GL_SGIX_flush_raster",
	"GL_SGIX_fog_offset",
	"GL_SGIX_fog_scale",
	"GL_SGIX_fragment_lighting",
	"GL_SGIX_framezoom",
	"GL_SGIX_igloo_interface",
	"GL_SGIX_impact_pixel_texture",
	"GL_SGIX_instruments",
	"GL_SGIX_interlace",
	"GL_SGIX_ir_instrument1",
	"GL_SGIX_list_priority",
	"GL_SGIX_pixel_texture",
	"GL_SGIX_pixel_tiles",
	"GL_SGIX_polynomial_ffd",
	"GL_SGIX_reference_plane",
	"GL_SGIX_resample",
	"GL_SGIX_scalebias_hint",
	"GL_SGIX_shadow",
	"GL_SGIX_shadow_ambient",
	"GL_SGIX_sprite",
	"GL_SGIX_subsample",
	"GL_SGIX_tag_sample_buffer",
	"GL_SGIX_texture_add_env",
	"GL_SGIX_texture_coordinate_clamp",
	"GL_SGIX_texture_lod_bias",
	"GL_SGIX_texture_multi_buffer",
	"GL_SGIX_texture_scale_bias",
	"GL_SGIX_texture_select",
	"GL_SGIX_vertex_preclip",
	"GL_SGIX_ycrcb",
	"GL_SGIX_ycrcba",
	"GL_SGIX_ycrcb_subsample",
	"GL_SUN_convolution_border_modes",
	"GL_SUN_global_alpha",
	"GL_SUN_mesh_array",
	"GL_SUN_slice_accum",
	"GL_SUN_triangle_list",
	"GL_SUN_vertex",
	"GL_SUNX_constant_data",
	"GL_WIN_phong_shading",
	"GL_WIN_specular_fog",
    "GL_KHR_texture_compression_astc_hdr",
    "GL_KHR_texture_compression_astc_ldr",
    "GL_KHR_blend_equation_advanced",
    "GL_KHR_blend_equation_advanced_coherent",
	//"GLX_EXT_swap_control_tear",
	"GL_NVX_gpu_memory_info",
    "GL_NVX_multiview_per_view_attributes"
};
//extra extension name that is reported as supported when irrbaw app is running in renderdoc
_NBL_STATIC_INLINE_CONSTEXPR const char* RUNNING_IN_RENDERDOC_EXTENSION_NAME = "GL_NBL_RUNNING_IN_RENDERDOC";


class COpenGLExtensionHandler
{
	public:
	enum EOpenGLFeatures {
		NBL_3DFX_multisample = 0,
		NBL_3DFX_tbuffer,
		NBL_3DFX_texture_compression_FXT1,
		NBL_AMD_blend_minmax_factor,
		NBL_AMD_conservative_depth,
		NBL_AMD_debug_output,
		NBL_AMD_depth_clamp_separate,
		NBL_AMD_draw_buffers_blend,
		NBL_AMD_multi_draw_indirect,
		NBL_AMD_name_gen_delete,
		NBL_AMD_performance_monitor,
		NBL_AMD_sample_positions,
		NBL_AMD_seamless_cubemap_per_texture,
		NBL_AMD_shader_stencil_export,
		NBL_AMD_texture_texture4,
		NBL_AMD_transform_feedback3_lines_triangles,
		NBL_AMD_vertex_shader_tesselator,
        NBL_AMD_gcn_shader,
        NBL_AMD_gpu_shader_half_float_fetch,
        NBL_AMD_shader_explicit_vertex_parameter,
        NBL_AMD_shader_fragment_mask,
        NBL_AMD_shader_image_load_store_lod,
        NBL_AMD_shader_trinary_minmax,
        NBL_AMD_texture_gather_bias_lod,
        NBL_AMD_vertex_shader_viewport_index,
        NBL_AMD_vertex_shader_layer,
        NBL_AMD_sparse_texture,
        NBL_AMD_shader_stencil_value_export,
        NBL_AMD_gpu_shader_int64,
        NBL_AMD_shader_ballot,
		NBL_APPLE_aux_depth_stencil,
		NBL_APPLE_client_storage,
		NBL_APPLE_element_array,
		NBL_APPLE_fence,
		NBL_APPLE_float_pixels,
		NBL_APPLE_flush_buffer_range,
		NBL_APPLE_object_purgeable,
		NBL_APPLE_rgb_422,
		NBL_APPLE_row_bytes,
		NBL_APPLE_specular_vector,
		NBL_APPLE_texture_range,
		NBL_APPLE_transform_hint,
		NBL_APPLE_vertex_array_object,
		NBL_APPLE_vertex_array_range,
		NBL_APPLE_vertex_program_evaluators,
		NBL_APPLE_ycbcr_422,
		NBL_ARB_base_instance,
		NBL_ARB_bindless_texture,
		NBL_ARB_buffer_storage,
		NBL_ARB_blend_func_extended,
		NBL_ARB_clip_control,
		NBL_ARB_cl_event,
		NBL_ARB_color_buffer_float,
		NBL_ARB_compatibility,
		NBL_ARB_compressed_texture_pixel_storage,
		NBL_ARB_compute_shader,
		NBL_ARB_conservative_depth,
		NBL_ARB_copy_buffer,
		NBL_ARB_debug_output,
		NBL_ARB_depth_buffer_float,
		NBL_ARB_depth_clamp,
		NBL_ARB_depth_texture,
		NBL_ARB_direct_state_access,
		NBL_ARB_draw_buffers,
		NBL_ARB_draw_buffers_blend,
		NBL_ARB_draw_elements_base_vertex,
		NBL_ARB_draw_indirect,
		NBL_ARB_draw_instanced,
		NBL_ARB_ES2_compatibility,
		NBL_ARB_explicit_attrib_location,
		NBL_ARB_explicit_uniform_location,
		NBL_ARB_fragment_coord_conventions,
		NBL_ARB_fragment_program,
		NBL_ARB_fragment_program_shadow,
		NBL_ARB_fragment_shader,
		NBL_ARB_fragment_shader_interlock,
		NBL_ARB_framebuffer_object,
		NBL_ARB_framebuffer_sRGB,
		NBL_ARB_geometry_shader4,
		NBL_ARB_get_program_binary,
		NBL_ARB_get_texture_sub_image,
		NBL_ARB_gpu_shader5,
		NBL_ARB_gpu_shader_fp64,
		NBL_ARB_half_float_pixel,
		NBL_ARB_half_float_vertex,
		NBL_ARB_imaging,
		NBL_ARB_instanced_arrays,
		NBL_ARB_indirect_parameters,
		NBL_ARB_internalformat_query,
		NBL_ARB_internalformat_query2,
		NBL_ARB_map_buffer_alignment,
		NBL_ARB_map_buffer_range,
		NBL_ARB_matrix_palette,
		NBL_ARB_multi_bind,
		NBL_ARB_multi_draw_indirect,
		NBL_ARB_multisample,
		NBL_ARB_multitexture,
		NBL_ARB_occlusion_query,
		NBL_ARB_occlusion_query2,
		NBL_ARB_pixel_buffer_object,
		NBL_ARB_point_parameters,
		NBL_ARB_point_sprite,
		NBL_ARB_program_interface_query,
		NBL_ARB_provoking_vertex,
		NBL_ARB_query_buffer_object,
		NBL_ARB_robustness,
		NBL_ARB_sample_shading,
		NBL_ARB_sampler_objects,
		NBL_ARB_seamless_cube_map,
		NBL_ARB_separate_shader_objects,
		NBL_ARB_shader_atomic_counters,
		NBL_ARB_shader_ballot,
		NBL_ARB_shader_bit_encoding,
		NBL_ARB_shader_draw_parameters,
		NBL_ARB_shader_group_vote,
		NBL_ARB_shader_image_load_store,
		NBL_ARB_shader_objects,
		NBL_ARB_shader_precision,
		NBL_ARB_shader_stencil_export,
		NBL_ARB_shader_subroutine,
		NBL_ARB_shader_texture_lod,
		NBL_ARB_shading_language_100,
		NBL_ARB_shading_language_420pack,
		NBL_ARB_shading_language_include,
		NBL_ARB_shading_language_packing,
		NBL_ARB_shadow,
		NBL_ARB_shadow_ambient,
		NBL_ARB_sync,
		NBL_ARB_tessellation_shader,
		NBL_ARB_texture_barrier,
		NBL_ARB_texture_border_clamp,
		NBL_ARB_texture_buffer_object,
		NBL_ARB_texture_buffer_object_rgb32,
		NBL_ARB_texture_buffer_range,
		NBL_ARB_texture_compression,
		NBL_ARB_texture_compression_bptc,
		NBL_ARB_texture_compression_rgtc,
		NBL_ARB_texture_cube_map,
		NBL_ARB_texture_cube_map_array,
		NBL_ARB_texture_env_add,
		NBL_ARB_texture_env_combine,
		NBL_ARB_texture_env_crossbar,
		NBL_ARB_texture_env_dot3,
		NBL_ARB_texture_float,
		NBL_ARB_texture_gather,
		NBL_ARB_texture_mirrored_repeat,
		NBL_ARB_texture_multisample,
		NBL_ARB_texture_non_power_of_two,
		NBL_ARB_texture_query_lod,
		NBL_ARB_texture_rectangle,
		NBL_ARB_texture_rg,
		NBL_ARB_texture_rgb10_a2ui,
		NBL_ARB_texture_stencil8,
		NBL_ARB_texture_storage,
		NBL_ARB_texture_storage_multisample,
		NBL_ARB_texture_swizzle,
		NBL_ARB_texture_view,
		NBL_ARB_timer_query,
		NBL_ARB_transform_feedback2,
		NBL_ARB_transform_feedback3,
		NBL_ARB_transform_feedback_instanced,
		NBL_ARB_transpose_matrix,
		NBL_ARB_uniform_buffer_object,
		NBL_ARB_vertex_array_bgra,
		NBL_ARB_vertex_array_object,
		NBL_ARB_vertex_attrib_64bit,
		NBL_ARB_vertex_attrib_binding,
		NBL_ARB_vertex_blend,
		NBL_ARB_vertex_buffer_object,
		NBL_ARB_vertex_program,
		NBL_ARB_vertex_shader,
		NBL_ARB_vertex_type_2_10_10_10_rev,
		NBL_ARB_viewport_array,
		NBL_ARB_window_pos,
        NBL_ARB_enhanced_layouts,
        NBL_ARB_cull_distance,
        NBL_ARB_derivative_control,
        NBL_ARB_shader_texture_image_samples,
        NBL_ARB_gpu_shader_int64,
        NBL_ARB_post_depth_coverage,
        NBL_ARB_shader_clock,
        NBL_ARB_shader_viewport_layer_array,
        NBL_ARB_sparse_texture2,
        NBL_ARB_sparse_texture_clamp,
        NBL_ARB_gl_spirv,
        NBL_ARB_spirv_extensions,
		NBL_ATI_draw_buffers,
		NBL_ATI_element_array,
		NBL_ATI_envmap_bumpmap,
		NBL_ATI_fragment_shader,
		NBL_ATI_map_object_buffer,
		NBL_ATI_meminfo,
		NBL_ATI_pixel_format_float,
		NBL_ATI_pn_triangles,
		NBL_ATI_separate_stencil,
		NBL_ATI_text_fragment_shader,
		NBL_ATI_texture_env_combine3,
		NBL_ATI_texture_float,
		NBL_ATI_texture_mirror_once,
		NBL_ATI_vertex_array_object,
		NBL_ATI_vertex_attrib_array_object,
		NBL_ATI_vertex_streams,
		NBL_EXT_422_pixels,
		NBL_EXT_abgr,
		NBL_EXT_bgra,
		NBL_EXT_bindable_uniform,
		NBL_EXT_blend_color,
		NBL_EXT_blend_equation_separate,
		NBL_EXT_blend_func_separate,
		NBL_EXT_blend_logic_op,
		NBL_EXT_blend_minmax,
		NBL_EXT_blend_subtract,
		NBL_EXT_clip_volume_hint,
		NBL_EXT_cmyka,
		NBL_EXT_color_subtable,
		NBL_EXT_compiled_vertex_array,
		NBL_EXT_convolution,
		NBL_EXT_coordinate_frame,
		NBL_EXT_copy_texture,
		NBL_EXT_cull_vertex,
		NBL_EXT_depth_bounds_test,
		NBL_EXT_direct_state_access,
		NBL_EXT_draw_buffers2,
		NBL_EXT_draw_instanced,
		NBL_EXT_draw_range_elements,
		NBL_EXT_fog_coord,
		NBL_EXT_framebuffer_blit,
		NBL_EXT_framebuffer_multisample,
		NBL_EXT_framebuffer_multisample_blit_scaled,
		NBL_EXT_framebuffer_object,
		NBL_EXT_framebuffer_sRGB,
		NBL_EXT_geometry_shader4,
		NBL_EXT_gpu_program_parameters,
		NBL_EXT_gpu_shader4,
		NBL_EXT_histogram,
		NBL_EXT_index_array_formats,
		NBL_EXT_index_func,
		NBL_EXT_index_material,
		NBL_EXT_index_texture,
		NBL_EXT_light_texture,
		NBL_EXT_misc_attribute,
		NBL_EXT_multi_draw_arrays,
		NBL_EXT_multisample,
		NBL_EXT_packed_depth_stencil,
		NBL_EXT_packed_float,
		NBL_EXT_packed_pixels,
		NBL_EXT_paletted_texture,
		NBL_EXT_pixel_buffer_object,
		NBL_EXT_pixel_transform,
		NBL_EXT_pixel_transform_color_table,
		NBL_EXT_point_parameters,
		NBL_EXT_polygon_offset,
		NBL_EXT_provoking_vertex,
		NBL_EXT_rescale_normal,
		NBL_EXT_secondary_color,
		NBL_EXT_separate_shader_objects,
		NBL_EXT_separate_specular_color,
		NBL_EXT_shader_image_load_store,
		NBL_EXT_shadow_funcs,
		NBL_EXT_shared_texture_palette,
		NBL_EXT_stencil_clear_tag,
		NBL_EXT_stencil_two_side,
		NBL_EXT_stencil_wrap,
		NBL_EXT_subtexture,
		NBL_EXT_texture,
		NBL_EXT_texture3D,
		NBL_EXT_texture_array,
		NBL_EXT_texture_buffer_object,
		NBL_EXT_texture_compression_latc,
		NBL_EXT_texture_compression_rgtc,
		NBL_EXT_texture_compression_s3tc,
		NBL_EXT_texture_cube_map,
		NBL_EXT_texture_env_add,
		NBL_EXT_texture_env_combine,
		NBL_EXT_texture_env_dot3,
		NBL_EXT_texture_filter_anisotropic,
		NBL_EXT_texture_integer,
		NBL_EXT_texture_lod_bias,
		NBL_EXT_texture_mirror_clamp,
		NBL_EXT_texture_object,
		NBL_EXT_texture_perturb_normal,
		NBL_EXT_texture_shared_exponent,
		NBL_EXT_texture_snorm,
		NBL_EXT_texture_sRGB,
		NBL_EXT_texture_sRGB_decode,
		NBL_EXT_texture_sRGB_R8,
		NBL_EXT_texture_sRGB_RG8,
		NBL_EXT_texture_swizzle,
		NBL_EXT_texture_view,
		NBL_EXT_timer_query,
		NBL_EXT_transform_feedback,
		NBL_EXT_vertex_array,
		NBL_EXT_vertex_array_bgra,
		NBL_EXT_vertex_attrib_64bit,
		NBL_EXT_vertex_shader,
		NBL_EXT_vertex_weighting,
		NBL_EXT_x11_sync_object,
        NBL_EXT_shader_pixel_local_storage,
        NBL_EXT_shader_pixel_local_storage2,
        NBL_EXT_shader_integer_mix,
        NBL_EXT_shader_image_load_formatted,
        NBL_EXT_post_depth_coverage,
        NBL_EXT_sparse_texture2,
        NBL_EXT_shader_framebuffer_fetch,
        NBL_EXT_shader_framebuffer_fetch_non_coherent,
		NBL_FfdMaskSGIX,
		NBL_GREMEDY_frame_terminator,
		NBL_GREMEDY_string_marker,
		NBL_HP_convolution_border_modes,
		NBL_HP_image_transform,
		NBL_HP_occlusion_test,
		NBL_HP_texture_lighting,
		NBL_IBM_cull_vertex,
		NBL_IBM_multimode_draw_arrays,
		NBL_IBM_rasterpos_clip,
		NBL_IBM_texture_mirrored_repeat,
		NBL_IBM_vertex_array_lists,
		NBL_INGR_blend_func_separate,
		NBL_INGR_color_clamp,
		NBL_INGR_interlace_read,
		NBL_INGR_palette_buffer,
		NBL_INTEL_fragment_shader_ordering,
		NBL_INTEL_parallel_arrays,
		NBL_INTEL_texture_scissor,
        NBL_INTEL_conservative_rasterization,
        NBL_INTEL_blackhole_render,
		NBL_KHR_debug,
		NBL_MESA_pack_invert,
		NBL_MESA_resize_buffers,
		NBL_MESA_window_pos,
		NBL_MESAX_texture_stack,
		NBL_MESA_ycbcr_texture,
		NBL_NV_blend_square,
		NBL_NV_conditional_render,
		NBL_NV_copy_depth_to_color,
		NBL_NV_copy_image,
		NBL_NV_depth_buffer_float,
		NBL_NV_depth_clamp,
		NBL_NV_evaluators,
		NBL_NV_explicit_multisample,
		NBL_NV_fence,
		NBL_NV_float_buffer,
		NBL_NV_fog_distance,
		NBL_NV_fragment_program,
		NBL_NV_fragment_program2,
		NBL_NV_fragment_program4,
		NBL_NV_fragment_program_option,
		NBL_NV_fragment_shader_interlock,
		NBL_NV_framebuffer_multisample_coverage,
		NBL_NV_geometry_program4,
		NBL_NV_geometry_shader4,
		NBL_NV_gpu_program4,
		NBL_NV_gpu_program5,
		NBL_NV_gpu_shader5,
		NBL_NV_half_float,
		NBL_NV_light_max_exponent,
		NBL_NV_multisample_coverage,
		NBL_NV_multisample_filter_hint,
		NBL_NV_occlusion_query,
		NBL_NV_packed_depth_stencil,
		NBL_NV_parameter_buffer_object,
		NBL_NV_parameter_buffer_object2,
		NBL_NV_pixel_data_range,
		NBL_NV_point_sprite,
		NBL_NV_present_video,
		NBL_NV_primitive_restart,
		NBL_NV_register_combiners,
		NBL_NV_register_combiners2,
		NBL_NV_shader_buffer_load,
		NBL_NV_shader_buffer_store,
		NBL_NV_shader_thread_group,
		NBL_NV_shader_thread_shuffle,
		NBL_NV_tessellation_program5,
		NBL_NV_texgen_emboss,
		NBL_NV_texgen_reflection,
		NBL_NV_texture_barrier,
		NBL_NV_texture_compression_vtc,
		NBL_NV_texture_env_combine4,
		NBL_NV_texture_expand_normal,
		NBL_NV_texture_multisample,
		NBL_NV_texture_rectangle,
		NBL_NV_texture_shader,
		NBL_NV_texture_shader2,
		NBL_NV_texture_shader3,
		NBL_NV_transform_feedback,
		NBL_NV_transform_feedback2,
		NBL_NV_vdpau_interop,
		NBL_NV_vertex_array_range,
		NBL_NV_vertex_array_range2,
		NBL_NV_vertex_attrib_integer_64bit,
		NBL_NV_vertex_buffer_unified_memory,
		NBL_NV_vertex_program,
		NBL_NV_vertex_program1_1,
		NBL_NV_vertex_program2,
		NBL_NV_vertex_program2_option,
		NBL_NV_vertex_program3,
		NBL_NV_vertex_program4,
		NBL_NV_video_capture,
        NBL_NV_viewport_array2,
        NBL_NV_stereo_view_rendering,
        NBL_NV_sample_mask_override_coverage,
        NBL_NV_geometry_shader_passthrough,
        NBL_NV_shader_subgroup_partitioned,
        NBL_NV_compute_shader_derivatives,
        NBL_NV_fragment_shader_barycentric,
        NBL_NV_mesh_shader,
        NBL_NV_shader_image_footprint,
        NBL_NV_shading_rate_image,
        NBL_NV_bindless_texture,
        NBL_NV_shader_atomic_float,
        NBL_NV_shader_atomic_int64,
        NBL_NV_sample_locations,
        NBL_NV_shader_atomic_fp16_vector,
        NBL_NV_command_list,
        NBL_NV_shader_atomic_float64,
        NBL_NV_conservative_raster_pre_snap,
        NBL_NV_shader_texture_footprint,
		NBL_OES_read_format,
		NBL_OML_interlace,
		NBL_OML_resample,
		NBL_OML_subsample,
        NBL_OVR_multiview,
        NBL_OVR_multiview2,
		NBL_PGI_misc_hints,
		NBL_PGI_vertex_hints,
		NBL_REND_screen_coordinates,
		NBL_S3_s3tc,
		NBL_SGI_color_matrix,
		NBL_SGI_color_table,
		NBL_SGI_depth_pass_instrument,
		NBL_SGIS_detail_texture,
		NBL_SGIS_fog_function,
		NBL_SGIS_generate_mipmap,
		NBL_SGIS_multisample,
		NBL_SGIS_pixel_texture,
		NBL_SGIS_point_line_texgen,
		NBL_SGIS_point_parameters,
		NBL_SGIS_sharpen_texture,
		NBL_SGIS_texture4D,
		NBL_SGIS_texture_border_clamp,
		NBL_SGIS_texture_color_mask,
		NBL_SGIS_texture_edge_clamp,
		NBL_SGIS_texture_filter4,
		NBL_SGIS_texture_lod,
		NBL_SGIS_texture_select,
		NBL_SGI_texture_color_table,
		NBL_SGIX_async,
		NBL_SGIX_async_histogram,
		NBL_SGIX_async_pixel,
		NBL_SGIX_blend_alpha_minmax,
		NBL_SGIX_calligraphic_fragment,
		NBL_SGIX_clipmap,
		NBL_SGIX_convolution_accuracy,
		NBL_SGIX_depth_pass_instrument,
		NBL_SGIX_depth_texture,
		NBL_SGIX_flush_raster,
		NBL_SGIX_fog_offset,
		NBL_SGIX_fog_scale,
		NBL_SGIX_fragment_lighting,
		NBL_SGIX_framezoom,
		NBL_SGIX_igloo_interface,
		NBL_SGIX_impact_pixel_texture,
		NBL_SGIX_instruments,
		NBL_SGIX_interlace,
		NBL_SGIX_ir_instrument1,
		NBL_SGIX_list_priority,
		NBL_SGIX_pixel_texture,
		NBL_SGIX_pixel_tiles,
		NBL_SGIX_polynomial_ffd,
		NBL_SGIX_reference_plane,
		NBL_SGIX_resample,
		NBL_SGIX_scalebias_hint,
		NBL_SGIX_shadow,
		NBL_SGIX_shadow_ambient,
		NBL_SGIX_sprite,
		NBL_SGIX_subsample,
		NBL_SGIX_tag_sample_buffer,
		NBL_SGIX_texture_add_env,
		NBL_SGIX_texture_coordinate_clamp,
		NBL_SGIX_texture_lod_bias,
		NBL_SGIX_texture_multi_buffer,
		NBL_SGIX_texture_scale_bias,
		NBL_SGIX_texture_select,
		NBL_SGIX_vertex_preclip,
		NBL_SGIX_ycrcb,
		NBL_SGIX_ycrcba,
		NBL_SGIX_ycrcb_subsample,
		NBL_SUN_convolution_border_modes,
		NBL_SUN_global_alpha,
		NBL_SUN_mesh_array,
		NBL_SUN_slice_accum,
		NBL_SUN_triangle_list,
		NBL_SUN_vertex,
		NBL_SUNX_constant_data,
		NBL_WIN_phong_shading,
		NBL_WIN_specular_fog,
        NBL_KHR_texture_compression_astc_hdr,
        NBL_KHR_texture_compression_astc_ldr,
        NBL_KHR_blend_equation_advanced,
        NBL_KHR_blend_equation_advanced_coherent,
		//NBL_GLX_EXT_swap_control_tear,
		NBL_NVX_gpu_memory_info,
        NBL_NVX_multiview_per_view_attributes,
		NBL_OpenGL_Feature_Count
	};
    _NBL_STATIC_INLINE_CONSTEXPR EOpenGLFeatures m_GLSLExtensions[]{
        NBL_AMD_gcn_shader,
        NBL_AMD_gpu_shader_half_float_fetch,
        NBL_AMD_shader_ballot,
        NBL_AMD_shader_explicit_vertex_parameter,
        NBL_AMD_shader_fragment_mask,
        NBL_AMD_shader_image_load_store_lod,
        NBL_AMD_shader_trinary_minmax,
        NBL_AMD_texture_gather_bias_lod,
        NBL_NVX_multiview_per_view_attributes,
        NBL_NV_viewport_array2,
        NBL_NV_stereo_view_rendering,
        NBL_NV_sample_mask_override_coverage,
        NBL_NV_geometry_shader_passthrough,
        NBL_NV_shader_subgroup_partitioned,
        NBL_NV_compute_shader_derivatives,
        NBL_NV_fragment_shader_barycentric,
        NBL_NV_mesh_shader,
        NBL_NV_shader_image_footprint,
        NBL_NV_shading_rate_image,
        NBL_ARB_shading_language_include,
        NBL_ARB_shader_stencil_export,
        NBL_ARB_enhanced_layouts,
        NBL_ARB_bindless_texture,
        NBL_ARB_shader_draw_parameters,
        NBL_ARB_shader_group_vote,
        NBL_ARB_cull_distance,
        NBL_ARB_derivative_control,
        NBL_ARB_shader_texture_image_samples,
        NBL_KHR_blend_equation_advanced,
        NBL_KHR_blend_equation_advanced_coherent,
        NBL_ARB_fragment_shader_interlock,
        NBL_ARB_gpu_shader_int64,
        NBL_ARB_post_depth_coverage,
        NBL_ARB_shader_ballot,
        NBL_ARB_shader_clock,
        NBL_ARB_shader_viewport_layer_array,
        NBL_ARB_sparse_texture2,
        NBL_ARB_sparse_texture_clamp,
        NBL_ARB_gl_spirv,
        NBL_ARB_spirv_extensions,
        NBL_AMD_shader_stencil_export,
        NBL_AMD_vertex_shader_viewport_index,
        NBL_AMD_vertex_shader_layer,
        NBL_NV_bindless_texture,
        NBL_NV_shader_atomic_float,
        NBL_AMD_sparse_texture,
        NBL_EXT_shader_integer_mix,
        NBL_INTEL_fragment_shader_ordering,
        NBL_AMD_shader_stencil_value_export,
        NBL_NV_shader_thread_group,
        NBL_NV_shader_thread_shuffle,
        NBL_EXT_shader_image_load_formatted,
        NBL_AMD_gpu_shader_int64,
        NBL_NV_shader_atomic_int64,
        NBL_EXT_post_depth_coverage,
        NBL_EXT_sparse_texture2,
        NBL_NV_fragment_shader_interlock,
        NBL_NV_sample_locations,
        NBL_NV_shader_atomic_fp16_vector,
        NBL_NV_command_list,
        NBL_OVR_multiview,
        NBL_OVR_multiview2,
        NBL_NV_shader_atomic_float64,
        NBL_INTEL_conservative_rasterization,
        NBL_NV_conservative_raster_pre_snap,
        NBL_EXT_shader_framebuffer_fetch,
        NBL_EXT_shader_framebuffer_fetch_non_coherent,
        NBL_INTEL_blackhole_render,
        NBL_NV_shader_texture_footprint,
        NBL_NV_gpu_shader5
    };

	static core::CLeakDebugger bufferLeaker;
	static core::CLeakDebugger textureLeaker;

	// deferred initialization
	void initExtensions(bool stencilBuffer);

	static void loadFunctions();

	bool isDeviceCompatibile(core::vector<std::string>* failedExtensions=NULL);

	//! queries the features of the driver, returns true if feature is available
	inline bool queryOpenGLFeature(EOpenGLFeatures feature) const
	{
		return FeatureAvailable[feature];
	}

	//! show all features with availablity
	void dump(std::string* outStr=NULL, bool onlyAvailable=false) const;

	void dumpFramebufferFormats() const;

	// Some variables for properties
	bool StencilBuffer;
	bool TextureCompressionExtension;

	// Some non-boolean properties
	//!
	static int32_t reqUBOAlignment;
	//!
	static int32_t reqSSBOAlignment;
	//!
	static int32_t reqTBOAlignment;
    //!
    static uint64_t maxUBOSize;
    //!
    static uint64_t maxSSBOSize;
    //!
    static uint64_t maxTBOSizeInTexels;
    //!
    static uint64_t maxBufferSize;
    //!
    static uint32_t maxUBOBindings;
    //!
    static uint32_t maxSSBOBindings;
    //! For vertex and fragment shaders
    //! If both the vertex shader and the fragment processing stage access the same texture image unit, then that counts as using two texture image units against this limit.
    static uint32_t maxTextureBindings;
    //! For compute shader
    static uint32_t maxTextureBindingsCompute;
    //!
    static uint32_t maxImageBindings;
	//!
	static int32_t minMemoryMapAlignment;
    //!
    static int32_t MaxComputeWGSize[3];
	//!
	static uint32_t MaxArrayTextureLayers;
	//! Maxmimum texture layers supported by the engine
	static uint8_t MaxTextureUnits;
	//! Maximal Anisotropy
	static uint8_t MaxAnisotropy;
	//! Number of user clipplanes
	static uint8_t MaxUserClipPlanes;
	//! Number of rendertargets available as MRTs
	static uint8_t MaxMultipleRenderTargets;
	//! Optimal number of indices per meshbuffer
	static uint32_t MaxIndices;
	//! Optimal number of vertices per meshbuffer
	static uint32_t MaxVertices;
	//! Maximal vertices handled by geometry shaders
	static uint32_t MaxGeometryVerticesOut;
	//! Maximal LOD Bias
	static float MaxTextureLODBias;
	//!
	static uint32_t MaxVertexStreams;
	//!
	static uint32_t MaxXFormFeedbackComponents;
	//!
	static uint32_t MaxGPUWaitTimeout;
	//! Gives the upper and lower bound on warp/wavefront/SIMD-lane size
	static uint32_t InvocationSubGroupSize[2];

    //TODO should be later changed to SPIR-V extensions enum like it is with OpenGL extensions
    //(however it does not have any implications on API)
    static GLuint SPIR_VextensionsCount;
    static core::smart_refctd_dynamic_array<const GLubyte*> SPIR_Vextensions;

	//! Minimal and maximal supported thickness for lines without smoothing
	GLfloat DimAliasedLine[2];
	//! Minimal and maximal supported thickness for points without smoothing
	GLfloat DimAliasedPoint[2];
	//! Minimal and maximal supported thickness for lines with smoothing
	GLfloat DimSmoothedLine[2];
	//! Minimal and maximal supported thickness for points with smoothing
	GLfloat DimSmoothedPoint[2];

	//! OpenGL version as Integer: 100*Major+Minor, i.e. 2.1 becomes 201
	static uint16_t Version;
	//! GLSL version as Integer: 100*Major+Minor
	static uint16_t ShaderLanguageVersion;

	static bool IsIntelGPU;
	static bool needsDSAFramebufferHack;

	//
    static bool extGlIsEnabledi(GLenum cap, GLuint index);
    static void extGlEnablei(GLenum cap, GLuint index);
    static void extGlDisablei(GLenum cap, GLuint index);
    static void extGlGetBooleani_v(GLenum pname, GLuint index, GLboolean* data);
    static void extGlGetFloati_v(GLenum pname, GLuint index, float* data);
    static void extGlGetInteger64v(GLenum pname, GLint64* data);
    static void extGlGetIntegeri_v(GLenum pname, GLuint index, GLint* data);
    static void extGlProvokingVertex(GLenum provokeMode);
    static void extGlClipControl(GLenum origin, GLenum depth);

    //
    static GLsync extGlFenceSync(GLenum condition, GLbitfield flags);
    static void extGlDeleteSync(GLsync sync);
    static GLenum extGlClientWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout);
    static void extGlWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout);

    // the above function definitions can stay, the rest towards the bottom are up for review
	// public access to the (loaded) extensions.
	static void extGlActiveTexture(GLenum target);
    static void extGlBindTextures(const GLuint& first, const GLsizei& count, const GLuint* textures, const GLenum* targets);
    static void extGlCreateTextures(GLenum target, GLsizei n, GLuint *textures);


    static void extGlTextureBuffer(GLuint texture, GLenum internalformat, GLuint buffer);
    static void extGlTextureBufferRange(GLuint texture, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizei length);
    static void extGlTextureStorage1D(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width);
    static void extGlTextureStorage2D(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height);
    static void extGlTextureStorage3D(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth);
    //multisample textures
    static void extGlTextureStorage2DMultisample(GLuint texture, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations);
    static void extGlTextureStorage3DMultisample(GLuint texture, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations);
	// views
	static void extGlTextureView(GLuint texture, GLenum target, GLuint origtexture, GLenum internalformat, GLuint minlevel, GLuint numlevels, GLuint minlayer, GLuint numlayers);
    // texture update functions
	static void extGlGetTextureSubImage(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLsizei bufSize, void* pixels);
	static void extGlGetCompressedTextureSubImage(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLsizei bufSize, void* pixels);
	static void extGlGetTextureImage(GLuint texture, GLenum target, GLint level, GLenum format, GLenum type, GLsizei bufSizeHint, void* pixels);
	static void extGlGetCompressedTextureImage(GLuint texture, GLenum target, GLint level, GLsizei bufSizeHint, void* pixels);
    static void extGlTextureSubImage1D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels);
    static void extGlTextureSubImage2D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels);
    static void extGlTextureSubImage3D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels);
    static void extGlCompressedTextureSubImage1D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *data);
    static void extGlCompressedTextureSubImage2D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data);
    static void extGlCompressedTextureSubImage3D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *data);
	static void extGlCopyImageSubData(GLuint srcName, GLenum srcTarget, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, GLuint dstName, GLenum dstTarget, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei srcWidth, GLsizei srcHeight, GLsizei srcDepth);
    static void extGlGenerateTextureMipmap(GLuint texture, GLenum target);
	// texture "parameter" functions
	static void extGlTextureParameterIuiv(GLuint texture, GLenum target, GLenum pname, const GLuint* params);
	static void extGlClampColor(GLenum target, GLenum clamp);

    static void extGlCreateSamplers(GLsizei n, GLuint* samplers);
    static void extGlDeleteSamplers(GLsizei n, GLuint* samplers);
    static void extGlBindSamplers(const GLuint& first, const GLsizei& count, const GLuint* samplers);
    static void extGlSamplerParameteri(GLuint sampler, GLenum pname, GLint param);
    static void extGlSamplerParameterf(GLuint sampler, GLenum pname, GLfloat param);
    static void extGlSamplerParameterfv(GLuint sampler, GLenum pname, const GLfloat* params);

    //
    static void extGlBindImageTexture(GLuint index, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format);
    static void extGlBindImageTextures(GLuint first, GLsizei count, const GLuint* textures, const GLenum* formats);

    //bindless textures
    static GLuint64 extGlGetTextureHandle(GLuint texture);
    static GLuint64 extGlGetTextureSamplerHandle(GLuint texture, GLuint sampler);
    static void extGlMakeTextureHandleResident(GLuint64 handle);
    static void extGlMakeTextureHandleNonResident(GLuint64 handle);
    static GLuint64 extGlGetImageHandle(GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum format);
    static void extGlMakeImageHandleResident(GLuint64 handle, GLenum access);
    static void extGlMakeImageHandleNonResident(GLuint64 handle);
    GLboolean extGlIsTextureHandleResident(GLuint64 handle);
    GLboolean extGlIsImageHandleResident(GLuint64 handle);

	static void extGlPointParameterf(GLint loc, GLfloat f);
	static void extGlPointParameterfv(GLint loc, const GLfloat *v);
	static void extGlStencilFuncSeparate(GLenum face, GLenum func, GLint ref, GLuint mask);
	static void extGlStencilOpSeparate(GLenum face, GLenum fail, GLenum zfail, GLenum zpass);
	static void extGlStencilMaskSeparate(GLenum face, GLuint mask);


	// shader programming
    static void extGlCreateProgramPipelines(GLsizei n, GLuint* pipelines);
    static void extGlDeleteProgramPipelines(GLsizei n, const GLuint* pipelines);
    static void extGlUseProgramStages(GLuint pipeline, GLbitfield stages, GLuint program);
	static GLuint extGlCreateShader(GLenum shaderType);
    static GLuint extGlCreateShaderProgramv(GLenum shaderType, GLsizei count, const char** strings);
	static void extGlShaderSource(GLuint shader, GLsizei numOfStrings, const char **strings, const GLint *lenOfStrings);
	static void extGlCompileShader(GLuint shader);
	static GLuint extGlCreateProgram(void);
	static void extGlAttachShader(GLuint program, GLuint shader);
    static void extGlTransformFeedbackVaryings(GLuint program, GLsizei count, const char** varyings, GLenum bufferMode);
    static void extGlLinkProgram(GLuint program);
	static void extGlUseProgram(GLuint prog);
	static void extGlDeleteProgram(GLuint object);
	static void extGlDeleteShader(GLuint shader);
	static void extGlGetAttachedShaders(GLuint program, GLsizei maxcount, GLsizei* count, GLuint* shaders);
	static void extGlGetShaderInfoLog(GLuint shader, GLsizei maxLength, GLsizei *length, GLchar *infoLog);
	static void extGlGetProgramInfoLog(GLuint program, GLsizei maxLength, GLsizei *length, GLchar *infoLog);
	static void extGlGetShaderiv(GLuint shader, GLenum type, GLint *param);
	static void extGlGetProgramiv(GLuint program, GLenum type, GLint *param);
	static GLint extGlGetUniformLocation(GLuint program, const char *name);
	static void extGlProgramUniform1fv(GLuint program, GLint loc, GLsizei count, const GLfloat *v);
	static void extGlProgramUniform2fv(GLuint program, GLint loc, GLsizei count, const GLfloat *v);
	static void extGlProgramUniform3fv(GLuint program, GLint loc, GLsizei count, const GLfloat *v);
	static void extGlProgramUniform4fv(GLuint program, GLint loc, GLsizei count, const GLfloat *v);
	static void extGlProgramUniform1iv(GLuint program, GLint loc, GLsizei count, const GLint *v);
	static void extGlProgramUniform2iv(GLuint program, GLint loc, GLsizei count, const GLint *v);
	static void extGlProgramUniform3iv(GLuint program, GLint loc, GLsizei count, const GLint *v);
	static void extGlProgramUniform4iv(GLuint program, GLint loc, GLsizei count, const GLint *v);
	static void extGlProgramUniform1uiv(GLuint program, GLint loc, GLsizei count, const GLuint *v);
	static void extGlProgramUniform2uiv(GLuint program, GLint loc, GLsizei count, const GLuint *v);
	static void extGlProgramUniform3uiv(GLuint program, GLint loc, GLsizei count, const GLuint *v);
	static void extGlProgramUniform4uiv(GLuint program, GLint loc, GLsizei count, const GLuint *v);
	static void extGlProgramUniformMatrix2fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v);
	static void extGlProgramUniformMatrix3fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v);
	static void extGlProgramUniformMatrix4fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v);
	static void extGlProgramUniformMatrix2x3fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v);
	static void extGlProgramUniformMatrix2x4fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v);
	static void extGlProgramUniformMatrix3x2fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v);
	static void extGlProgramUniformMatrix3x4fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v);
	static void extGlProgramUniformMatrix4x2fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v);
	static void extGlProgramUniformMatrix4x3fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v);
	static void extGlGetActiveUniform(GLuint program, GLuint index, GLsizei maxlength, GLsizei *length, GLint *size, GLenum *type, GLchar *name);
	static void extGlBindProgramPipeline(GLuint pipeline);
    static void extGlGetProgramBinary(GLuint program, GLsizei bufSize, GLsizei* length, GLenum* binaryFormat, void* binary);
    static void extGlProgramBinary(GLuint program, GLenum binaryFormat, const void* binary, GLsizei length);

	//compute
    static void extGlMemoryBarrier(GLbitfield barriers);
    static void extGlDispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z);
    static void extGlDispatchComputeIndirect(GLintptr indirect);

	// framebuffer objects
	static void extGlDeleteFramebuffers(GLsizei n, const GLuint *framebuffers);
	static void extGlCreateFramebuffers(GLsizei n, GLuint *framebuffers);
	static void extGlBindFramebuffer(GLenum target, GLuint framebuffer);
	static GLenum extGlCheckNamedFramebufferStatus(GLuint framebuffer, GLenum target);
	static void extGlNamedFramebufferTexture(GLuint framebuffer, GLenum attachment, GLuint texture, GLint level);
	static void extGlNamedFramebufferTextureLayer(GLuint framebuffer, GLenum attachment, GLuint texture, GLenum textureType, GLint level, GLint layer);
	static void extGlBlitNamedFramebuffer(GLuint readFramebuffer, GLuint drawFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);
    static void extGlNamedFramebufferReadBuffer(GLuint framebuffer, GLenum mode);
	static void extGlNamedFramebufferDrawBuffers(GLuint framebuffer, GLsizei n, const GLenum *bufs);
	static void extGlNamedFramebufferDrawBuffer(GLuint framebuffer, GLenum buf);
	static void extGlClearNamedFramebufferiv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLint* value);
	static void extGlClearNamedFramebufferuiv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLuint* value);
	static void extGlClearNamedFramebufferfv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLfloat* value);
	static void extGlClearNamedFramebufferfi(GLuint framebuffer, GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil);

	static void extGlActiveStencilFace(GLenum face);

	// vertex buffer object
	static void extGlCreateBuffers(GLsizei n, GLuint *buffers);
	static void extGlBindBuffer(const GLenum& target, const GLuint& buffer);
    static void extGlBindBuffersBase(const GLenum& target, const GLuint& first, const GLsizei& count, const GLuint* buffers);
    static void extGlBindBuffersRange(const GLenum& target, const GLuint& first, const GLsizei& count, const GLuint* buffers, const GLintptr* offsets, const GLsizeiptr* sizes);
	static void extGlDeleteBuffers(GLsizei n, const GLuint *buffers);
    static void extGlNamedBufferStorage(GLuint buffer, GLsizeiptr size, const void *data, GLbitfield flags);
	static void extGlNamedBufferSubData (GLuint buffer, GLintptr offset, GLsizeiptr size, const void *data);
	static void extGlGetNamedBufferSubData (GLuint buffer, GLintptr offset, GLsizeiptr size, void *data);
    static void* extGlMapNamedBuffer(GLuint buffer, GLbitfield access);
    static void* extGlMapNamedBufferRange(GLuint buffer, GLintptr offset, GLsizeiptr length, GLbitfield access);
    static void extGlFlushMappedNamedBufferRange(GLuint buffer, GLintptr offset, GLsizeiptr length);
    static GLboolean extGlUnmapNamedBuffer(GLuint buffer);
    static void extGlClearNamedBufferData(GLuint buffer, GLenum internalformat, GLenum format, GLenum type, const void *data);
    static void extGlClearNamedBufferSubData(GLuint buffer, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format, GLenum type, const void *data);
    static void extGlCopyNamedBufferSubData(GLuint readBuffer, GLuint writeBuffer, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size);
	static GLboolean extGlIsBuffer (GLuint buffer);
	static void extGlGetNamedBufferParameteriv(const GLuint& buffer, const GLenum& value, GLint* data);
	static void extGlGetNamedBufferParameteri64v(const GLuint& buffer, const GLenum& value, GLint64* data);

	//vao
	static void extGlCreateVertexArrays(GLsizei n, GLuint *arrays);
	static void extGlDeleteVertexArrays(GLsizei n, GLuint *arrays);
    static void extGlBindVertexArray(GLuint vaobj);
	static void extGlVertexArrayElementBuffer(GLuint vaobj, GLuint buffer);
	static void extGlVertexArrayVertexBuffer(GLuint vaobj, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride);
	static void extGlVertexArrayAttribBinding(GLuint vaobj, GLuint attribindex, GLuint bindingindex);
	static void extGlEnableVertexArrayAttrib(GLuint vaobj, GLuint index);
	static void extGlDisableVertexArrayAttrib(GLuint vaobj, GLuint index);
	static void extGlVertexArrayAttribFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset);
	static void extGlVertexArrayAttribIFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset);
	static void extGlVertexArrayAttribLFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset);
	static void extGlVertexArrayBindingDivisor(GLuint vaobj, GLuint bindingindex, GLuint divisor);

	//transform feedback
	static void extGlCreateTransformFeedbacks(GLsizei n, GLuint* ids);
    static void extGlDeleteTransformFeedbacks(GLsizei n, const GLuint* ids);
    static void extGlBindTransformFeedback(GLenum target, GLuint id);
    static void extGlBeginTransformFeedback(GLenum primitiveMode);
    static void extGlPauseTransformFeedback();
    static void extGlResumeTransformFeedback();
    static void extGlEndTransformFeedback();
    static void extGlTransformFeedbackBufferBase(GLuint xfb, GLuint index, GLuint buffer);
    static void extGlTransformFeedbackBufferRange(GLuint xfb, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size);


    //draw
    static void extGlPrimitiveRestartIndex(GLuint index);
    static void extGlDrawArraysInstanced(GLenum mode, GLint first, GLsizei count, GLsizei instancecount);
    static void extGlDrawArraysInstancedBaseInstance(GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance);
	static void extGlDrawElementsInstancedBaseVertex(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex);
	static void extGlDrawElementsInstancedBaseVertexBaseInstance(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance);
    static void extGlDrawTransformFeedback(GLenum mode, GLuint id);
    static void extGlDrawTransformFeedbackInstanced(GLenum mode, GLuint id, GLsizei instancecount);
    static void extGlDrawTransformFeedbackStream(GLenum mode, GLuint id, GLuint stream);
    static void extGlDrawTransformFeedbackStreamInstanced(GLenum mode, GLuint id, GLuint stream, GLsizei instancecount);
    static void extGlDrawArraysIndirect(GLenum mode, const void* indirect);
    static void extGlDrawElementsIndirect(GLenum mode, GLenum type, const void *indirect);
    static void extGlMultiDrawArraysIndirect(GLenum mode, const void* indirect, GLsizei drawcount, GLsizei stride);
    static void extGlMultiDrawElementsIndirect(GLenum mode, GLenum type, const void *indirect, GLsizei drawcount, GLsizei stride);
    static void extGlMultiDrawArraysIndirectCount(GLenum mode, const void *indirect, GLintptr drawcount, GLintptr maxdrawcount, GLsizei stride);
    static void extGlMultiDrawElementsIndirectCount(GLenum mode, GLenum type, const void *indirect, GLintptr drawcount, GLintptr maxdrawcount, GLsizei stride);

	// ROP
	static void extGlBlendColor(float red, float green, float blue, float alpha);
    static void extGlDepthRangeIndexed(GLuint index, GLdouble nearVal, GLdouble farVal);
    static void extGlViewportIndexedfv(GLuint index, const GLfloat* v);
    static void extGlScissorIndexedv(GLuint index, const GLint* v);
    static void extGlSampleCoverage(float value, bool invert);
    static void extGlSampleMaski(GLuint maskNumber, GLbitfield mask);
    static void extGlMinSampleShading(float value);
    static void extGlBlendEquationSeparatei(GLuint buf, GLenum modeRGB, GLenum modeAlpha);
    static void extGlBlendFuncSeparatei(GLuint buf, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha);
    static void extGlColorMaski(GLuint buf, GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha);

	//
	static void extGlBlendFuncSeparate(GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha);
	static void extGlColorMaskIndexed(GLuint buf, GLboolean r, GLboolean g, GLboolean b, GLboolean a);
	static void extGlEnableIndexed(GLenum target, GLuint index);
	static void extGlDisableIndexed(GLenum target, GLuint index);
	static void extGlBlendFuncIndexed(GLuint buf, GLenum src, GLenum dst);
	static void extGlBlendEquationIndexed(GLuint buf, GLenum mode);
	static void extGlProgramParameteri(GLuint program, GLenum pname, GLint value);
	static void extGlPatchParameterfv(GLenum pname, const float* values);
    static void extGlPatchParameteri(GLenum pname, GLuint value);

	// queries
	static void extGlCreateQueries(GLenum target, GLsizei n, GLuint *ids);
	static void extGlDeleteQueries(GLsizei n, const GLuint *ids);
	static GLboolean extGlIsQuery(GLuint id);
	static void extGlBeginQuery(GLenum target, GLuint id);
	static void extGlEndQuery(GLenum target);
	static void extGlBeginQueryIndexed(GLenum target, GLuint index, GLuint id);
	static void extGlEndQueryIndexed(GLenum target, GLuint index);
	static void extGlGetQueryObjectuiv(GLuint id, GLenum pname, GLuint *params);
	static void extGlGetQueryObjectui64v(GLuint id, GLenum pname, GLuint64 *params);
	static void extGlGetQueryBufferObjectuiv(GLuint id, GLuint buffer, GLenum pname, GLintptr offset);
	static void extGlGetQueryBufferObjectui64v(GLuint id, GLuint buffer, GLenum pname, GLintptr offset);
	static void extGlQueryCounter(GLuint id, GLenum target);
	static void extGlBeginConditionalRender(GLuint id, GLenum mode);
	static void extGlEndConditionalRender();

	//
	static void extGlTextureBarrier();

	// generic vsync setting method for several extensions
	static void extGlSwapInterval(int interval);

	// blend operations
	static void extGlBlendEquation(GLenum mode);

    // ARB_internalformat_query
    static void extGlGetInternalformativ(GLenum target, GLenum internalformat, GLenum pname, GLsizei bufSize, GLint* params);
    static void extGlGetInternalformati64v(GLenum target, GLenum internalformat, GLenum pname, GLsizei bufSize, GLint64* params);

	// the global feature array
	static bool FeatureAvailable[NBL_OpenGL_Feature_Count];


    //
    static PFNGLISENABLEDIPROC pGlIsEnabledi;
    static PFNGLENABLEIPROC pGlEnablei;
    static PFNGLDISABLEIPROC pGlDisablei;
    static PFNGLGETBOOLEANI_VPROC pGlGetBooleani_v;
    static PFNGLGETFLOATI_VPROC pGlGetFloati_v;
    static PFNGLGETINTEGER64VPROC pGlGetInteger64v;
    static PFNGLGETINTEGERI_VPROC pGlGetIntegeri_v;
    static PFNGLGETSTRINGIPROC pGlGetStringi;
    static PFNGLPROVOKINGVERTEXPROC pGlProvokingVertex;
    static PFNGLCLIPCONTROLPROC pGlClipControl;

    //fences
    static PFNGLFENCESYNCPROC pGlFenceSync;
    static PFNGLDELETESYNCPROC pGlDeleteSync;
    static PFNGLCLIENTWAITSYNCPROC pGlClientWaitSync;
    static PFNGLWAITSYNCPROC pGlWaitSync;

    //textures
    static PFNGLACTIVETEXTUREPROC pGlActiveTexture;
    static PFNGLBINDTEXTURESPROC pGlBindTextures; //NULL
    static PFNGLCREATETEXTURESPROC pGlCreateTextures; //NULL
    static PFNGLTEXSTORAGE1DPROC pGlTexStorage1D;
    static PFNGLTEXSTORAGE2DPROC pGlTexStorage2D;
    static PFNGLTEXSTORAGE3DPROC pGlTexStorage3D;
    static PFNGLTEXSTORAGE2DMULTISAMPLEPROC pGlTexStorage2DMultisample;
    static PFNGLTEXSTORAGE3DMULTISAMPLEPROC pGlTexStorage3DMultisample;
    static PFNGLTEXBUFFERPROC pGlTexBuffer;
    static PFNGLTEXBUFFERRANGEPROC pGlTexBufferRange;
    static PFNGLTEXTURESTORAGE1DPROC pGlTextureStorage1D; //NULL
    static PFNGLTEXTURESTORAGE2DPROC pGlTextureStorage2D; //NULL
    static PFNGLTEXTURESTORAGE3DPROC pGlTextureStorage3D; //NULL
    static PFNGLTEXTURESTORAGE2DMULTISAMPLEPROC pGlTextureStorage2DMultisample;
    static PFNGLTEXTURESTORAGE3DMULTISAMPLEPROC pGlTextureStorage3DMultisample;
	static PFNGLTEXTUREVIEWPROC pGlTextureView;
    static PFNGLTEXTUREBUFFERPROC pGlTextureBuffer; //NULL
    static PFNGLTEXTUREBUFFERRANGEPROC pGlTextureBufferRange; //NULL
    static PFNGLTEXTURESTORAGE1DEXTPROC pGlTextureStorage1DEXT;
    static PFNGLTEXTURESTORAGE2DEXTPROC pGlTextureStorage2DEXT;
    static PFNGLTEXTURESTORAGE3DEXTPROC pGlTextureStorage3DEXT;
    static PFNGLTEXTUREBUFFEREXTPROC pGlTextureBufferEXT;
    static PFNGLTEXTUREBUFFERRANGEEXTPROC pGlTextureBufferRangeEXT;
    static PFNGLTEXTURESTORAGE2DMULTISAMPLEEXTPROC pGlTextureStorage2DMultisampleEXT;
    static PFNGLTEXTURESTORAGE3DMULTISAMPLEEXTPROC pGlTextureStorage3DMultisampleEXT;
	static PFNGLGETTEXTURESUBIMAGEPROC pGlGetTextureSubImage;
	static PFNGLGETCOMPRESSEDTEXTURESUBIMAGEPROC pGlGetCompressedTextureSubImage;
	static PFNGLGETTEXTUREIMAGEPROC pGlGetTextureImage;
	static PFNGLGETTEXTUREIMAGEEXTPROC pGlGetTextureImageEXT;
	static PFNGLGETCOMPRESSEDTEXTUREIMAGEPROC pGlGetCompressedTextureImage;
	static PFNGLGETCOMPRESSEDTEXTUREIMAGEEXTPROC pGlGetCompressedTextureImageEXT;
	static PFNGLGETCOMPRESSEDTEXIMAGEPROC pGlGetCompressedTexImage;
    static PFNGLTEXSUBIMAGE3DPROC pGlTexSubImage3D;
    static PFNGLMULTITEXSUBIMAGE1DEXTPROC pGlMultiTexSubImage1DEXT;
    static PFNGLMULTITEXSUBIMAGE2DEXTPROC pGlMultiTexSubImage2DEXT;
    static PFNGLMULTITEXSUBIMAGE3DEXTPROC pGlMultiTexSubImage3DEXT;
    static PFNGLTEXTURESUBIMAGE1DPROC pGlTextureSubImage1D; //NULL
    static PFNGLTEXTURESUBIMAGE2DPROC pGlTextureSubImage2D; //NULL
    static PFNGLTEXTURESUBIMAGE3DPROC pGlTextureSubImage3D; //NULL
    static PFNGLTEXTURESUBIMAGE1DEXTPROC pGlTextureSubImage1DEXT;
    static PFNGLTEXTURESUBIMAGE2DEXTPROC pGlTextureSubImage2DEXT;
    static PFNGLTEXTURESUBIMAGE3DEXTPROC pGlTextureSubImage3DEXT;
    static PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC pGlCompressedTexSubImage1D;
    static PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC pGlCompressedTexSubImage2D;
    static PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC pGlCompressedTexSubImage3D;
    static PFNGLCOMPRESSEDTEXTURESUBIMAGE1DPROC pGlCompressedTextureSubImage1D; //NULL
    static PFNGLCOMPRESSEDTEXTURESUBIMAGE2DPROC pGlCompressedTextureSubImage2D; //NULL
    static PFNGLCOMPRESSEDTEXTURESUBIMAGE3DPROC pGlCompressedTextureSubImage3D; //NULL
    static PFNGLCOMPRESSEDTEXTURESUBIMAGE1DEXTPROC pGlCompressedTextureSubImage1DEXT;
    static PFNGLCOMPRESSEDTEXTURESUBIMAGE2DEXTPROC pGlCompressedTextureSubImage2DEXT;
    static PFNGLCOMPRESSEDTEXTURESUBIMAGE3DEXTPROC pGlCompressedTextureSubImage3DEXT;
    static PFNGLCOPYIMAGESUBDATAPROC pGlCopyImageSubData;
	static PFNGLTEXTUREPARAMETERIUIVPROC pGlTextureParameterIuiv;
	static PFNGLTEXTUREPARAMETERIUIVEXTPROC pGlTextureParameterIuivEXT;
	static PFNGLTEXPARAMETERIUIVPROC pGlTexParameterIuiv;
    static PFNGLGENERATEMIPMAPPROC pGlGenerateMipmap;
    static PFNGLGENERATETEXTUREMIPMAPPROC pGlGenerateTextureMipmap; //NULL
    static PFNGLGENERATETEXTUREMIPMAPEXTPROC pGlGenerateTextureMipmapEXT;
    static PFNGLCLAMPCOLORPROC pGlClampColor;

    //samplers
    static PFNGLGENSAMPLERSPROC pGlGenSamplers;
    static PFNGLCREATESAMPLERSPROC pGlCreateSamplers;
    static PFNGLDELETESAMPLERSPROC pGlDeleteSamplers;
    static PFNGLBINDSAMPLERPROC pGlBindSampler;
    static PFNGLBINDSAMPLERSPROC pGlBindSamplers;
    static PFNGLSAMPLERPARAMETERIPROC pGlSamplerParameteri;
    static PFNGLSAMPLERPARAMETERFPROC pGlSamplerParameterf;
    static PFNGLSAMPLERPARAMETERFVPROC pGlSamplerParameterfv;

    //
    static PFNGLBINDIMAGETEXTUREPROC pGlBindImageTexture;
    static PFNGLBINDIMAGETEXTURESPROC pGlBindImageTextures;
    
    //bindless textures
    //ARB
    static PFNGLGETTEXTUREHANDLEARBPROC pGlGetTextureHandleARB;
    static PFNGLGETTEXTURESAMPLERHANDLEARBPROC pGlGetTextureSamplerHandleARB;
    static PFNGLMAKETEXTUREHANDLERESIDENTARBPROC pGlMakeTextureHandleResidentARB;
    static PFNGLMAKETEXTUREHANDLENONRESIDENTARBPROC pGlMakeTextureHandleNonResidentARB;
    static PFNGLGETIMAGEHANDLEARBPROC pGlGetImageHandleARB;
    static PFNGLMAKEIMAGEHANDLERESIDENTARBPROC pGlMakeImageHandleResidentARB;
    static PFNGLMAKEIMAGEHANDLENONRESIDENTARBPROC pGlMakeImageHandleNonResidentARB;
    static PFNGLISTEXTUREHANDLERESIDENTARBPROC pGlIsTextureHandleResidentARB;
    static PFNGLISIMAGEHANDLERESIDENTARBPROC pGlIsImageHandleResidentARB;
    //NV
    static PFNGLGETTEXTUREHANDLENVPROC pGlGetTextureHandleNV;
    static PFNGLGETTEXTURESAMPLERHANDLENVPROC pGlGetTextureSamplerHandleNV;
    static PFNGLMAKETEXTUREHANDLERESIDENTNVPROC pGlMakeTextureHandleResidentNV;
    static PFNGLMAKETEXTUREHANDLENONRESIDENTNVPROC pGlMakeTextureHandleNonResidentNV;
    static PFNGLGETIMAGEHANDLENVPROC pGlGetImageHandleNV;
    static PFNGLMAKEIMAGEHANDLERESIDENTNVPROC pGlMakeImageHandleResidentNV;
    static PFNGLMAKEIMAGEHANDLENONRESIDENTNVPROC pGlMakeImageHandleNonResidentNV;
    static PFNGLISTEXTUREHANDLERESIDENTNVPROC pGlIsTextureHandleResidentNV;
    static PFNGLISIMAGEHANDLERESIDENTNVPROC pGlIsImageHandleResidentNV;

    // stuff
    static PFNGLBINDBUFFERBASEPROC pGlBindBufferBase;
    static PFNGLBINDBUFFERRANGEPROC pGlBindBufferRange;
    static PFNGLBINDBUFFERSBASEPROC pGlBindBuffersBase;
    static PFNGLBINDBUFFERSRANGEPROC pGlBindBuffersRange;

    //shaders
    static PFNGLCREATEPROGRAMPIPELINESPROC pGlCreateProgramPipelines;
    static PFNGLDELETEPROGRAMPIPELINESPROC pGlDeleteProgramPipelines;
    static PFNGLUSEPROGRAMSTAGESPROC pGlUseProgramStages;
    static PFNGLBINDATTRIBLOCATIONPROC pGlBindAttribLocation; //NULL
    static PFNGLCREATEPROGRAMPROC pGlCreateProgram;
    static PFNGLUSEPROGRAMPROC pGlUseProgram;
    static PFNGLDELETEPROGRAMPROC pGlDeleteProgram;
    static PFNGLDELETESHADERPROC pGlDeleteShader;
    static PFNGLGETATTACHEDSHADERSPROC pGlGetAttachedShaders;
    static PFNGLCREATESHADERPROC pGlCreateShader;
    static PFNGLCREATESHADERPROGRAMVPROC pGlCreateShaderProgramv;
    static PFNGLSHADERSOURCEPROC pGlShaderSource;
    static PFNGLCOMPILESHADERPROC pGlCompileShader;
    static PFNGLATTACHSHADERPROC pGlAttachShader;
    static PFNGLTRANSFORMFEEDBACKVARYINGSPROC pGlTransformFeedbackVaryings;
    static PFNGLLINKPROGRAMPROC pGlLinkProgram;
    static PFNGLGETSHADERINFOLOGPROC pGlGetShaderInfoLog;
    static PFNGLGETPROGRAMINFOLOGPROC pGlGetProgramInfoLog;
    static PFNGLGETSHADERIVPROC pGlGetShaderiv;
    static PFNGLGETSHADERIVPROC pGlGetProgramiv;
    static PFNGLGETUNIFORMLOCATIONPROC pGlGetUniformLocation;
    static PFNGLPROGRAMUNIFORM1FVPROC pGlProgramUniform1fv;
    static PFNGLPROGRAMUNIFORM2FVPROC pGlProgramUniform2fv;
    static PFNGLPROGRAMUNIFORM3FVPROC pGlProgramUniform3fv;
    static PFNGLPROGRAMUNIFORM4FVPROC pGlProgramUniform4fv;
    static PFNGLPROGRAMUNIFORM1IVPROC pGlProgramUniform1iv;
    static PFNGLPROGRAMUNIFORM2IVPROC pGlProgramUniform2iv;
    static PFNGLPROGRAMUNIFORM3IVPROC pGlProgramUniform3iv;
    static PFNGLPROGRAMUNIFORM4IVPROC pGlProgramUniform4iv;
    static PFNGLPROGRAMUNIFORM1UIVPROC pGlProgramUniform1uiv;
    static PFNGLPROGRAMUNIFORM2UIVPROC pGlProgramUniform2uiv;
    static PFNGLPROGRAMUNIFORM3UIVPROC pGlProgramUniform3uiv;
    static PFNGLPROGRAMUNIFORM4UIVPROC pGlProgramUniform4uiv;
    static PFNGLPROGRAMUNIFORMMATRIX2FVPROC pGlProgramUniformMatrix2fv;
    static PFNGLPROGRAMUNIFORMMATRIX3FVPROC pGlProgramUniformMatrix3fv;
    static PFNGLPROGRAMUNIFORMMATRIX4FVPROC pGlProgramUniformMatrix4fv;
    static PFNGLPROGRAMUNIFORMMATRIX2X3FVPROC pGlProgramUniformMatrix2x3fv;
    static PFNGLPROGRAMUNIFORMMATRIX2X4FVPROC pGlProgramUniformMatrix2x4fv;
    static PFNGLPROGRAMUNIFORMMATRIX3X2FVPROC pGlProgramUniformMatrix3x2fv;
    static PFNGLPROGRAMUNIFORMMATRIX3X4FVPROC pGlProgramUniformMatrix3x4fv;
    static PFNGLPROGRAMUNIFORMMATRIX4X2FVPROC pGlProgramUniformMatrix4x2fv;
    static PFNGLPROGRAMUNIFORMMATRIX4X3FVPROC pGlProgramUniformMatrix4x3fv;
    static PFNGLGETACTIVEUNIFORMPROC pGlGetActiveUniform;
    static PFNGLPOINTPARAMETERFPROC  pGlPointParameterf;
    static PFNGLPOINTPARAMETERFVPROC pGlPointParameterfv;
    static PFNGLBINDPROGRAMPIPELINEPROC pGlBindProgramPipeline;
    static PFNGLGETPROGRAMBINARYPROC pGlGetProgramBinary;
    static PFNGLPROGRAMBINARYPROC pGlProgramBinary;

	// Compute
	static PFNGLMEMORYBARRIERPROC pGlMemoryBarrier;
	static PFNGLDISPATCHCOMPUTEPROC pGlDispatchCompute;
	static PFNGLDISPATCHCOMPUTEINDIRECTPROC pGlDispatchComputeIndirect;

    //ROP
	static PFNGLBLENDCOLORPROC pGlBlendColor;
    static PFNGLDEPTHRANGEINDEXEDPROC pGlDepthRangeIndexed;
    static PFNGLVIEWPORTINDEXEDFVPROC pGlViewportIndexedfv;
    static PFNGLSCISSORINDEXEDVPROC pGlScissorIndexedv;
    static PFNGLSAMPLECOVERAGEPROC pGlSampleCoverage;
	static PFNGLSAMPLEMASKIPROC pGlSampleMaski;
	static PFNGLMINSAMPLESHADINGPROC pGlMinSampleShading;
    static PFNGLBLENDEQUATIONSEPARATEIPROC pGlBlendEquationSeparatei;
    static PFNGLBLENDFUNCSEPARATEIPROC pGlBlendFuncSeparatei;
    static PFNGLCOLORMASKIPROC pGlColorMaski;
    static PFNGLSTENCILFUNCSEPARATEPROC pGlStencilFuncSeparate;
    static PFNGLSTENCILOPSEPARATEPROC pGlStencilOpSeparate;
    static PFNGLSTENCILMASKSEPARATEPROC pGlStencilMaskSeparate;

    // ARB framebuffer object
    static PFNGLBLITNAMEDFRAMEBUFFERPROC pGlBlitNamedFramebuffer; //NULL
    static PFNGLBLITFRAMEBUFFERPROC pGlBlitFramebuffer;
    static PFNGLDELETEFRAMEBUFFERSPROC pGlDeleteFramebuffers;
    static PFNGLCREATEFRAMEBUFFERSPROC pGlCreateFramebuffers; //NULL
    static PFNGLBINDFRAMEBUFFERPROC pGlBindFramebuffer;
    static PFNGLGENFRAMEBUFFERSPROC pGlGenFramebuffers;
    static PFNGLCHECKFRAMEBUFFERSTATUSPROC pGlCheckFramebufferStatus;
    static PFNGLCHECKNAMEDFRAMEBUFFERSTATUSPROC pGlCheckNamedFramebufferStatus; //NULL
    static PFNGLCHECKNAMEDFRAMEBUFFERSTATUSEXTPROC pGlCheckNamedFramebufferStatusEXT;
    static PFNGLFRAMEBUFFERTEXTUREPROC pGlFramebufferTexture;
    static PFNGLNAMEDFRAMEBUFFERTEXTUREPROC pGlNamedFramebufferTexture; //NULL
    static PFNGLNAMEDFRAMEBUFFERTEXTUREEXTPROC pGlNamedFramebufferTextureEXT;
    static PFNGLFRAMEBUFFERTEXTURELAYERPROC pGlFramebufferTextureLayer;
    static PFNGLNAMEDFRAMEBUFFERTEXTURELAYERPROC pGlNamedFramebufferTextureLayer; //NULL
    static PFNGLNAMEDFRAMEBUFFERTEXTURELAYEREXTPROC pGlNamedFramebufferTextureLayerEXT;
	static PFNGLFRAMEBUFFERTEXTURE2DPROC pGlFramebufferTexture2D;
	static PFNGLNAMEDFRAMEBUFFERTEXTURE2DEXTPROC pGlNamedFramebufferTexture2DEXT;

    //! REMOVE ALL BELOW
    // EXT framebuffer object
    static PFNGLACTIVESTENCILFACEEXTPROC pGlActiveStencilFaceEXT; //NULL
    static PFNGLNAMEDFRAMEBUFFERREADBUFFERPROC pGlNamedFramebufferReadBuffer; //NULL
    static PFNGLFRAMEBUFFERREADBUFFEREXTPROC pGlFramebufferReadBufferEXT;
    static PFNGLNAMEDFRAMEBUFFERDRAWBUFFERPROC pGlNamedFramebufferDrawBuffer; //NULL
    static PFNGLFRAMEBUFFERDRAWBUFFEREXTPROC pGlFramebufferDrawBufferEXT;
    static PFNGLDRAWBUFFERSPROC pGlDrawBuffers;
    static PFNGLNAMEDFRAMEBUFFERDRAWBUFFERSPROC pGlNamedFramebufferDrawBuffers; //NULL
    static PFNGLFRAMEBUFFERDRAWBUFFERSEXTPROC pGlFramebufferDrawBuffersEXT;
    static PFNGLCLEARNAMEDFRAMEBUFFERIVPROC pGlClearNamedFramebufferiv; //NULL
    static PFNGLCLEARNAMEDFRAMEBUFFERUIVPROC pGlClearNamedFramebufferuiv; //NULL
    static PFNGLCLEARNAMEDFRAMEBUFFERFVPROC pGlClearNamedFramebufferfv; //NULL
    static PFNGLCLEARNAMEDFRAMEBUFFERFIPROC pGlClearNamedFramebufferfi; //NULL
    static PFNGLCLEARBUFFERIVPROC pGlClearBufferiv;
    static PFNGLCLEARBUFFERUIVPROC pGlClearBufferuiv;
    static PFNGLCLEARBUFFERFVPROC pGlClearBufferfv;
    static PFNGLCLEARBUFFERFIPROC pGlClearBufferfi;
    //
    static PFNGLGENBUFFERSPROC pGlGenBuffers;
    static PFNGLCREATEBUFFERSPROC pGlCreateBuffers; //NULL
    static PFNGLBINDBUFFERPROC pGlBindBuffer;
    static PFNGLDELETEBUFFERSPROC pGlDeleteBuffers;
    static PFNGLBUFFERSTORAGEPROC pGlBufferStorage;
    static PFNGLNAMEDBUFFERSTORAGEPROC pGlNamedBufferStorage; //NULL
    static PFNGLNAMEDBUFFERSTORAGEEXTPROC pGlNamedBufferStorageEXT;
    static PFNGLBUFFERSUBDATAPROC pGlBufferSubData;
    static PFNGLNAMEDBUFFERSUBDATAPROC pGlNamedBufferSubData; //NULL
    static PFNGLNAMEDBUFFERSUBDATAEXTPROC pGlNamedBufferSubDataEXT;
    static PFNGLGETBUFFERSUBDATAPROC pGlGetBufferSubData;
    static PFNGLGETNAMEDBUFFERSUBDATAPROC pGlGetNamedBufferSubData; //NULL
    static PFNGLGETNAMEDBUFFERSUBDATAEXTPROC pGlGetNamedBufferSubDataEXT;
    static PFNGLMAPBUFFERPROC pGlMapBuffer;
    static PFNGLMAPNAMEDBUFFERPROC pGlMapNamedBuffer; //NULL
    static PFNGLMAPNAMEDBUFFEREXTPROC pGlMapNamedBufferEXT;
    static PFNGLMAPBUFFERRANGEPROC pGlMapBufferRange;
    static PFNGLMAPNAMEDBUFFERRANGEPROC pGlMapNamedBufferRange; //NULL
    static PFNGLMAPNAMEDBUFFERRANGEEXTPROC pGlMapNamedBufferRangeEXT;
    static PFNGLFLUSHMAPPEDBUFFERRANGEPROC pGlFlushMappedBufferRange;
    static PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEPROC pGlFlushMappedNamedBufferRange; //NULL
    static PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEEXTPROC pGlFlushMappedNamedBufferRangeEXT;
    static PFNGLUNMAPBUFFERPROC pGlUnmapBuffer;
    static PFNGLUNMAPNAMEDBUFFERPROC pGlUnmapNamedBuffer; //NULL
    static PFNGLUNMAPNAMEDBUFFEREXTPROC pGlUnmapNamedBufferEXT;
    static PFNGLCLEARBUFFERDATAPROC pGlClearBufferData;
    static PFNGLCLEARNAMEDBUFFERDATAPROC pGlClearNamedBufferData; //NULL
    static PFNGLCLEARNAMEDBUFFERDATAEXTPROC pGlClearNamedBufferDataEXT;
    static PFNGLCLEARBUFFERSUBDATAPROC pGlClearBufferSubData;
    static PFNGLCLEARNAMEDBUFFERSUBDATAPROC pGlClearNamedBufferSubData; //NULL
    static PFNGLCLEARNAMEDBUFFERSUBDATAEXTPROC pGlClearNamedBufferSubDataEXT;
    static PFNGLCOPYBUFFERSUBDATAPROC pGlCopyBufferSubData;
    static PFNGLCOPYNAMEDBUFFERSUBDATAPROC pGlCopyNamedBufferSubData; //NULL
    static PFNGLNAMEDCOPYBUFFERSUBDATAEXTPROC pGlNamedCopyBufferSubDataEXT;
    static PFNGLISBUFFERPROC pGlIsBuffer;
    static PFNGLGETNAMEDBUFFERPARAMETERI64VPROC pGlGetNamedBufferParameteri64v;
    static PFNGLGETBUFFERPARAMETERI64VPROC pGlGetBufferParameteri64v;
    static PFNGLGETNAMEDBUFFERPARAMETERIVPROC pGlGetNamedBufferParameteriv;
    static PFNGLGETNAMEDBUFFERPARAMETERIVEXTPROC pGlGetNamedBufferParameterivEXT;
    static PFNGLGETBUFFERPARAMETERIVPROC pGlGetBufferParameteriv;
    //vao
    static PFNGLGENVERTEXARRAYSPROC pGlGenVertexArrays;
    static PFNGLCREATEVERTEXARRAYSPROC pGlCreateVertexArrays; //NULL
    static PFNGLDELETEVERTEXARRAYSPROC pGlDeleteVertexArrays;
    static PFNGLBINDVERTEXARRAYPROC pGlBindVertexArray;
    static PFNGLVERTEXARRAYELEMENTBUFFERPROC pGlVertexArrayElementBuffer; //NULL
    static PFNGLBINDVERTEXBUFFERPROC pGlBindVertexBuffer;
    static PFNGLVERTEXARRAYVERTEXBUFFERPROC pGlVertexArrayVertexBuffer; //NULL
    static PFNGLVERTEXARRAYBINDVERTEXBUFFEREXTPROC pGlVertexArrayBindVertexBufferEXT;
    static PFNGLVERTEXATTRIBBINDINGPROC pGlVertexAttribBinding;
    static PFNGLVERTEXARRAYATTRIBBINDINGPROC pGlVertexArrayAttribBinding; //NULL
    static PFNGLVERTEXARRAYVERTEXATTRIBBINDINGEXTPROC pGlVertexArrayVertexAttribBindingEXT;
    static PFNGLENABLEVERTEXATTRIBARRAYPROC pGlEnableVertexAttribArray;
    static PFNGLENABLEVERTEXARRAYATTRIBPROC pGlEnableVertexArrayAttrib; //NULL
    static PFNGLENABLEVERTEXARRAYATTRIBEXTPROC pGlEnableVertexArrayAttribEXT;
    static PFNGLDISABLEVERTEXATTRIBARRAYPROC pGlDisableVertexAttribArray;
    static PFNGLDISABLEVERTEXARRAYATTRIBPROC pGlDisableVertexArrayAttrib; //NULL
    static PFNGLDISABLEVERTEXARRAYATTRIBEXTPROC pGlDisableVertexArrayAttribEXT;
    static PFNGLVERTEXATTRIBFORMATPROC pGlVertexAttribFormat;
    static PFNGLVERTEXATTRIBIFORMATPROC pGlVertexAttribIFormat;
    static PFNGLVERTEXATTRIBLFORMATPROC pGlVertexAttribLFormat;
    static PFNGLVERTEXARRAYATTRIBFORMATPROC pGlVertexArrayAttribFormat; //NULL
    static PFNGLVERTEXARRAYATTRIBIFORMATPROC pGlVertexArrayAttribIFormat; //NULL
    static PFNGLVERTEXARRAYATTRIBLFORMATPROC pGlVertexArrayAttribLFormat; //NULL
    static PFNGLVERTEXARRAYVERTEXATTRIBFORMATEXTPROC pGlVertexArrayVertexAttribFormatEXT;
    static PFNGLVERTEXARRAYVERTEXATTRIBIFORMATEXTPROC pGlVertexArrayVertexAttribIFormatEXT;
    static PFNGLVERTEXARRAYVERTEXATTRIBLFORMATEXTPROC pGlVertexArrayVertexAttribLFormatEXT;
    static PFNGLVERTEXARRAYBINDINGDIVISORPROC pGlVertexArrayBindingDivisor; //NULL
    static PFNGLVERTEXARRAYVERTEXBINDINGDIVISOREXTPROC pGlVertexArrayVertexBindingDivisorEXT;
    static PFNGLVERTEXBINDINGDIVISORPROC pGlVertexBindingDivisor;
    //
    static PFNGLPRIMITIVERESTARTINDEXPROC pGlPrimitiveRestartIndex;
    static PFNGLDRAWARRAYSINSTANCEDPROC pGlDrawArraysInstanced;
    static PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC pGlDrawArraysInstancedBaseInstance;
	static PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXPROC pGlDrawElementsInstancedBaseVertex;
	static PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXBASEINSTANCEPROC pGlDrawElementsInstancedBaseVertexBaseInstance;
	static PFNGLDRAWTRANSFORMFEEDBACKPROC pGlDrawTransformFeedback;
	static PFNGLDRAWTRANSFORMFEEDBACKINSTANCEDPROC pGlDrawTransformFeedbackInstanced;
	static PFNGLDRAWTRANSFORMFEEDBACKSTREAMPROC pGlDrawTransformFeedbackStream;
	static PFNGLDRAWTRANSFORMFEEDBACKSTREAMINSTANCEDPROC pGlDrawTransformFeedbackStreamInstanced;
	static PFNGLDRAWARRAYSINDIRECTPROC pGlDrawArraysIndirect;
	static PFNGLDRAWELEMENTSINDIRECTPROC pGlDrawElementsIndirect;
	static PFNGLMULTIDRAWARRAYSINDIRECTPROC pGlMultiDrawArraysIndirect;
	static PFNGLMULTIDRAWELEMENTSINDIRECTPROC pGlMultiDrawElementsIndirect;
    static PFNGLMULTIDRAWARRAYSINDIRECTCOUNTPROC pGlMultiDrawArrysIndirectCount;
    static PFNGLMULTIDRAWELEMENTSINDIRECTCOUNTPROC pGlMultiDrawElementsIndirectCount;
	//
	static PFNGLCREATETRANSFORMFEEDBACKSPROC pGlCreateTransformFeedbacks;
	static PFNGLGENTRANSFORMFEEDBACKSPROC pGlGenTransformFeedbacks;
	static PFNGLDELETETRANSFORMFEEDBACKSPROC pGlDeleteTransformFeedbacks;
	static PFNGLBINDTRANSFORMFEEDBACKPROC pGlBindTransformFeedback;
	static PFNGLBEGINTRANSFORMFEEDBACKPROC pGlBeginTransformFeedback;
	static PFNGLPAUSETRANSFORMFEEDBACKPROC pGlPauseTransformFeedback;
	static PFNGLRESUMETRANSFORMFEEDBACKPROC pGlResumeTransformFeedback;
	static PFNGLENDTRANSFORMFEEDBACKPROC pGlEndTransformFeedback;
	static PFNGLTRANSFORMFEEDBACKBUFFERBASEPROC pGlTransformFeedbackBufferBase;
	static PFNGLTRANSFORMFEEDBACKBUFFERRANGEPROC pGlTransformFeedbackBufferRange;

    static PFNGLGETINTERNALFORMATIVPROC pGlGetInternalformativ;
    static PFNGLGETINTERNALFORMATI64VPROC pGlGetInternalformati64v;

    //! REMOVE ALL BELOW
    static PFNGLBLENDFUNCSEPARATEPROC pGlBlendFuncSeparate;
    static PFNGLBLENDFUNCINDEXEDAMDPROC pGlBlendFuncIndexedAMD; //NULL
    static PFNGLBLENDFUNCIPROC pGlBlendFunciARB;
    static PFNGLBLENDEQUATIONINDEXEDAMDPROC pGlBlendEquationIndexedAMD; //NULL
    static PFNGLBLENDEQUATIONIPROC pGlBlendEquationiARB; //NULL
    //
    static PFNGLPROGRAMPARAMETERIPROC pGlProgramParameteri;
    static PFNGLPATCHPARAMETERIPROC pGlPatchParameteri;
    static PFNGLPATCHPARAMETERFVPROC pGlPatchParameterfv;
    //
    static PFNGLCREATEQUERIESPROC pGlCreateQueries;
    static PFNGLGENQUERIESPROC pGlGenQueries;
    static PFNGLDELETEQUERIESPROC pGlDeleteQueries;
    static PFNGLISQUERYPROC pGlIsQuery;
    static PFNGLBEGINQUERYPROC pGlBeginQuery;
    static PFNGLENDQUERYPROC pGlEndQuery;
    static PFNGLBEGINQUERYINDEXEDPROC pGlBeginQueryIndexed;
    static PFNGLENDQUERYINDEXEDPROC pGlEndQueryIndexed;
    static PFNGLGETQUERYIVPROC pGlGetQueryiv;
    static PFNGLGETQUERYOBJECTUIVPROC pGlGetQueryObjectuiv;
    static PFNGLGETQUERYOBJECTUI64VPROC pGlGetQueryObjectui64v;
    static PFNGLGETQUERYBUFFEROBJECTUIVPROC pGlGetQueryBufferObjectuiv;
    static PFNGLGETQUERYBUFFEROBJECTUI64VPROC pGlGetQueryBufferObjectui64v;
    static PFNGLQUERYCOUNTERPROC pGlQueryCounter;
    static PFNGLBEGINCONDITIONALRENDERPROC pGlBeginConditionalRender;
    static PFNGLENDCONDITIONALRENDERPROC pGlEndConditionalRender;
    //
    static PFNGLTEXTUREBARRIERPROC pGlTextureBarrier;
    static PFNGLTEXTUREBARRIERNVPROC pGlTextureBarrierNV;
    //
    static PFNGLBLENDEQUATIONEXTPROC pGlBlendEquationEXT;
    static PFNGLBLENDEQUATIONPROC pGlBlendEquation;

    // the following can stay also
    static PFNGLDEBUGMESSAGECONTROLPROC pGlDebugMessageControl;
    static PFNGLDEBUGMESSAGECONTROLARBPROC pGlDebugMessageControlARB;
    static PFNGLDEBUGMESSAGECALLBACKPROC pGlDebugMessageCallback;
    static PFNGLDEBUGMESSAGECALLBACKARBPROC pGlDebugMessageCallbackARB;

    // os specific stuff for swapchain
    #if defined(WGL_EXT_swap_control)
    static PFNWGLSWAPINTERVALEXTPROC pWglSwapIntervalEXT;
    #endif
    #if defined(GLX_SGI_swap_control)
    static PFNGLXSWAPINTERVALSGIPROC pGlxSwapIntervalSGI;
    #endif
    #if defined(GLX_EXT_swap_control)
    static PFNGLXSWAPINTERVALEXTPROC pGlxSwapIntervalEXT;
    #endif
    #if defined(GLX_MESA_swap_control)
    static PFNGLXSWAPINTERVALMESAPROC pGlxSwapIntervalMESA;
    #endif
protected:
	// constructor
	COpenGLExtensionHandler();
private:
    static bool functionsAlreadyLoaded;

    static int32_t pixelUnpackAlignment;
};




inline bool COpenGLExtensionHandler::extGlIsEnabledi(GLenum cap, GLuint index)
{
    return pGlIsEnabledi(cap,index);
}
inline void COpenGLExtensionHandler::extGlEnablei(GLenum cap, GLuint index)
{
    pGlEnablei(cap,index);
}
inline void COpenGLExtensionHandler::extGlDisablei(GLenum cap, GLuint index)
{
    pGlDisablei(cap,index);
}
inline void COpenGLExtensionHandler::extGlGetBooleani_v(GLenum pname, GLuint index, GLboolean* data)
{
    pGlGetBooleani_v(pname,index,data);
}
inline void COpenGLExtensionHandler::extGlGetFloati_v(GLenum pname, GLuint index, float* data)
{
    pGlGetFloati_v(pname,index,data);
}
inline void COpenGLExtensionHandler::extGlGetInteger64v(GLenum pname, GLint64* data)
{
	pGlGetInteger64v(pname, data);
}
inline void COpenGLExtensionHandler::extGlGetIntegeri_v(GLenum pname, GLuint index, GLint* data)
{
    pGlGetIntegeri_v(pname,index,data);
}
inline void COpenGLExtensionHandler::extGlProvokingVertex(GLenum provokeMode)
{
    pGlProvokingVertex(provokeMode);
}
inline void COpenGLExtensionHandler::extGlClipControl(GLenum origin, GLenum depth)
{
	pGlClipControl(origin,depth);
}



inline GLsync COpenGLExtensionHandler::extGlFenceSync(GLenum condition, GLbitfield flags)
{
	return pGlFenceSync(condition,flags);
}

inline void COpenGLExtensionHandler::extGlDeleteSync(GLsync sync)
{
	pGlDeleteSync(sync);
}

inline GLenum COpenGLExtensionHandler::extGlClientWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout)
{
	return pGlClientWaitSync(sync,flags,timeout);
}

inline void COpenGLExtensionHandler::extGlWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout)
{
	pGlWaitSync(sync,flags,timeout);
}

inline void COpenGLExtensionHandler::extGlActiveTexture(GLenum target) // kill this function in favour of multibind (but need to upgrade OpenGL tracker)
{
	pGlActiveTexture(target);
}

inline void COpenGLExtensionHandler::extGlBindTextures(const GLuint& first, const GLsizei& count, const GLuint* textures, const GLenum* targets)
{
    const GLenum supportedTargets[] = { GL_TEXTURE_1D,GL_TEXTURE_2D, // GL 1.x
                                        GL_TEXTURE_3D,GL_TEXTURE_RECTANGLE,GL_TEXTURE_CUBE_MAP, // GL 2.x
                                        GL_TEXTURE_1D_ARRAY,GL_TEXTURE_2D_ARRAY,GL_TEXTURE_BUFFER, // GL 3.x
                                        GL_TEXTURE_CUBE_MAP_ARRAY,GL_TEXTURE_2D_MULTISAMPLE,GL_TEXTURE_2D_MULTISAMPLE_ARRAY}; // GL 4.x

    if (Version>=440||FeatureAvailable[NBL_ARB_multi_bind])
		pGlBindTextures(first,count,textures);
    else
    {
        int32_t activeTex = 0;
        glGetIntegerv(GL_ACTIVE_TEXTURE,&activeTex);

        for (GLsizei i=0; i<count; i++)
        {
            GLuint texture = textures ? textures[i]:0;

            GLuint unit = first+i;
			pGlActiveTexture(GL_TEXTURE0+unit);

            if (texture)
                glBindTexture(targets[i],texture);
            else
            {
                for (size_t j=0; j<sizeof(supportedTargets)/sizeof(GLenum); j++)
                    glBindTexture(supportedTargets[j],0);
            }
        }

		pGlActiveTexture(activeTex);
    }
}

inline void COpenGLExtensionHandler::extGlCreateTextures(GLenum target, GLsizei n, GLuint *textures)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
		pGlCreateTextures(target,n,textures);
    else
        glGenTextures(n,textures);
}

inline void COpenGLExtensionHandler::extGlTextureBuffer(GLuint texture, GLenum internalformat, GLuint buffer)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
		pGlTextureBuffer(texture,internalformat,buffer);
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
		pGlTextureBufferEXT(texture,GL_TEXTURE_BUFFER,internalformat,buffer);
    else
    {
        GLint bound;
        glGetIntegerv(GL_TEXTURE_BINDING_BUFFER, &bound);
        glBindTexture(GL_TEXTURE_BUFFER, texture);
		pGlTexBuffer(GL_TEXTURE_BUFFER,internalformat,buffer);
        glBindTexture(GL_TEXTURE_BUFFER, bound);
    }
}

inline void COpenGLExtensionHandler::extGlTextureBufferRange(GLuint texture, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizei length)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {

        if (pGlTextureBufferRange)
            pGlTextureBufferRange(texture,internalformat,buffer,offset,length);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlTextureBufferRangeEXT)
            pGlTextureBufferRangeEXT(texture,GL_TEXTURE_BUFFER,internalformat,buffer,offset,length);
    }
    else
    {
        GLint bound;
        glGetIntegerv(GL_TEXTURE_BINDING_BUFFER, &bound);
        glBindTexture(GL_TEXTURE_BUFFER, texture);
        if (pGlTexBufferRange)
            pGlTexBufferRange(GL_TEXTURE_BUFFER,internalformat,buffer,offset,length);
        glBindTexture(GL_TEXTURE_BUFFER, bound);
    }
}

inline void COpenGLExtensionHandler::extGlTextureStorage1D(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlTextureStorage1D)
            pGlTextureStorage1D(texture,levels,internalformat,width);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlTextureStorage1DEXT)
            pGlTextureStorage1DEXT(texture,target,levels,internalformat,width);
    }
    else if (pGlTexStorage1D)
    {
        GLint bound;
        switch (target)
        {
            case GL_TEXTURE_1D:
                glGetIntegerv(GL_TEXTURE_BINDING_1D, &bound);
                break;
            default:
                os::Printer::log("DevSH would like to ask you what are you doing!!??\n",ELL_ERROR);
                return;
        }
        glBindTexture(target, texture);
        pGlTexStorage1D(target,levels,internalformat,width);
        glBindTexture(target, bound);
    }
}
inline void COpenGLExtensionHandler::extGlTextureStorage2D(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlTextureStorage2D)
            pGlTextureStorage2D(texture,levels,internalformat,width,height);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlTextureStorage2DEXT)
            pGlTextureStorage2DEXT(texture,target,levels,internalformat,width,height);
    }
    else if (pGlTexStorage2D)
    {
        GLint bound;
        switch (target)
        {
            case GL_TEXTURE_1D_ARRAY:
                glGetIntegerv(GL_TEXTURE_BINDING_1D_ARRAY, &bound);
                break;
            case GL_TEXTURE_2D:
                glGetIntegerv(GL_TEXTURE_BINDING_2D, &bound);
                break;
            case GL_TEXTURE_CUBE_MAP:
                glGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &bound);
                break;
            case GL_TEXTURE_RECTANGLE:
                glGetIntegerv(GL_TEXTURE_BINDING_RECTANGLE, &bound);
                break;
            default:
                os::Printer::log("DevSH would like to ask you what are you doing!!??\n",ELL_ERROR);
                return;
        }
        glBindTexture(target, texture);
        pGlTexStorage2D(target,levels,internalformat,width,height);
        glBindTexture(target, bound);
    }
}
inline void COpenGLExtensionHandler::extGlTextureStorage3D(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
		pGlTextureStorage3D(texture,levels,internalformat,width,height,depth);
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
		pGlTextureStorage3DEXT(texture,target,levels,internalformat,width,height,depth);
    else
    {
        GLint bound;
        switch (target)
        {
            case GL_TEXTURE_2D_ARRAY:
                glGetIntegerv(GL_TEXTURE_BINDING_2D_ARRAY, &bound);
                break;
            case GL_TEXTURE_3D:
                glGetIntegerv(GL_TEXTURE_BINDING_3D, &bound);
                break;
            case GL_TEXTURE_CUBE_MAP_ARRAY:
                glGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP_ARRAY, &bound);
                break;
            default:
                os::Printer::log("DevSH would like to ask you what are you doing!!??\n",ELL_ERROR);
                return;
        }
        glBindTexture(target, texture);
        pGlTexStorage3D(target,levels,internalformat,width,height,depth);
        glBindTexture(target, bound);
    }
}

inline void COpenGLExtensionHandler::extGlTextureStorage2DMultisample(GLuint texture, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
		pGlTextureStorage2DMultisample(texture,samples,internalformat,width,height,fixedsamplelocations);
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
		pGlTextureStorage2DMultisampleEXT(texture,target,samples,internalformat,width,height,fixedsamplelocations);
	else
    {
        GLint bound;
        if (target!=GL_TEXTURE_2D_MULTISAMPLE)
        {
            os::Printer::log("DevSH would like to ask you what are you doing!!??\n",ELL_ERROR);
            return;
        }
        else
            glGetIntegerv(GL_TEXTURE_BINDING_2D_MULTISAMPLE, &bound);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, texture);
        pGlTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE,samples,internalformat,width,height,fixedsamplelocations);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, bound);
    }
}
inline void COpenGLExtensionHandler::extGlTextureStorage3DMultisample(GLuint texture, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
		pGlTextureStorage3DMultisample(texture,samples,internalformat,width,height,depth,fixedsamplelocations);
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
		pGlTextureStorage3DMultisampleEXT(texture,target,samples,internalformat,width,height,depth,fixedsamplelocations);
	else
    {
        GLint bound;
        if (target!=GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
        {
            os::Printer::log("DevSH would like to ask you what are you doing!!??\n",ELL_ERROR);
            return;
        }
        else
            glGetIntegerv(GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY, &bound);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, texture);
        pGlTexStorage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY,samples,internalformat,width,height,depth,fixedsamplelocations);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, bound);
    }
}

inline void COpenGLExtensionHandler::extGlTextureView(GLuint texture, GLenum target, GLuint origtexture, GLenum internalformat, GLuint minlevel, GLuint numlevels, GLuint minlayer, GLuint numlayers)
{
	pGlTextureView(texture,target,origtexture,internalformat,minlevel,numlevels,minlayer,numlayers);
}

inline void COpenGLExtensionHandler::extGlGetTextureSubImage(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLsizei bufSize, void* pixels)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_get_texture_sub_image])
		pGlGetTextureSubImage(texture, level, xoffset, yoffset, zoffset, width, height, depth, format, type, bufSize, pixels);
#ifdef _NBL_DEBUG
	else
		os::Printer::log("EDF_GET_TEXTURE_SUB_IMAGE Not Available! Tell DevSH to implement!\n", ELL_ERROR);
#endif // _NBL_DEBUG
}

inline void COpenGLExtensionHandler::extGlGetCompressedTextureSubImage(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLsizei bufSize, void* pixels)
{
	if (Version >= 450 || FeatureAvailable[NBL_ARB_get_texture_sub_image])
		extGlGetCompressedTextureSubImage(texture, level, xoffset, yoffset, zoffset, width, height, depth, bufSize, pixels);
#ifdef _NBL_DEBUG
	else
		os::Printer::log("EDF_GET_TEXTURE_SUB_IMAGE Not Available! Tell DevSH to implement!\n", ELL_ERROR);
#endif // _NBL_DEBUG
}

inline void COpenGLExtensionHandler::extGlGetTextureImage(GLuint texture, GLenum target, GLint level, GLenum format, GLenum type, GLsizei bufSizeHint, void* pixels)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
		pGlGetTextureImage(texture, level, format, type, bufSizeHint, pixels);
	else if (FeatureAvailable[NBL_EXT_direct_state_access])
		pGlGetTextureImageEXT(texture, target, level, format, type, pixels);
    else
    {
        GLint bound = 0;
        switch (target)
        {
            case GL_TEXTURE_1D:
                glGetIntegerv(GL_TEXTURE_BINDING_1D, &bound);
                break;
			case GL_TEXTURE_1D_ARRAY:
				glGetIntegerv(GL_TEXTURE_BINDING_1D_ARRAY, &bound);
				break;
			case GL_TEXTURE_2D:
				glGetIntegerv(GL_TEXTURE_BINDING_2D, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
			case GL_TEXTURE_CUBE_MAP:
				glGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &bound);
				break;
			case GL_TEXTURE_RECTANGLE:
				glGetIntegerv(GL_TEXTURE_BINDING_RECTANGLE, &bound);
				break;
			case GL_TEXTURE_2D_ARRAY:
				glGetIntegerv(GL_TEXTURE_BINDING_2D_ARRAY, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP_ARRAY:
				glGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP_ARRAY, &bound);
				break;
			case GL_TEXTURE_3D:
				glGetIntegerv(GL_TEXTURE_BINDING_3D, &bound);
				break;
            default:
				break;
        }
        glBindTexture(target, texture);
		glGetTexImage(target, level, format, type, pixels);
        glBindTexture(target, bound);
    }
}

inline void COpenGLExtensionHandler::extGlGetCompressedTextureImage(GLuint texture, GLenum target, GLint level, GLsizei bufSizeHint, void* pixels)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
		pGlGetCompressedTextureImage(texture, level, bufSizeHint, pixels);
	else if (FeatureAvailable[NBL_EXT_direct_state_access])
		pGlGetCompressedTextureImageEXT(texture, target, level, pixels);
    else
    {
        GLint bound = 0;
        switch (target)
        {
            case GL_TEXTURE_1D:
                glGetIntegerv(GL_TEXTURE_BINDING_1D, &bound);
                break;
			case GL_TEXTURE_1D_ARRAY:
				glGetIntegerv(GL_TEXTURE_BINDING_1D_ARRAY, &bound);
				break;
			case GL_TEXTURE_2D:
				glGetIntegerv(GL_TEXTURE_BINDING_2D, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
			case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
			case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
			case GL_TEXTURE_CUBE_MAP:
				glGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &bound);
				break;
			case GL_TEXTURE_RECTANGLE:
				glGetIntegerv(GL_TEXTURE_BINDING_RECTANGLE, &bound);
				break;
			case GL_TEXTURE_2D_ARRAY:
				glGetIntegerv(GL_TEXTURE_BINDING_2D_ARRAY, &bound);
				break;
			case GL_TEXTURE_CUBE_MAP_ARRAY:
				glGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP_ARRAY, &bound);
				break;
			case GL_TEXTURE_3D:
				glGetIntegerv(GL_TEXTURE_BINDING_3D, &bound);
				break;
            default:
				break;
        }
        glBindTexture(target, texture);
		pGlGetCompressedTexImage(target, level, pixels);
        glBindTexture(target, bound);
    }
}

inline void COpenGLExtensionHandler::extGlTextureSubImage1D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
		pGlTextureSubImage1D(texture, level, xoffset, width,format, type, pixels);
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
		pGlTextureSubImage1DEXT(texture, target, level, xoffset, width,format, type, pixels);
    else
    {
        GLint bound;
        switch (target)
        {
            case GL_TEXTURE_1D:
                glGetIntegerv(GL_TEXTURE_BINDING_1D, &bound);
                break;
            default:
                os::Printer::log("DevSH would like to ask you what are you doing!!??\n",ELL_ERROR);
                return;
        }
        glBindTexture(target, texture);
        glTexSubImage1D(target, level, xoffset, width,format, type, pixels);
        glBindTexture(target, bound);
    }
}
inline void COpenGLExtensionHandler::extGlTextureSubImage2D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
		pGlTextureSubImage2D(texture, level, xoffset, yoffset,width, height,format, type, pixels);
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
		pGlTextureSubImage2DEXT(texture, target, level, xoffset, yoffset,width, height,format, type, pixels);
    else
    {
        GLint bound;
        switch (target)
        {
            case GL_TEXTURE_1D_ARRAY:
                glGetIntegerv(GL_TEXTURE_BINDING_1D_ARRAY, &bound);
                break;
            case GL_TEXTURE_2D:
                glGetIntegerv(GL_TEXTURE_BINDING_2D, &bound);
                break;
            case GL_TEXTURE_2D_MULTISAMPLE:
                glGetIntegerv(GL_TEXTURE_BINDING_2D_MULTISAMPLE, &bound);
                break;
            case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
            case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
            case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
            case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
            case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
                glGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &bound);
                break;
            case GL_TEXTURE_RECTANGLE:
                glGetIntegerv(GL_TEXTURE_BINDING_RECTANGLE, &bound);
                break;
            default:
                os::Printer::log("DevSH would like to ask you what are you doing!!??\n",ELL_ERROR);
                return;
        }
        glBindTexture(target, texture);
        glTexSubImage2D(target, level, xoffset, yoffset,width, height,format, type, pixels);
        glBindTexture(target, bound);
    }
}
inline void COpenGLExtensionHandler::extGlTextureSubImage3D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
		pGlTextureSubImage3D(texture, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
		pGlTextureSubImage3DEXT(texture, target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
    else
    {
        GLint bound;
        switch (target)
        {
            case GL_TEXTURE_2D_ARRAY:
                glGetIntegerv(GL_TEXTURE_BINDING_2D_ARRAY, &bound);
                break;
            case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
                glGetIntegerv(GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY, &bound);
                break;
            case GL_TEXTURE_3D:
                glGetIntegerv(GL_TEXTURE_BINDING_3D, &bound);
                break;
            case GL_TEXTURE_CUBE_MAP:
                glGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &bound);
                break;
            case GL_TEXTURE_CUBE_MAP_ARRAY:
                glGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP_ARRAY, &bound);
                break;
            default:
                os::Printer::log("DevSH would like to ask you what are you doing!!??\n",ELL_ERROR);
                return;
        }
        glBindTexture(target, texture);
        pGlTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
        glBindTexture(target, bound);
    }
}
inline void COpenGLExtensionHandler::extGlCompressedTextureSubImage1D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *data)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlCompressedTextureSubImage1D)
            pGlCompressedTextureSubImage1D(texture, level, xoffset, width,format, imageSize, data);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlCompressedTextureSubImage1DEXT)
            pGlCompressedTextureSubImage1DEXT(texture, target, level, xoffset, width,format, imageSize, data);
    }
    else if (pGlCompressedTexSubImage1D)
    {
        GLint bound;
        switch (target)
        {
            case GL_TEXTURE_1D:
                glGetIntegerv(GL_TEXTURE_BINDING_1D, &bound);
                break;
            default:
                os::Printer::log("DevSH would like to ask you what are you doing!!??\n",ELL_ERROR);
                return;
        }
        glBindTexture(target, texture);
        pGlCompressedTexSubImage1D(target, level, xoffset, width,format, imageSize, data);
        glBindTexture(target, bound);
    }
}
inline void COpenGLExtensionHandler::extGlCompressedTextureSubImage2D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlCompressedTextureSubImage2D)
            pGlCompressedTextureSubImage2D(texture, level, xoffset, yoffset,width, height,format, imageSize, data);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlCompressedTextureSubImage2DEXT)
            pGlCompressedTextureSubImage2DEXT(texture, target, level, xoffset, yoffset,width, height,format, imageSize, data);
    }
    else if (pGlCompressedTexSubImage2D)
    {
        GLint bound;
        switch (target)
        {
            case GL_TEXTURE_1D_ARRAY:
                glGetIntegerv(GL_TEXTURE_BINDING_1D_ARRAY, &bound);
                break;
            case GL_TEXTURE_2D:
                glGetIntegerv(GL_TEXTURE_BINDING_2D, &bound);
                break;
            case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
            case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
            case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
            case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
            case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
                glGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &bound);
                break;
            default:
                os::Printer::log("DevSH would like to ask you what are you doing!!??\n",ELL_ERROR);
                return;
        }
        glBindTexture(target, texture);
        pGlCompressedTexSubImage2D(target, level, xoffset, yoffset,width, height,format, imageSize, data);
        glBindTexture(target, bound);
    }
}
inline void COpenGLExtensionHandler::extGlCompressedTextureSubImage3D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *data)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlCompressedTextureSubImage3D)
            pGlCompressedTextureSubImage3D(texture, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlCompressedTextureSubImage3DEXT)
            pGlCompressedTextureSubImage3DEXT(texture, target, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
    }
    else if (pGlCompressedTexSubImage3D)
    {
        GLint bound;
        switch (target)
        {
            case GL_TEXTURE_2D_ARRAY:
                glGetIntegerv(GL_TEXTURE_BINDING_2D_ARRAY, &bound);
                break;
            case GL_TEXTURE_3D:
                glGetIntegerv(GL_TEXTURE_BINDING_3D, &bound);
                break;
            case GL_TEXTURE_CUBE_MAP:
                glGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &bound);
                break;
            case GL_TEXTURE_CUBE_MAP_ARRAY:
                glGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP_ARRAY, &bound);
                break;
            default:
                os::Printer::log("DevSH would like to ask you what are you doing!!??\n",ELL_ERROR);
                return;
        }
        glBindTexture(target, texture);
        pGlCompressedTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
        glBindTexture(target, bound);
    }
}

inline void COpenGLExtensionHandler::extGlCopyImageSubData(GLuint srcName, GLenum srcTarget, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, GLuint dstName, GLenum dstTarget, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei srcWidth, GLsizei srcHeight, GLsizei srcDepth)
{
	pGlCopyImageSubData(srcName,srcTarget,srcLevel,srcX,srcY,srcZ,dstName,dstTarget,dstLevel,dstX,dstY,dstZ,srcWidth,srcHeight,srcDepth);
}

inline void COpenGLExtensionHandler::extGlGenerateTextureMipmap(GLuint texture, GLenum target)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlGenerateTextureMipmap)
            pGlGenerateTextureMipmap(texture);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlGenerateTextureMipmapEXT)
            pGlGenerateTextureMipmapEXT(texture,target);
    }
    else
    {
        GLint bound;
        switch (target)
        {
            case GL_TEXTURE_1D:
                glGetIntegerv(GL_TEXTURE_BINDING_1D, &bound);
                break;
            case GL_TEXTURE_1D_ARRAY:
                glGetIntegerv(GL_TEXTURE_BINDING_1D_ARRAY, &bound);
                break;
            case GL_TEXTURE_2D:
                glGetIntegerv(GL_TEXTURE_BINDING_2D, &bound);
                break;
            case GL_TEXTURE_2D_ARRAY:
                glGetIntegerv(GL_TEXTURE_BINDING_2D_ARRAY, &bound);
                break;
            case GL_TEXTURE_2D_MULTISAMPLE:
                glGetIntegerv(GL_TEXTURE_BINDING_2D_MULTISAMPLE, &bound);
                break;
            case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
                glGetIntegerv(GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY, &bound);
                break;
            case GL_TEXTURE_3D:
                glGetIntegerv(GL_TEXTURE_BINDING_3D, &bound);
                break;
            case GL_TEXTURE_BUFFER:
                glGetIntegerv(GL_TEXTURE_BINDING_BUFFER, &bound);
                break;
            case GL_TEXTURE_CUBE_MAP:
                glGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &bound);
                break;
            case GL_TEXTURE_CUBE_MAP_ARRAY:
                glGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP_ARRAY, &bound);
                break;
            case GL_TEXTURE_RECTANGLE:
                glGetIntegerv(GL_TEXTURE_BINDING_RECTANGLE, &bound);
                break;
            default:
                os::Printer::log("DevSH would like to ask you what are you doing!!??\n",ELL_ERROR);
                return;
        }
        glBindTexture(target, texture);
        pGlGenerateMipmap(target);
        glBindTexture(target, bound);
    }
}

inline void COpenGLExtensionHandler::extGlTextureParameterIuiv(GLuint texture, GLenum target, GLenum pname, const GLuint* params)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        pGlTextureParameterIuiv(texture,pname,params);
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
		pGlTextureParameterIuivEXT(texture,target,pname,params);
	else
	{
        GLint bound;
        switch (target)
        {
            case GL_TEXTURE_1D:
                glGetIntegerv(GL_TEXTURE_BINDING_1D, &bound);
                break;
            case GL_TEXTURE_1D_ARRAY:
                glGetIntegerv(GL_TEXTURE_BINDING_1D_ARRAY, &bound);
                break;
            case GL_TEXTURE_2D:
                glGetIntegerv(GL_TEXTURE_BINDING_2D, &bound);
                break;
            case GL_TEXTURE_2D_ARRAY:
                glGetIntegerv(GL_TEXTURE_BINDING_2D_ARRAY, &bound);
                break;
            case GL_TEXTURE_2D_MULTISAMPLE:
                glGetIntegerv(GL_TEXTURE_BINDING_2D_MULTISAMPLE, &bound);
                break;
            case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
                glGetIntegerv(GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY, &bound);
                break;
            case GL_TEXTURE_3D:
                glGetIntegerv(GL_TEXTURE_BINDING_3D, &bound);
                break;
            case GL_TEXTURE_BUFFER:
                glGetIntegerv(GL_TEXTURE_BINDING_BUFFER, &bound);
                break;
            case GL_TEXTURE_CUBE_MAP:
                glGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP, &bound);
                break;
            case GL_TEXTURE_CUBE_MAP_ARRAY:
                glGetIntegerv(GL_TEXTURE_BINDING_CUBE_MAP_ARRAY, &bound);
                break;
            case GL_TEXTURE_RECTANGLE:
                glGetIntegerv(GL_TEXTURE_BINDING_RECTANGLE, &bound);
                break;
            default:
                os::Printer::log("DevSH would like to ask you what are you doing!!??\n",ELL_ERROR);
                return;
        }
        glBindTexture(target, texture);
		pGlTexParameterIuiv(target,pname,params);
        glBindTexture(target, bound);
	}
}

inline void COpenGLExtensionHandler::extGlClampColor(GLenum target, GLenum clamp)
{
    if (pGlClampColor)
        pGlClampColor(GL_CLAMP_READ_COLOR,clamp);
}

inline void COpenGLExtensionHandler::extGlCreateSamplers(GLsizei n, GLuint* samplers)
{
    if (pGlCreateSamplers)
        pGlCreateSamplers(n, samplers);
    else if (pGlGenSamplers)
        pGlGenSamplers(n, samplers);
    else memset(samplers, 0, 4*n);
}

inline void COpenGLExtensionHandler::extGlDeleteSamplers(GLsizei n, GLuint* samplers)
{
    if (pGlDeleteSamplers)
        pGlDeleteSamplers(n,samplers);
}

inline void COpenGLExtensionHandler::extGlBindSamplers(const GLuint& first, const GLsizei& count, const GLuint* samplers)
{
    if (Version>=440||FeatureAvailable[NBL_ARB_multi_bind])
    {
        if (pGlBindSamplers)
            pGlBindSamplers(first,count,samplers);
    }
    else
    {
        for (GLsizei i=0; i<count; i++)
        {
            GLuint unit = first+i;
            if (samplers)
            {
                if (pGlBindSampler)
                    pGlBindSampler(unit,samplers[i]);
            }
            else
            {
                if (pGlBindSampler)
                    pGlBindSampler(unit,0);
            }
        }
    }
}

inline void COpenGLExtensionHandler::extGlSamplerParameteri(GLuint sampler, GLenum pname, GLint param)
{
    if (pGlSamplerParameteri)
        pGlSamplerParameteri(sampler,pname,param);
}

inline void COpenGLExtensionHandler::extGlSamplerParameterf(GLuint sampler, GLenum pname, GLfloat param)
{
    if (pGlSamplerParameterf)
        pGlSamplerParameterf(sampler,pname,param);
}

inline void COpenGLExtensionHandler::extGlSamplerParameterfv(GLuint sampler, GLenum pname, const GLfloat* params)
{
    if (pGlSamplerParameterfv)
        pGlSamplerParameterfv(sampler, pname, params);
}


inline void COpenGLExtensionHandler::extGlBindImageTexture(GLuint index, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format)
{
    if (pGlBindImageTexture)
        pGlBindImageTexture(index,texture,level,layered,layer,access,format);
}


inline void COpenGLExtensionHandler::extGlBindImageTextures(GLuint first, GLsizei count, const GLuint* textures, const GLenum* formats)
{
    if (pGlBindImageTextures && !IsIntelGPU) // Intel is a special boy, as always
        pGlBindImageTextures(first, count, textures);
    else
    {
        for (GLsizei i=0; i<count; i++)
        {
            if (!textures || textures[i] == 0u)
                extGlBindImageTexture(first+i, 0u, 0u, GL_FALSE, 0, GL_READ_WRITE, GL_R8);
            else
                extGlBindImageTexture(first+i, textures[i], 0, GL_TRUE, 0, GL_READ_WRITE, formats[i]);
        }
    }
}

inline GLuint64 COpenGLExtensionHandler::extGlGetTextureHandle(GLuint texture)
{
    if (pGlGetTextureHandleARB)
        return pGlGetTextureHandleARB(texture);
    else if (pGlGetTextureHandleNV)
        return pGlGetTextureHandleNV(texture);
    return 0ull;
}

inline GLuint64 COpenGLExtensionHandler::extGlGetTextureSamplerHandle(GLuint texture, GLuint sampler)
{
    if (pGlGetTextureSamplerHandleARB)
        return pGlGetTextureSamplerHandleARB(texture, sampler);
    else if (pGlGetTextureSamplerHandleNV)
        return pGlGetTextureSamplerHandleNV(texture, sampler);
    return 0ull;
}

inline void COpenGLExtensionHandler::extGlMakeTextureHandleResident(GLuint64 handle)
{
    if (pGlMakeTextureHandleResidentARB)
        return pGlMakeTextureHandleResidentARB(handle);
    else if (pGlMakeTextureHandleResidentNV)
        return pGlMakeTextureHandleResidentNV(handle);
}

inline void COpenGLExtensionHandler::extGlMakeTextureHandleNonResident(GLuint64 handle)
{
    if (pGlMakeTextureHandleNonResidentARB)
        return pGlMakeTextureHandleNonResidentARB(handle);
    else if (pGlMakeTextureHandleNonResidentNV)
        return pGlMakeTextureHandleNonResidentNV(handle);
}

inline GLuint64 COpenGLExtensionHandler::extGlGetImageHandle(GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum format)
{
    if (pGlGetImageHandleARB)
        return pGlGetImageHandleARB(texture, level, layered, layer, format);
    else if (pGlGetImageHandleNV)
        return pGlGetImageHandleNV(texture, level, layered, layer, format);
    return 0ull;
}

inline void COpenGLExtensionHandler::extGlMakeImageHandleResident(GLuint64 handle, GLenum access)
{
    if (pGlMakeImageHandleResidentARB)
        return pGlMakeImageHandleResidentARB(handle, access);
    else if (pGlMakeImageHandleResidentNV)
        return pGlMakeImageHandleResidentNV(handle, access);
}

inline void COpenGLExtensionHandler::extGlMakeImageHandleNonResident(GLuint64 handle)
{
    if (pGlMakeImageHandleNonResidentARB)
        return pGlMakeImageHandleNonResidentARB(handle);
    else if (pGlMakeImageHandleNonResidentNV)
        return pGlMakeImageHandleNonResidentNV(handle);
}

inline GLboolean COpenGLExtensionHandler::extGlIsTextureHandleResident(GLuint64 handle)
{
    if (pGlIsTextureHandleResidentARB)
        return pGlIsTextureHandleResidentARB(handle);
    else if (pGlIsTextureHandleResidentNV)
        return pGlIsTextureHandleResidentNV(handle);
    return false;
}

inline GLboolean COpenGLExtensionHandler::extGlIsImageHandleResident(GLuint64 handle)
{
    if (pGlIsTextureHandleResidentARB)
        return pGlIsTextureHandleResidentARB(handle);
    else if (pGlIsTextureHandleResidentNV)
        return pGlIsTextureHandleResidentNV(handle);
    return false;
}


inline void COpenGLExtensionHandler::extGlCreateProgramPipelines(GLsizei n, GLuint * pipelines)
{
    if (pGlCreateProgramPipelines)
        pGlCreateProgramPipelines(n, pipelines);
}

inline void COpenGLExtensionHandler::extGlDeleteProgramPipelines(GLsizei n, const GLuint * pipelines)
{
    if (pGlDeleteProgramPipelines)
        pGlDeleteProgramPipelines(n, pipelines);
}

inline void COpenGLExtensionHandler::extGlUseProgramStages(GLuint pipeline, GLbitfield stages, GLuint program)
{
    if (pGlUseProgramStages)
        pGlUseProgramStages(pipeline, stages, program);
}

inline GLuint COpenGLExtensionHandler::extGlCreateShader(GLenum shaderType)
{
	if (pGlCreateShader)
		return pGlCreateShader(shaderType);
	return 0;
}

inline GLuint COpenGLExtensionHandler::extGlCreateShaderProgramv(GLenum shaderType, GLsizei count, const char** strings)
{
    if (pGlCreateShaderProgramv)
        return pGlCreateShaderProgramv(shaderType, count, strings);
    return 0;
}

inline void COpenGLExtensionHandler::extGlShaderSource(GLuint shader, GLsizei numOfStrings, const char **strings, const GLint *lenOfStrings)
{
	if (pGlShaderSource)
		pGlShaderSource(shader, numOfStrings, strings, lenOfStrings);
}

inline void COpenGLExtensionHandler::extGlCompileShader(GLuint shader)
{
	if (pGlCompileShader)
		pGlCompileShader(shader);
}

inline GLuint COpenGLExtensionHandler::extGlCreateProgram(void)
{
	if (pGlCreateProgram)
		return pGlCreateProgram();
	return 0;
}

inline void COpenGLExtensionHandler::extGlAttachShader(GLuint program, GLuint shader)
{
	if (pGlAttachShader)
		pGlAttachShader(program, shader);
}


inline void COpenGLExtensionHandler::extGlLinkProgram(GLuint program)
{
	if (pGlLinkProgram)
		pGlLinkProgram(program);
}

inline void COpenGLExtensionHandler::extGlTransformFeedbackVaryings(GLuint program, GLsizei count, const char** varyings, GLenum bufferMode)
{
	if (pGlTransformFeedbackVaryings)
        pGlTransformFeedbackVaryings(program,count,varyings,bufferMode);
}

inline void COpenGLExtensionHandler::extGlUseProgram(GLuint prog)
{
	if (pGlUseProgram)
		pGlUseProgram(prog);
}

inline void COpenGLExtensionHandler::extGlDeleteProgram(GLuint object)
{
	if (pGlDeleteProgram)
		pGlDeleteProgram(object);
}

inline void COpenGLExtensionHandler::extGlDeleteShader(GLuint shader)
{
	if (pGlDeleteShader)
		pGlDeleteShader(shader);
}

inline void COpenGLExtensionHandler::extGlGetAttachedShaders(GLuint program, GLsizei maxcount, GLsizei* count, GLuint* shaders)
{
	if (count)
		*count=0;
	if (pGlGetAttachedShaders)
		pGlGetAttachedShaders(program, maxcount, count, shaders);
}

inline void COpenGLExtensionHandler::extGlGetShaderInfoLog(GLuint shader, GLsizei maxLength, GLsizei *length, GLchar *infoLog)
{
	if (length)
		*length=0;
	if (pGlGetShaderInfoLog)
		pGlGetShaderInfoLog(shader, maxLength, length, infoLog);
}

inline void COpenGLExtensionHandler::extGlGetProgramInfoLog(GLuint program, GLsizei maxLength, GLsizei *length, GLchar *infoLog)
{
	if (length)
		*length=0;
	if (pGlGetProgramInfoLog)
		pGlGetProgramInfoLog(program, maxLength, length, infoLog);
}


inline void COpenGLExtensionHandler::extGlGetShaderiv(GLuint shader, GLenum type, GLint *param)
{
	if (pGlGetShaderiv)
		pGlGetShaderiv(shader, type, param);
}

inline void COpenGLExtensionHandler::extGlGetProgramiv(GLuint program, GLenum type, GLint *param)
{
	if (pGlGetProgramiv)
		pGlGetProgramiv(program, type, param);
}

inline GLint COpenGLExtensionHandler::extGlGetUniformLocation(GLuint program, const char *name)
{
	if (pGlGetUniformLocation)
		return pGlGetUniformLocation(program, name);
	return -1;
}

inline void COpenGLExtensionHandler::extGlProgramUniform1fv(GLuint program, GLint loc, GLsizei count, const GLfloat *v)
{
	if (pGlProgramUniform1fv)
		pGlProgramUniform1fv(program, loc, count, v);
}

inline void COpenGLExtensionHandler::extGlProgramUniform2fv(GLuint program, GLint loc, GLsizei count, const GLfloat *v)
{
	if (pGlProgramUniform2fv)
		pGlProgramUniform2fv(program, loc, count, v);
}

inline void COpenGLExtensionHandler::extGlProgramUniform3fv(GLuint program, GLint loc, GLsizei count, const GLfloat *v)
{
	if (pGlProgramUniform3fv)
		pGlProgramUniform3fv(program, loc, count, v);
}

inline void COpenGLExtensionHandler::extGlProgramUniform4fv(GLuint program, GLint loc, GLsizei count, const GLfloat *v)
{
	if (pGlProgramUniform4fv)
		pGlProgramUniform4fv(program, loc, count, v);
}

inline void COpenGLExtensionHandler::extGlProgramUniform1iv(GLuint program, GLint loc, GLsizei count, const GLint *v)
{
	if (pGlProgramUniform1iv)
		pGlProgramUniform1iv(program, loc, count, v);
}

inline void COpenGLExtensionHandler::extGlProgramUniform2iv(GLuint program, GLint loc, GLsizei count, const GLint *v)
{
	if (pGlProgramUniform2iv)
		pGlProgramUniform2iv(program, loc, count, v);
}

inline void COpenGLExtensionHandler::extGlProgramUniform3iv(GLuint program, GLint loc, GLsizei count, const GLint *v)
{
	if (pGlProgramUniform3iv)
		pGlProgramUniform3iv(program, loc, count, v);
}

inline void COpenGLExtensionHandler::extGlProgramUniform4iv(GLuint program, GLint loc, GLsizei count, const GLint *v)
{
	if (pGlProgramUniform4iv)
		pGlProgramUniform4iv(program, loc, count, v);
}

inline void COpenGLExtensionHandler::extGlProgramUniform1uiv(GLuint program, GLint loc, GLsizei count, const GLuint *v)
{
	if (pGlProgramUniform1uiv)
		pGlProgramUniform1uiv(program, loc, count, v);
}

inline void COpenGLExtensionHandler::extGlProgramUniform2uiv(GLuint program, GLint loc, GLsizei count, const GLuint *v)
{
	if (pGlProgramUniform2uiv)
		pGlProgramUniform2uiv(program, loc, count, v);
}

inline void COpenGLExtensionHandler::extGlProgramUniform3uiv(GLuint program, GLint loc, GLsizei count, const GLuint *v)
{
	if (pGlProgramUniform3uiv)
		pGlProgramUniform3uiv(program, loc, count, v);
}

inline void COpenGLExtensionHandler::extGlProgramUniform4uiv(GLuint program, GLint loc, GLsizei count, const GLuint *v)
{
	if (pGlProgramUniform4uiv)
		pGlProgramUniform4uiv(program, loc, count, v);
}

inline void COpenGLExtensionHandler::extGlProgramUniformMatrix2fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v)
{
	if (pGlProgramUniformMatrix2fv)
		pGlProgramUniformMatrix2fv(program, loc, count, transpose, v);
}

inline void COpenGLExtensionHandler::extGlProgramUniformMatrix3fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v)
{
	if (pGlProgramUniformMatrix3fv)
		pGlProgramUniformMatrix3fv(program, loc, count, transpose, v);
}

inline void COpenGLExtensionHandler::extGlProgramUniformMatrix4fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v)
{
	if (pGlProgramUniformMatrix4fv)
		pGlProgramUniformMatrix4fv(program, loc, count, transpose, v);
}

inline void COpenGLExtensionHandler::extGlProgramUniformMatrix2x3fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v)
{
	if (pGlProgramUniformMatrix2x3fv)
		pGlProgramUniformMatrix2x3fv(program, loc, count, transpose, v);
}
inline void COpenGLExtensionHandler::extGlProgramUniformMatrix2x4fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v)
{
	if (pGlProgramUniformMatrix2x4fv)
		pGlProgramUniformMatrix2x4fv(program, loc, count, transpose, v);
}
inline void COpenGLExtensionHandler::extGlProgramUniformMatrix3x2fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v)
{
	if (pGlProgramUniformMatrix3x2fv)
		pGlProgramUniformMatrix3x2fv(program, loc, count, transpose, v);
}
inline void COpenGLExtensionHandler::extGlProgramUniformMatrix3x4fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v)
{
	if (pGlProgramUniformMatrix3x4fv)
		pGlProgramUniformMatrix3x4fv(program, loc, count, transpose, v);
}
inline void COpenGLExtensionHandler::extGlProgramUniformMatrix4x2fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v)
{
	if (pGlProgramUniformMatrix4x2fv)
		pGlProgramUniformMatrix4x2fv(program, loc, count, transpose, v);
}
inline void COpenGLExtensionHandler::extGlProgramUniformMatrix4x3fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v)
{
	if (pGlProgramUniformMatrix4x3fv)
		pGlProgramUniformMatrix4x3fv(program, loc, count, transpose, v);
}



inline void COpenGLExtensionHandler::extGlGetActiveUniform(GLuint program,
		GLuint index, GLsizei maxlength, GLsizei *length,
		GLint *size, GLenum *type, GLchar *name)
{
	if (length)
		*length=0;
	if (pGlGetActiveUniform)
		pGlGetActiveUniform(program, index, maxlength, length, size, type, name);
}

inline void COpenGLExtensionHandler::extGlBindProgramPipeline(GLuint pipeline)
{
	if (pGlBindProgramPipeline)
		pGlBindProgramPipeline(pipeline);
}

inline void COpenGLExtensionHandler::extGlGetProgramBinary(GLuint program, GLsizei bufSize, GLsizei* length, GLenum* binaryFormat, void* binary)
{
    if (pGlGetProgramBinary)
        pGlGetProgramBinary(program, bufSize, length, binaryFormat, binary);
}

inline void COpenGLExtensionHandler::extGlProgramBinary(GLuint program, GLenum binaryFormat, const void* binary, GLsizei length)
{
    if (pGlProgramBinary)
        pGlProgramBinary(program, binaryFormat, binary, length);
}



inline void COpenGLExtensionHandler::extGlMemoryBarrier(GLbitfield barriers)
{
	if (pGlMemoryBarrier)
		pGlMemoryBarrier(barriers);
}

inline void COpenGLExtensionHandler::extGlDispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z)
{
	if (pGlDispatchCompute)
		pGlDispatchCompute(num_groups_x,num_groups_y,num_groups_z);
}

inline void COpenGLExtensionHandler::extGlDispatchComputeIndirect(GLintptr indirect)
{
	if (pGlDispatchComputeIndirect)
		pGlDispatchComputeIndirect(indirect);
}




inline void COpenGLExtensionHandler::extGlPointParameterf(GLint loc, GLfloat f)
{
	if (pGlPointParameterf)
		pGlPointParameterf(loc, f);
}

inline void COpenGLExtensionHandler::extGlPointParameterfv(GLint loc, const GLfloat *v)
{
	if (pGlPointParameterfv)
		pGlPointParameterfv(loc, v);
}

inline void COpenGLExtensionHandler::extGlStencilFuncSeparate(GLenum face, GLenum func, GLint ref, GLuint mask)
{
    pGlStencilFuncSeparate(face, func, ref, mask);
}

inline void COpenGLExtensionHandler::extGlStencilOpSeparate(GLenum face, GLenum fail, GLenum zfail, GLenum zpass)
{
    pGlStencilOpSeparate(face, fail, zfail, zpass);
}

inline void COpenGLExtensionHandler::extGlStencilMaskSeparate(GLenum face, GLuint mask)
{
    pGlStencilMaskSeparate(face, mask);
}


inline void COpenGLExtensionHandler::extGlBindFramebuffer(GLenum target, GLuint framebuffer)
{
    pGlBindFramebuffer(target, framebuffer);
}

inline void COpenGLExtensionHandler::extGlDeleteFramebuffers(GLsizei n, const GLuint *framebuffers)
{
    pGlDeleteFramebuffers(n, framebuffers);
}

inline void COpenGLExtensionHandler::extGlCreateFramebuffers(GLsizei n, GLuint *framebuffers)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlCreateFramebuffers(n, framebuffers);
            return;
        }
    }

    pGlGenFramebuffers(n, framebuffers);
}

inline GLenum COpenGLExtensionHandler::extGlCheckNamedFramebufferStatus(GLuint framebuffer, GLenum target)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
            return pGlCheckNamedFramebufferStatus(framebuffer,target);
        else if (FeatureAvailable[NBL_EXT_direct_state_access])
            return pGlCheckNamedFramebufferStatusEXT(framebuffer,target);
    }

    GLenum retval;
    GLuint bound;
    glGetIntegerv(target==GL_READ_FRAMEBUFFER ? GL_READ_FRAMEBUFFER_BINDING:GL_DRAW_FRAMEBUFFER_BINDING,reinterpret_cast<GLint*>(&bound));

    if (bound!=framebuffer)
        pGlBindFramebuffer(target,framebuffer);
    retval = pGlCheckFramebufferStatus(target);
    if (bound!=framebuffer)
        pGlBindFramebuffer(target,bound);

    return retval;
}

inline void COpenGLExtensionHandler::extGlNamedFramebufferTexture(GLuint framebuffer, GLenum attachment, GLuint texture, GLint level)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlNamedFramebufferTexture(framebuffer, attachment, texture, level);
            return;
        }
        else if (FeatureAvailable[NBL_EXT_direct_state_access])
        {
            pGlNamedFramebufferTextureEXT(framebuffer, attachment, texture, level);
            return;
        }
    }

    GLuint bound;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING,reinterpret_cast<GLint*>(&bound));

    if (bound!=framebuffer)
        pGlBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
    pGlFramebufferTexture(GL_FRAMEBUFFER,attachment,texture,level);
    if (bound!=framebuffer)
        pGlBindFramebuffer(GL_FRAMEBUFFER,bound);
}

inline void COpenGLExtensionHandler::extGlNamedFramebufferTextureLayer(GLuint framebuffer, GLenum attachment, GLuint texture, GLenum textureType, GLint level, GLint layer)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlNamedFramebufferTextureLayer(framebuffer, attachment, texture, level, layer);
            return;
        }
    }

	if (textureType!=GL_TEXTURE_CUBE_MAP)
	{
		if (!needsDSAFramebufferHack && FeatureAvailable[NBL_EXT_direct_state_access])
		{
            pGlNamedFramebufferTextureLayerEXT(framebuffer, attachment, texture, level, layer);
		}
		else
		{
			GLuint bound;
			glGetIntegerv(GL_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(&bound));

			if (bound != framebuffer)
				pGlBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
			pGlFramebufferTextureLayer(GL_FRAMEBUFFER, attachment, texture, level, layer);
			if (bound != framebuffer)
				pGlBindFramebuffer(GL_FRAMEBUFFER, bound);
		}
	}
	else
	{
		constexpr GLenum CubeMapFaceToCubeMapFaceGLenum[IGPUImageView::ECMF_COUNT] = {
			GL_TEXTURE_CUBE_MAP_POSITIVE_X,GL_TEXTURE_CUBE_MAP_NEGATIVE_X,GL_TEXTURE_CUBE_MAP_POSITIVE_Y,GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,GL_TEXTURE_CUBE_MAP_POSITIVE_Z,GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
		};

		if (!needsDSAFramebufferHack && FeatureAvailable[NBL_EXT_direct_state_access])
		{
            pGlNamedFramebufferTexture2DEXT(framebuffer, attachment, CubeMapFaceToCubeMapFaceGLenum[layer], texture, level);
		}
		else
		{
			GLuint bound;
			glGetIntegerv(GL_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(&bound));

			if (bound != framebuffer)
				pGlBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
			pGlFramebufferTexture2D(GL_FRAMEBUFFER, attachment, CubeMapFaceToCubeMapFaceGLenum[layer], texture, level);
			if (bound != framebuffer)
				pGlBindFramebuffer(GL_FRAMEBUFFER, bound);
		}
	}
}

inline void COpenGLExtensionHandler::extGlBlitNamedFramebuffer(GLuint readFramebuffer, GLuint drawFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlBlitNamedFramebuffer(readFramebuffer, drawFramebuffer, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
            return;
        }
    }

    GLint boundReadFBO = -1;
    GLint boundDrawFBO = -1;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING,&boundReadFBO);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&boundDrawFBO);

    if (static_cast<GLint>(readFramebuffer)!=boundReadFBO)
        extGlBindFramebuffer(GL_READ_FRAMEBUFFER,readFramebuffer);
    if (static_cast<GLint>(drawFramebuffer)!=boundDrawFBO)
        extGlBindFramebuffer(GL_DRAW_FRAMEBUFFER,drawFramebuffer);

    pGlBlitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);

    if (static_cast<GLint>(readFramebuffer)!=boundReadFBO)
        extGlBindFramebuffer(GL_READ_FRAMEBUFFER,boundReadFBO);
    if (static_cast<GLint>(drawFramebuffer)!=boundDrawFBO)
        extGlBindFramebuffer(GL_DRAW_FRAMEBUFFER,boundDrawFBO);
}

//! there should be a GL 3.1 thing for this
inline void COpenGLExtensionHandler::extGlActiveStencilFace(GLenum face)
{
    pGlActiveStencilFaceEXT(face);
}

inline void COpenGLExtensionHandler::extGlNamedFramebufferReadBuffer(GLuint framebuffer, GLenum mode)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlNamedFramebufferReadBuffer(framebuffer, mode);
            return;
        }
        else if (FeatureAvailable[NBL_EXT_direct_state_access])
        {
            pGlFramebufferReadBufferEXT(framebuffer, mode);
            return;
        }
    }

    GLint boundFBO;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING,&boundFBO);

    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        pGlBindFramebuffer(GL_READ_FRAMEBUFFER,framebuffer);
    glReadBuffer(mode);
    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        pGlBindFramebuffer(GL_READ_FRAMEBUFFER,boundFBO);
}

inline void COpenGLExtensionHandler::extGlNamedFramebufferDrawBuffer(GLuint framebuffer, GLenum buf)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlNamedFramebufferDrawBuffer(framebuffer, buf);
            return;
        }
        else if (FeatureAvailable[NBL_EXT_direct_state_access])
        {
            pGlFramebufferDrawBufferEXT(framebuffer, buf);
            return;
        }
    }

    GLint boundFBO;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&boundFBO);

    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        pGlBindFramebuffer(GL_DRAW_FRAMEBUFFER,framebuffer);
    glDrawBuffer(buf);
    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        pGlBindFramebuffer(GL_DRAW_FRAMEBUFFER,boundFBO);
}

inline void COpenGLExtensionHandler::extGlNamedFramebufferDrawBuffers(GLuint framebuffer, GLsizei n, const GLenum *bufs)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlNamedFramebufferDrawBuffers(framebuffer, n, bufs);
            return;
        }
        else if (FeatureAvailable[NBL_EXT_direct_state_access])
        {
            pGlFramebufferDrawBuffersEXT(framebuffer, n, bufs);
            return;
        }
    }

    GLint boundFBO;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&boundFBO);

    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        pGlBindFramebuffer(GL_DRAW_FRAMEBUFFER,framebuffer);
    pGlDrawBuffers(n,bufs);
    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        pGlBindFramebuffer(GL_DRAW_FRAMEBUFFER,boundFBO);
}

inline void COpenGLExtensionHandler::extGlClearNamedFramebufferiv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLint* value)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlClearNamedFramebufferiv(framebuffer, buffer, drawbuffer, value);
            return;
        }
    }

    GLint boundFBO = -1;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&boundFBO);
    if (boundFBO<0)
        return;

    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        extGlBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
    pGlClearBufferiv(buffer, drawbuffer, value);
    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        extGlBindFramebuffer(GL_FRAMEBUFFER,boundFBO);
}

inline void COpenGLExtensionHandler::extGlClearNamedFramebufferuiv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLuint* value)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlClearNamedFramebufferuiv(framebuffer, buffer, drawbuffer, value);
            return;
        }
    }

    GLint boundFBO = -1;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&boundFBO);
    if (boundFBO<0)
        return;

    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        extGlBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
    pGlClearBufferuiv(buffer, drawbuffer, value);
    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        extGlBindFramebuffer(GL_FRAMEBUFFER,boundFBO);
}

inline void COpenGLExtensionHandler::extGlClearNamedFramebufferfv(GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLfloat* value)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlClearNamedFramebufferfv(framebuffer, buffer, drawbuffer, value);
            return;
        }
    }

    GLint boundFBO = -1;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&boundFBO);
    if (boundFBO<0)
        return;

    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        extGlBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
    pGlClearBufferfv(buffer, drawbuffer, value);
    if (static_cast<GLuint>(boundFBO)!=framebuffer)
        extGlBindFramebuffer(GL_FRAMEBUFFER,boundFBO);
}

inline void COpenGLExtensionHandler::extGlClearNamedFramebufferfi(GLuint framebuffer, GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
        {
            pGlClearNamedFramebufferfi(framebuffer, buffer, drawbuffer, depth, stencil);
            return;
        }
    }

    GLint boundFBO = -1;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&boundFBO);
    if (boundFBO<0)
        return;
    extGlBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
    pGlClearBufferfi(buffer, drawbuffer, depth, stencil);
    extGlBindFramebuffer(GL_FRAMEBUFFER,boundFBO);
}


inline void COpenGLExtensionHandler::extGlCreateBuffers(GLsizei n, GLuint *buffers)
{
#ifdef OPENGL_LEAK_DEBUG
    for (size_t i=0; i<n; i++)
        COpenGLExtensionHandler::bufferLeaker.registerObj(buffers);
#endif // OPENGL_LEAK_DEBUG

    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlCreateBuffers)
            pGlCreateBuffers(n, buffers);
        else if (buffers)
            memset(buffers,0,n*sizeof(GLuint));
    }
    else
    {
        if (pGlGenBuffers)
            pGlGenBuffers(n, buffers);
        else if (buffers)
            memset(buffers,0,n*sizeof(GLuint));
    }
}

inline void COpenGLExtensionHandler::extGlDeleteBuffers(GLsizei n, const GLuint *buffers)
{
#ifdef OPENGL_LEAK_DEBUG
    for (size_t i=0; i<n; i++)
        COpenGLExtensionHandler::bufferLeaker.deregisterObj(buffers);
#endif // OPENGL_LEAK_DEBUG

	if (pGlDeleteBuffers)
		pGlDeleteBuffers(n, buffers);
}

inline void COpenGLExtensionHandler::extGlBindBuffer(const GLenum& target, const GLuint& buffer)
{
	if (pGlBindBuffer)
		pGlBindBuffer(target, buffer);
}

inline void COpenGLExtensionHandler::extGlBindBuffersBase(const GLenum& target, const GLuint& first, const GLsizei& count, const GLuint* buffers)
{
    if (Version>=440||FeatureAvailable[NBL_ARB_multi_bind])
    {
        if (pGlBindBuffersBase)
            pGlBindBuffersBase(target,first,count,buffers);
    }
    else
    {
        for (GLsizei i=0; i<count; i++)
        {
            if (pGlBindBufferBase)
                pGlBindBufferBase(target,first+i,buffers ? buffers[i]:0);
        }
    }
}

inline void COpenGLExtensionHandler::extGlBindBuffersRange(const GLenum& target, const GLuint& first, const GLsizei& count, const GLuint* buffers, const GLintptr* offsets, const GLsizeiptr* sizes)
{
    if (Version>=440||FeatureAvailable[NBL_ARB_multi_bind])
    {
        if (pGlBindBuffersRange)
            pGlBindBuffersRange(target,first,count,buffers,offsets,sizes);
    }
    else
    {
        for (GLsizei i=0; i<count; i++)
        {
            if (buffers)
            {
                if (pGlBindBufferRange)
                    pGlBindBufferRange(target,first+i,buffers[i],offsets[i],sizes[i]);
            }
            else
            {
                if (pGlBindBufferBase)
                    pGlBindBufferBase(target,first+i,0);
            }
        }
    }
}

inline void COpenGLExtensionHandler::extGlNamedBufferStorage(GLuint buffer, GLsizeiptr size, const void *data, GLbitfield flags)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlNamedBufferStorage)
            pGlNamedBufferStorage(buffer,size,data,flags);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlNamedBufferStorageEXT)
            pGlNamedBufferStorageEXT(buffer,size,data,flags);
    }
    else if (pGlBufferStorage&&pGlBindBuffer)
    {
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        pGlBufferStorage(GL_ARRAY_BUFFER, size, data, flags);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
    }
}

inline void COpenGLExtensionHandler::extGlNamedBufferSubData(GLuint buffer, GLintptr offset, GLsizeiptr size, const void *data)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlNamedBufferSubData)
            pGlNamedBufferSubData(buffer,offset,size,data);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlNamedBufferSubDataEXT)
            pGlNamedBufferSubDataEXT(buffer,offset,size,data);
    }
    else if (pGlBufferSubData&&pGlBindBuffer)
    {
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        pGlBufferSubData(GL_ARRAY_BUFFER, offset, size, data);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
    }
}

inline void COpenGLExtensionHandler::extGlGetNamedBufferSubData(GLuint buffer, GLintptr offset, GLsizeiptr size, void *data)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlGetNamedBufferSubData)
            pGlGetNamedBufferSubData(buffer,offset,size,data);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlGetNamedBufferSubDataEXT)
            pGlGetNamedBufferSubDataEXT(buffer,offset,size,data);
    }
    else if (pGlGetBufferSubData&&pGlBindBuffer)
    {
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        pGlGetBufferSubData(GL_ARRAY_BUFFER, offset, size, data);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
    }
}

inline void *COpenGLExtensionHandler::extGlMapNamedBuffer(GLuint buffer, GLbitfield access)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlMapNamedBuffer)
            return pGlMapNamedBuffer(buffer,access);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlMapNamedBufferEXT)
            return pGlMapNamedBufferEXT(buffer,access);
    }
    else if (pGlMapBuffer&&pGlBindBuffer)
    {
        GLvoid* retval;
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        retval = pGlMapBuffer(GL_ARRAY_BUFFER,access);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
        return retval;
    }
    return NULL;
}

inline void *COpenGLExtensionHandler::extGlMapNamedBufferRange(GLuint buffer, GLintptr offset, GLsizeiptr length, GLbitfield access)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlMapNamedBufferRange)
            return pGlMapNamedBufferRange(buffer,offset,length,access);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlMapNamedBufferRangeEXT)
            return pGlMapNamedBufferRangeEXT(buffer,offset,length,access);
    }
    else if (pGlMapBufferRange&&pGlBindBuffer)
    {
        GLvoid* retval;
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        retval = pGlMapBufferRange(GL_ARRAY_BUFFER,offset,length,access);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
        return retval;
    }
    return NULL;
}

inline void COpenGLExtensionHandler::extGlFlushMappedNamedBufferRange(GLuint buffer, GLintptr offset, GLsizeiptr length)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlFlushMappedNamedBufferRange)
            pGlFlushMappedNamedBufferRange(buffer,offset,length);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlFlushMappedNamedBufferRangeEXT)
            pGlFlushMappedNamedBufferRangeEXT(buffer,offset,length);
    }
    else if (pGlFlushMappedBufferRange&&pGlBindBuffer)
    {
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        pGlFlushMappedBufferRange(GL_ARRAY_BUFFER, offset, length);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
    }
}

inline GLboolean COpenGLExtensionHandler::extGlUnmapNamedBuffer(GLuint buffer)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlUnmapNamedBuffer)
            return pGlUnmapNamedBuffer(buffer);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlUnmapNamedBufferEXT)
            return pGlUnmapNamedBufferEXT(buffer);
    }
    else if (pGlUnmapBuffer&&pGlBindBuffer)
    {
        GLboolean retval;
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        retval = pGlUnmapBuffer(GL_ARRAY_BUFFER);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
        return retval;
    }
    return false;
}

inline void COpenGLExtensionHandler::extGlClearNamedBufferData(GLuint buffer, GLenum internalformat, GLenum format, GLenum type, const void *data)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlClearNamedBufferData)
            pGlClearNamedBufferData(buffer,internalformat,format,type,data);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlClearNamedBufferDataEXT)
            pGlClearNamedBufferDataEXT(buffer,internalformat,format,type,data);
    }
    else if (pGlClearBufferData&&pGlBindBuffer)
    {
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        pGlClearBufferData(GL_ARRAY_BUFFER, internalformat,format,type,data);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
    }
}

inline void COpenGLExtensionHandler::extGlClearNamedBufferSubData(GLuint buffer, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format, GLenum type, const void *data)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlClearNamedBufferSubData)
            pGlClearNamedBufferSubData(buffer,internalformat,offset,size,format,type,data);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlClearNamedBufferSubDataEXT)
            pGlClearNamedBufferSubDataEXT(buffer,internalformat,offset,size,format,type,data);
    }
    else if (pGlClearBufferSubData&&pGlBindBuffer)
    {
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        pGlClearBufferSubData(GL_ARRAY_BUFFER, internalformat,offset,size,format,type,data);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
    }
}

inline void COpenGLExtensionHandler::extGlCopyNamedBufferSubData(GLuint readBuffer, GLuint writeBuffer, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlCopyNamedBufferSubData)
            pGlCopyNamedBufferSubData(readBuffer, writeBuffer, readOffset, writeOffset, size);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlNamedCopyBufferSubDataEXT)
            pGlNamedCopyBufferSubDataEXT(readBuffer, writeBuffer, readOffset, writeOffset, size);
    }
    else if (pGlCopyBufferSubData&&pGlBindBuffer)
    {
        GLint boundRead,boundWrite;
        glGetIntegerv(GL_COPY_READ_BUFFER_BINDING,&boundRead);
        glGetIntegerv(GL_COPY_WRITE_BUFFER_BINDING,&boundWrite);
        pGlBindBuffer(GL_COPY_READ_BUFFER,readBuffer);
        pGlBindBuffer(GL_COPY_WRITE_BUFFER,writeBuffer);
        pGlCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, readOffset, writeOffset, size);
        pGlBindBuffer(GL_COPY_READ_BUFFER,boundRead);
        pGlBindBuffer(GL_COPY_WRITE_BUFFER,boundWrite);
    }
}

inline GLboolean COpenGLExtensionHandler::extGlIsBuffer(GLuint buffer)
{
	if (pGlIsBuffer)
		return pGlIsBuffer(buffer);
	return false;
}

inline void COpenGLExtensionHandler::extGlGetNamedBufferParameteriv(const GLuint& buffer, const GLenum& value, GLint* data)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlGetNamedBufferParameteriv)
            pGlGetNamedBufferParameteriv(buffer, value, data);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlGetNamedBufferParameterivEXT)
            pGlGetNamedBufferParameterivEXT(buffer, value, data);
    }
    else if (pGlGetBufferParameteriv&&pGlBindBuffer)
    {
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        pGlGetBufferParameteriv(GL_ARRAY_BUFFER, value, data);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
    }
}

inline void COpenGLExtensionHandler::extGlGetNamedBufferParameteri64v(const GLuint& buffer, const GLenum& value, GLint64* data)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlGetNamedBufferParameteri64v)
            pGlGetNamedBufferParameteri64v(buffer, value, data);
    }
    else if (pGlGetBufferParameteri64v&&pGlBindBuffer)
    {
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        pGlGetBufferParameteri64v(GL_ARRAY_BUFFER, value, data);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
    }
}


inline void COpenGLExtensionHandler::extGlCreateVertexArrays(GLsizei n, GLuint *arrays)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlCreateVertexArrays)
            pGlCreateVertexArrays(n,arrays);
    }
    else
    {
        if (pGlGenVertexArrays)
            pGlGenVertexArrays(n,arrays);
        else
            memset(arrays,0,sizeof(GLuint)*n);
    }
}

inline void COpenGLExtensionHandler::extGlDeleteVertexArrays(GLsizei n, GLuint *arrays)
{
    if (pGlDeleteVertexArrays)
        pGlDeleteVertexArrays(n,arrays);
}

inline void COpenGLExtensionHandler::extGlBindVertexArray(GLuint vaobj)
{
    if (pGlBindVertexArray)
        pGlBindVertexArray(vaobj);
}

inline void COpenGLExtensionHandler::extGlVertexArrayElementBuffer(GLuint vaobj, GLuint buffer)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlVertexArrayElementBuffer)
            pGlVertexArrayElementBuffer(vaobj,buffer);
    }
    else if (pGlBindBuffer&&pGlBindVertexArray)
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
        pGlBindVertexArray(vaobj);
        pGlBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
        pGlBindVertexArray(restoreVertexArray);
    }
}

inline void COpenGLExtensionHandler::extGlVertexArrayVertexBuffer(GLuint vaobj, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlVertexArrayVertexBuffer)
            pGlVertexArrayVertexBuffer(vaobj,bindingindex,buffer,offset,stride);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlVertexArrayBindVertexBufferEXT)
            pGlVertexArrayBindVertexBufferEXT(vaobj,bindingindex,buffer,offset,stride);
    }
    else if (pGlBindVertexBuffer&&pGlBindVertexArray)
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
        pGlBindVertexArray(vaobj);
        pGlBindVertexBuffer(bindingindex,buffer,offset,stride);
        pGlBindVertexArray(restoreVertexArray);
    }
}

inline void COpenGLExtensionHandler::extGlVertexArrayAttribBinding(GLuint vaobj, GLuint attribindex, GLuint bindingindex)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlVertexArrayAttribBinding)
            pGlVertexArrayAttribBinding(vaobj,attribindex,bindingindex);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlVertexArrayVertexAttribBindingEXT)
            pGlVertexArrayVertexAttribBindingEXT(vaobj,attribindex,bindingindex);
    }
    else if (pGlVertexAttribBinding&&pGlBindVertexArray)
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
        pGlBindVertexArray(vaobj);
        pGlVertexAttribBinding(attribindex,bindingindex);
        pGlBindVertexArray(restoreVertexArray);
    }
}

inline void COpenGLExtensionHandler::extGlEnableVertexArrayAttrib(GLuint vaobj, GLuint index)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlEnableVertexArrayAttrib)
            pGlEnableVertexArrayAttrib(vaobj,index);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlEnableVertexArrayAttribEXT)
            pGlEnableVertexArrayAttribEXT(vaobj,index);
    }
    else if (pGlEnableVertexAttribArray&&pGlBindVertexArray)
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
        pGlBindVertexArray(vaobj);
        pGlEnableVertexAttribArray(index);
        pGlBindVertexArray(restoreVertexArray);
    }
}

inline void COpenGLExtensionHandler::extGlDisableVertexArrayAttrib(GLuint vaobj, GLuint index)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlDisableVertexArrayAttrib)
            pGlDisableVertexArrayAttrib(vaobj,index);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlDisableVertexArrayAttribEXT)
            pGlDisableVertexArrayAttribEXT(vaobj,index);
    }
    else if (pGlDisableVertexAttribArray&&pGlBindVertexArray)
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
        pGlBindVertexArray(vaobj);
        pGlDisableVertexAttribArray(index);
        pGlBindVertexArray(restoreVertexArray);
    }
}

inline void COpenGLExtensionHandler::extGlVertexArrayAttribFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlVertexArrayAttribFormat)
            pGlVertexArrayAttribFormat(vaobj,attribindex,size,type,normalized,relativeoffset);
    }
    else if (!IsIntelGPU&&FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlVertexArrayVertexAttribFormatEXT)
            pGlVertexArrayVertexAttribFormatEXT(vaobj,attribindex,size,type,normalized,relativeoffset);
    }
    else if (pGlVertexAttribFormat&&pGlBindVertexArray)
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
        pGlBindVertexArray(vaobj);
        pGlVertexAttribFormat(attribindex,size,type,normalized,relativeoffset);
        pGlBindVertexArray(restoreVertexArray);
    }
}

inline void COpenGLExtensionHandler::extGlVertexArrayAttribIFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlVertexArrayAttribIFormat)
            pGlVertexArrayAttribIFormat(vaobj,attribindex,size,type,relativeoffset);
    }
    else if (!IsIntelGPU&&FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlVertexArrayVertexAttribIFormatEXT)
            pGlVertexArrayVertexAttribIFormatEXT(vaobj,attribindex,size,type,relativeoffset);
    }
    else if (pGlVertexAttribIFormat&&pGlBindVertexArray)
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
        pGlBindVertexArray(vaobj);
        pGlVertexAttribIFormat(attribindex,size,type,relativeoffset);
        pGlBindVertexArray(restoreVertexArray);
    }
}

inline void COpenGLExtensionHandler::extGlVertexArrayAttribLFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlVertexArrayAttribLFormat)
            pGlVertexArrayAttribLFormat(vaobj,attribindex,size,type,relativeoffset);
    }
    else if (!IsIntelGPU&&FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlVertexArrayVertexAttribLFormatEXT)
            pGlVertexArrayVertexAttribLFormatEXT(vaobj,attribindex,size,type,relativeoffset);
    }
    else if (pGlVertexAttribLFormat&&pGlBindVertexArray)
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
        pGlBindVertexArray(vaobj);
        pGlVertexAttribLFormat(attribindex,size,type,relativeoffset);
        pGlBindVertexArray(restoreVertexArray);
    }
}

inline void COpenGLExtensionHandler::extGlVertexArrayBindingDivisor(GLuint vaobj, GLuint bindingindex, GLuint divisor)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlVertexArrayBindingDivisor)
            pGlVertexArrayBindingDivisor(vaobj,bindingindex,divisor);
    }
    else if (FeatureAvailable[NBL_EXT_direct_state_access])
    {
        if (pGlVertexArrayVertexBindingDivisorEXT)
            pGlVertexArrayVertexBindingDivisorEXT(vaobj,bindingindex,divisor);
    }
    else if (pGlVertexBindingDivisor&&pGlBindVertexArray)
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
        pGlBindVertexArray(vaobj);
        pGlVertexBindingDivisor(bindingindex,divisor);
        pGlBindVertexArray(restoreVertexArray);
    }
}



inline void COpenGLExtensionHandler::extGlCreateTransformFeedbacks(GLsizei n, GLuint* ids)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlCreateTransformFeedbacks)
            pGlCreateTransformFeedbacks(n,ids);
    }
    else
    {
        if (pGlGenTransformFeedbacks)
            pGlGenTransformFeedbacks(n,ids);
    }
}

inline void COpenGLExtensionHandler::extGlDeleteTransformFeedbacks(GLsizei n, const GLuint* ids)
{
    if (pGlDeleteTransformFeedbacks)
        pGlDeleteTransformFeedbacks(n,ids);
}

inline void COpenGLExtensionHandler::extGlBindTransformFeedback(GLenum target, GLuint id)
{
    if (pGlBindTransformFeedback)
        pGlBindTransformFeedback(target,id);
}

inline void COpenGLExtensionHandler::extGlBeginTransformFeedback(GLenum primitiveMode)
{
    if (pGlBeginTransformFeedback)
        pGlBeginTransformFeedback(primitiveMode);
}

inline void COpenGLExtensionHandler::extGlPauseTransformFeedback()
{
    if (pGlPauseTransformFeedback)
        pGlPauseTransformFeedback();
}

inline void COpenGLExtensionHandler::extGlResumeTransformFeedback()
{
    if (pGlResumeTransformFeedback)
        pGlResumeTransformFeedback();
}

inline void COpenGLExtensionHandler::extGlEndTransformFeedback()
{
    if (pGlEndTransformFeedback)
        pGlEndTransformFeedback();
}

inline void COpenGLExtensionHandler::extGlTransformFeedbackBufferBase(GLuint xfb, GLuint index, GLuint buffer)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlTransformFeedbackBufferBase)
            pGlTransformFeedbackBufferBase(xfb,index,buffer);
    }
    else
    {
        GLint restoreXFormFeedback;
        glGetIntegerv(GL_TRANSFORM_FEEDBACK_BINDING, &restoreXFormFeedback);
        extGlBindTransformFeedback(GL_TRANSFORM_FEEDBACK,xfb);
        extGlBindBuffersBase(GL_TRANSFORM_FEEDBACK_BUFFER,index,1,&buffer);
        extGlBindTransformFeedback(GL_TRANSFORM_FEEDBACK,restoreXFormFeedback);
    }
}

inline void COpenGLExtensionHandler::extGlTransformFeedbackBufferRange(GLuint xfb, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlTransformFeedbackBufferRange)
            pGlTransformFeedbackBufferRange(xfb,index,buffer,offset,size);
    }
    else
    {
        GLint restoreXFormFeedback;
        glGetIntegerv(GL_TRANSFORM_FEEDBACK_BINDING, &restoreXFormFeedback);
        extGlBindTransformFeedback(GL_TRANSFORM_FEEDBACK,xfb);
        extGlBindBuffersRange(GL_TRANSFORM_FEEDBACK_BUFFER,index,1,&buffer,&offset,&size);
        extGlBindTransformFeedback(GL_TRANSFORM_FEEDBACK,restoreXFormFeedback);
    }
}


inline void COpenGLExtensionHandler::extGlPrimitiveRestartIndex(GLuint index)
{
    if (pGlPrimitiveRestartIndex)
        pGlPrimitiveRestartIndex(index);
}

inline void COpenGLExtensionHandler::extGlDrawArraysInstanced(GLenum mode, GLint first, GLsizei count, GLsizei instancecount)
{
    if (pGlDrawArraysInstanced)
        pGlDrawArraysInstanced(mode,first,count,instancecount);
}

inline void COpenGLExtensionHandler::extGlDrawArraysInstancedBaseInstance(GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance)
{
    if (pGlDrawArraysInstancedBaseInstance)
        pGlDrawArraysInstancedBaseInstance(mode,first,count,instancecount,baseinstance);
}

inline void COpenGLExtensionHandler::extGlDrawElementsInstancedBaseVertex(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex)
{
    if (pGlDrawElementsInstancedBaseVertex)
        pGlDrawElementsInstancedBaseVertex(mode,count,type,indices,instancecount,basevertex);
}

inline void COpenGLExtensionHandler::extGlDrawElementsInstancedBaseVertexBaseInstance(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance)
{
    if (pGlDrawElementsInstancedBaseVertexBaseInstance)
        pGlDrawElementsInstancedBaseVertexBaseInstance(mode,count,type,indices,instancecount,basevertex,baseinstance);
}

inline void COpenGLExtensionHandler::extGlDrawTransformFeedback(GLenum mode, GLuint id)
{
    if (pGlDrawTransformFeedback)
        pGlDrawTransformFeedback(mode,id);
}

inline void COpenGLExtensionHandler::extGlDrawTransformFeedbackInstanced(GLenum mode, GLuint id, GLsizei instancecount)
{
    if (pGlDrawTransformFeedbackInstanced)
        pGlDrawTransformFeedbackInstanced(mode,id,instancecount);
}

inline void COpenGLExtensionHandler::extGlDrawTransformFeedbackStream(GLenum mode, GLuint id, GLuint stream)
{
    if (pGlDrawTransformFeedbackStream)
        pGlDrawTransformFeedbackStream(mode,id,stream);
}

inline void COpenGLExtensionHandler::extGlDrawTransformFeedbackStreamInstanced(GLenum mode, GLuint id, GLuint stream, GLsizei instancecount)
{
    if (pGlDrawTransformFeedbackStreamInstanced)
        pGlDrawTransformFeedbackStreamInstanced(mode,id,stream,instancecount);
}

inline void COpenGLExtensionHandler::extGlDrawArraysIndirect(GLenum mode, const void *indirect)
{
    if (pGlDrawArraysIndirect)
        pGlDrawArraysIndirect(mode,indirect);
}

inline void COpenGLExtensionHandler::extGlDrawElementsIndirect(GLenum mode, GLenum type, const void *indirect)
{
    if (pGlDrawElementsIndirect)
        pGlDrawElementsIndirect(mode,type,indirect);
}

inline void COpenGLExtensionHandler::extGlMultiDrawArraysIndirect(GLenum mode, const void *indirect, GLsizei drawcount, GLsizei stride)
{
    if (pGlMultiDrawArraysIndirect)
        pGlMultiDrawArraysIndirect(mode,indirect,drawcount,stride);
}

inline void COpenGLExtensionHandler::extGlMultiDrawElementsIndirect(GLenum mode, GLenum type, const void *indirect, GLsizei drawcount, GLsizei stride)
{
    if (pGlMultiDrawElementsIndirect)
        pGlMultiDrawElementsIndirect(mode,type,indirect,drawcount,stride);
}

inline void COpenGLExtensionHandler::extGlMultiDrawArraysIndirectCount(GLenum mode, const void * indirect, GLintptr drawcount, GLintptr maxdrawcount, GLsizei stride)
{
    if (pGlMultiDrawArrysIndirectCount)
        pGlMultiDrawArrysIndirectCount(mode, indirect, drawcount, maxdrawcount, stride);
}

inline void COpenGLExtensionHandler::extGlMultiDrawElementsIndirectCount(GLenum mode, GLenum type, const void * indirect, GLintptr drawcount, GLintptr maxdrawcount, GLsizei stride)
{
    if (pGlMultiDrawElementsIndirectCount)
        pGlMultiDrawElementsIndirectCount(mode, type, indirect, drawcount, maxdrawcount, stride);
}



// ROP
inline void COpenGLExtensionHandler::extGlBlendColor(float red, float green, float blue, float alpha)
{
	if (pGlBlendColor)
		pGlBlendColor(red,green,blue,alpha);
}
inline void COpenGLExtensionHandler::extGlDepthRangeIndexed(GLuint index, GLdouble nearVal, GLdouble farVal)
{
	if (pGlDepthRangeIndexed)
		pGlDepthRangeIndexed(index,nearVal,farVal);
}
inline void COpenGLExtensionHandler::extGlViewportIndexedfv(GLuint index, const GLfloat* v)
{
	if (pGlViewportIndexedfv)
		pGlViewportIndexedfv(index,v);
}
inline void COpenGLExtensionHandler::extGlScissorIndexedv(GLuint index, const GLint* v)
{
	if (pGlScissorIndexedv)
		pGlScissorIndexedv(index,v);
}
inline void COpenGLExtensionHandler::extGlSampleCoverage(float value, bool invert)
{
	if (pGlSampleCoverage)
		pGlSampleCoverage(value,invert);
}
inline void COpenGLExtensionHandler::extGlSampleMaski(GLuint maskNumber, GLbitfield mask)
{
	if (pGlSampleMaski)
		pGlSampleMaski(maskNumber,mask);
}
inline void COpenGLExtensionHandler::extGlMinSampleShading(float value)
{
	if (pGlMinSampleShading)
		pGlMinSampleShading(value);
}
inline void COpenGLExtensionHandler::extGlBlendEquationSeparatei(GLuint buf, GLenum modeRGB, GLenum modeAlpha)
{
	if (pGlBlendEquationSeparatei)
		pGlBlendEquationSeparatei(buf,modeRGB,modeAlpha);
}
inline void COpenGLExtensionHandler::extGlBlendFuncSeparatei(GLuint buf, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha)
{
	if (pGlBlendFuncSeparatei)
		pGlBlendFuncSeparatei(buf,srcRGB,dstRGB,srcAlpha,dstAlpha);
}
inline void COpenGLExtensionHandler::extGlColorMaski(GLuint buf, GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha)
{
	if (pGlColorMaski)
		pGlColorMaski(buf,red,green,blue,alpha);
}



inline void COpenGLExtensionHandler::extGlBlendFuncSeparate(GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha)
{
	if (pGlBlendFuncSeparate)
		pGlBlendFuncSeparate(srcRGB,dstRGB,srcAlpha,dstAlpha);
}


inline void COpenGLExtensionHandler::extGlColorMaskIndexed(GLuint buf, GLboolean r, GLboolean g, GLboolean b, GLboolean a)
{
	if (pGlColorMaski)
		pGlColorMaski(buf, r, g, b, a);
}


inline void COpenGLExtensionHandler::extGlEnableIndexed(GLenum target, GLuint index)
{
	if (pGlEnablei)
		pGlEnablei(target, index);
}

inline void COpenGLExtensionHandler::extGlDisableIndexed(GLenum target, GLuint index)
{
	if (pGlDisablei)
		pGlDisablei(target, index);
}

inline void COpenGLExtensionHandler::extGlBlendFuncIndexed(GLuint buf, GLenum src, GLenum dst)
{
	pGlBlendFunciARB(buf, src, dst);
}

inline void COpenGLExtensionHandler::extGlBlendEquationIndexed(GLuint buf, GLenum mode)
{
	pGlBlendEquationiARB(buf, mode);
}

inline void COpenGLExtensionHandler::extGlPatchParameterfv(GLenum pname, const float* values)
{
	pGlPatchParameterfv(pname, values);
}

inline void COpenGLExtensionHandler::extGlPatchParameteri(GLenum pname, GLuint value)
{
	pGlPatchParameteri(pname, value);
}

inline void COpenGLExtensionHandler::extGlProgramParameteri(GLuint program, GLenum pname, GLint value)
{
	pGlProgramParameteri(program, pname, value);
}

inline void COpenGLExtensionHandler::extGlCreateQueries(GLenum target, GLsizei n, GLuint *ids)
{
    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlCreateQueries)
            pGlCreateQueries(target, n, ids);
    }
    else
    {
        if (pGlGenQueries)
            pGlGenQueries(n, ids);
    }
}

inline void COpenGLExtensionHandler::extGlDeleteQueries(GLsizei n, const GLuint *ids)
{
	if (pGlDeleteQueries)
		pGlDeleteQueries(n, ids);
}

inline GLboolean COpenGLExtensionHandler::extGlIsQuery(GLuint id)
{
	if (pGlIsQuery)
		return pGlIsQuery(id);
	return false;
}

inline void COpenGLExtensionHandler::extGlBeginQuery(GLenum target, GLuint id)
{
	if (pGlBeginQuery)
		pGlBeginQuery(target, id);
}

inline void COpenGLExtensionHandler::extGlEndQuery(GLenum target)
{
	if (pGlEndQuery)
		pGlEndQuery(target);
}

inline void COpenGLExtensionHandler::extGlBeginQueryIndexed(GLenum target, GLuint index, GLuint id)
{
	if (pGlBeginQueryIndexed)
		pGlBeginQueryIndexed(target, index, id);
}

inline void COpenGLExtensionHandler::extGlEndQueryIndexed(GLenum target, GLuint index)
{
	if (pGlEndQueryIndexed)
		pGlEndQueryIndexed(target, index);
}


inline void COpenGLExtensionHandler::extGlGetQueryObjectuiv(GLuint id, GLenum pname, GLuint *params)
{
	if (pGlGetQueryObjectuiv)
		pGlGetQueryObjectuiv(id, pname, params);
}

inline void COpenGLExtensionHandler::extGlGetQueryObjectui64v(GLuint id, GLenum pname, GLuint64 *params)
{
	if (pGlGetQueryObjectui64v)
		pGlGetQueryObjectui64v(id, pname, params);
}

inline void COpenGLExtensionHandler::extGlGetQueryBufferObjectuiv(GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
{
    if (Version<440 && !FeatureAvailable[NBL_ARB_query_buffer_object])
    {
#ifdef _DEBuG
        os::Printer::log("GL_ARB_query_buffer_object unsupported!\n");
#endif // _DEBuG
        return;
    }

    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlGetQueryBufferObjectuiv)
            pGlGetQueryBufferObjectuiv(id, buffer, pname, offset);
    }
    else
    {
        GLint restoreQueryBuffer;
        glGetIntegerv(GL_QUERY_BUFFER_BINDING, &restoreQueryBuffer);
        pGlBindBuffer(GL_QUERY_BUFFER,id);
        if (pGlGetQueryObjectuiv)
            pGlGetQueryObjectuiv(id, pname, reinterpret_cast<GLuint*>(offset));
        pGlBindBuffer(GL_QUERY_BUFFER,restoreQueryBuffer);
    }
}

inline void COpenGLExtensionHandler::extGlGetQueryBufferObjectui64v(GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
{
    if (Version<440 && !FeatureAvailable[NBL_ARB_query_buffer_object])
    {
#ifdef _DEBuG
        os::Printer::log("GL_ARB_query_buffer_object unsupported!\n");
#endif // _DEBuG
        return;
    }

    if (Version>=450||FeatureAvailable[NBL_ARB_direct_state_access])
    {
        if (pGlGetQueryBufferObjectui64v)
            pGlGetQueryBufferObjectui64v(id, buffer, pname, offset);
    }
    else
    {
        GLint restoreQueryBuffer;
        glGetIntegerv(GL_QUERY_BUFFER_BINDING, &restoreQueryBuffer);
        pGlBindBuffer(GL_QUERY_BUFFER,id);
        if (pGlGetQueryObjectui64v)
            pGlGetQueryObjectui64v(id, pname, reinterpret_cast<GLuint64*>(offset));
        pGlBindBuffer(GL_QUERY_BUFFER,restoreQueryBuffer);
    }
}

inline void COpenGLExtensionHandler::extGlQueryCounter(GLuint id, GLenum target)
{
	if (pGlQueryCounter)
		pGlQueryCounter(id, target);
}

inline void COpenGLExtensionHandler::extGlBeginConditionalRender(GLuint id, GLenum mode)
{
	if (pGlBeginConditionalRender)
		pGlBeginConditionalRender(id, mode);
}

inline void COpenGLExtensionHandler::extGlEndConditionalRender()
{
	if (pGlEndConditionalRender)
		pGlEndConditionalRender();
}


inline void COpenGLExtensionHandler::extGlTextureBarrier()
{
	if (FeatureAvailable[NBL_ARB_texture_barrier])
		pGlTextureBarrier();
	else if (FeatureAvailable[NBL_NV_texture_barrier])
		pGlTextureBarrierNV();
#ifdef _NBL_DEBUG
    else
        os::Printer::log("EDF_TEXTURE_BARRIER Not Available!\n",ELL_ERROR);
#endif // _NBL_DEBUG
}


inline void COpenGLExtensionHandler::extGlSwapInterval(int interval)
{
	// we have wglext, so try to use that
#if defined(_NBL_WINDOWS_API_) && defined(_NBL_COMPILE_WITH_WINDOWS_DEVICE_)
#ifdef WGL_EXT_swap_control
	if (pWglSwapIntervalEXT)
		pWglSwapIntervalEXT(interval);
#endif
#endif
#ifdef _NBL_COMPILE_WITH_X11_DEVICE_
	//TODO: Check GLX_EXT_swap_control and GLX_MESA_swap_control
#ifdef GLX_SGI_swap_control
	// does not work with interval==0
	if (interval && pGlxSwapIntervalSGI)
		pGlxSwapIntervalSGI(interval);
#elif defined(GLX_EXT_swap_control)
	Display *dpy = glXGetCurrentDisplay();
	GLXDrawable drawable = glXGetCurrentDrawable();

	if (pGlxSwapIntervalEXT)
		pGlxSwapIntervalEXT(dpy, drawable, interval);
#elif defined(GLX_MESA_swap_control)
	if (pGlxSwapIntervalMESA)
		pGlxSwapIntervalMESA(interval);
#endif
#endif
}

inline void COpenGLExtensionHandler::extGlBlendEquation(GLenum mode)
{
	pGlBlendEquation(mode);
}

inline void COpenGLExtensionHandler::extGlGetInternalformativ(GLenum target, GLenum internalformat, GLenum pname, GLsizei bufSize, GLint* params)
{
    if (Version>=460 || FeatureAvailable[NBL_ARB_internalformat_query])
    {
        if (pGlGetInternalformativ)
            pGlGetInternalformativ(target, internalformat, pname, bufSize, params);
    }
}

inline void COpenGLExtensionHandler::extGlGetInternalformati64v(GLenum target, GLenum internalformat, GLenum pname, GLsizei bufSize, GLint64* params)
{
    if (Version>=460 || FeatureAvailable[NBL_ARB_internalformat_query])
    {
        if (pGlGetInternalformati64v)
            pGlGetInternalformati64v(target, internalformat, pname, bufSize, params);
    }
}

}
}

#endif

#endif

