#ifndef __NBL_C_OPEN_GL_FEATURE_MAP_INCLUDED__
#define __NBL_C_OPEN_GL_FEATURE_MAP_INCLUDED__

// TODO review this whole file

#include "nbl/macros.h"
#include <cstdint>
#include <cstring>

namespace nbl {
namespace video
{

class COpenGLFeatureMap
{
public:
	constexpr inline static const char* const OpenGLFeatureStrings[] = {
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
	"GL_EXT_draw_buffers_indexed",
	"GL_EXT_draw_instanced",
	"GL_EXT_draw_range_elements",
	"GL_EXT_fog_coord",
	"GL_EXT_framebuffer_blit",
	"GL_EXT_framebuffer_multisample",
	"GL_EXT_framebuffer_multisample_blit_scaled",
	"GL_EXT_framebuffer_object",
	"GL_EXT_framebuffer_sRGB",
	"GL_EXT_geometry_shader",
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
	"GL_EXT_multisample_compatibility",
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
	"GL_EXT_tessellation_shader",
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
	"GL_EXT_texture_mirror_clamp_to_edge",
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
	"GL_EXT_copy_image",
	"GL_EXT_draw_elements_base_vertex",
	"GL_EXT_base_instance",
	"GL_EXT_multi_draw_indirect",
	"GL_EXT_depth_clamp",
	"GL_EXT_clip_control",
	"GL_EXT_clip_cull_distance",
	"GL_EXT_debug_label",
	"GL_EXT_nonuniform_qualifier",
	"GL_EXT_disjoint_timer_query",
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
	"GL_NV_polygon_mode",
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
	"GL_NV_viewport_array",
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
	"GL_NV_conservative_raster",
	"GL_NV_conservative_raster_dilate",
	"GL_NV_conservative_raster_pre_snap",
	"GL_NV_conservative_raster_pre_snap_triangles",
	"GL_NV_conservative_raster_underestimation",
	"GL_NV_shader_texture_footprint",
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
	"GL_KHR_shader_subgroup",
	//"GLX_EXT_swap_control_tear",
	"GL_NVX_gpu_memory_info",
	"GL_NVX_multiview_per_view_attributes",
	"GL_OES_read_format",
	"GL_OES_texture_compression_astc_hdr",
	"GL_OES_texture_compression_astc_ldr",
	"GL_OES_texture_border_clamp",
	"GL_OES_texture_cube_map_array",
	"GL_OES_sample_shading",
	"GL_OES_copy_image",
	"GL_OES_viewport_array",
	"GL_OES_draw_buffers_indexed",
	"GL_OES_draw_elements_base_vertex",
	"GL_OES_geometry_shader",
	"GL_OES_tessellation_shader"
};
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
		NBL_EXT_draw_buffers_indexed,
		NBL_EXT_draw_instanced,
		NBL_EXT_draw_range_elements,
		NBL_EXT_fog_coord,
		NBL_EXT_framebuffer_blit,
		NBL_EXT_framebuffer_multisample,
		NBL_EXT_framebuffer_multisample_blit_scaled,
		NBL_EXT_framebuffer_object,
		NBL_EXT_framebuffer_sRGB,
		NBL_EXT_geometry_shader,
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
		NBL_EXT_multisample_compatibility,
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
		NBL_EXT_tessellation_shader,
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
		NBL_EXT_texture_mirror_clamp_to_edge,
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
		NBL_EXT_copy_image,
		NBL_EXT_draw_elements_base_vertex,
		NBL_EXT_base_instance,
		NBL_EXT_multi_draw_indirect,
		NBL_EXT_depth_clamp,
		NBL_EXT_clip_control,
		NBL_EXT_clip_cull_distance,
		NBL_EXT_debug_label,
		NBL_EXT_nonuniform_qualifier,
		NBL_EXT_disjoint_timer_query,
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
		NBL_NV_polygon_mode,
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
		NBL_NV_viewport_array,
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
		NBL_NV_conservative_raster,
		NBL_NV_conservative_raster_dilate,
		NBL_NV_conservative_raster_pre_snap,
		NBL_NV_conservative_raster_pre_snap_triangles,
		NBL_NV_conservative_raster_underestimation,
		NBL_NV_shader_texture_footprint,
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
		NBL_KHR_shader_subgroup,
		NBL_NVX_gpu_memory_info,
		NBL_NVX_multiview_per_view_attributes,
		NBL_OES_read_format,
		NBL_OES_texture_compression_astc_hdr,
		NBL_OES_texture_compression_astc_ldr,
		NBL_OES_texture_border_clamp,
		NBL_OES_texture_cube_map_array,
		NBL_OES_sample_shading,
		NBL_OES_copy_image,
		NBL_OES_viewport_array,
		NBL_OES_draw_buffers_indexed,
		NBL_OES_draw_elements_base_vertex,
		NBL_OES_geometry_shader,
		NBL_OES_tessellation_shader,

		NBL_OpenGL_Feature_Count
	};

	static inline constexpr EOpenGLFeatures m_GLSLExtensions[]{
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

	bool FeatureAvailable[NBL_OpenGL_Feature_Count];

	//Version of OpenGL multiplied by 100 - 4.4 becomes 440
	uint16_t Version = 0;
	uint16_t ShaderLanguageVersion = 0;

	//!
	uint32_t maxUBOBindings;
	//!
	uint32_t maxSSBOBindings;
	//! For vertex and fragment shaders
	//! If both the vertex shader and the fragment processing stage access the same texture image unit, then that counts as using two texture image units against this limit.
	uint32_t maxTextureBindings;
	//! For compute shader
	uint32_t maxTextureBindingsCompute;
	//!
	uint32_t maxImageBindings;
	//! Number of rendertargets available as MRTs
	uint8_t MaxMultipleRenderTargets;
	//! Maximal LOD Bias
	float MaxTextureLODBias;

	bool isIntelGPU = false;
	// seems to be always true in our current code (COpenGLExtensionHandler, COpenGLDriver)
	bool needsDSAFramebufferHack = true;

	bool runningInRenderDoc = false;

	COpenGLFeatureMap()
	{
		memset(FeatureAvailable, 0, sizeof(FeatureAvailable));
	}

	bool isFeatureAvailable(EOpenGLFeatures feature) const
	{
		return FeatureAvailable[feature];
	}
};

} //end of namespace video
} //end of namespace irr

#endif