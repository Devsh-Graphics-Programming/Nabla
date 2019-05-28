// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in Irrlicht.h

#ifndef __C_OPEN_GL_FEATURE_MAP_H_INCLUDED__
#define __C_OPEN_GL_FEATURE_MAP_H_INCLUDED__

#include "IrrCompileConfig.h"
#ifdef _IRR_COMPILE_WITH_OPENGL_

#define _IRR_OPENGL_USE_EXTPOINTER_

#include "IMaterialRendererServices.h"
#include "irr/core/Types.h"
#include "irr/macros.h"
#include "os.h"
#include "coreutil.h"

#include "COpenGLStateManager.h"
#include "COpenGLCubemapTexture.h"

namespace irr
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


E_SHADER_CONSTANT_TYPE getIrrUniformType(GLenum oglType);


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
	"GL_OES_read_format",
	"GL_OML_interlace",
	"GL_OML_resample",
	"GL_OML_subsample",
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
	//"GLX_EXT_swap_control_tear",
	"GL_NVX_gpu_memory_info"
};


class COpenGLExtensionHandler
{
	public:
	enum EOpenGLFeatures {
		IRR_3DFX_multisample = 0,
		IRR_3DFX_tbuffer,
		IRR_3DFX_texture_compression_FXT1,
		IRR_AMD_blend_minmax_factor,
		IRR_AMD_conservative_depth,
		IRR_AMD_debug_output,
		IRR_AMD_depth_clamp_separate,
		IRR_AMD_draw_buffers_blend,
		IRR_AMD_multi_draw_indirect,
		IRR_AMD_name_gen_delete,
		IRR_AMD_performance_monitor,
		IRR_AMD_sample_positions,
		IRR_AMD_seamless_cubemap_per_texture,
		IRR_AMD_shader_stencil_export,
		IRR_AMD_texture_texture4,
		IRR_AMD_transform_feedback3_lines_triangles,
		IRR_AMD_vertex_shader_tesselator,
		IRR_APPLE_aux_depth_stencil,
		IRR_APPLE_client_storage,
		IRR_APPLE_element_array,
		IRR_APPLE_fence,
		IRR_APPLE_float_pixels,
		IRR_APPLE_flush_buffer_range,
		IRR_APPLE_object_purgeable,
		IRR_APPLE_rgb_422,
		IRR_APPLE_row_bytes,
		IRR_APPLE_specular_vector,
		IRR_APPLE_texture_range,
		IRR_APPLE_transform_hint,
		IRR_APPLE_vertex_array_object,
		IRR_APPLE_vertex_array_range,
		IRR_APPLE_vertex_program_evaluators,
		IRR_APPLE_ycbcr_422,
		IRR_ARB_base_instance,
		IRR_ARB_bindless_texture,
		IRR_ARB_buffer_storage,
		IRR_ARB_blend_func_extended,
		IRR_ARB_clip_control,
		IRR_ARB_cl_event,
		IRR_ARB_color_buffer_float,
		IRR_ARB_compatibility,
		IRR_ARB_compressed_texture_pixel_storage,
		IRR_ARB_compute_shader,
		IRR_ARB_conservative_depth,
		IRR_ARB_copy_buffer,
		IRR_ARB_debug_output,
		IRR_ARB_depth_buffer_float,
		IRR_ARB_depth_clamp,
		IRR_ARB_depth_texture,
		IRR_ARB_direct_state_access,
		IRR_ARB_draw_buffers,
		IRR_ARB_draw_buffers_blend,
		IRR_ARB_draw_elements_base_vertex,
		IRR_ARB_draw_indirect,
		IRR_ARB_draw_instanced,
		IRR_ARB_ES2_compatibility,
		IRR_ARB_explicit_attrib_location,
		IRR_ARB_explicit_uniform_location,
		IRR_ARB_fragment_coord_conventions,
		IRR_ARB_fragment_program,
		IRR_ARB_fragment_program_shadow,
		IRR_ARB_fragment_shader,
		IRR_ARB_fragment_shader_interlock,
		IRR_ARB_framebuffer_object,
		IRR_ARB_framebuffer_sRGB,
		IRR_ARB_geometry_shader4,
		IRR_ARB_get_program_binary,
		IRR_ARB_get_texture_sub_image,
		IRR_ARB_gpu_shader5,
		IRR_ARB_gpu_shader_fp64,
		IRR_ARB_half_float_pixel,
		IRR_ARB_half_float_vertex,
		IRR_ARB_imaging,
		IRR_ARB_instanced_arrays,
		IRR_ARB_indirect_parameters,
		IRR_ARB_internalformat_query,
		IRR_ARB_internalformat_query2,
		IRR_ARB_map_buffer_alignment,
		IRR_ARB_map_buffer_range,
		IRR_ARB_matrix_palette,
		IRR_ARB_multi_bind,
		IRR_ARB_multi_draw_indirect,
		IRR_ARB_multisample,
		IRR_ARB_multitexture,
		IRR_ARB_occlusion_query,
		IRR_ARB_occlusion_query2,
		IRR_ARB_pixel_buffer_object,
		IRR_ARB_point_parameters,
		IRR_ARB_point_sprite,
		IRR_ARB_program_interface_query,
		IRR_ARB_provoking_vertex,
		IRR_ARB_query_buffer_object,
		IRR_ARB_robustness,
		IRR_ARB_sample_shading,
		IRR_ARB_sampler_objects,
		IRR_ARB_seamless_cube_map,
		IRR_ARB_separate_shader_objects,
		IRR_ARB_shader_atomic_counters,
		IRR_ARB_shader_ballot,
		IRR_ARB_shader_bit_encoding,
		IRR_ARB_shader_draw_parameters,
		IRR_ARB_shader_group_vote,
		IRR_ARB_shader_image_load_store,
		IRR_ARB_shader_objects,
		IRR_ARB_shader_precision,
		IRR_ARB_shader_stencil_export,
		IRR_ARB_shader_subroutine,
		IRR_ARB_shader_texture_lod,
		IRR_ARB_shading_language_100,
		IRR_ARB_shading_language_420pack,
		IRR_ARB_shading_language_include,
		IRR_ARB_shading_language_packing,
		IRR_ARB_shadow,
		IRR_ARB_shadow_ambient,
		IRR_ARB_sync,
		IRR_ARB_tessellation_shader,
		IRR_ARB_texture_barrier,
		IRR_ARB_texture_border_clamp,
		IRR_ARB_texture_buffer_object,
		IRR_ARB_texture_buffer_object_rgb32,
		IRR_ARB_texture_buffer_range,
		IRR_ARB_texture_compression,
		IRR_ARB_texture_compression_bptc,
		IRR_ARB_texture_compression_rgtc,
		IRR_ARB_texture_cube_map,
		IRR_ARB_texture_cube_map_array,
		IRR_ARB_texture_env_add,
		IRR_ARB_texture_env_combine,
		IRR_ARB_texture_env_crossbar,
		IRR_ARB_texture_env_dot3,
		IRR_ARB_texture_float,
		IRR_ARB_texture_gather,
		IRR_ARB_texture_mirrored_repeat,
		IRR_ARB_texture_multisample,
		IRR_ARB_texture_non_power_of_two,
		IRR_ARB_texture_query_lod,
		IRR_ARB_texture_rectangle,
		IRR_ARB_texture_rg,
		IRR_ARB_texture_rgb10_a2ui,
		IRR_ARB_texture_stencil8,
		IRR_ARB_texture_storage,
		IRR_ARB_texture_storage_multisample,
		IRR_ARB_texture_swizzle,
		IRR_ARB_texture_view,
		IRR_ARB_timer_query,
		IRR_ARB_transform_feedback2,
		IRR_ARB_transform_feedback3,
		IRR_ARB_transform_feedback_instanced,
		IRR_ARB_transpose_matrix,
		IRR_ARB_uniform_buffer_object,
		IRR_ARB_vertex_array_bgra,
		IRR_ARB_vertex_array_object,
		IRR_ARB_vertex_attrib_64bit,
		IRR_ARB_vertex_attrib_binding,
		IRR_ARB_vertex_blend,
		IRR_ARB_vertex_buffer_object,
		IRR_ARB_vertex_program,
		IRR_ARB_vertex_shader,
		IRR_ARB_vertex_type_2_10_10_10_rev,
		IRR_ARB_viewport_array,
		IRR_ARB_window_pos,
		IRR_ATI_draw_buffers,
		IRR_ATI_element_array,
		IRR_ATI_envmap_bumpmap,
		IRR_ATI_fragment_shader,
		IRR_ATI_map_object_buffer,
		IRR_ATI_meminfo,
		IRR_ATI_pixel_format_float,
		IRR_ATI_pn_triangles,
		IRR_ATI_separate_stencil,
		IRR_ATI_text_fragment_shader,
		IRR_ATI_texture_env_combine3,
		IRR_ATI_texture_float,
		IRR_ATI_texture_mirror_once,
		IRR_ATI_vertex_array_object,
		IRR_ATI_vertex_attrib_array_object,
		IRR_ATI_vertex_streams,
		IRR_EXT_422_pixels,
		IRR_EXT_abgr,
		IRR_EXT_bgra,
		IRR_EXT_bindable_uniform,
		IRR_EXT_blend_color,
		IRR_EXT_blend_equation_separate,
		IRR_EXT_blend_func_separate,
		IRR_EXT_blend_logic_op,
		IRR_EXT_blend_minmax,
		IRR_EXT_blend_subtract,
		IRR_EXT_clip_volume_hint,
		IRR_EXT_cmyka,
		IRR_EXT_color_subtable,
		IRR_EXT_compiled_vertex_array,
		IRR_EXT_convolution,
		IRR_EXT_coordinate_frame,
		IRR_EXT_copy_texture,
		IRR_EXT_cull_vertex,
		IRR_EXT_depth_bounds_test,
		IRR_EXT_direct_state_access,
		IRR_EXT_draw_buffers2,
		IRR_EXT_draw_instanced,
		IRR_EXT_draw_range_elements,
		IRR_EXT_fog_coord,
		IRR_EXT_framebuffer_blit,
		IRR_EXT_framebuffer_multisample,
		IRR_EXT_framebuffer_multisample_blit_scaled,
		IRR_EXT_framebuffer_object,
		IRR_EXT_framebuffer_sRGB,
		IRR_EXT_geometry_shader4,
		IRR_EXT_gpu_program_parameters,
		IRR_EXT_gpu_shader4,
		IRR_EXT_histogram,
		IRR_EXT_index_array_formats,
		IRR_EXT_index_func,
		IRR_EXT_index_material,
		IRR_EXT_index_texture,
		IRR_EXT_light_texture,
		IRR_EXT_misc_attribute,
		IRR_EXT_multi_draw_arrays,
		IRR_EXT_multisample,
		IRR_EXT_packed_depth_stencil,
		IRR_EXT_packed_float,
		IRR_EXT_packed_pixels,
		IRR_EXT_paletted_texture,
		IRR_EXT_pixel_buffer_object,
		IRR_EXT_pixel_transform,
		IRR_EXT_pixel_transform_color_table,
		IRR_EXT_point_parameters,
		IRR_EXT_polygon_offset,
		IRR_EXT_provoking_vertex,
		IRR_EXT_rescale_normal,
		IRR_EXT_secondary_color,
		IRR_EXT_separate_shader_objects,
		IRR_EXT_separate_specular_color,
		IRR_EXT_shader_image_load_store,
		IRR_EXT_shadow_funcs,
		IRR_EXT_shared_texture_palette,
		IRR_EXT_stencil_clear_tag,
		IRR_EXT_stencil_two_side,
		IRR_EXT_stencil_wrap,
		IRR_EXT_subtexture,
		IRR_EXT_texture,
		IRR_EXT_texture3D,
		IRR_EXT_texture_array,
		IRR_EXT_texture_buffer_object,
		IRR_EXT_texture_compression_latc,
		IRR_EXT_texture_compression_rgtc,
		IRR_EXT_texture_compression_s3tc,
		IRR_EXT_texture_cube_map,
		IRR_EXT_texture_env_add,
		IRR_EXT_texture_env_combine,
		IRR_EXT_texture_env_dot3,
		IRR_EXT_texture_filter_anisotropic,
		IRR_EXT_texture_integer,
		IRR_EXT_texture_lod_bias,
		IRR_EXT_texture_mirror_clamp,
		IRR_EXT_texture_object,
		IRR_EXT_texture_perturb_normal,
		IRR_EXT_texture_shared_exponent,
		IRR_EXT_texture_snorm,
		IRR_EXT_texture_sRGB,
		IRR_EXT_texture_sRGB_decode,
		IRR_EXT_texture_swizzle,
		IRR_EXT_texture_view,
		IRR_EXT_timer_query,
		IRR_EXT_transform_feedback,
		IRR_EXT_vertex_array,
		IRR_EXT_vertex_array_bgra,
		IRR_EXT_vertex_attrib_64bit,
		IRR_EXT_vertex_shader,
		IRR_EXT_vertex_weighting,
		IRR_EXT_x11_sync_object,
		IRR_FfdMaskSGIX,
		IRR_GREMEDY_frame_terminator,
		IRR_GREMEDY_string_marker,
		IRR_HP_convolution_border_modes,
		IRR_HP_image_transform,
		IRR_HP_occlusion_test,
		IRR_HP_texture_lighting,
		IRR_IBM_cull_vertex,
		IRR_IBM_multimode_draw_arrays,
		IRR_IBM_rasterpos_clip,
		IRR_IBM_texture_mirrored_repeat,
		IRR_IBM_vertex_array_lists,
		IRR_INGR_blend_func_separate,
		IRR_INGR_color_clamp,
		IRR_INGR_interlace_read,
		IRR_INGR_palette_buffer,
		IRR_INTEL_fragment_shader_ordering,
		IRR_INTEL_parallel_arrays,
		IRR_INTEL_texture_scissor,
		IRR_KHR_debug,
		IRR_MESA_pack_invert,
		IRR_MESA_resize_buffers,
		IRR_MESA_window_pos,
		IRR_MESAX_texture_stack,
		IRR_MESA_ycbcr_texture,
		IRR_NV_blend_square,
		IRR_NV_conditional_render,
		IRR_NV_copy_depth_to_color,
		IRR_NV_copy_image,
		IRR_NV_depth_buffer_float,
		IRR_NV_depth_clamp,
		IRR_NV_evaluators,
		IRR_NV_explicit_multisample,
		IRR_NV_fence,
		IRR_NV_float_buffer,
		IRR_NV_fog_distance,
		IRR_NV_fragment_program,
		IRR_NV_fragment_program2,
		IRR_NV_fragment_program4,
		IRR_NV_fragment_program_option,
		IRR_NV_fragment_shader_interlock,
		IRR_NV_framebuffer_multisample_coverage,
		IRR_NV_geometry_program4,
		IRR_NV_geometry_shader4,
		IRR_NV_gpu_program4,
		IRR_NV_gpu_program5,
		IRR_NV_gpu_shader5,
		IRR_NV_half_float,
		IRR_NV_light_max_exponent,
		IRR_NV_multisample_coverage,
		IRR_NV_multisample_filter_hint,
		IRR_NV_occlusion_query,
		IRR_NV_packed_depth_stencil,
		IRR_NV_parameter_buffer_object,
		IRR_NV_parameter_buffer_object2,
		IRR_NV_pixel_data_range,
		IRR_NV_point_sprite,
		IRR_NV_present_video,
		IRR_NV_primitive_restart,
		IRR_NV_register_combiners,
		IRR_NV_register_combiners2,
		IRR_NV_shader_buffer_load,
		IRR_NV_shader_buffer_store,
		IRR_NV_shader_thread_group,
		IRR_NV_shader_thread_shuffle,
		IRR_NV_tessellation_program5,
		IRR_NV_texgen_emboss,
		IRR_NV_texgen_reflection,
		IRR_NV_texture_barrier,
		IRR_NV_texture_compression_vtc,
		IRR_NV_texture_env_combine4,
		IRR_NV_texture_expand_normal,
		IRR_NV_texture_multisample,
		IRR_NV_texture_rectangle,
		IRR_NV_texture_shader,
		IRR_NV_texture_shader2,
		IRR_NV_texture_shader3,
		IRR_NV_transform_feedback,
		IRR_NV_transform_feedback2,
		IRR_NV_vdpau_interop,
		IRR_NV_vertex_array_range,
		IRR_NV_vertex_array_range2,
		IRR_NV_vertex_attrib_integer_64bit,
		IRR_NV_vertex_buffer_unified_memory,
		IRR_NV_vertex_program,
		IRR_NV_vertex_program1_1,
		IRR_NV_vertex_program2,
		IRR_NV_vertex_program2_option,
		IRR_NV_vertex_program3,
		IRR_NV_vertex_program4,
		IRR_NV_video_capture,
		IRR_OES_read_format,
		IRR_OML_interlace,
		IRR_OML_resample,
		IRR_OML_subsample,
		IRR_PGI_misc_hints,
		IRR_PGI_vertex_hints,
		IRR_REND_screen_coordinates,
		IRR_S3_s3tc,
		IRR_SGI_color_matrix,
		IRR_SGI_color_table,
		IRR_SGI_depth_pass_instrument,
		IRR_SGIS_detail_texture,
		IRR_SGIS_fog_function,
		IRR_SGIS_generate_mipmap,
		IRR_SGIS_multisample,
		IRR_SGIS_pixel_texture,
		IRR_SGIS_point_line_texgen,
		IRR_SGIS_point_parameters,
		IRR_SGIS_sharpen_texture,
		IRR_SGIS_texture4D,
		IRR_SGIS_texture_border_clamp,
		IRR_SGIS_texture_color_mask,
		IRR_SGIS_texture_edge_clamp,
		IRR_SGIS_texture_filter4,
		IRR_SGIS_texture_lod,
		IRR_SGIS_texture_select,
		IRR_SGI_texture_color_table,
		IRR_SGIX_async,
		IRR_SGIX_async_histogram,
		IRR_SGIX_async_pixel,
		IRR_SGIX_blend_alpha_minmax,
		IRR_SGIX_calligraphic_fragment,
		IRR_SGIX_clipmap,
		IRR_SGIX_convolution_accuracy,
		IRR_SGIX_depth_pass_instrument,
		IRR_SGIX_depth_texture,
		IRR_SGIX_flush_raster,
		IRR_SGIX_fog_offset,
		IRR_SGIX_fog_scale,
		IRR_SGIX_fragment_lighting,
		IRR_SGIX_framezoom,
		IRR_SGIX_igloo_interface,
		IRR_SGIX_impact_pixel_texture,
		IRR_SGIX_instruments,
		IRR_SGIX_interlace,
		IRR_SGIX_ir_instrument1,
		IRR_SGIX_list_priority,
		IRR_SGIX_pixel_texture,
		IRR_SGIX_pixel_tiles,
		IRR_SGIX_polynomial_ffd,
		IRR_SGIX_reference_plane,
		IRR_SGIX_resample,
		IRR_SGIX_scalebias_hint,
		IRR_SGIX_shadow,
		IRR_SGIX_shadow_ambient,
		IRR_SGIX_sprite,
		IRR_SGIX_subsample,
		IRR_SGIX_tag_sample_buffer,
		IRR_SGIX_texture_add_env,
		IRR_SGIX_texture_coordinate_clamp,
		IRR_SGIX_texture_lod_bias,
		IRR_SGIX_texture_multi_buffer,
		IRR_SGIX_texture_scale_bias,
		IRR_SGIX_texture_select,
		IRR_SGIX_vertex_preclip,
		IRR_SGIX_ycrcb,
		IRR_SGIX_ycrcba,
		IRR_SGIX_ycrcb_subsample,
		IRR_SUN_convolution_border_modes,
		IRR_SUN_global_alpha,
		IRR_SUN_mesh_array,
		IRR_SUN_slice_accum,
		IRR_SUN_triangle_list,
		IRR_SUN_vertex,
		IRR_SUNX_constant_data,
		IRR_WIN_phong_shading,
		IRR_WIN_specular_fog,
        IRR_KHR_texture_compression_astc_hdr,
        IRR_KHR_texture_compression_astc_ldr,
		//IRR_GLX_EXT_swap_control_tear,
		IRR_NVX_gpu_memory_info,
		IRR_OpenGL_Feature_Count
	};



	static core::LeakDebugger bufferLeaker;
	static core::LeakDebugger textureLeaker;

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
    static uint64_t maxTBOSize;
    //!
    static uint64_t maxBufferSize;
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
    //texture update functions
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
    static void extGlCopyTextureSubImage1D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width);
    static void extGlCopyTextureSubImage2D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height);
    static void extGlCopyTextureSubImage3D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height);
    static void extGlGenerateTextureMipmap(GLuint texture, GLenum target);
    static void extGlClampColor(GLenum target, GLenum clamp);
    static void setPixelUnpackAlignment(const uint32_t &pitchInBytes, void* ptr, const uint32_t& minimumAlignment=1);

    static void extGlGenSamplers(GLsizei n, GLuint* samplers);
    static void extGlDeleteSamplers(GLsizei n, GLuint* samplers);
    static void extGlBindSamplers(const GLuint& first, const GLsizei& count, const GLuint* samplers);
    static void extGlSamplerParameteri(GLuint sampler, GLenum pname, GLint param);
    static void extGlSamplerParameterf(GLuint sampler, GLenum pname, GLfloat param);

    //
    static void extGlBindImageTexture(GLuint index, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format);


	static void extGlPointParameterf(GLint loc, GLfloat f);
	static void extGlPointParameterfv(GLint loc, const GLfloat *v);
	static void extGlStencilFuncSeparate(GLenum face, GLenum func, GLint ref, GLuint mask);
	static void extGlStencilOpSeparate(GLenum face, GLenum fail, GLenum zfail, GLenum zpass);
	static void extGlStencilMaskSeparate(GLenum face, GLuint mask);


	// shader programming
	static GLuint extGlCreateShader(GLenum shaderType);
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
	static bool FeatureAvailable[IRR_OpenGL_Feature_Count];


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
    static PFNGLCOPYTEXSUBIMAGE3DPROC pGlCopyTexSubImage3D;
    static PFNGLCOPYTEXTURESUBIMAGE1DPROC pGlCopyTextureSubImage1D;
    static PFNGLCOPYTEXTURESUBIMAGE2DPROC pGlCopyTextureSubImage2D;
    static PFNGLCOPYTEXTURESUBIMAGE3DPROC pGlCopyTextureSubImage3D;
    static PFNGLCOPYTEXTURESUBIMAGE1DEXTPROC pGlCopyTextureSubImage1DEXT;
    static PFNGLCOPYTEXTURESUBIMAGE2DEXTPROC pGlCopyTextureSubImage2DEXT;
    static PFNGLCOPYTEXTURESUBIMAGE3DEXTPROC pGlCopyTextureSubImage3DEXT;
    static PFNGLGENERATEMIPMAPPROC pGlGenerateMipmap;
    static PFNGLGENERATETEXTUREMIPMAPPROC pGlGenerateTextureMipmap; //NULL
    static PFNGLGENERATETEXTUREMIPMAPEXTPROC pGlGenerateTextureMipmapEXT;
    static PFNGLCLAMPCOLORPROC pGlClampColor;

    //samplers
    static PFNGLGENSAMPLERSPROC pGlGenSamplers;
    static PFNGLCREATESAMPLERSPROC pGlCreateSamplers; //NULL
    static PFNGLDELETESAMPLERSPROC pGlDeleteSamplers;
    static PFNGLBINDSAMPLERPROC pGlBindSampler;
    static PFNGLBINDSAMPLERSPROC pGlBindSamplers;
    static PFNGLSAMPLERPARAMETERIPROC pGlSamplerParameteri;
    static PFNGLSAMPLERPARAMETERFPROC pGlSamplerParameterf;

    //
    static PFNGLBINDIMAGETEXTUREPROC pGlBindImageTexture;

    // stuff
    static PFNGLBINDBUFFERBASEPROC pGlBindBufferBase;
    static PFNGLBINDBUFFERRANGEPROC pGlBindBufferRange;
    static PFNGLBINDBUFFERSBASEPROC pGlBindBuffersBase;
    static PFNGLBINDBUFFERSRANGEPROC pGlBindBuffersRange;

    //shaders
    static PFNGLBINDATTRIBLOCATIONPROC pGlBindAttribLocation; //NULL
    static PFNGLCREATEPROGRAMPROC pGlCreateProgram;
    static PFNGLUSEPROGRAMPROC pGlUseProgram;
    static PFNGLDELETEPROGRAMPROC pGlDeleteProgram;
    static PFNGLDELETESHADERPROC pGlDeleteShader;
    static PFNGLGETATTACHEDSHADERSPROC pGlGetAttachedShaders;
    static PFNGLCREATESHADERPROC pGlCreateShader;
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

    if (Version>=440||FeatureAvailable[IRR_ARB_multi_bind])
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
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
		pGlCreateTextures(target,n,textures);
    else
        glGenTextures(n,textures);
}

inline void COpenGLExtensionHandler::extGlTextureBuffer(GLuint texture, GLenum internalformat, GLuint buffer)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
		pGlTextureBuffer(texture,internalformat,buffer);
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
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
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlTextureBufferRange)
            pGlTextureBufferRange(texture,internalformat,buffer,offset,length);
#else
        glTextureBufferRange(texture,internalformat,buffer,offset,length);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlTextureBufferRangeEXT)
            pGlTextureBufferRangeEXT(texture,GL_TEXTURE_BUFFER,internalformat,buffer,offset,length);
#else
        glTextureBufferRangeEXT(texture,GL_TEXTURE_BUFFER,internalformat,buffer,offset,length);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
    else
    {
        GLint bound;
        glGetIntegerv(GL_TEXTURE_BINDING_BUFFER, &bound);
        glBindTexture(GL_TEXTURE_BUFFER, texture);
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlTexBufferRange)
            pGlTexBufferRange(GL_TEXTURE_BUFFER,internalformat,buffer,offset,length);
#else
        glTexBufferRange(GL_TEXTURE_BUFFER,internalformat,buffer,offset,length);
#endif // _IRR_OPENGL_USE_EXTPOINTER
        glBindTexture(GL_TEXTURE_BUFFER, bound);
    }
}

inline void COpenGLExtensionHandler::extGlTextureStorage1D(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlTextureStorage1D)
            pGlTextureStorage1D(texture,levels,internalformat,width);
#else
        glTextureStorage1D(texture,levels,internalformat,width);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlTextureStorage1DEXT)
            pGlTextureStorage1DEXT(texture,target,levels,internalformat,width);
#else
        glTextureStorage1DEXT(texture,target,levels,internalformat,width);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlTexStorage1D)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
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
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlTexStorage1D(target,levels,internalformat,width);
#else
        glTexStorage1D(target,levels,internalformat,width);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
        glBindTexture(target, bound);
    }
}
inline void COpenGLExtensionHandler::extGlTextureStorage2D(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlTextureStorage2D)
            pGlTextureStorage2D(texture,levels,internalformat,width,height);
#else
        glTextureStorage2D(texture,levels,internalformat,width,height);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlTextureStorage2DEXT)
            pGlTextureStorage2DEXT(texture,target,levels,internalformat,width,height);
#else
        glTextureStorage2DEXT(texture,target,levels,internalformat,width,height);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlTexStorage2D)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
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
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlTexStorage2D(target,levels,internalformat,width,height);
#else
        glTexStorage2D(target,levels,internalformat,width,height);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
        glBindTexture(target, bound);
    }
}
inline void COpenGLExtensionHandler::extGlTextureStorage3D(GLuint texture, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
		pGlTextureStorage3D(texture,levels,internalformat,width,height,depth);
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
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
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
		pGlTextureStorage2DMultisample(texture,samples,internalformat,width,height,fixedsamplelocations);
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
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
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
		pGlTextureStorage3DMultisample(texture,samples,internalformat,width,height,depth,fixedsamplelocations);
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
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

inline void COpenGLExtensionHandler::extGlGetTextureSubImage(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLsizei bufSize, void* pixels)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_get_texture_sub_image])
		pGlGetTextureSubImage(texture, level, xoffset, yoffset, zoffset, width, height, depth, format, type, bufSize, pixels);
#ifdef _IRR_DEBUG
	else
		os::Printer::log("EDF_GET_TEXTURE_SUB_IMAGE Not Available!\n", ELL_ERROR);
#endif // _IRR_DEBUG
}

inline void COpenGLExtensionHandler::extGlGetCompressedTextureSubImage(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLsizei bufSize, void* pixels)
{
	if (Version >= 450 || FeatureAvailable[IRR_ARB_get_texture_sub_image])
		extGlGetCompressedTextureSubImage(texture, level, xoffset, yoffset, zoffset, width, height, depth, bufSize, pixels);
#ifdef _IRR_DEBUG
	else
		os::Printer::log("EDF_GET_TEXTURE_SUB_IMAGE Not Available!\n", ELL_ERROR);
#endif // _IRR_DEBUG
}

inline void COpenGLExtensionHandler::extGlGetTextureImage(GLuint texture, GLenum target, GLint level, GLenum format, GLenum type, GLsizei bufSizeHint, void* pixels)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
		pGlGetTextureImage(texture, level, format, type, bufSizeHint, pixels);
	else if (FeatureAvailable[IRR_EXT_direct_state_access])
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
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
		pGlGetCompressedTextureImage(texture, level, bufSizeHint, pixels);
	else if (FeatureAvailable[IRR_EXT_direct_state_access])
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
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
		pGlTextureSubImage1D(texture, level, xoffset, width,format, type, pixels);
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
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
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
		pGlTextureSubImage2D(texture, level, xoffset, yoffset,width, height,format, type, pixels);
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
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
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
		pGlTextureSubImage3D(texture, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
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
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlCompressedTextureSubImage1D)
            pGlCompressedTextureSubImage1D(texture, level, xoffset, width,format, imageSize, data);
#else
        glCompressedTextureSubImage1D(texture, level, xoffset, width,format, imageSize, data);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlCompressedTextureSubImage1DEXT)
            pGlCompressedTextureSubImage1DEXT(texture, target, level, xoffset, width,format, imageSize, data);
#else
        glCompressedTextureSubImage1DEXT(texture, target, level, xoffset, width,format, imageSize, data);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlCompressedTexSubImage1D)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
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
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlCompressedTexSubImage1D(target, level, xoffset, width,format, imageSize, data);
#else
        glCompressedTexSubImage1D(target, level, xoffset, width,format, imageSize, data);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
        glBindTexture(target, bound);
    }
}
inline void COpenGLExtensionHandler::extGlCompressedTextureSubImage2D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlCompressedTextureSubImage2D)
            pGlCompressedTextureSubImage2D(texture, level, xoffset, yoffset,width, height,format, imageSize, data);
#else
        glCompressedTextureSubImage2D(texture, level, xoffset, yoffset,width, height,format, imageSize, data);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlCompressedTextureSubImage2DEXT)
            pGlCompressedTextureSubImage2DEXT(texture, target, level, xoffset, yoffset,width, height,format, imageSize, data);
#else
        glCompressedTextureSubImage2DEXT(texture, target, level, xoffset, yoffset,width, height,format, imageSize, data);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlCompressedTexSubImage2D)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
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
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlCompressedTexSubImage2D(target, level, xoffset, yoffset,width, height,format, imageSize, data);
#else
        glCompressedTexSubImage2D(target, level, xoffset, yoffset,width, height,format, imageSize, data);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
        glBindTexture(target, bound);
    }
}
inline void COpenGLExtensionHandler::extGlCompressedTextureSubImage3D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *data)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlCompressedTextureSubImage3D)
            pGlCompressedTextureSubImage3D(texture, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
#else
        glCompressedTextureSubImage3D(texture, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlCompressedTextureSubImage3DEXT)
            pGlCompressedTextureSubImage3DEXT(texture, target, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
#else
        glCompressedTextureSubImage3DEXT(texture, target, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlCompressedTexSubImage3D)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
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
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlCompressedTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
#else
        glCompressedTexSubImage3D(target, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
        glBindTexture(target, bound);
    }
}

inline void COpenGLExtensionHandler::extGlCopyTextureSubImage1D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlCopyTextureSubImage1D)
            pGlCopyTextureSubImage1D(texture, level, xoffset, x, y, width);
#else
        glCopyTextureSubImage1D(texture, level, xoffset, x, y, width);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlCopyTextureSubImage1DEXT)
            pGlCopyTextureSubImage1DEXT(texture, target, level, xoffset, x, y, width);
#else
        glCopyTextureSubImage1DEXT(texture, target, level, xoffset, x, y, width);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlCopyTexSubImage3D)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
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
        glCopyTexSubImage1D(target, level, xoffset, x, y, width);
        glBindTexture(target, bound);
    }
}
inline void COpenGLExtensionHandler::extGlCopyTextureSubImage2D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlCopyTextureSubImage2D)
            pGlCopyTextureSubImage2D(texture, level, xoffset, yoffset, x, y, width, height);
#else
        glCopyTextureSubImage2D(texture, level, xoffset, yoffset, x, y, width, height);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlCopyTextureSubImage2DEXT)
            pGlCopyTextureSubImage2DEXT(texture, target, level, xoffset, yoffset, x, y, width, height);
#else
        glCopyTextureSubImage2DEXT(texture, target, level, xoffset, yoffset, x, y, width, height);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlCopyTexSubImage3D)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
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
        glCopyTexSubImage2D(target, level, xoffset, yoffset, x, y, width, height);
        glBindTexture(target, bound);
    }
}
inline void COpenGLExtensionHandler::extGlCopyTextureSubImage3D(GLuint texture, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlCopyTextureSubImage3D)
            pGlCopyTextureSubImage3D(texture, level, xoffset, yoffset, zoffset, x, y, width, height);
#else
        glCopyTextureSubImage3D(texture, level, xoffset, yoffset, zoffset, x, y, width, height);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlCopyTextureSubImage3DEXT)
            pGlCopyTextureSubImage3DEXT(texture, target, level, xoffset, yoffset, zoffset, x, y, width, height);
#else
        glCopyTextureSubImage3DEXT(texture, target, level, xoffset, yoffset, zoffset, x, y, width, height);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlCopyTexSubImage3D)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
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
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlCopyTexSubImage3D(target, level, xoffset, yoffset, zoffset, x, y, width, height);
#else
        glCopyTexSubImage3D(target, level, xoffset, yoffset, zoffset, x, y, width, height);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
        glBindTexture(target, bound);
    }
}

inline void COpenGLExtensionHandler::extGlGenerateTextureMipmap(GLuint texture, GLenum target)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlGenerateTextureMipmap)
            pGlGenerateTextureMipmap(texture);
#else
        glGenerateTextureMipmap(texture);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlGenerateTextureMipmapEXT)
            pGlGenerateTextureMipmapEXT(texture,target);
#else
        glGenerateTextureMipmapEXT(texture,target);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlGenerateMipmap)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
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
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlGenerateMipmap(target);
#else
        glGenerateMipmap(target);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
        glBindTexture(target, bound);
    }
}

inline void COpenGLExtensionHandler::extGlClampColor(GLenum target, GLenum clamp)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlClampColor)
        pGlClampColor(GL_CLAMP_READ_COLOR,clamp);
#else
    glClampColor(GL_CLAMP_READ_COLOR,clamp);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
}

inline void COpenGLExtensionHandler::setPixelUnpackAlignment(const uint32_t &pitchInBytes, void* ptr, const uint32_t& minimumAlignment)
{
#if _MSC_VER && !__INTEL_COMPILER
    DWORD textureUploadAlignment,textureUploadAlignment2;
    if (!_BitScanForward(&textureUploadAlignment,core::max_(pitchInBytes,minimumAlignment)))
        textureUploadAlignment = 3;
    if (!_BitScanForward64(&textureUploadAlignment2,*reinterpret_cast<size_t*>(&ptr)))
        textureUploadAlignment2 = 3;
#else
    int32_t textureUploadAlignment = __builtin_ffs(core::max_(pitchInBytes,minimumAlignment));
    if (textureUploadAlignment)
        textureUploadAlignment--;
    else
        textureUploadAlignment = 3;
    int32_t textureUploadAlignment2 = __builtin_ffs(*reinterpret_cast<size_t*>(&ptr));
    if (textureUploadAlignment2)
        textureUploadAlignment2--;
    else
        textureUploadAlignment2 = 3;
#endif
    textureUploadAlignment = core::min_(core::min_((int32_t)textureUploadAlignment,(int32_t)textureUploadAlignment2),(int32_t)3);

    if (textureUploadAlignment==pixelUnpackAlignment)
        return;

    glPixelStorei(GL_UNPACK_ALIGNMENT,0x1u<<textureUploadAlignment);
    pixelUnpackAlignment = textureUploadAlignment;
}

inline void COpenGLExtensionHandler::extGlGenSamplers(GLsizei n, GLuint* samplers)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlGenSamplers)
        pGlGenSamplers(n,samplers);
#else
    glGenSamplers(n,samplers);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
}

inline void COpenGLExtensionHandler::extGlDeleteSamplers(GLsizei n, GLuint* samplers)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlDeleteSamplers)
        pGlDeleteSamplers(n,samplers);
#else
    glDeleteSamplers(n,samplers);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
}

inline void COpenGLExtensionHandler::extGlBindSamplers(const GLuint& first, const GLsizei& count, const GLuint* samplers)
{
    if (Version>=440||FeatureAvailable[IRR_ARB_multi_bind])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlBindSamplers)
            pGlBindSamplers(first,count,samplers);
    #else
        glBindSamplers(first,count,samplers);
    #endif // _IRR_OPENGL_USE_EXTPOINTER_
    }
    else
    {
        for (GLsizei i=0; i<count; i++)
        {
            GLuint unit = first+i;
            if (samplers)
            {
            #ifdef _IRR_OPENGL_USE_EXTPOINTER_
                if (pGlBindSampler)
                    pGlBindSampler(unit,samplers[i]);
            #else
                glBindSampler(unit,samplers[i]);
            #endif // _IRR_OPENGL_USE_EXTPOINTER_
            }
            else
            {
            #ifdef _IRR_OPENGL_USE_EXTPOINTER_
                if (pGlBindSampler)
                    pGlBindSampler(unit,0);
            #else
                glBindSampler(unit,0);
            #endif // _IRR_OPENGL_USE_EXTPOINTER_
            }
        }
    }
}

inline void COpenGLExtensionHandler::extGlSamplerParameteri(GLuint sampler, GLenum pname, GLint param)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlSamplerParameteri)
        pGlSamplerParameteri(sampler,pname,param);
#else
    glSamplerParameteri(sampler,pname,param);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
}

inline void COpenGLExtensionHandler::extGlSamplerParameterf(GLuint sampler, GLenum pname, GLfloat param)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlSamplerParameterf)
        pGlSamplerParameterf(sampler,pname,param);
#else
    glSamplerParameterf(sampler,pname,param);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
}


inline void COpenGLExtensionHandler::extGlBindImageTexture(GLuint index, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlBindImageTexture)
        pGlBindImageTexture(index,texture,level,layered,layer,access,format);
#else
    glBindImageTexture(index,texture,level,layered,layer,access,format);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
}



inline GLuint COpenGLExtensionHandler::extGlCreateShader(GLenum shaderType)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlCreateShader)
		return pGlCreateShader(shaderType);
#else
	return glCreateShader(shaderType);
#endif
	return 0;
}

inline void COpenGLExtensionHandler::extGlShaderSource(GLuint shader, GLsizei numOfStrings, const char **strings, const GLint *lenOfStrings)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlShaderSource)
		pGlShaderSource(shader, numOfStrings, strings, lenOfStrings);
#else
	glShaderSource(shader, numOfStrings, strings, (GLint *)lenOfStrings);
#endif
}

inline void COpenGLExtensionHandler::extGlCompileShader(GLuint shader)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlCompileShader)
		pGlCompileShader(shader);
#else
	glCompileShader(shader);
#endif
}

inline GLuint COpenGLExtensionHandler::extGlCreateProgram(void)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlCreateProgram)
		return pGlCreateProgram();
#else
	return glCreateProgram();
#endif
	return 0;
}

inline void COpenGLExtensionHandler::extGlAttachShader(GLuint program, GLuint shader)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlAttachShader)
		pGlAttachShader(program, shader);
#else
	glAttachShader(program, shader);
#endif
}


inline void COpenGLExtensionHandler::extGlLinkProgram(GLuint program)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlLinkProgram)
		pGlLinkProgram(program);
#else
	glLinkProgram(program);
#endif
}

inline void COpenGLExtensionHandler::extGlTransformFeedbackVaryings(GLuint program, GLsizei count, const char** varyings, GLenum bufferMode)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlTransformFeedbackVaryings)
        pGlTransformFeedbackVaryings(program,count,varyings,bufferMode);
#else
	glTransformFeedbackVaryings(program,count,varyings,bufferMode);
#endif
}

inline void COpenGLExtensionHandler::extGlUseProgram(GLuint prog)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlUseProgram)
		pGlUseProgram(prog);
#else
	glUseProgram(prog);
#endif
}

inline void COpenGLExtensionHandler::extGlDeleteProgram(GLuint object)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlDeleteProgram)
		pGlDeleteProgram(object);
#else
	glDeleteProgram(object);
#endif
}

inline void COpenGLExtensionHandler::extGlDeleteShader(GLuint shader)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlDeleteShader)
		pGlDeleteShader(shader);
#else
	glDeleteShader(shader);
#endif
}

inline void COpenGLExtensionHandler::extGlGetAttachedShaders(GLuint program, GLsizei maxcount, GLsizei* count, GLuint* shaders)
{
	if (count)
		*count=0;
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlGetAttachedShaders)
		pGlGetAttachedShaders(program, maxcount, count, shaders);
#else
	glGetAttachedShaders(program, maxcount, count, shaders);
#endif
}

inline void COpenGLExtensionHandler::extGlGetShaderInfoLog(GLuint shader, GLsizei maxLength, GLsizei *length, GLchar *infoLog)
{
	if (length)
		*length=0;
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlGetShaderInfoLog)
		pGlGetShaderInfoLog(shader, maxLength, length, infoLog);
#else
	glGetShaderInfoLog(shader, maxLength, length, infoLog);
#endif
}

inline void COpenGLExtensionHandler::extGlGetProgramInfoLog(GLuint program, GLsizei maxLength, GLsizei *length, GLchar *infoLog)
{
	if (length)
		*length=0;
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlGetProgramInfoLog)
		pGlGetProgramInfoLog(program, maxLength, length, infoLog);
#else
	glGetProgramInfoLog(program, maxLength, length, infoLog);
#endif
}


inline void COpenGLExtensionHandler::extGlGetShaderiv(GLuint shader, GLenum type, GLint *param)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlGetShaderiv)
		pGlGetShaderiv(shader, type, param);
#else
	glGetShaderiv(shader, type, param);
#endif
}

inline void COpenGLExtensionHandler::extGlGetProgramiv(GLuint program, GLenum type, GLint *param)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlGetProgramiv)
		pGlGetProgramiv(program, type, param);
#else
	glGetProgramiv(program, type, param);
#endif
}

inline GLint COpenGLExtensionHandler::extGlGetUniformLocation(GLuint program, const char *name)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlGetUniformLocation)
		return pGlGetUniformLocation(program, name);
#else
	return glGetUniformLocation(program, name);
#endif
	return -1;
}

inline void COpenGLExtensionHandler::extGlProgramUniform1fv(GLuint program, GLint loc, GLsizei count, const GLfloat *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniform1fv)
		pGlProgramUniform1fv(program, loc, count, v);
#else
	glProgramUniform1fv(program, loc, count, v);
#endif
}

inline void COpenGLExtensionHandler::extGlProgramUniform2fv(GLuint program, GLint loc, GLsizei count, const GLfloat *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniform2fv)
		pGlProgramUniform2fv(program, loc, count, v);
#else
	glProgramUniform2fv(program, loc, count, v);
#endif
}

inline void COpenGLExtensionHandler::extGlProgramUniform3fv(GLuint program, GLint loc, GLsizei count, const GLfloat *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniform3fv)
		pGlProgramUniform3fv(program, loc, count, v);
#else
	glProgramUniform3fv(program, loc, count, v);
#endif
}

inline void COpenGLExtensionHandler::extGlProgramUniform4fv(GLuint program, GLint loc, GLsizei count, const GLfloat *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniform4fv)
		pGlProgramUniform4fv(program, loc, count, v);
#else
	glProgramUniform4fv(program, loc, count, v);
#endif
}

inline void COpenGLExtensionHandler::extGlProgramUniform1iv(GLuint program, GLint loc, GLsizei count, const GLint *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniform1iv)
		pGlProgramUniform1iv(program, loc, count, v);
#else
	glProgramUniform1iv(program, loc, count, v);
#endif
}

inline void COpenGLExtensionHandler::extGlProgramUniform2iv(GLuint program, GLint loc, GLsizei count, const GLint *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniform2iv)
		pGlProgramUniform2iv(program, loc, count, v);
#else
	glProgramUniform2iv(program, loc, count, v);
#endif
}

inline void COpenGLExtensionHandler::extGlProgramUniform3iv(GLuint program, GLint loc, GLsizei count, const GLint *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniform3iv)
		pGlProgramUniform3iv(program, loc, count, v);
#else
	glProgramUniform3iv(program, loc, count, v);
#endif
}

inline void COpenGLExtensionHandler::extGlProgramUniform4iv(GLuint program, GLint loc, GLsizei count, const GLint *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniform4iv)
		pGlProgramUniform4iv(program, loc, count, v);
#else
	glProgramUniform4iv(program, loc, count, v);
#endif
}

inline void COpenGLExtensionHandler::extGlProgramUniform1uiv(GLuint program, GLint loc, GLsizei count, const GLuint *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniform1uiv)
		pGlProgramUniform1uiv(program, loc, count, v);
#else
	glProgramUniform1uiv(program, loc, count, v);
#endif
}

inline void COpenGLExtensionHandler::extGlProgramUniform2uiv(GLuint program, GLint loc, GLsizei count, const GLuint *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniform2uiv)
		pGlProgramUniform2uiv(program, loc, count, v);
#else
	glProgramUniform2uiv(program, loc, count, v);
#endif
}

inline void COpenGLExtensionHandler::extGlProgramUniform3uiv(GLuint program, GLint loc, GLsizei count, const GLuint *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniform3uiv)
		pGlProgramUniform3uiv(program, loc, count, v);
#else
	glProgramUniform3uiv(program, loc, count, v);
#endif
}

inline void COpenGLExtensionHandler::extGlProgramUniform4uiv(GLuint program, GLint loc, GLsizei count, const GLuint *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniform4uiv)
		pGlProgramUniform4uiv(program, loc, count, v);
#else
	glProgramUniform4uiv(program, loc, count, v);
#endif
}

inline void COpenGLExtensionHandler::extGlProgramUniformMatrix2fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniformMatrix2fv)
		pGlProgramUniformMatrix2fv(program, loc, count, transpose, v);
#else
	glProgramUniformMatrix2fv(program, loc, count, transpose, v);
#endif
}

inline void COpenGLExtensionHandler::extGlProgramUniformMatrix3fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniformMatrix3fv)
		pGlProgramUniformMatrix3fv(program, loc, count, transpose, v);
#else
	glProgramUniformMatrix3fv(program, loc, count, transpose, v);
#endif
}

inline void COpenGLExtensionHandler::extGlProgramUniformMatrix4fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniformMatrix4fv)
		pGlProgramUniformMatrix4fv(program, loc, count, transpose, v);
#else
	glProgramUniformMatrix4fv(program, loc, count, transpose, v);
#endif
}

inline void COpenGLExtensionHandler::extGlProgramUniformMatrix2x3fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniformMatrix2x3fv)
		pGlProgramUniformMatrix2x3fv(program, loc, count, transpose, v);
#else
	glProgramUniformMatrix2x3fv(program, loc, count, transpose, v);
#endif
}
inline void COpenGLExtensionHandler::extGlProgramUniformMatrix2x4fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniformMatrix2x4fv)
		pGlProgramUniformMatrix2x4fv(program, loc, count, transpose, v);
#else
	glProgramUniformMatrix2x4fv(program, loc, count, transpose, v);
#endif
}
inline void COpenGLExtensionHandler::extGlProgramUniformMatrix3x2fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniformMatrix3x2fv)
		pGlProgramUniformMatrix3x2fv(program, loc, count, transpose, v);
#else
	glProgramUniformMatrix3x2fv(program, loc, count, transpose, v);
#endif
}
inline void COpenGLExtensionHandler::extGlProgramUniformMatrix3x4fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniformMatrix3x4fv)
		pGlProgramUniformMatrix3x4fv(program, loc, count, transpose, v);
#else
	glProgramUniformMatrix3x4fv(program, loc, count, transpose, v);
#endif
}
inline void COpenGLExtensionHandler::extGlProgramUniformMatrix4x2fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniformMatrix4x2fv)
		pGlProgramUniformMatrix4x2fv(program, loc, count, transpose, v);
#else
	glProgramUniformMatrix4x2fv(program, loc, count, transpose, v);
#endif
}
inline void COpenGLExtensionHandler::extGlProgramUniformMatrix4x3fv(GLuint program, GLint loc, GLsizei count, GLboolean transpose, const GLfloat *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlProgramUniformMatrix4x3fv)
		pGlProgramUniformMatrix4x3fv(program, loc, count, transpose, v);
#else
	glProgramUniformMatrix4x3fv(program, loc, count, transpose, v);
#endif
}



inline void COpenGLExtensionHandler::extGlGetActiveUniform(GLuint program,
		GLuint index, GLsizei maxlength, GLsizei *length,
		GLint *size, GLenum *type, GLchar *name)
{
	if (length)
		*length=0;
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlGetActiveUniform)
		pGlGetActiveUniform(program, index, maxlength, length, size, type, name);
#else
	glGetActiveUniform(program, index, maxlength, length, size, type, name);
#endif
}

inline void COpenGLExtensionHandler::extGlBindProgramPipeline(GLuint pipeline)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlBindProgramPipeline)
		pGlBindProgramPipeline(pipeline);
#else
	glBindProgramPipeline(pipeline);
#endif
}



inline void COpenGLExtensionHandler::extGlMemoryBarrier(GLbitfield barriers)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlMemoryBarrier)
		pGlMemoryBarrier(barriers);
#else
	glMemoryBarrier(barriers);
#endif
}

inline void COpenGLExtensionHandler::extGlDispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlDispatchCompute)
		pGlDispatchCompute(num_groups_x,num_groups_y,num_groups_z);
#else
	glDispatchCompute(num_groups_x,num_groups_y,num_groups_z);
#endif
}

inline void COpenGLExtensionHandler::extGlDispatchComputeIndirect(GLintptr indirect)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlDispatchComputeIndirect)
		pGlDispatchComputeIndirect(indirect);
#else
	glDispatchComputeIndirect(indirect);
#endif
}




inline void COpenGLExtensionHandler::extGlPointParameterf(GLint loc, GLfloat f)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlPointParameterf)
		pGlPointParameterf(loc, f);
#else
	glPointParameterf(loc, f);
#endif
}

inline void COpenGLExtensionHandler::extGlPointParameterfv(GLint loc, const GLfloat *v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlPointParameterfv)
		pGlPointParameterfv(loc, v);
#else
	glPointParameterfv(loc, v);
#endif
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
        if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
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
        if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
            return pGlCheckNamedFramebufferStatus(framebuffer,target);
        else if (FeatureAvailable[IRR_EXT_direct_state_access])
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
        if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
        {
            pGlNamedFramebufferTexture(framebuffer, attachment, texture, level);
            return;
        }
        else if (FeatureAvailable[IRR_EXT_direct_state_access])
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
        if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
        {
            pGlNamedFramebufferTextureLayer(framebuffer, attachment, texture, level, layer);
            return;
        }
    }

	if (textureType!=GL_TEXTURE_CUBE_MAP)
	{
		if (!needsDSAFramebufferHack && FeatureAvailable[IRR_EXT_direct_state_access])
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
		if (!needsDSAFramebufferHack && FeatureAvailable[IRR_EXT_direct_state_access])
		{
            pGlNamedFramebufferTexture2DEXT(framebuffer, attachment, COpenGLCubemapTexture::faceEnumToGLenum((ITexture::E_CUBE_MAP_FACE)layer), texture, level);
		}
		else
		{
			GLuint bound;
			glGetIntegerv(GL_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(&bound));

			if (bound != framebuffer)
				pGlBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
			pGlFramebufferTexture2D(GL_FRAMEBUFFER, attachment, COpenGLCubemapTexture::faceEnumToGLenum((ITexture::E_CUBE_MAP_FACE)layer), texture, level);
			if (bound != framebuffer)
				pGlBindFramebuffer(GL_FRAMEBUFFER, bound);
		}
	}
}

inline void COpenGLExtensionHandler::extGlBlitNamedFramebuffer(GLuint readFramebuffer, GLuint drawFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter)
{
    if (!needsDSAFramebufferHack)
    {
        if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
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
        if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
        {
            pGlNamedFramebufferReadBuffer(framebuffer, mode);
            return;
        }
        else if (FeatureAvailable[IRR_EXT_direct_state_access])
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
        if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
        {
            pGlNamedFramebufferDrawBuffer(framebuffer, buf);
            return;
        }
        else if (FeatureAvailable[IRR_EXT_direct_state_access])
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
        if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
        {
            pGlNamedFramebufferDrawBuffers(framebuffer, n, bufs);
            return;
        }
        else if (FeatureAvailable[IRR_EXT_direct_state_access])
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
        if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
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
        if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
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
        if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
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
        if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
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

    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlCreateBuffers)
            pGlCreateBuffers(n, buffers);
        else if (buffers)
            memset(buffers,0,n*sizeof(GLuint));
    #else
        glCreateBuffers(n, buffers);
    #endif
    }
    else
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlGenBuffers)
            pGlGenBuffers(n, buffers);
        else if (buffers)
            memset(buffers,0,n*sizeof(GLuint));
    #else
        glGenBuffers(n, buffers);
    #endif
    }
}

inline void COpenGLExtensionHandler::extGlDeleteBuffers(GLsizei n, const GLuint *buffers)
{
#ifdef OPENGL_LEAK_DEBUG
    for (size_t i=0; i<n; i++)
        COpenGLExtensionHandler::bufferLeaker.deregisterObj(buffers);
#endif // OPENGL_LEAK_DEBUG


#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlDeleteBuffers)
		pGlDeleteBuffers(n, buffers);
#else
	glDeleteBuffers(n, buffers);
#endif
}

inline void COpenGLExtensionHandler::extGlBindBuffer(const GLenum& target, const GLuint& buffer)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlBindBuffer)
		pGlBindBuffer(target, buffer);
#else
	glBindBuffer(target, buffer);
#endif
}

inline void COpenGLExtensionHandler::extGlBindBuffersBase(const GLenum& target, const GLuint& first, const GLsizei& count, const GLuint* buffers)
{
    if (Version>=440||FeatureAvailable[IRR_ARB_multi_bind])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlBindBuffersBase)
            pGlBindBuffersBase(target,first,count,buffers);
    #else
        glBindBuffersBase(target,first,count,buffers);
    #endif
    }
    else
    {
        for (GLsizei i=0; i<count; i++)
        {
        #ifdef _IRR_OPENGL_USE_EXTPOINTER_
            if (pGlBindBufferBase)
                pGlBindBufferBase(target,first+i,buffers ? buffers[i]:0);
        #else
            glBindBufferBase(target,first+i,buffers ? buffers[i]:0);
        #endif
        }
    }
}

inline void COpenGLExtensionHandler::extGlBindBuffersRange(const GLenum& target, const GLuint& first, const GLsizei& count, const GLuint* buffers, const GLintptr* offsets, const GLsizeiptr* sizes)
{
    if (Version>=440||FeatureAvailable[IRR_ARB_multi_bind])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlBindBuffersRange)
            pGlBindBuffersRange(target,first,count,buffers,offsets,sizes);
    #else
        glBindBuffersRange(target,first,count,buffers,offsets,sizes);
    #endif
    }
    else
    {
        for (GLsizei i=0; i<count; i++)
        {
            if (buffers)
            {
            #ifdef _IRR_OPENGL_USE_EXTPOINTER_
                if (pGlBindBufferRange)
                    pGlBindBufferRange(target,first+i,buffers[i],offsets[i],sizes[i]);
            #else
                glBindBufferRange(target,first+i,buffers[i],offsets[i],sizes[i]);
            #endif
            }
            else
            {
            #ifdef _IRR_OPENGL_USE_EXTPOINTER_
                if (pGlBindBufferBase)
                    pGlBindBufferBase(target,first+i,0);
            #else
                glBindBufferBase(target,first+i,0);
            #endif
            }
        }
    }
}

inline void COpenGLExtensionHandler::extGlNamedBufferStorage(GLuint buffer, GLsizeiptr size, const void *data, GLbitfield flags)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlNamedBufferStorage)
            pGlNamedBufferStorage(buffer,size,data,flags);
    #else
        glNamedBufferStorage(buffer,size,data,flags);
    #endif
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlNamedBufferStorageEXT)
            pGlNamedBufferStorageEXT(buffer,size,data,flags);
    #else
        glNamedBufferStorageEXT(buffer,size,data,flags);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlBufferStorage&&pGlBindBuffer)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        pGlBufferStorage(GL_ARRAY_BUFFER, size, data, flags);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
#else
        glBindBuffer(GL_ARRAY_BUFFER,buffer);
        glBufferStorage(GL_ARRAY_BUFFER, size, data, flags);
        glBindBuffer(GL_ARRAY_BUFFER,bound);
#endif
    }
}

inline void COpenGLExtensionHandler::extGlNamedBufferSubData(GLuint buffer, GLintptr offset, GLsizeiptr size, const void *data)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlNamedBufferSubData)
            pGlNamedBufferSubData(buffer,offset,size,data);
    #else
        glNamedBufferSubData(buffer,offset,size,data);
    #endif
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlNamedBufferSubDataEXT)
            pGlNamedBufferSubDataEXT(buffer,offset,size,data);
    #else
        glNamedBufferSubDataEXT(buffer,offset,size,data);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlBufferSubData&&pGlBindBuffer)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        pGlBufferSubData(GL_ARRAY_BUFFER, offset, size, data);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
#else
        glBindBuffer(GL_ARRAY_BUFFER,buffer);
        glBufferSubData(GL_ARRAY_BUFFER, offset, size, data);
        glBindBuffer(GL_ARRAY_BUFFER,bound);
#endif
    }
}

inline void COpenGLExtensionHandler::extGlGetNamedBufferSubData(GLuint buffer, GLintptr offset, GLsizeiptr size, void *data)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlGetNamedBufferSubData)
            pGlGetNamedBufferSubData(buffer,offset,size,data);
    #else
        glGetNamedBufferSubData(buffer,offset,size,data);
    #endif
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlGetNamedBufferSubDataEXT)
            pGlGetNamedBufferSubDataEXT(buffer,offset,size,data);
    #else
        glGetNamedBufferSubDataEXT(buffer,offset,size,data);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlGetBufferSubData&&pGlBindBuffer)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        pGlGetBufferSubData(GL_ARRAY_BUFFER, offset, size, data);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
#else
        glBindBuffer(GL_ARRAY_BUFFER,buffer);
        glGetBufferSubData(GL_ARRAY_BUFFER, offset, size, data);
        glBindBuffer(GL_ARRAY_BUFFER,bound);
#endif
    }
}

inline void *COpenGLExtensionHandler::extGlMapNamedBuffer(GLuint buffer, GLbitfield access)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlMapNamedBuffer)
            return pGlMapNamedBuffer(buffer,access);
    #else
        return glMapNamedBuffer(buffer,access);
    #endif
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlMapNamedBufferEXT)
            return pGlMapNamedBufferEXT(buffer,access);
    #else
        return glMapNamedBufferEXT(buffer,access);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlMapBuffer&&pGlBindBuffer)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        GLvoid* retval;
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        retval = pGlMapBuffer(GL_ARRAY_BUFFER,access);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
#else
        glBindBuffer(GL_ARRAY_BUFFER,buffer);
        retval = glMapBuffer(GL_ARRAY_BUFFER,access);
        glBindBuffer(GL_ARRAY_BUFFER,bound);
#endif
        return retval;
    }
    return NULL;
}

inline void *COpenGLExtensionHandler::extGlMapNamedBufferRange(GLuint buffer, GLintptr offset, GLsizeiptr length, GLbitfield access)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlMapNamedBufferRange)
            return pGlMapNamedBufferRange(buffer,offset,length,access);
    #else
        return glMapNamedBufferRange(buffer,offset,length,access);
    #endif
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlMapNamedBufferRangeEXT)
            return pGlMapNamedBufferRangeEXT(buffer,offset,length,access);
    #else
        return glMapNamedBufferRangeEXT(buffer,offset,length,access);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlMapBufferRange&&pGlBindBuffer)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        GLvoid* retval;
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        retval = pGlMapBufferRange(GL_ARRAY_BUFFER,offset,length,access);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
#else
        glBindBuffer(GL_ARRAY_BUFFER,buffer);
        retval = glMapBufferRange(GL_ARRAY_BUFFER,offset,length,access);
        glBindBuffer(GL_ARRAY_BUFFER,bound);
#endif
        return retval;
    }
    return NULL;
}

inline void COpenGLExtensionHandler::extGlFlushMappedNamedBufferRange(GLuint buffer, GLintptr offset, GLsizeiptr length)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlFlushMappedNamedBufferRange)
            pGlFlushMappedNamedBufferRange(buffer,offset,length);
    #else
        glFlushMappedNamedBufferRange(buffer,offset,length);
    #endif
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlFlushMappedNamedBufferRangeEXT)
            pGlFlushMappedNamedBufferRangeEXT(buffer,offset,length);
    #else
        glFlushMappedNamedBufferRangeEXT(buffer,offset,length);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlFlushMappedBufferRange&&pGlBindBuffer)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        pGlFlushMappedBufferRange(GL_ARRAY_BUFFER, offset, length);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
#else
        glBindBuffer(GL_ARRAY_BUFFER,buffer);
        glFlushMappedBufferRange(GL_ARRAY_BUFFER, offset, length);
        glBindBuffer(GL_ARRAY_BUFFER,bound);
#endif
    }
}

inline GLboolean COpenGLExtensionHandler::extGlUnmapNamedBuffer(GLuint buffer)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlUnmapNamedBuffer)
            return pGlUnmapNamedBuffer(buffer);
    #else
        return glUnmapNamedBuffer(buffer);
    #endif
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlUnmapNamedBufferEXT)
            return pGlUnmapNamedBufferEXT(buffer);
    #else
        return glUnmapNamedBufferEXT(buffer);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlUnmapBuffer&&pGlBindBuffer)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        GLboolean retval;
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        retval = pGlUnmapBuffer(GL_ARRAY_BUFFER);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
#else
        glBindBuffer(GL_ARRAY_BUFFER,buffer);
        retval = glUnmapBuffer(GL_ARRAY_BUFFER);
        glBindBuffer(GL_ARRAY_BUFFER,bound);
#endif
        return retval;
    }
    return false;
}

inline void COpenGLExtensionHandler::extGlClearNamedBufferData(GLuint buffer, GLenum internalformat, GLenum format, GLenum type, const void *data)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlClearNamedBufferData)
            pGlClearNamedBufferData(buffer,internalformat,format,type,data);
    #else
        glClearNamedBufferData(buffer,internalformat,format,type,data);
    #endif
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlClearNamedBufferDataEXT)
            pGlClearNamedBufferDataEXT(buffer,internalformat,format,type,data);
    #else
        glClearNamedBufferDataEXT(buffer,internalformat,format,type,data);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlClearBufferData&&pGlBindBuffer)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        pGlClearBufferData(GL_ARRAY_BUFFER, internalformat,format,type,data);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
#else
        glBindBuffer(GL_ARRAY_BUFFER,buffer);
        glClearBufferData(GL_ARRAY_BUFFER, internalformat,format,type,data);
        glBindBuffer(GL_ARRAY_BUFFER,bound);
#endif
    }
}

inline void COpenGLExtensionHandler::extGlClearNamedBufferSubData(GLuint buffer, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format, GLenum type, const void *data)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlClearNamedBufferSubData)
            pGlClearNamedBufferSubData(buffer,internalformat,offset,size,format,type,data);
    #else
        glClearNamedBufferSubData(buffer,internalformat,offset,size,format,type,data);
    #endif
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlClearNamedBufferSubDataEXT)
            pGlClearNamedBufferSubDataEXT(buffer,internalformat,offset,size,format,type,data);
    #else
        glClearNamedBufferSubDataEXT(buffer,internalformat,offset,size,format,type,data);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlClearBufferSubData&&pGlBindBuffer)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        pGlClearBufferSubData(GL_ARRAY_BUFFER, internalformat,offset,size,format,type,data);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
#else
        glBindBuffer(GL_ARRAY_BUFFER,buffer);
        glClearBufferSubData(GL_ARRAY_BUFFER, internalformat,offset,size,format,type,data);
        glBindBuffer(GL_ARRAY_BUFFER,bound);
#endif
    }
}

inline void COpenGLExtensionHandler::extGlCopyNamedBufferSubData(GLuint readBuffer, GLuint writeBuffer, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlCopyNamedBufferSubData)
            pGlCopyNamedBufferSubData(readBuffer, writeBuffer, readOffset, writeOffset, size);
    #else
        glCopyNamedBufferSubData(readBuffer, writeBuffer, readOffset, writeOffset, size);
    #endif
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlNamedCopyBufferSubDataEXT)
            pGlNamedCopyBufferSubDataEXT(readBuffer, writeBuffer, readOffset, writeOffset, size);
    #else
        glNamedCopyBufferSubDataEXT(readBuffer, writeBuffer, readOffset, writeOffset, size);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlCopyBufferSubData&&pGlBindBuffer)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        GLint boundRead,boundWrite;
        glGetIntegerv(GL_COPY_READ_BUFFER_BINDING,&boundRead);
        glGetIntegerv(GL_COPY_WRITE_BUFFER_BINDING,&boundWrite);
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindBuffer(GL_COPY_READ_BUFFER,readBuffer);
        pGlBindBuffer(GL_COPY_WRITE_BUFFER,writeBuffer);
        pGlCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, readOffset, writeOffset, size);
        pGlBindBuffer(GL_COPY_READ_BUFFER,boundRead);
        pGlBindBuffer(GL_COPY_WRITE_BUFFER,boundWrite);
#else
        glBindBuffer(GL_COPY_READ_BUFFER,readBuffer);
        glBindBuffer(GL_COPY_WRITE_BUFFER,writeBuffer);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, readOffset, writeOffset, size);
        glBindBuffer(GL_COPY_READ_BUFFER,boundRead);
        glBindBuffer(GL_COPY_WRITE_BUFFER,boundWrite);
#endif
    }
}

inline GLboolean COpenGLExtensionHandler::extGlIsBuffer(GLuint buffer)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlIsBuffer)
		return pGlIsBuffer(buffer);
	return false;
#else
	return glIsBuffer(buffer);
#endif
}

inline void COpenGLExtensionHandler::extGlGetNamedBufferParameteriv(const GLuint& buffer, const GLenum& value, GLint* data)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlGetNamedBufferParameteriv)
            pGlGetNamedBufferParameteriv(buffer, value, data);
    #else
        glGetNamedBufferParameteriv(buffer, value, data);
    #endif
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlGetNamedBufferParameterivEXT)
            pGlGetNamedBufferParameterivEXT(buffer, value, data);
    #else
        glGetNamedBufferParameterivEXT(buffer, value, data);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlGetBufferParameteriv&&pGlBindBuffer)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        pGlGetBufferParameteriv(GL_ARRAY_BUFFER, value, data);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
#else
        glBindBuffer(GL_ARRAY_BUFFER,buffer);
        glGetBufferParameteriv(GL_ARRAY_BUFFER, value, data);
        glBindBuffer(GL_ARRAY_BUFFER,bound);
#endif
    }
}

inline void COpenGLExtensionHandler::extGlGetNamedBufferParameteri64v(const GLuint& buffer, const GLenum& value, GLint64* data)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlGetNamedBufferParameteri64v)
            pGlGetNamedBufferParameteri64v(buffer, value, data);
    #else
        glGetNamedBufferParameteri64v(buffer, value, data);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlGetBufferParameteri64v&&pGlBindBuffer)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        GLint bound;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING,&bound);
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindBuffer(GL_ARRAY_BUFFER,buffer);
        pGlGetBufferParameteri64v(GL_ARRAY_BUFFER, value, data);
        pGlBindBuffer(GL_ARRAY_BUFFER,bound);
#else
        glBindBuffer(GL_ARRAY_BUFFER,buffer);
        glGetBufferParameteri64v(GL_ARRAY_BUFFER, value, data);
        glBindBuffer(GL_ARRAY_BUFFER,bound);
#endif
    }
}


inline void COpenGLExtensionHandler::extGlCreateVertexArrays(GLsizei n, GLuint *arrays)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlCreateVertexArrays)
            pGlCreateVertexArrays(n,arrays);
    #else
        glCreateVertexArrays(n,arrays);
    #endif
    }
    else
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlGenVertexArrays)
            pGlGenVertexArrays(n,arrays);
        else
            memset(arrays,0,sizeof(GLuint)*n);
    #else
        glGenVertexArrays(n,arrays);
    #endif
    }
}

inline void COpenGLExtensionHandler::extGlDeleteVertexArrays(GLsizei n, GLuint *arrays)
{
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlDeleteVertexArrays)
        pGlDeleteVertexArrays(n,arrays);
    #else
    glDeleteVertexArrays(n,arrays);
    #endif
}

inline void COpenGLExtensionHandler::extGlBindVertexArray(GLuint vaobj)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlBindVertexArray)
        pGlBindVertexArray(vaobj);
#else
    glBindVertexArray(vaobj);
#endif
}

inline void COpenGLExtensionHandler::extGlVertexArrayElementBuffer(GLuint vaobj, GLuint buffer)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlVertexArrayElementBuffer)
            pGlVertexArrayElementBuffer(vaobj,buffer);
    #else
        glVertexArrayElementBuffer(vaobj,buffer);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlBindBuffer&&pGlBindVertexArray)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindVertexArray(vaobj);
        pGlBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
        pGlBindVertexArray(restoreVertexArray);
    #else
        glBindVertexArray(vaobj);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
        glBindVertexArray(restoreVertexArray);
    #endif
    }
}

inline void COpenGLExtensionHandler::extGlVertexArrayVertexBuffer(GLuint vaobj, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlVertexArrayVertexBuffer)
            pGlVertexArrayVertexBuffer(vaobj,bindingindex,buffer,offset,stride);
    #else
        glVertexArrayVertexBuffer(vaobj,bindingindex,buffer,offset,stride);
    #endif
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlVertexArrayBindVertexBufferEXT)
            pGlVertexArrayBindVertexBufferEXT(vaobj,bindingindex,buffer,offset,stride);
    #else
        glVertexArrayBindVertexBufferEXT(vaobj,bindingindex,buffer,offset,stride);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlBindVertexBuffer&&pGlBindVertexArray)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindVertexArray(vaobj);
        pGlBindVertexBuffer(bindingindex,buffer,offset,stride);
        pGlBindVertexArray(restoreVertexArray);
    #else
        glBindVertexArray(vaobj);
        glBindVertexBuffer(bindingindex,buffer,offset,stride);
        glBindVertexArray(restoreVertexArray);
    #endif
    }
}

inline void COpenGLExtensionHandler::extGlVertexArrayAttribBinding(GLuint vaobj, GLuint attribindex, GLuint bindingindex)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlVertexArrayAttribBinding)
            pGlVertexArrayAttribBinding(vaobj,attribindex,bindingindex);
    #else
        glVertexArrayAttribBinding(vaobj,attribindex,bindingindex);
    #endif
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlVertexArrayVertexAttribBindingEXT)
            pGlVertexArrayVertexAttribBindingEXT(vaobj,attribindex,bindingindex);
    #else
        glVertexArrayVertexAttribBindingEXT(vaobj,attribindex,bindingindex);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlVertexAttribBinding&&pGlBindVertexArray)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindVertexArray(vaobj);
        pGlVertexAttribBinding(attribindex,bindingindex);
        pGlBindVertexArray(restoreVertexArray);
    #else
        glBindVertexArray(vaobj);
        glVertexAttribBinding(attribindex,bindingindex);
        glBindVertexArray(restoreVertexArray);
    #endif
    }
}

inline void COpenGLExtensionHandler::extGlEnableVertexArrayAttrib(GLuint vaobj, GLuint index)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlEnableVertexArrayAttrib)
            pGlEnableVertexArrayAttrib(vaobj,index);
    #else
        glEnableVertexArrayAttrib(vaobj,index);
    #endif
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlEnableVertexArrayAttribEXT)
            pGlEnableVertexArrayAttribEXT(vaobj,index);
    #else
        glEnableVertexArrayAttribEXT(vaobj,index);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlEnableVertexAttribArray&&pGlBindVertexArray)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindVertexArray(vaobj);
        pGlEnableVertexAttribArray(index);
        pGlBindVertexArray(restoreVertexArray);
    #else
        glBindVertexArray(vaobj);
        glEnableVertexAttribArray(index);
        glBindVertexArray(restoreVertexArray);
    #endif
    }
}

inline void COpenGLExtensionHandler::extGlDisableVertexArrayAttrib(GLuint vaobj, GLuint index)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlDisableVertexArrayAttrib)
            pGlDisableVertexArrayAttrib(vaobj,index);
    #else
        glDisableVertexArrayAttrib(vaobj,index);
    #endif
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlDisableVertexArrayAttribEXT)
            pGlDisableVertexArrayAttribEXT(vaobj,index);
    #else
        glDisableVertexArrayAttribEXT(vaobj,index);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlDisableVertexAttribArray&&pGlBindVertexArray)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindVertexArray(vaobj);
        pGlDisableVertexAttribArray(index);
        pGlBindVertexArray(restoreVertexArray);
    #else
        glBindVertexArray(vaobj);
        glDisableVertexAttribArray(index);
        glBindVertexArray(restoreVertexArray);
    #endif
    }
}

inline void COpenGLExtensionHandler::extGlVertexArrayAttribFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlVertexArrayAttribFormat)
            pGlVertexArrayAttribFormat(vaobj,attribindex,size,type,normalized,relativeoffset);
    #else
        glVertexArrayAttribFormat(vaobj,attribindex,size,type,normalized,relativeoffset);
    #endif
    }
    else if (!IsIntelGPU&&FeatureAvailable[IRR_EXT_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlVertexArrayVertexAttribFormatEXT)
            pGlVertexArrayVertexAttribFormatEXT(vaobj,attribindex,size,type,normalized,relativeoffset);
    #else
        glVertexArrayVertexAttribFormatEXT(vaobj,attribindex,size,type,normalized,relativeoffset);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlVertexAttribFormat&&pGlBindVertexArray)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindVertexArray(vaobj);
        pGlVertexAttribFormat(attribindex,size,type,normalized,relativeoffset);
        pGlBindVertexArray(restoreVertexArray);
    #else
        glBindVertexArray(vaobj);
        glVertexAttribFormat(attribindex,size,type,normalized,relativeoffset);
        glBindVertexArray(restoreVertexArray);
    #endif
    }
}

inline void COpenGLExtensionHandler::extGlVertexArrayAttribIFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlVertexArrayAttribIFormat)
            pGlVertexArrayAttribIFormat(vaobj,attribindex,size,type,relativeoffset);
    #else
        glVertexArrayAttribIFormat(vaobj,attribindex,size,type,relativeoffset);
    #endif
    }
    else if (!IsIntelGPU&&FeatureAvailable[IRR_EXT_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlVertexArrayVertexAttribIFormatEXT)
            pGlVertexArrayVertexAttribIFormatEXT(vaobj,attribindex,size,type,relativeoffset);
    #else
        glVertexArrayVertexAttribIFormatEXT(vaobj,attribindex,size,type,relativeoffset);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlVertexAttribIFormat&&pGlBindVertexArray)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindVertexArray(vaobj);
        pGlVertexAttribIFormat(attribindex,size,type,relativeoffset);
        pGlBindVertexArray(restoreVertexArray);
    #else
        glBindVertexArray(vaobj);
        glVertexAttribIFormat(attribindex,size,type,relativeoffset);
        glBindVertexArray(restoreVertexArray);
    #endif
    }
}

inline void COpenGLExtensionHandler::extGlVertexArrayAttribLFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlVertexArrayAttribLFormat)
            pGlVertexArrayAttribLFormat(vaobj,attribindex,size,type,relativeoffset);
    #else
        pGlVertexArrayAttribLFormat(vaobj,attribindex,size,type,relativeoffset);
    #endif
    }
    else if (!IsIntelGPU&&FeatureAvailable[IRR_EXT_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlVertexArrayVertexAttribLFormatEXT)
            pGlVertexArrayVertexAttribLFormatEXT(vaobj,attribindex,size,type,relativeoffset);
    #else
        pGlVertexArrayVertexAttribLFormatEXT(vaobj,attribindex,size,type,relativeoffset);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlVertexAttribLFormat&&pGlBindVertexArray)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindVertexArray(vaobj);
        pGlVertexAttribLFormat(attribindex,size,type,relativeoffset);
        pGlBindVertexArray(restoreVertexArray);
    #else
        glBindVertexArray(vaobj);
        glVertexAttribLFormat(attribindex,size,type,relativeoffset);
        glBindVertexArray(restoreVertexArray);
    #endif
    }
}

inline void COpenGLExtensionHandler::extGlVertexArrayBindingDivisor(GLuint vaobj, GLuint bindingindex, GLuint divisor)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlVertexArrayBindingDivisor)
            pGlVertexArrayBindingDivisor(vaobj,bindingindex,divisor);
    #else
        glVertexArrayBindingDivisor(vaobj,bindingindex,divisor);
    #endif
    }
    else if (FeatureAvailable[IRR_EXT_direct_state_access])
    {
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlVertexArrayVertexBindingDivisorEXT)
            pGlVertexArrayVertexBindingDivisorEXT(vaobj,bindingindex,divisor);
    #else
        glVertexArrayVertexBindingDivisorEXT(vaobj,bindingindex,divisor);
    #endif
    }
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    else if (pGlVertexBindingDivisor&&pGlBindVertexArray)
#else
    else
#endif // _IRR_OPENGL_USE_EXTPOINTER_
    {
        // Save the previous bound vertex array
        GLint restoreVertexArray;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &restoreVertexArray);
    #ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindVertexArray(vaobj);
        pGlVertexBindingDivisor(bindingindex,divisor);
        pGlBindVertexArray(restoreVertexArray);
    #else
        glBindVertexArray(vaobj);
        glVertexBindingDivisor(bindingindex,divisor);
        glBindVertexArray(restoreVertexArray);
    #endif
    }
}



inline void COpenGLExtensionHandler::extGlCreateTransformFeedbacks(GLsizei n, GLuint* ids)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlCreateTransformFeedbacks)
            pGlCreateTransformFeedbacks(n,ids);
#else
        glCreateTransformFeedbacks(n,ids);
#endif
    }
    else
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlGenTransformFeedbacks)
            pGlGenTransformFeedbacks(n,ids);
#else
        glGenTransformFeedbacks(n,ids);
#endif
    }
}

inline void COpenGLExtensionHandler::extGlDeleteTransformFeedbacks(GLsizei n, const GLuint* ids)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlDeleteTransformFeedbacks)
        pGlDeleteTransformFeedbacks(n,ids);
#else
    glDeleteTransformFeedbacks(n,ids);
#endif
}

inline void COpenGLExtensionHandler::extGlBindTransformFeedback(GLenum target, GLuint id)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlBindTransformFeedback)
        pGlBindTransformFeedback(target,id);
#else
    glBindTransformFeedback(target,id);
#endif
}

inline void COpenGLExtensionHandler::extGlBeginTransformFeedback(GLenum primitiveMode)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlBeginTransformFeedback)
        pGlBeginTransformFeedback(primitiveMode);
#else
    glBeginTransformFeedback(primitiveMode);
#endif
}

inline void COpenGLExtensionHandler::extGlPauseTransformFeedback()
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlPauseTransformFeedback)
        pGlPauseTransformFeedback();
#else
    glPauseTransformFeedback();
#endif
}

inline void COpenGLExtensionHandler::extGlResumeTransformFeedback()
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlResumeTransformFeedback)
        pGlResumeTransformFeedback();
#else
    glResumeTransformFeedback();
#endif
}

inline void COpenGLExtensionHandler::extGlEndTransformFeedback()
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlEndTransformFeedback)
        pGlEndTransformFeedback();
#else
    glEndTransformFeedback();
#endif
}

inline void COpenGLExtensionHandler::extGlTransformFeedbackBufferBase(GLuint xfb, GLuint index, GLuint buffer)
{
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlTransformFeedbackBufferBase)
            pGlTransformFeedbackBufferBase(xfb,index,buffer);
#else
        glTransformFeedbackBufferBase(xfb,index,buffer);
#endif
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
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlTransformFeedbackBufferRange)
            pGlTransformFeedbackBufferRange(xfb,index,buffer,offset,size);
#else
        glTransformFeedbackBufferRange(xfb,index,buffer,offset,size);
#endif
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
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlPrimitiveRestartIndex)
        pGlPrimitiveRestartIndex(index);
#else
    glPrimitiveRestartIndex(index);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
}

inline void COpenGLExtensionHandler::extGlDrawArraysInstanced(GLenum mode, GLint first, GLsizei count, GLsizei instancecount)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlDrawArraysInstanced)
        pGlDrawArraysInstanced(mode,first,count,instancecount);
#else
    glDrawArraysInstanced(mode,first,count,instancecount);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
}

inline void COpenGLExtensionHandler::extGlDrawArraysInstancedBaseInstance(GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlDrawArraysInstancedBaseInstance)
        pGlDrawArraysInstancedBaseInstance(mode,first,count,instancecount,baseinstance);
#else
    glDrawArraysInstancedBaseInstance(mode,first,count,instancecount,baseinstance);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
}

inline void COpenGLExtensionHandler::extGlDrawElementsInstancedBaseVertex(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlDrawElementsInstancedBaseVertex)
        pGlDrawElementsInstancedBaseVertex(mode,count,type,indices,instancecount,basevertex);
#else
    glDrawElementsInstancedBaseVertex(mode,count,type,indices,instancecount,basevertex);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
}

inline void COpenGLExtensionHandler::extGlDrawElementsInstancedBaseVertexBaseInstance(GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlDrawElementsInstancedBaseVertexBaseInstance)
        pGlDrawElementsInstancedBaseVertexBaseInstance(mode,count,type,indices,instancecount,basevertex,baseinstance);
#else
    glDrawElementsInstancedBaseVertexBaseInstance(mode,count,type,indices,instancecount,basevertex,baseinstance);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
}

inline void COpenGLExtensionHandler::extGlDrawTransformFeedback(GLenum mode, GLuint id)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlDrawTransformFeedback)
        pGlDrawTransformFeedback(mode,id);
#else
    glDrawTransformFeedback(mode,id);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
}

inline void COpenGLExtensionHandler::extGlDrawTransformFeedbackInstanced(GLenum mode, GLuint id, GLsizei instancecount)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlDrawTransformFeedbackInstanced)
        pGlDrawTransformFeedbackInstanced(mode,id,instancecount);
#else
    glDrawTransformFeedbackInstanced(mode,id,instancecount);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
}

inline void COpenGLExtensionHandler::extGlDrawTransformFeedbackStream(GLenum mode, GLuint id, GLuint stream)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlDrawTransformFeedbackStream)
        pGlDrawTransformFeedbackStream(mode,id,stream);
#else
    glDrawTransformFeedbackStream(mode,id,stream);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
}

inline void COpenGLExtensionHandler::extGlDrawTransformFeedbackStreamInstanced(GLenum mode, GLuint id, GLuint stream, GLsizei instancecount)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlDrawTransformFeedbackStreamInstanced)
        pGlDrawTransformFeedbackStreamInstanced(mode,id,stream,instancecount);
#else
    glDrawTransformFeedbackStreamInstanced(mode,id,stream,instancecount);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
}

inline void COpenGLExtensionHandler::extGlDrawArraysIndirect(GLenum mode, const void *indirect)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlDrawArraysIndirect)
        pGlDrawArraysIndirect(mode,indirect);
#else
    glDrawArraysIndirect(mode,indirect);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
}

inline void COpenGLExtensionHandler::extGlDrawElementsIndirect(GLenum mode, GLenum type, const void *indirect)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlDrawElementsIndirect)
        pGlDrawElementsIndirect(mode,type,indirect);
#else
    glDrawElementsIndirect(mode,type,indirect);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
}

inline void COpenGLExtensionHandler::extGlMultiDrawArraysIndirect(GLenum mode, const void *indirect, GLsizei drawcount, GLsizei stride)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlMultiDrawArraysIndirect)
        pGlMultiDrawArraysIndirect(mode,indirect,drawcount,stride);
#else
    glMultiDrawArraysIndirect(mode,indirect,drawcount,stride);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
}

inline void COpenGLExtensionHandler::extGlMultiDrawElementsIndirect(GLenum mode, GLenum type, const void *indirect, GLsizei drawcount, GLsizei stride)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
    if (pGlMultiDrawElementsIndirect)
        pGlMultiDrawElementsIndirect(mode,type,indirect,drawcount,stride);
#else
    glMultiDrawElementsIndirect(mode,type,indirect,drawcount,stride);
#endif // _IRR_OPENGL_USE_EXTPOINTER_
}




// ROP
inline void COpenGLExtensionHandler::extGlBlendColor(float red, float green, float blue, float alpha)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlBlendColor)
		pGlBlendColor(red,green,blue,alpha);
#else
	glBlendColor(red, green, blue, alpha);
#endif
}
inline void COpenGLExtensionHandler::extGlDepthRangeIndexed(GLuint index, GLdouble nearVal, GLdouble farVal)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlDepthRangeIndexed)
		pGlDepthRangeIndexed(index,nearVal,farVal);
#else
	glDepthRangeIndexed(index,nearVal,farVal);
#endif
}
inline void COpenGLExtensionHandler::extGlViewportIndexedfv(GLuint index, const GLfloat* v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlViewportIndexedfv)
		pGlViewportIndexedfv(index,v);
#else
	glViewportIndexedfv(index,v);
#endif
}
inline void COpenGLExtensionHandler::extGlScissorIndexedv(GLuint index, const GLint* v)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlScissorIndexedv)
		pGlScissorIndexedv(index,v);
#else
	glScissorIndexedv(index,v);
#endif
}
inline void COpenGLExtensionHandler::extGlSampleCoverage(float value, bool invert)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlSampleCoverage)
		pGlSampleCoverage(value,invert);
#else
	glSampleCoverage(value,invert);
#endif
}
inline void COpenGLExtensionHandler::extGlSampleMaski(GLuint maskNumber, GLbitfield mask)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlSampleMaski)
		pGlSampleMaski(maskNumber,mask);
#else
	glSampleMaski(maskNumber,mask);
#endif
}
inline void COpenGLExtensionHandler::extGlMinSampleShading(float value)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlMinSampleShading)
		pGlMinSampleShading(value);
#else
	glMinSampleShading(value);
#endif
}
inline void COpenGLExtensionHandler::extGlBlendEquationSeparatei(GLuint buf, GLenum modeRGB, GLenum modeAlpha)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlBlendEquationSeparatei)
		pGlBlendEquationSeparatei(buf,modeRGB,modeAlpha);
#else
	glBlendEquationSeparatei(buf,modeRGB,modeAlpha);
#endif
}
inline void COpenGLExtensionHandler::extGlBlendFuncSeparatei(GLuint buf, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlBlendFuncSeparatei)
		pGlBlendFuncSeparatei(buf,srcRGB,dstRGB,srcAlpha,dstAlpha);
#else
	glBlendFuncSeparatei(buf,srcRGB,dstRGB,srcAlpha,dstAlpha);
#endif
}
inline void COpenGLExtensionHandler::extGlColorMaski(GLuint buf, GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlColorMaski)
		pGlColorMaski(buf,red,green,blue,alpha);
#else
	glColorMaski(buf,red,green,blue,alpha);
#endif
}



inline void COpenGLExtensionHandler::extGlBlendFuncSeparate(GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlBlendFuncSeparate)
		pGlBlendFuncSeparate(srcRGB,dstRGB,srcAlpha,dstAlpha);
#else
	glBlendFuncSeparate(srcRGB,dstRGB,srcAlpha,dstAlpha);
#endif
}


inline void COpenGLExtensionHandler::extGlColorMaskIndexed(GLuint buf, GLboolean r, GLboolean g, GLboolean b, GLboolean a)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlColorMaski)
		pGlColorMaski(buf, r, g, b, a);
#else
	glColorMaski(buf,r,g,b,a);
#endif
}


inline void COpenGLExtensionHandler::extGlEnableIndexed(GLenum target, GLuint index)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlEnablei)
		pGlEnablei(target, index);
#else
    glEnablei(target, index);
#endif
}

inline void COpenGLExtensionHandler::extGlDisableIndexed(GLenum target, GLuint index)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlDisablei)
		pGlDisablei(target, index);
#else
	glDisablei(target, index);
#endif
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
    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlCreateQueries)
            pGlCreateQueries(target, n, ids);
#else
        glCreateQueries(target, n, ids);
#endif
    }
    else
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlGenQueries)
            pGlGenQueries(n, ids);
#else
        glGenQueries(n, ids);
#endif
    }
}

inline void COpenGLExtensionHandler::extGlDeleteQueries(GLsizei n, const GLuint *ids)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlDeleteQueries)
		pGlDeleteQueries(n, ids);
#else
	glDeleteQueries(n, ids);
#endif
}

inline GLboolean COpenGLExtensionHandler::extGlIsQuery(GLuint id)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlIsQuery)
		return pGlIsQuery(id);
	return false;
#else
	return glIsQuery(id);
#endif
}

inline void COpenGLExtensionHandler::extGlBeginQuery(GLenum target, GLuint id)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlBeginQuery)
		pGlBeginQuery(target, id);
#else
	glBeginQuery(target, id);
#endif
}

inline void COpenGLExtensionHandler::extGlEndQuery(GLenum target)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlEndQuery)
		pGlEndQuery(target);
#else
	glEndQuery(target);
#endif
}

inline void COpenGLExtensionHandler::extGlBeginQueryIndexed(GLenum target, GLuint index, GLuint id)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlBeginQueryIndexed)
		pGlBeginQueryIndexed(target, index, id);
#else
	glBeginQueryIndexed(target, index, id);
#endif
}

inline void COpenGLExtensionHandler::extGlEndQueryIndexed(GLenum target, GLuint index)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlEndQueryIndexed)
		pGlEndQueryIndexed(target, index);
#else
	glEndQueryIndexed(target, index);
#endif
}


inline void COpenGLExtensionHandler::extGlGetQueryObjectuiv(GLuint id, GLenum pname, GLuint *params)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlGetQueryObjectuiv)
		pGlGetQueryObjectuiv(id, pname, params);
#else
	glGetQueryObjectuiv(id, pname, params);
#endif
}

inline void COpenGLExtensionHandler::extGlGetQueryObjectui64v(GLuint id, GLenum pname, GLuint64 *params)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlGetQueryObjectui64v)
		pGlGetQueryObjectui64v(id, pname, params);
#else
    glGetQueryObjectui64v(id, pname, params);
#endif
}

inline void COpenGLExtensionHandler::extGlGetQueryBufferObjectuiv(GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
{
    if (Version<440 && !FeatureAvailable[IRR_ARB_query_buffer_object])
    {
#ifdef _DEBuG
        os::Printer::log("GL_ARB_query_buffer_object unsupported!\n");
#endif // _DEBuG
        return;
    }

    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlGetQueryBufferObjectuiv)
            pGlGetQueryBufferObjectuiv(id, buffer, pname, offset);
#else
        glGetQueryBufferObjectuiv(id, buffer, pname, offset);
#endif
    }
    else
    {
        GLint restoreQueryBuffer;
        glGetIntegerv(GL_QUERY_BUFFER_BINDING, &restoreQueryBuffer);
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindBuffer(GL_QUERY_BUFFER,id);
        if (pGlGetQueryObjectuiv)
            pGlGetQueryObjectuiv(id, pname, reinterpret_cast<GLuint*>(offset));
        pGlBindBuffer(GL_QUERY_BUFFER,restoreQueryBuffer);
#else
        glBindBuffer(GL_QUERY_BUFFER,id);
        glGetQueryObjectuiv(id, pname, reinterpret_cast<GLuint*>(offset));
        glBindBuffer(GL_QUERY_BUFFER,restoreQueryBuffer);
#endif
    }
}

inline void COpenGLExtensionHandler::extGlGetQueryBufferObjectui64v(GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
{
    if (Version<440 && !FeatureAvailable[IRR_ARB_query_buffer_object])
    {
#ifdef _DEBuG
        os::Printer::log("GL_ARB_query_buffer_object unsupported!\n");
#endif // _DEBuG
        return;
    }

    if (Version>=450||FeatureAvailable[IRR_ARB_direct_state_access])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlGetQueryBufferObjectui64v)
            pGlGetQueryBufferObjectui64v(id, buffer, pname, offset);
#else
        glGetQueryBufferObjectui64v(id, buffer, pname, offset);
#endif
    }
    else
    {
        GLint restoreQueryBuffer;
        glGetIntegerv(GL_QUERY_BUFFER_BINDING, &restoreQueryBuffer);
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        pGlBindBuffer(GL_QUERY_BUFFER,id);
        if (pGlGetQueryObjectui64v)
            pGlGetQueryObjectui64v(id, pname, reinterpret_cast<GLuint64*>(offset));
        pGlBindBuffer(GL_QUERY_BUFFER,restoreQueryBuffer);
#else
        glBindBuffer(GL_QUERY_BUFFER,id);
        glGetQueryObjectui64v(id, pname, reinterpret_cast<GLuint64*>(offset));
        glBindBuffer(GL_QUERY_BUFFER,restoreQueryBuffer);
#endif
    }
}

inline void COpenGLExtensionHandler::extGlQueryCounter(GLuint id, GLenum target)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlQueryCounter)
		pGlQueryCounter(id, target);
#else
    glQueryCounter(id, target);
#endif
}

inline void COpenGLExtensionHandler::extGlBeginConditionalRender(GLuint id, GLenum mode)
{
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
	if (pGlBeginConditionalRender)
		pGlBeginConditionalRender(id, mode);
#else
    glBeginConditionalRender(id, mode);
#endif
}

inline void COpenGLExtensionHandler::extGlEndConditionalRender()
{
	if (pGlEndConditionalRender)
		pGlEndConditionalRender();
}


inline void COpenGLExtensionHandler::extGlTextureBarrier()
{
	if (FeatureAvailable[IRR_ARB_texture_barrier])
		pGlTextureBarrier();
	else if (FeatureAvailable[IRR_NV_texture_barrier])
		pGlTextureBarrierNV();
#ifdef _IRR_DEBUG
    else
        os::Printer::log("EDF_TEXTURE_BARRIER Not Available!\n",ELL_ERROR);
#endif // _IRR_DEBUG
}


inline void COpenGLExtensionHandler::extGlSwapInterval(int interval)
{
	// we have wglext, so try to use that
#if defined(_IRR_WINDOWS_API_) && defined(_IRR_COMPILE_WITH_WINDOWS_DEVICE_)
#ifdef WGL_EXT_swap_control
	if (pWglSwapIntervalEXT)
		pWglSwapIntervalEXT(interval);
#endif
#endif
#ifdef _IRR_COMPILE_WITH_X11_DEVICE_
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
    if (Version>=460 || FeatureAvailable[IRR_ARB_internalformat_query])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlGetInternalformativ)
            pGlGetInternalformativ(target, internalformat, pname, bufSize, params);
#else
        glGetInternalformativ(target, internalformat, pname, bufSize, params);
#endif
    }
}

inline void COpenGLExtensionHandler::extGlGetInternalformati64v(GLenum target, GLenum internalformat, GLenum pname, GLsizei bufSize, GLint64* params)
{
    if (Version>=460 || FeatureAvailable[IRR_ARB_internalformat_query])
    {
#ifdef _IRR_OPENGL_USE_EXTPOINTER_
        if (pGlGetInternalformati64v)
            pGlGetInternalformati64v(target, internalformat, pname, bufSize, params);
#else
        glGetInternalformati64v(target, internalformat, pname, bufSize, params);
#endif
    }
}

}
}

#endif

#endif

