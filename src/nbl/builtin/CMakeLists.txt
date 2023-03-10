find_package(Python3 REQUIRED)

macro(LIST_BUILTIN_RESOURCE _BUNDLE_NAME_ _LBR_PATH_)
	set(_OPTIONAL_ALIAS_ "${ARGV2}")
	
	list(APPEND ${_BUNDLE_NAME_} "${_LBR_PATH_}")
	set(${_BUNDLE_NAME_} ${${_BUNDLE_NAME_}} PARENT_SCOPE) # override
	
	list(APPEND _LBR_${_BUNDLE_NAME_}_ "${_LBR_PATH_},${_OPTIONAL_ALIAS_}")
	set(_LBR_${_BUNDLE_NAME_}_ ${_LBR_${_BUNDLE_NAME_}_} PARENT_SCOPE) # override
endmacro()

LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/utils/acceleration_structures.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/shader/loader/mtl/vertex_impl.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/shader/loader/mtl/vertex_no_uv.vert") # TODO: remove naming tautology
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/shader/loader/mtl/vertex_uv.vert") # TODO: remove naming tautology
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/shader/loader/mtl/fragment_impl.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/shader/loader/mtl/fragment_no_uv.frag") # TODO: remove naming tautology
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/shader/loader/mtl/fragment_uv.frag") # TODO: remove naming tautology
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/shader/loader/gltf/common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/shader/loader/gltf/vertex_impl.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/shader/loader/gltf/uv.vert")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/shader/loader/gltf/color.vert")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/shader/loader/gltf/no_uv_color.vert")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/shader/loader/gltf/fragment_impl.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/shader/loader/gltf/uv.frag")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/shader/loader/gltf/color.frag")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/shader/loader/gltf/no_uv_color.frag")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/specialized_shader/fullscreentriangle.vert")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/material/lambertian/singletexture/specialized_shader.vert")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/material/lambertian/singletexture/specialized_shader.frag")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/material/debug/vertex_color/specialized_shader.vert")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/material/debug/vertex_normal/specialized_shader.vert")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/material/debug/vertex_normal/specialized_shader.frag")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/material/debug/vertex_uv/specialized_shader.frag")
# generic GLSL headers after this line
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/macros.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/algorithm.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ieee754.glsl")
# barycentric
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/barycentric/extensions.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/barycentric/frag.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/barycentric/vert.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/barycentric/utils.glsl")
# bump mapping
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bump_mapping/fragment.glsl") # TODO: rename to `frag.glsl`
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bump_mapping/utils.glsl")
# bxdf
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/common_samples.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/fresnel.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/ndf/common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/ndf/blinn_phong.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/ndf/beckmann.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/ndf/ggx.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/geom/smith/common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/geom/smith/beckmann.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/geom/smith/ggx.glsl")
# brdf
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/brdf/diffuse/lambert.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/brdf/diffuse/oren_nayar.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/brdf/specular/blinn_phong.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/brdf/specular/beckmann.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/brdf/specular/ggx.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/brdf/diffuse/fresnel_correction.glsl")
# bsdf
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/bsdf/diffuse/lambert.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/bsdf/specular/common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/bsdf/specular/dielectric.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/bsdf/specular/beckmann.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/bxdf/bsdf/specular/ggx.glsl")
# colorspace
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/colorspace/EOTF.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/colorspace/OETF.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/colorspace/decodeCIEXYZ.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/colorspace/encodeCIEXYZ.glsl")
# Culling and LoD System
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/culling_lod_selection/dispatch_indirect_params.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/culling_lod_selection/draw_instance_count_scan_override.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/culling_lod_selection/input_descriptor_set.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/culling_lod_selection/instance_cull_and_lod_select.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/culling_lod_selection/instance_draw_count_scan_override.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/culling_lod_selection/instance_draw_cull.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/culling_lod_selection/instance_ref_counting_sort_scatter.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/culling_lod_selection/potentially_visible_instance_draw_struct.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/culling_lod_selection/output_descriptor_modifiers.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/culling_lod_selection/output_descriptor_set.glsl")
# VK_FORMAT support
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/format/decode.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/format/encode.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/format/constants.glsl")
# asset loaders (mostly for pipelines)
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/loader/mtl/common.glsl")
# LoD Library
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/lod_library/structs.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/lod_library/descriptor_set.glsl")
# math and limits
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/math/constants.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/math/complex.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/math/functions.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/math/quaternions.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/math/typeless_arithmetic.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/limits/numeric.glsl")
# material_compiler
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/material_compiler/common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/material_compiler/common_declarations.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/material_compiler/common_invariant_declarations.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/material_compiler/rasterization/impl.glsl")
# property pool
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/property_pool/transfer.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/property_pool/copy.comp")
# random numbers
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/random/xoroshiro.glsl")
# sampling
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/sampling/bilinear.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/sampling/box_muller_transform.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/sampling/concentric_mapping.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/sampling/cos_weighted.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/sampling/linear.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/sampling/projected_spherical_triangle.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/sampling/spherical_rectangle.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/sampling/spherical_triangle.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/sampling/quantized_sequence.glsl")
# global exclusive scan
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/scan/direct.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/scan/declarations.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/scan/descriptors.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/scan/default_scheduler.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/scan/indirect.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/scan/parameters_struct.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/scan/virtual_workgroup.glsl")
# faster and easier scan
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/scanning_append/scanning_append.glsl")
# scene
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/scene/animation.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/scene/keyframe.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/scene/node.glsl")
# shapes
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/shapes/aabb.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/shapes/frustum.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/shapes/rectangle.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/shapes/triangle.glsl")
# skinning
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/skinning/cache_descriptor_set.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/skinning/cache_update.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/skinning/debug.vert")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/skinning/linear.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/skinning/render_descriptor_set.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/skinning/update_descriptor_set.glsl")
# subgroup emulation
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/subgroup/shared_arithmetic_portability.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/subgroup/basic_portability.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/subgroup/arithmetic_portability.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/subgroup/arithmetic_portability_impl.glsl")
##LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/subgroup/fft.glsl")
# utilities
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/utils/indirect_commands.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/utils/common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/utils/culling.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/utils/compressed_normal_matrix_t.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/utils/normal_decode.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/utils/normal_encode.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/utils/transform.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/utils/morton.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/utils/surface_transform_transformedDerivatives.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/utils/surface_transform.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/utils/surface_transform_e.h")
# virtual texturing
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/virtual_texturing/descriptors.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/virtual_texturing/extensions.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/virtual_texturing/impl_functions.glsl")
# workgroup "intrinsics"
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/workgroup/shared_arithmetic.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/workgroup/shared_ballot.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/workgroup/shared_clustered.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/workgroup/shared_fft.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/workgroup/shared_shuffle.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/workgroup/shared_shuffle_relative.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/workgroup/shared_vote.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/workgroup/arithmetic.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/workgroup/ballot.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/workgroup/basic.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/workgroup/clustered.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/workgroup/fft.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/workgroup/shuffle.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/workgroup/shuffle_relative.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/workgroup/vote.glsl")
#transform_tree
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/transform_tree/global_transform_update.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/transform_tree/global_transform_and_normal_matrix_update.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/transform_tree/global_transform_update_common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/transform_tree/global_transform_update_descriptor_set.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/transform_tree/modification_request_range.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/transform_tree/pool_descriptor_set.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/transform_tree/render_descriptor_set.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/transform_tree/relative_transform_modification.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/transform_tree/relative_transform_update.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/transform_tree/relative_transform_update_common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/transform_tree/relative_transform_update_descriptor_set.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/transform_tree/debug.vert")
# ext shouldn't be built into the engine, but there's no harm including some non-dynamic GLSL source to make life easier
#LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/.glsl")
# radix sort
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/FFT/default_compute_fft.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/FFT/fft.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/FFT/parameters_struct.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/FFT/parameters.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/FFT/types.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/LumaMeter/common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/LumaMeter/impl.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/ToneMapper/operators.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/RadeonRays/intersection.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/RadeonRays/ray.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/MitsubaLoader/instance_data_struct.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/MitsubaLoader/instance_data_descriptor.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/MitsubaLoader/material_compiler_compatibility_impl.glsl")
# OIT
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/OIT/oit.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/OIT/insert_node.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/OIT/resolve.frag")
# virtual geometry
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/virtual_geometry/descriptors.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/virtual_geometry/virtual_attribute.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/virtual_geometry/virtual_attribute_fetch.glsl")
# depth pyramid generator
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/DepthPyramidGenerator/common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/DepthPyramidGenerator/push_constants_struct_common.h")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/DepthPyramidGenerator/depth_pyramid_generator_impl.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/ext/DepthPyramidGenerator/virtual_work_group.glsl")
# blit
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/blit/formats_encode.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/blit/parameters.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/blit/multi_dimensional_array_addressing.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/blit/default_compute_common.comp")

LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/blit/default_compute_blit.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/blit/blit/blit.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/blit/blit/descriptors.glsl")

LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/blit/default_compute_alpha_test.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/blit/alpha_test/alpha_test.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/blit/alpha_test/descriptors.glsl")

LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/blit/default_compute_normalization.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/blit/normalization/normalization.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/blit/normalization/descriptors.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "nbl/builtin/glsl/blit/normalization/shared_normalization.glsl")

function(ADD_CUSTOM_BUILTIN_RESOURCES _TARGET_NAME_ _BUNDLE_NAME_ _BUNDLE_SEARCH_DIRECTORY_ _NAMESPACE_ _OUTPUT_DIRECTORY_)
	file(MAKE_DIRECTORY "${_OUTPUT_DIRECTORY_}")

	set(NBL_TEMPLATE_APK_RESOURCES_ARCHIVE_HEADER "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/template/CArchive.h.in")
	set(NBL_BUILTIN_HEADER_GEN_PY "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/builtinHeaderGen.py")
	set(NBL_BUILTIN_DATA_GEN_PY "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/builtinDataGen.py")
	set(NBL_BS_HEADER_FILENAME "builtinResources.h")
	set(NBL_BS_DATA_SOURCE_FILENAME "builtinResourceData.cpp")
	
	string(REPLACE "::" "/" _PATH_PREFIX_ "${_NAMESPACE_}")
	string(REPLACE "::" "_" _GUARD_SUFFIX_ "${_NAMESPACE_}")
	string(REGEX REPLACE "^[0-9]+\." "" _GUARD_SUFFIX_ ${_GUARD_SUFFIX_})
	string(TOUPPER ${_GUARD_SUFFIX_} _GUARD_SUFFIX_)
	string(MAKE_C_IDENTIFIER ${_GUARD_SUFFIX_} _GUARD_SUFFIX_)
	
	set(_ITR_ 0)
	foreach(X IN LISTS _LBR_${_BUNDLE_NAME_}_)
		set(_CURRENT_ITEM_ "${X}")
		string(FIND "${_CURRENT_ITEM_}" "," _FOUND_ REVERSE)
		
		string(REGEX REPLACE ",.*" "" _CURRENT_PATH_ "${_CURRENT_ITEM_}")
		string(REGEX REPLACE ".*," "" _CURRENT_ALIAS_ "${_CURRENT_ITEM_}")
		
		if(EXISTS "${_BUNDLE_SEARCH_DIRECTORY_}/${_CURRENT_PATH_}")
			list(APPEND NBL_DEPENDENCY_FILES "${_BUNDLE_SEARCH_DIRECTORY_}/${_CURRENT_PATH_}")
			file(SIZE "${_BUNDLE_SEARCH_DIRECTORY_}/${_CURRENT_PATH_}" _FILE_SIZE_)
			
			string(APPEND _RESOURCES_INIT_LIST_ "\t\t\t\t\t{\"${_CURRENT_PATH_}\", ${_FILE_SIZE_}, 0xdeadbeefu, ${_ITR_}, nbl::system::IFileArchive::E_ALLOCATOR_TYPE::EAT_NULL},\n") # initializer list
		else()
			message(FATAL_ERROR "!") # TODO: set GENERATED property, therefore we could turn some input into output and list it as builtin resource 
		endif()	
		
		math(EXPR _ITR_ "${_ITR_} + 1")
	endforeach()
	
	configure_file("${NBL_TEMPLATE_APK_RESOURCES_ARCHIVE_HEADER}" "${_OUTPUT_DIRECTORY_}/CArchive.h")
	
	list(APPEND NBL_DEPENDENCY_FILES "${NBL_BUILTIN_HEADER_GEN_PY}")
	list(APPEND NBL_DEPENDENCY_FILES "${NBL_BUILTIN_DATA_GEN_PY}")

	set(NBL_RESOURCES_LIST_FILE "${_OUTPUT_DIRECTORY_}/resources.txt")

	string(REPLACE ";" "," RESOURCES_ARGS "${${_BUNDLE_NAME_}}")
	file(WRITE "${NBL_RESOURCES_LIST_FILE}" "${RESOURCES_ARGS}")

	set(NBL_BUILTIN_RESOURCES_HEADER "${_OUTPUT_DIRECTORY_}/${NBL_BS_HEADER_FILENAME}")
	set(NBL_BUILTIN_RESOURCE_DATA_SOURCE "${_OUTPUT_DIRECTORY_}/${NBL_BS_DATA_SOURCE_FILENAME}")
	
	set(_NBL_SOURCES_
		"${NBL_BUILTIN_RESOURCES_HEADER}"
		"${NBL_BUILTIN_RESOURCE_DATA_SOURCE}"
	)

	add_custom_command(
		OUTPUT "${NBL_BUILTIN_RESOURCES_HEADER}" "${NBL_BUILTIN_RESOURCE_DATA_SOURCE}"
		COMMAND "${PYTHON_EXECUTABLE}" "${NBL_BUILTIN_HEADER_GEN_PY}" "${NBL_BUILTIN_RESOURCES_HEADER}" "${_BUNDLE_SEARCH_DIRECTORY_}" "${NBL_RESOURCES_LIST_FILE}" "${_NAMESPACE_}" "${_GUARD_SUFFIX_}"
		COMMAND "${PYTHON_EXECUTABLE}" "${NBL_BUILTIN_DATA_GEN_PY}" "${NBL_BUILTIN_RESOURCE_DATA_SOURCE}" "${_BUNDLE_SEARCH_DIRECTORY_}" "${NBL_RESOURCES_LIST_FILE}" "${_NAMESPACE_}" "${NBL_BS_HEADER_FILENAME}"
		COMMENT "Generating built-in resources"
		DEPENDS ${NBL_DEPENDENCY_FILES}
		VERBATIM  
	)
	
	add_library(${_TARGET_NAME_} STATIC
		${_NBL_SOURCES_}
	)
	
	target_include_directories(${_TARGET_NAME_} PUBLIC "${NBL_ROOT_PATH}/include")
	
	if(NBL_DYNAMIC_MSVC_RUNTIME)
		set_property(TARGET ${_TARGET_NAME_} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
	else()
		set_property(TARGET ${_TARGET_NAME_} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
	endif()
	
	list(APPEND _NBL_BUILTIN_RESOURCES_ARCHIVE_APK_SOURCES_ "${_OUTPUT_DIRECTORY_}/CArchive.h")
	set(_NBL_BUILTIN_RESOURCES_ARCHIVE_APK_SOURCES_ ${_NBL_BUILTIN_RESOURCES_ARCHIVE_APK_SOURCES_} PARENT_SCOPE)
	
	list(APPEND _NBL_BUILTIN_RESOURCES_LIBRARIES_ ${_TARGET_NAME_})
	set(_NBL_BUILTIN_RESOURCES_LIBRARIES_ ${_NBL_BUILTIN_RESOURCES_LIBRARIES_} PARENT_SCOPE) # override
endfunction()

macro(NBL_ADD_BUILTIN_RESOURCES)
	ADD_CUSTOM_BUILTIN_RESOURCES(nblBuiltinResourceData NBL_RESOURCES_TO_EMBED "${NBL_ROOT_PATH}/include" "nbl::builtin" "${NBL_ROOT_PATH_BINARY}/include/nbl/builtin")
endmacro()

