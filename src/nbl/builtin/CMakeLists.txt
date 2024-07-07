include("${CMAKE_CURRENT_SOURCE_DIR}/utils.cmake")

LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "shader/loader/mtl/vertex_impl.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "shader/loader/mtl/vertex_no_uv.vert") # TODO: remove naming tautology
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "shader/loader/mtl/vertex_uv.vert") # TODO: remove naming tautology
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "shader/loader/mtl/fragment_impl.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "shader/loader/mtl/fragment_no_uv.frag") # TODO: remove naming tautology
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "shader/loader/mtl/fragment_uv.frag") # TODO: remove naming tautology
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "shader/loader/gltf/common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "shader/loader/gltf/vertex_impl.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "shader/loader/gltf/uv.vert")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "shader/loader/gltf/color.vert")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "shader/loader/gltf/no_uv_color.vert")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "shader/loader/gltf/fragment_impl.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "shader/loader/gltf/uv.frag")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "shader/loader/gltf/color.frag")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "shader/loader/gltf/no_uv_color.frag")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "material/lambertian/singletexture/specialized_shader.vert")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "material/lambertian/singletexture/specialized_shader.frag")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "material/debug/vertex_color/specialized_shader.vert")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "material/debug/vertex_normal/specialized_shader.vert")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "material/debug/vertex_normal/specialized_shader.frag")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "material/debug/vertex_uv/specialized_shader.frag")
# generic GLSL headers after this line
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/macros.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/algorithm.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ieee754.glsl")
# barycentric
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/barycentric/extensions.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/barycentric/frag.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/barycentric/vert.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/barycentric/utils.glsl")
#bda
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/bda/__ref.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/bda/__ptr.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/bda/bda_accessor.hlsl")
# bump mapping
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bump_mapping/fragment.glsl") # TODO: rename to `frag.glsl`
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bump_mapping/utils.glsl")
# bxdf
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/common_samples.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/fresnel.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/ndf/common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/ndf/blinn_phong.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/ndf/beckmann.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/ndf/ggx.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/geom/smith/common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/geom/smith/beckmann.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/geom/smith/ggx.glsl")
# brdf
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/brdf/diffuse/lambert.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/brdf/diffuse/oren_nayar.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/brdf/specular/blinn_phong.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/brdf/specular/beckmann.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/brdf/specular/ggx.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/brdf/diffuse/fresnel_correction.glsl")
# bsdf
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/bsdf/diffuse/lambert.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/bsdf/specular/common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/bsdf/specular/dielectric.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/bsdf/specular/beckmann.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/bxdf/bsdf/specular/ggx.glsl")
# colorspace
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/colorspace/EOTF.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/colorspace/OETF.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/colorspace/decodeCIEXYZ.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/colorspace/encodeCIEXYZ.glsl")
# Culling and LoD System
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/culling_lod_selection/dispatch_indirect_params.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/culling_lod_selection/draw_instance_count_scan_override.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/culling_lod_selection/input_descriptor_set.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/culling_lod_selection/instance_cull_and_lod_select.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/culling_lod_selection/instance_draw_count_scan_override.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/culling_lod_selection/instance_draw_cull.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/culling_lod_selection/instance_ref_counting_sort_scatter.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/culling_lod_selection/potentially_visible_instance_draw_struct.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/culling_lod_selection/output_descriptor_modifiers.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/culling_lod_selection/output_descriptor_set.glsl")
# VK_FORMAT support
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/format/decode.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/format/encode.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/format/constants.glsl")
# asset loaders (mostly for pipelines)
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/loader/mtl/common.glsl")
# LoD Library
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/lod_library/structs.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/lod_library/descriptor_set.glsl")
# math and limits
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/math/constants.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/math/complex.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/math/functions.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/math/quaternions.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/math/typeless_arithmetic.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/limits/numeric.glsl")
# material_compiler
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/material_compiler/common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/material_compiler/common_declarations.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/material_compiler/common_invariant_declarations.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/material_compiler/rasterization/impl.glsl")
# property pool
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/property_pool/transfer.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/property_pool/copy.comp")
# random numbers
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/random/xoroshiro.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/random/xoroshiro.hlsl")
# sampling
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/sampling/bilinear.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/sampling/box_muller_transform.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/sampling/concentric_mapping.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/sampling/cos_weighted.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/sampling/linear.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/sampling/projected_spherical_triangle.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/sampling/spherical_rectangle.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/sampling/spherical_triangle.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/sampling/quantized_sequence.glsl")
# global exclusive scan
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/scan/direct.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/scan/declarations.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/scan/descriptors.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/scan/default_scheduler.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/scan/indirect.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/scan/parameters_struct.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/scan/virtual_workgroup.glsl")
# faster and easier scan
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/scanning_append/scanning_append.glsl")
# scene
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/scene/animation.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/scene/keyframe.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/scene/node.glsl")
# shapes
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/shapes/aabb.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/shapes/frustum.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/shapes/rectangle.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/shapes/triangle.glsl")
# skinning
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/skinning/cache_descriptor_set.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/skinning/cache_update.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/skinning/debug.vert")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/skinning/linear.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/skinning/render_descriptor_set.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/skinning/update_descriptor_set.glsl")
# subgroup emulation
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/subgroup/arithmetic_portability.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/subgroup/arithmetic_portability_impl.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/subgroup/basic_portability.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/subgroup/fft.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/subgroup/shared_arithmetic_portability.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/subgroup/shared_shuffle_portability.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/subgroup/shuffle_portability.glsl")
# utilities
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/utils/common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/utils/culling.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/utils/compressed_normal_matrix_t.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/utils/normal_decode.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/utils/normal_encode.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/utils/transform.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/utils/morton.glsl")
# virtual texturing
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/virtual_texturing/descriptors.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/virtual_texturing/extensions.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/virtual_texturing/impl_functions.glsl")
# workgroup "intrinsics"
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/workgroup/arithmetic.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/workgroup/basic.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/workgroup/ballot.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/workgroup/fft.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/workgroup/shared_arithmetic.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/workgroup/shared_ballot.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/workgroup/shared_fft.glsl")
#transform_tree
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/transform_tree/global_transform_update.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/transform_tree/global_transform_and_normal_matrix_update.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/transform_tree/global_transform_update_common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/transform_tree/global_transform_update_descriptor_set.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/transform_tree/modification_request_range.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/transform_tree/pool_descriptor_set.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/transform_tree/render_descriptor_set.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/transform_tree/relative_transform_modification.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/transform_tree/relative_transform_update.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/transform_tree/relative_transform_update_common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/transform_tree/relative_transform_update_descriptor_set.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/transform_tree/debug.vert")
# ext shouldn't be built into the engine, but there's no harm including some non-dynamic GLSL source to make life easier
#LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/.glsl")
# radix sort
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/FFT/default_compute_fft.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/FFT/fft.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/FFT/parameters_struct.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/FFT/parameters.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/FFT/types.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/LumaMeter/common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/LumaMeter/impl.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/ToneMapper/operators.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/RadeonRays/intersection.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/RadeonRays/ray.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/MitsubaLoader/instance_data_struct.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/MitsubaLoader/instance_data_descriptor.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/MitsubaLoader/material_compiler_compatibility_impl.glsl")
# OIT
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/OIT/oit.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/OIT/insert_node.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/OIT/resolve.frag")
# virtual geometry
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/virtual_geometry/descriptors.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/virtual_geometry/virtual_attribute.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/virtual_geometry/virtual_attribute_fetch.glsl")
# depth pyramid generator
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/DepthPyramidGenerator/common.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/DepthPyramidGenerator/push_constants_struct_common.h")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/DepthPyramidGenerator/depth_pyramid_generator_impl.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/ext/DepthPyramidGenerator/virtual_work_group.glsl")
# blit
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/blit/formats_encode.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/blit/parameters.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/blit/multi_dimensional_array_addressing.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/blit/default_compute_common.comp")

LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/blit/default_compute_blit.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/blit/blit/blit.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/blit/blit/descriptors.glsl")

LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/blit/default_compute_alpha_test.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/blit/alpha_test/alpha_test.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/blit/alpha_test/descriptors.glsl")

LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/blit/default_compute_normalization.comp")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/blit/normalization/normalization.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/blit/normalization/descriptors.glsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "glsl/blit/normalization/shared_normalization.glsl")

# HLSL
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/macros.h")
#spirv intrinsics
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/spirv_intrinsics/core.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/spirv_intrinsics/fragment_shader_pixel_interlock.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/spirv_intrinsics/subgroup_arithmetic.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/spirv_intrinsics/subgroup_ballot.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/spirv_intrinsics/subgroup_basic.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/spirv_intrinsics/subgroup_shuffle.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/spirv_intrinsics/subgroup_vote.hlsl")
#C++ compatibility
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/cpp_compat.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/cpp_compat/intrinsics.h")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/cpp_compat/matrix.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/cpp_compat/promote.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/cpp_compat/vector.hlsl")
#glsl compat
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/glsl_compat/core.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/glsl_compat/subgroup_arithmetic.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/glsl_compat/subgroup_ballot.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/glsl_compat/subgroup_basic.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/glsl_compat/subgroup_shuffle.hlsl")
#stdlib
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/algorithm.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/bit.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/concepts.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/functional.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/limits.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/type_traits.hlsl")
#metaprogramming
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/mpl.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/member_test_macros.hlsl")
#device capability
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/device_capabilities_traits.hlsl")
#Basic math constants
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/numbers.hlsl")
#Complex math
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/complex.hlsl")
#linear algebra
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/math/linalg/fast_affine.hlsl")
# TODO: rename `equations` to `polynomials` probably
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/math/geometry.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/math/equations/quadratic.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/math/equations/cubic.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/math/equations/quartic.hlsl")
#extra math
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/math/quadrature/gauss_legendre/gauss_legendre.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/math/quadrature/gauss_legendre/impl.hlsl")
#acceleration structures
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/acceleration_structures.hlsl")
#colorspace
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/colorspace/EOTF.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/colorspace/OETF.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/colorspace/decodeCIEXYZ.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/colorspace/encodeCIEXYZ.hlsl")
#barycentrics
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/barycentric/utils.hlsl")
#scanning append
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/scanning_append.hlsl")
#shapes
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/shapes/circle.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/shapes/ellipse.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/shapes/line.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/shapes/beziers.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/shapes/util.hlsl")
#FFT
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/fft/common.hlsl")
#sort
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/sort/common.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/sort/counting.hlsl")
#subgroup
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/subgroup/ballot.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/subgroup/basic.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/subgroup/arithmetic_portability.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/subgroup/arithmetic_portability_impl.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/subgroup/fft.hlsl")
#shared header between C++ and HLSL
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/surface_transform.h")
#workgroup
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/workgroup/arithmetic.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/workgroup/basic.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/workgroup/ballot.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/workgroup/broadcast.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/workgroup/scratch_size.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/workgroup/shared_scan.hlsl")
#global scan
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/workgroup/declarations.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/workgroup/default_scheduler.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/workgroup/descriptors.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/workgroup/direct.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/workgroup/scan_scheduler.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/workgroup/scheduler.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/workgroup/virtual_workgroup.hlsl")
#Extensions
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl")
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/ext/FullScreenTriangle/default.vert.hlsl")
#memory
LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED "hlsl/memory.hlsl")

ADD_CUSTOM_BUILTIN_RESOURCES(nblBuiltinResourceData NBL_RESOURCES_TO_EMBED "${NBL_ROOT_PATH}/include" "nbl/builtin" "nbl::builtin" "${NBL_ROOT_PATH_BINARY}/include" "${NBL_ROOT_PATH_BINARY}/src" "STATIC" "INTERNAL")
