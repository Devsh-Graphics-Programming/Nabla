function(nbl_add_builtin_resources)
  # generate files
  foreach(X IN LISTS ARGV0)
    list(APPEND builtin_dependencies_list  ${CMAKE_SOURCE_DIR}/include/${X})
  endforeach()
  list(APPEND builtin_dependencies_list  ${CMAKE_CURRENT_SOURCE_DIR}/builtin/builtinHeaderGen.py)
  list(APPEND builtin_dependencies_list  ${CMAKE_CURRENT_SOURCE_DIR}/builtin/builtinDataGen.py)

  # Just in case if not found yet (3rd parties should've found it already though regardless of platform)
  find_package(Python3 REQUIRED)

  set(RESOURCES_FILE ${CMAKE_BINARY_DIR}/include/nbl/builtin/resources.txt)

  string(REPLACE ";" "," RESOURCES_ARGS "${ARGV}")
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/nbl/builtin/)
  file(WRITE ${RESOURCES_FILE} ${RESOURCES_ARGS})
  add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/include/nbl/builtin/builtinResources.h" "${CMAKE_CURRENT_BINARY_DIR}/builtin/builtinResourceData.cpp"
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/builtin/builtinHeaderGen.py ${CMAKE_BINARY_DIR}/include/nbl/builtin/builtinResources.h ${CMAKE_SOURCE_DIR}/include ${RESOURCES_FILE}
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/builtin/builtinDataGen.py ${CMAKE_CURRENT_BINARY_DIR}/builtin/builtinResourceData.cpp ${CMAKE_SOURCE_DIR}/include ${RESOURCES_FILE}
    COMMENT "Generated built-in resources"
    DEPENDS ${builtin_dependencies_list}
    VERBATIM  
  )
endfunction()

set(nbl_resources_to_embed
  "nbl/builtin/glsl/utils/acceleration_structures.glsl"
  "nbl/builtin/shader/loader/mtl/vertex_impl.glsl"
  "nbl/builtin/shader/loader/mtl/vertex_no_uv.vert" # TODO: remove naming tautology
  "nbl/builtin/shader/loader/mtl/vertex_uv.vert" # TODO: remove naming tautology
  "nbl/builtin/shader/loader/mtl/fragment_impl.glsl"
  "nbl/builtin/shader/loader/mtl/fragment_no_uv.frag" # TODO: remove naming tautology
  "nbl/builtin/shader/loader/mtl/fragment_uv.frag" # TODO: remove naming tautology
  "nbl/builtin/shader/loader/gltf/common.glsl"
  "nbl/builtin/shader/loader/gltf/vertex_impl.glsl"
  "nbl/builtin/shader/loader/gltf/uv.vert"
  "nbl/builtin/shader/loader/gltf/color.vert"
  "nbl/builtin/shader/loader/gltf/no_uv_color.vert"
  "nbl/builtin/shader/loader/gltf/fragment_impl.glsl"
  "nbl/builtin/shader/loader/gltf/uv.frag"
  "nbl/builtin/shader/loader/gltf/color.frag"
  "nbl/builtin/shader/loader/gltf/no_uv_color.frag"
  "nbl/builtin/specialized_shader/fullscreentriangle.vert"
  "nbl/builtin/material/lambertian/singletexture/specialized_shader.vert"
  "nbl/builtin/material/lambertian/singletexture/specialized_shader.frag"
  "nbl/builtin/material/debug/vertex_color/specialized_shader.vert"
  "nbl/builtin/material/debug/vertex_normal/specialized_shader.vert"
  "nbl/builtin/material/debug/vertex_normal/specialized_shader.frag"
  "nbl/builtin/material/debug/vertex_uv/specialized_shader.frag"
  # generic GLSL headers after this line
  "nbl/builtin/glsl/macros.glsl"
  "nbl/builtin/glsl/algorithm.glsl"
  # barycentric
  "nbl/builtin/glsl/barycentric/extensions.glsl"
  "nbl/builtin/glsl/barycentric/frag.glsl"
  "nbl/builtin/glsl/barycentric/vert.glsl"
  "nbl/builtin/glsl/barycentric/utils.glsl"
  # bump mapping
  "nbl/builtin/glsl/bump_mapping/fragment.glsl" # TODO: rename to `frag.glsl`
  "nbl/builtin/glsl/bump_mapping/utils.glsl"
  # bxdf
  "nbl/builtin/glsl/bxdf/common.glsl"
  "nbl/builtin/glsl/bxdf/common_samples.glsl"
  "nbl/builtin/glsl/bxdf/fresnel.glsl"
  "nbl/builtin/glsl/bxdf/ndf/common.glsl"
  "nbl/builtin/glsl/bxdf/ndf/blinn_phong.glsl"
  "nbl/builtin/glsl/bxdf/ndf/beckmann.glsl"
  "nbl/builtin/glsl/bxdf/ndf/ggx.glsl"
  "nbl/builtin/glsl/bxdf/geom/smith/common.glsl"
  "nbl/builtin/glsl/bxdf/geom/smith/beckmann.glsl"
  "nbl/builtin/glsl/bxdf/geom/smith/ggx.glsl"
  # brdf
  "nbl/builtin/glsl/bxdf/brdf/diffuse/lambert.glsl"
  "nbl/builtin/glsl/bxdf/brdf/diffuse/oren_nayar.glsl"
  "nbl/builtin/glsl/bxdf/brdf/specular/blinn_phong.glsl"
  "nbl/builtin/glsl/bxdf/brdf/specular/beckmann.glsl"
  "nbl/builtin/glsl/bxdf/brdf/specular/ggx.glsl"
  "nbl/builtin/glsl/bxdf/brdf/diffuse/fresnel_correction.glsl"
  # bsdf
  "nbl/builtin/glsl/bxdf/bsdf/diffuse/lambert.glsl"
  "nbl/builtin/glsl/bxdf/bsdf/specular/common.glsl"
  "nbl/builtin/glsl/bxdf/bsdf/specular/dielectric.glsl"
  "nbl/builtin/glsl/bxdf/bsdf/specular/beckmann.glsl"
  "nbl/builtin/glsl/bxdf/bsdf/specular/ggx.glsl"
  # colorspace
  "nbl/builtin/glsl/colorspace/EOTF.glsl"
  "nbl/builtin/glsl/colorspace/OETF.glsl"
  "nbl/builtin/glsl/colorspace/decodeCIEXYZ.glsl"
  "nbl/builtin/glsl/colorspace/encodeCIEXYZ.glsl"
  # Culling and LoD System
  "nbl/builtin/glsl/culling_lod_selection/dispatch_indirect_params.glsl"
  "nbl/builtin/glsl/culling_lod_selection/draw_instance_count_scan_override.glsl"
  "nbl/builtin/glsl/culling_lod_selection/input_descriptor_set.glsl"
  "nbl/builtin/glsl/culling_lod_selection/instance_cull_and_lod_select.comp"
  "nbl/builtin/glsl/culling_lod_selection/instance_draw_count_scan_override.glsl"
  "nbl/builtin/glsl/culling_lod_selection/instance_draw_cull.comp"
  "nbl/builtin/glsl/culling_lod_selection/instance_ref_counting_sort_scatter.comp"
  "nbl/builtin/glsl/culling_lod_selection/potentially_visible_instance_draw_struct.glsl"
  "nbl/builtin/glsl/culling_lod_selection/output_descriptor_modifiers.glsl"
  "nbl/builtin/glsl/culling_lod_selection/output_descriptor_set.glsl"
  # VK_FORMAT support
  "nbl/builtin/glsl/format/decode.glsl"
  "nbl/builtin/glsl/format/encode.glsl"
  "nbl/builtin/glsl/format/constants.glsl"
  # asset loaders (mostly for pipelines)
  "nbl/builtin/glsl/loader/mtl/common.glsl"
  # LoD Library
  "nbl/builtin/glsl/lod_library/structs.glsl"
  "nbl/builtin/glsl/lod_library/descriptor_set.glsl"
  # math and limits
  "nbl/builtin/glsl/math/constants.glsl"
  "nbl/builtin/glsl/math/complex.glsl"
  "nbl/builtin/glsl/math/functions.glsl"
  "nbl/builtin/glsl/math/quaternions.glsl"
  "nbl/builtin/glsl/math/typeless_arithmetic.glsl"
  "nbl/builtin/glsl/limits/numeric.glsl"
  # material_compiler
  "nbl/builtin/glsl/material_compiler/common.glsl"
  "nbl/builtin/glsl/material_compiler/common_declarations.glsl"
  "nbl/builtin/glsl/material_compiler/common_invariant_declarations.glsl"
  "nbl/builtin/glsl/material_compiler/rasterization/impl.glsl"
  # property pool
  "nbl/builtin/glsl/property_pool/transfer.glsl"
  "nbl/builtin/glsl/property_pool/copy.comp"
  # random numbers
  "nbl/builtin/glsl/random/xoroshiro.glsl"
  # sampling
  "nbl/builtin/glsl/sampling/bilinear.glsl"
  "nbl/builtin/glsl/sampling/box_muller_transform.glsl"
  "nbl/builtin/glsl/sampling/concentric_mapping.glsl"
  "nbl/builtin/glsl/sampling/cos_weighted.glsl"
  "nbl/builtin/glsl/sampling/linear.glsl"
  "nbl/builtin/glsl/sampling/projected_spherical_triangle.glsl"
  "nbl/builtin/glsl/sampling/spherical_rectangle.glsl"
  "nbl/builtin/glsl/sampling/spherical_triangle.glsl"
  "nbl/builtin/glsl/sampling/quantized_sequence.glsl"
  # global exclusive scan
  "nbl/builtin/glsl/scan/direct.comp"
  "nbl/builtin/glsl/scan/declarations.glsl"
  "nbl/builtin/glsl/scan/descriptors.glsl"
  "nbl/builtin/glsl/scan/default_scheduler.glsl"
  "nbl/builtin/glsl/scan/indirect.comp"
  "nbl/builtin/glsl/scan/parameters_struct.glsl"
  "nbl/builtin/glsl/scan/virtual_workgroup.glsl"
  # scene
  "nbl/builtin/glsl/scene/animation.glsl"
  "nbl/builtin/glsl/scene/keyframe.glsl"
  "nbl/builtin/glsl/scene/node.glsl"
  # shapes
  "nbl/builtin/glsl/shapes/aabb.glsl"
  "nbl/builtin/glsl/shapes/frustum.glsl"
  "nbl/builtin/glsl/shapes/rectangle.glsl"
  "nbl/builtin/glsl/shapes/triangle.glsl"
  # skinning
  "nbl/builtin/glsl/skinning/cache_descriptor_set.glsl"
  "nbl/builtin/glsl/skinning/cache_update.comp"
  "nbl/builtin/glsl/skinning/debug.vert"
  "nbl/builtin/glsl/skinning/linear.glsl"
  "nbl/builtin/glsl/skinning/render_descriptor_set.glsl"
  "nbl/builtin/glsl/skinning/update_descriptor_set.glsl"
  # subgroup emulation
  "nbl/builtin/glsl/subgroup/shared_arithmetic_portability.glsl"
  "nbl/builtin/glsl/subgroup/basic_portability.glsl"
  "nbl/builtin/glsl/subgroup/arithmetic_portability.glsl"
  "nbl/builtin/glsl/subgroup/arithmetic_portability_impl.glsl"
  ##"nbl/builtin/glsl/subgroup/fft.glsl"
  # utilities
  "nbl/builtin/glsl/utils/indirect_commands.glsl"
  "nbl/builtin/glsl/utils/common.glsl"
  "nbl/builtin/glsl/utils/culling.glsl"
  "nbl/builtin/glsl/utils/compressed_normal_matrix_t.glsl"
  "nbl/builtin/glsl/utils/normal_decode.glsl"
  "nbl/builtin/glsl/utils/normal_encode.glsl"
  "nbl/builtin/glsl/utils/transform.glsl"
  "nbl/builtin/glsl/utils/morton.glsl"
  # virtual texturing
  "nbl/builtin/glsl/virtual_texturing/descriptors.glsl"
  "nbl/builtin/glsl/virtual_texturing/extensions.glsl"
  "nbl/builtin/glsl/virtual_texturing/impl_functions.glsl"
  # workgroup "intrinsics"
  "nbl/builtin/glsl/workgroup/shared_arithmetic.glsl"
  "nbl/builtin/glsl/workgroup/shared_ballot.glsl"
  "nbl/builtin/glsl/workgroup/shared_clustered.glsl"
  "nbl/builtin/glsl/workgroup/shared_fft.glsl"
  "nbl/builtin/glsl/workgroup/shared_shuffle.glsl"
  "nbl/builtin/glsl/workgroup/shared_shuffle_relative.glsl"
  "nbl/builtin/glsl/workgroup/shared_vote.glsl"
  "nbl/builtin/glsl/workgroup/arithmetic.glsl"
  "nbl/builtin/glsl/workgroup/ballot.glsl"
  "nbl/builtin/glsl/workgroup/basic.glsl"
  "nbl/builtin/glsl/workgroup/clustered.glsl"
  "nbl/builtin/glsl/workgroup/fft.glsl"
  "nbl/builtin/glsl/workgroup/shuffle.glsl"
  "nbl/builtin/glsl/workgroup/shuffle_relative.glsl"
  "nbl/builtin/glsl/workgroup/vote.glsl"
  #transform_tree
  "nbl/builtin/glsl/transform_tree/global_transform_update.comp"
  "nbl/builtin/glsl/transform_tree/global_transform_and_normal_matrix_update.comp"
  "nbl/builtin/glsl/transform_tree/global_transform_update_common.glsl"
  "nbl/builtin/glsl/transform_tree/global_transform_update_descriptor_set.glsl"
  "nbl/builtin/glsl/transform_tree/modification_request_range.glsl"
  "nbl/builtin/glsl/transform_tree/pool_descriptor_set.glsl"
  "nbl/builtin/glsl/transform_tree/render_descriptor_set.glsl"
  "nbl/builtin/glsl/transform_tree/relative_transform_modification.glsl"
  "nbl/builtin/glsl/transform_tree/relative_transform_update.comp"
  "nbl/builtin/glsl/transform_tree/relative_transform_update_common.glsl"
  "nbl/builtin/glsl/transform_tree/relative_transform_update_descriptor_set.glsl"
  "nbl/builtin/glsl/transform_tree/debug.vert"
  # ext shouldn't be built into the engine, but there's no harm including some non-dynamic GLSL source to make life easier
  #"nbl/builtin/glsl/ext/.glsl"
  # radix sort
  "nbl/builtin/glsl/ext/FFT/default_compute_fft.comp"
  "nbl/builtin/glsl/ext/FFT/fft.glsl"
  "nbl/builtin/glsl/ext/FFT/parameters_struct.glsl"
  "nbl/builtin/glsl/ext/FFT/parameters.glsl"
  "nbl/builtin/glsl/ext/FFT/types.glsl"
  "nbl/builtin/glsl/ext/LumaMeter/common.glsl"
  "nbl/builtin/glsl/ext/LumaMeter/impl.glsl"
  "nbl/builtin/glsl/ext/ToneMapper/operators.glsl"
  "nbl/builtin/glsl/ext/RadeonRays/intersection.glsl"
  "nbl/builtin/glsl/ext/RadeonRays/ray.glsl"
  "nbl/builtin/glsl/ext/MitsubaLoader/instance_data_struct.glsl"
  "nbl/builtin/glsl/ext/MitsubaLoader/instance_data_descriptor.glsl"
  "nbl/builtin/glsl/ext/MitsubaLoader/material_compiler_compatibility_impl.glsl"
  # OIT
  "nbl/builtin/glsl/ext/OIT/oit.glsl"
  "nbl/builtin/glsl/ext/OIT/insert_node.glsl"
  "nbl/builtin/glsl/ext/OIT/resolve.frag"
  # virtual geometry
  "nbl/builtin/glsl/virtual_geometry/descriptors.glsl"
  "nbl/builtin/glsl/virtual_geometry/virtual_attribute.glsl"
  "nbl/builtin/glsl/virtual_geometry/virtual_attribute_fetch.glsl"
  # depth pyramid generator
  "nbl/builtin/glsl/ext/DepthPyramidGenerator/common.glsl"
  "nbl/builtin/glsl/ext/DepthPyramidGenerator/push_constants_struct_common.h"
  "nbl/builtin/glsl/ext/DepthPyramidGenerator/depth_pyramid_generator_impl.glsl"
  "nbl/builtin/glsl/ext/DepthPyramidGenerator/virtual_work_group.glsl"
PARENT_SCOPE)

