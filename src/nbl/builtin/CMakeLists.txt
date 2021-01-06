function(nbl_add_builtin_resources)
  # generate files
  foreach(X IN LISTS ARGV0)
    list(APPEND builtin_dependencies_list  ${CMAKE_SOURCE_DIR}/include/${X})
  endforeach()
  list(APPEND builtin_dependencies_list  ${CMAKE_CURRENT_SOURCE_DIR}/builtin/builtinHeaderGen.py)
  list(APPEND builtin_dependencies_list  ${CMAKE_CURRENT_SOURCE_DIR}/builtin/builtinDataGen.py)

  add_custom_command(
	OUTPUT "${CMAKE_BINARY_DIR}/include/nbl/builtin/builtinResources.h" "${CMAKE_CURRENT_BINARY_DIR}/builtin/builtinResourceData.cpp"
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/builtin/builtinHeaderGen.py ${CMAKE_BINARY_DIR}/include/nbl/builtin/builtinResources.h ${CMAKE_SOURCE_DIR}/include "${ARGV}"
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/builtin/builtinDataGen.py ${CMAKE_CURRENT_BINARY_DIR}/builtin/builtinResourceData.cpp ${CMAKE_SOURCE_DIR}/include "${ARGV}"
    COMMENT "Generated built-in resources"
    DEPENDS ${builtin_dependencies_list}
    VERBATIM  
  )
endfunction()

set(nbl_resources_to_embed
  "nbl/builtin/shaders/loaders/mtl/vertex_impl.glsl"
  "nbl/builtin/shaders/loaders/mtl/vertex_no_uv.vert"
  "nbl/builtin/shaders/loaders/mtl/vertex_uv.vert"
  "nbl/builtin/shaders/loaders/mtl/fragment_impl.glsl"
  "nbl/builtin/shaders/loaders/mtl/fragment_no_uv.frag"
  "nbl/builtin/shaders/loaders/mtl/fragment_uv.frag"
  "nbl/builtin/specializedshaders/fullscreentriangle.vert"
  "nbl/builtin/materials/lambertian/singletexture/specializedshader.vert"
  "nbl/builtin/materials/lambertian/singletexture/specializedshader.frag"
  "nbl/builtin/materials/debug/vertex_color/specializedshader.vert"
  "nbl/builtin/materials/debug/vertex_normal/specializedshader.vert"
  "nbl/builtin/materials/debug/vertex_normal/specializedshader.frag"
  "nbl/builtin/materials/debug/vertex_uv/specializedshader.frag"
  # generic GLSL headers after this line
  "nbl/builtin/glsl/macros.glsl"
  "nbl/builtin/glsl/algorithm.glsl"
  # bump mapping
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
  # VK_FORMAT support
  "nbl/builtin/glsl/format/decode.glsl"
  "nbl/builtin/glsl/format/encode.glsl"
  "nbl/builtin/glsl/format/constants.glsl"
  # asset loaders (mostly for pipelines)
  "nbl/builtin/glsl/loaders/mtl/common.glsl"
  # math and limits
  "nbl/builtin/glsl/math/functions.glsl"
  "nbl/builtin/glsl/math/constants.glsl"
  "nbl/builtin/glsl/math/quaternions.glsl"
  "nbl/builtin/glsl/math/typeless_arithmetic.glsl"
  "nbl/builtin/glsl/limits/numeric.glsl"
  # material_compiler
  "nbl/builtin/glsl/material_compiler/common.glsl"
  "nbl/builtin/glsl/material_compiler/common_declarations.glsl"
  "nbl/builtin/glsl/material_compiler/common_invariant_declarations.glsl"
  "nbl/builtin/glsl/material_compiler/rasterization/impl.glsl"
  # random numbers
  "nbl/builtin/glsl/random/xoroshiro.glsl"
  # sampling
  "nbl/builtin/glsl/sampling/bilinear.glsl"
  "nbl/builtin/glsl/sampling/concentric_mapping.glsl"
  "nbl/builtin/glsl/sampling/cos_weighted.glsl"
  "nbl/builtin/glsl/sampling/linear.glsl"
  "nbl/builtin/glsl/sampling/projected_spherical_triangle.glsl"
  #"nbl/builtin/glsl/sampling/spherical_rectangle.glsl"
  "nbl/builtin/glsl/sampling/spherical_triangle.glsl"
  # shapes
  #"nbl/builtin/glsl/shapes/rectangle.glsl"
  "nbl/builtin/glsl/shapes/triangle.glsl"
  # subgroup emulation
  "nbl/builtin/glsl/subgroup/shared_arithmetic_portability.glsl"
  "nbl/builtin/glsl/subgroup/basic_portability.glsl"
  "nbl/builtin/glsl/subgroup/arithmetic_portability.glsl"
  "nbl/builtin/glsl/subgroup/arithmetic_portability_impl.glsl"
  # utilities
  "nbl/builtin/glsl/utils/indirect_commands.glsl"
  "nbl/builtin/glsl/utils/common.glsl"
  "nbl/builtin/glsl/utils/culling.glsl"
  "nbl/builtin/glsl/utils/normal_decode.glsl"
  "nbl/builtin/glsl/utils/normal_encode.glsl"
  "nbl/builtin/glsl/utils/transform.glsl"
  # virtual texturing
  "nbl/builtin/glsl/virtual_texturing/descriptors.glsl"
  "nbl/builtin/glsl/virtual_texturing/extensions.glsl"
  "nbl/builtin/glsl/virtual_texturing/impl_functions.glsl"
  # workgroup "intrinsics"
  "nbl/builtin/glsl/workgroup/shared_arithmetic.glsl"
  "nbl/builtin/glsl/workgroup/shared_ballot.glsl"
  "nbl/builtin/glsl/workgroup/shared_clustered.glsl"
  "nbl/builtin/glsl/workgroup/shared_shuffle.glsl"
  "nbl/builtin/glsl/workgroup/shared_shuffle_relative.glsl"
  "nbl/builtin/glsl/workgroup/shared_vote.glsl"
  "nbl/builtin/glsl/workgroup/arithmetic.glsl"
  "nbl/builtin/glsl/workgroup/ballot.glsl"
  "nbl/builtin/glsl/workgroup/basic.glsl"
  "nbl/builtin/glsl/workgroup/clustered.glsl"
  "nbl/builtin/glsl/workgroup/shuffle.glsl"
  "nbl/builtin/glsl/workgroup/shuffle_relative.glsl"
  "nbl/builtin/glsl/workgroup/vote.glsl"
  # ext shouldn't be built into the engine, but there's no harm including some non-dynamic GLSL source to make life easier
  #"nbl/builtin/glsl/ext/.glsl"
  "nbl/builtin/glsl/ext/LumaMeter/common.glsl"
  "nbl/builtin/glsl/ext/LumaMeter/impl.glsl"
  "nbl/builtin/glsl/ext/ToneMapper/operators.glsl"
  "nbl/builtin/glsl/ext/RadeonRays/intersection.glsl"
  "nbl/builtin/glsl/ext/RadeonRays/ray.glsl"
  "nbl/builtin/glsl/ext/MitsubaLoader/instance_data_struct.glsl"
  "nbl/builtin/glsl/ext/MitsubaLoader/material_compiler_compatibility_impl.glsl"
PARENT_SCOPE)

