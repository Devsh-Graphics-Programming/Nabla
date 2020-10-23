function(irr_add_builtin_resources)
  # generate files
  foreach(X IN LISTS ARGV0)
    list(APPEND builtin_dependencies_list  ${CMAKE_SOURCE_DIR}/include/${X})
  endforeach()
  list(APPEND builtin_dependencies_list  ${CMAKE_CURRENT_SOURCE_DIR}/builtin/builtinHeaderGen.py)
  list(APPEND builtin_dependencies_list  ${CMAKE_CURRENT_SOURCE_DIR}/builtin/builtinDataGen.py)

  add_custom_command(
	OUTPUT "${CMAKE_BINARY_DIR}/include/irr/builtin/builtinResources.h" "${CMAKE_CURRENT_BINARY_DIR}/builtin/builtinResourceData.cpp"
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/builtin/builtinHeaderGen.py ${CMAKE_BINARY_DIR}/include/irr/builtin/builtinResources.h ${CMAKE_SOURCE_DIR}/include "${ARGV}"
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/builtin/builtinDataGen.py ${CMAKE_CURRENT_BINARY_DIR}/builtin/builtinResourceData.cpp ${CMAKE_SOURCE_DIR}/include "${ARGV}"
    COMMENT "Generated built-in resources"
    DEPENDS ${builtin_dependencies_list}
    VERBATIM  
  )
endfunction()

set(irr_resources_to_embed
  "irr/builtin/glsl/math/functions.glsl"
  "irr/builtin/glsl/math/constants.glsl"
  "irr/builtin/glsl/math/quaternions.glsl"
  "irr/builtin/glsl/limits/numeric.glsl"
  "irr/builtin/shaders/loaders/mtl/vertex_impl.glsl"
  "irr/builtin/shaders/loaders/mtl/vertex_no_uv.vert"
  "irr/builtin/shaders/loaders/mtl/vertex_uv.vert"
  "irr/builtin/shaders/loaders/mtl/fragment_impl.glsl"
  "irr/builtin/shaders/loaders/mtl/fragment_no_uv.frag"
  "irr/builtin/shaders/loaders/mtl/fragment_uv.frag"
  "irr/builtin/specializedshaders/fullscreentriangle.vert"
  "irr/builtin/materials/lambertian/singletexture/specializedshader.vert"
  "irr/builtin/materials/lambertian/singletexture/specializedshader.frag"
  "irr/builtin/materials/debug/vertex_color/specializedshader.vert"
  "irr/builtin/materials/debug/vertex_normal/specializedshader.vert"
  "irr/builtin/materials/debug/vertex_normal/specializedshader.frag"
  "irr/builtin/materials/debug/vertex_uv/specializedshader.frag"
  "irr/builtin/glsl/broken_driver_workarounds/amd.glsl"
  "irr/builtin/glsl/bump_mapping/utils.glsl"
  "irr/builtin/glsl/colorspace/EOTF.glsl"
  "irr/builtin/glsl/colorspace/OETF.glsl"
  "irr/builtin/glsl/colorspace/decodeCIEXYZ.glsl"
  "irr/builtin/glsl/colorspace/encodeCIEXYZ.glsl"
  "irr/builtin/glsl/loaders/mtl/common.glsl"
  "irr/builtin/glsl/utils/indirect_commands.glsl"
  "irr/builtin/glsl/utils/common.glsl"
  "irr/builtin/glsl/utils/culling.glsl"
  "irr/builtin/glsl/utils/transform.glsl"
  "irr/builtin/glsl/virtual_texturing/descriptors.glsl"
  "irr/builtin/glsl/virtual_texturing/extensions.glsl"
  "irr/builtin/glsl/virtual_texturing/impl_functions.glsl"
  "irr/builtin/glsl/format/decode.glsl"
  "irr/builtin/glsl/format/constants.glsl"
  "irr/builtin/glsl/random/xoroshiro.glsl"
  #material_compiler
  "irr/builtin/material_compiler/glsl/common.glsl"
  "irr/builtin/material_compiler/glsl/common_declarations.glsl"
  "irr/builtin/material_compiler/glsl/rasterization/impl.glsl"
  #shapes
  "irr/builtin/glsl/shapes/triangle.glsl"
  #sampling
  "irr/builtin/glsl/sampling/bilinear.glsl"
  "irr/builtin/glsl/sampling/concentric_mapping.glsl"
  "irr/builtin/glsl/sampling/cos_weighted.glsl"
  "irr/builtin/glsl/sampling/linear.glsl"
  "irr/builtin/glsl/sampling/projected_spherical_triangle.glsl"
  "irr/builtin/glsl/sampling/spherical_triangle.glsl"
  #bxdf
  "irr/builtin/glsl/bxdf/common.glsl"
  "irr/builtin/glsl/bxdf/common_samples.glsl"
  "irr/builtin/glsl/bxdf/fresnel.glsl"
  "irr/builtin/glsl/bxdf/ndf/common.glsl"
  "irr/builtin/glsl/bxdf/ndf/blinn_phong.glsl"
  "irr/builtin/glsl/bxdf/ndf/beckmann.glsl"
  "irr/builtin/glsl/bxdf/ndf/ggx.glsl"
  "irr/builtin/glsl/bxdf/geom/smith/common.glsl"
  "irr/builtin/glsl/bxdf/geom/smith/beckmann.glsl"
  "irr/builtin/glsl/bxdf/geom/smith/ggx.glsl"
  "irr/builtin/glsl/bxdf/brdf/diffuse/lambert.glsl"
  "irr/builtin/glsl/bxdf/brdf/diffuse/oren_nayar.glsl"
  "irr/builtin/glsl/bxdf/brdf/specular/blinn_phong.glsl"
  "irr/builtin/glsl/bxdf/brdf/specular/beckmann.glsl"
  "irr/builtin/glsl/bxdf/brdf/specular/ggx.glsl"
  "irr/builtin/glsl/bxdf/brdf/diffuse/fresnel_correction.glsl"
  "irr/builtin/glsl/bxdf/bsdf/diffuse/lambert.glsl"
  "irr/builtin/glsl/bxdf/bsdf/specular/common.glsl"
  "irr/builtin/glsl/bxdf/bsdf/specular/dielectric.glsl"
  "irr/builtin/glsl/bxdf/bsdf/specular/beckmann.glsl"
  "irr/builtin/glsl/bxdf/bsdf/specular/ggx.glsl"
 # ext shouldn't be built into the engine, but there's no harm including some non-dynamic GLSL source to make life easier
  #"irr/builtin/glsl/ext/.glsl"
  "irr/builtin/glsl/ext/LumaMeter/common.glsl"
  "irr/builtin/glsl/ext/LumaMeter/impl.glsl"
  "irr/builtin/glsl/ext/ToneMapper/operators.glsl"
  PARENT_SCOPE)

