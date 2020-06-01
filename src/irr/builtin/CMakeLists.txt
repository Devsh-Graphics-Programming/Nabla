function(irr_add_builtin_resources)
  # generate files
  foreach(X IN LISTS ARGV0)
    list(APPEND builtin_dependencies_list  ${CMAKE_SOURCE_DIR}/include/${X})
  endforeach()


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
  "irr/builtin/specializedshaders/fullscreentriangle.vert"
  "irr/builtin/materials/lambertian/singletexture/specializedshader.vert"
  "irr/builtin/materials/lambertian/singletexture/specializedshader.frag"
  "irr/builtin/materials/debug/vertex_color/specializedshader.vert"
  "irr/builtin/materials/debug/vertex_normal/specializedshader.vert"
  "irr/builtin/materials/debug/vertex_normal/specializedshader.frag"
  "irr/builtin/materials/debug/vertex_uv/specializedshader.frag"
  "irr/builtin/glsl/broken_driver_workarounds/amd.glsl"
  "irr/builtin/glsl/colorspace/EOTF.glsl"
  "irr/builtin/glsl/colorspace/OETF.glsl"
  "irr/builtin/glsl/colorspace/decodeCIEXYZ.glsl"
  "irr/builtin/glsl/colorspace/encodeCIEXYZ.glsl"
  "irr/builtin/glsl/utils/vertex.glsl"
  "irr/builtin/glsl/virtual_texturing/descriptors.glsl"
  "irr/builtin/glsl/virtual_texturing/extensions.glsl"
  "irr/builtin/glsl/virtual_texturing/impl_functions.glsl"
  PARENT_SCOPE)

