function (set_builtin_resource_dependencies)
  add_custom_target(builtin_resources
   DEPENDS ${CMAKE_BINARY_DIR}/include/irr/builtin/builtinResources.h ${CMAKE_CURRENT_BINARY_DIR}/builtinResourceData.cpp
   )
  add_dependencies(Irrlicht builtin_resources)
  add_dependencies(IrrlichtServer builtin_resources)

endfunction()

set(irr_builtin_resource_paths PARENT_SCOPE
    "irr/builtin/materials/lambertian/singletexture/specializedshader.vert"
    "irr/builtin/materials/lambertian/singletexture/specializedshader.frag"
    "irr/builtin/materials/lambertian/no_texture/specializedshader.vert"
    "irr/builtin/materials/lambertian/no_texture/specializedshader.frag")

