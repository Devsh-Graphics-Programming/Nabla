  function(irr_add_builtin_resource)
  # generate files
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/builtinResources.h" "${CMAKE_CURRENT_BINARY_DIR}/builtinResourceData.cpp"
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/builtinHeaderGen.py ${CMAKE_CURRENT_BINARY_DIR}/builtinResources.h ${CMAKE_SOURCE_DIR} "${ARGV}"
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/builtinDataGen.py ${CMAKE_CURRENT_BINARY_DIR}/builtinResourceData.cpp ${CMAKE_SOURCE_DIR} "${ARGV}"
    COMMENT "Generated built-in resources"
    VERBATIM  
  )

  # set up dependency
  add_custom_target(builtin_resources DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/builtinResources.h" "${CMAKE_CURRENT_BINARY_DIR}/builtinResourceData.cpp")
  add_dependencies(Irrlicht builtin_resources)

endfunction()


if(IRR_EMBED_BUILTIN_RESOURCES)
  #fill this list with resource paths relative to top level of IrrlichtBaW's source
  irr_add_builtin_resource(
    "src/irr/builtin/materials/lambertian/singletexture/specializedshader_vertex"
    "src/irr/builtin/materials/lambertian/singletexture/specializedshader_fragment"
    "src/irr/builtin/materials/lambertian/no_texture/specializedshader_vertex"
    "src/irr/builtin/materials/lambertian/no_texture/specializedshader_fragment"
  )
endif()