function(irr_add_builtin_resource)
  # generate files
  foreach(X IN LISTS ARGV)
    list(APPEND builtin_dependencies_list  ${CMAKE_SOURCE_DIR}/include/${X})
  endforeach()

  add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/include/irr/builtin/builtinResources.h" "${CMAKE_CURRENT_BINARY_DIR}/builtinResourceData.cpp"
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/builtinHeaderGen.py ${CMAKE_BINARY_DIR}/include/irr/builtin/builtinResources.h ${CMAKE_SOURCE_DIR}/include "${ARGV}"
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/builtinDataGen.py ${CMAKE_CURRENT_BINARY_DIR}/builtinResourceData.cpp ${CMAKE_SOURCE_DIR}/include "${ARGV}"
    COMMENT "Generated built-in resources"
    DEPENDS ${builtin_dependencies_list} # havent found if it needs to be  dependencies_list  or  ${dependencies_list}  or  "${dependencies_list}"
    VERBATIM  
  )

  # set up dependency
  add_custom_target(builtin_resources
   DEPENDS ${CMAKE_BINARY_DIR}/include/irr/builtin/builtinResources.h ${CMAKE_CURRENT_BINARY_DIR}/builtinResourceData.cpp)
  add_dependencies(Irrlicht builtin_resources)

endfunction()


if(IRR_EMBED_BUILTIN_RESOURCES)
  #fill this list with resource paths relative to IrrlichtBaW's source/include
  irr_add_builtin_resource(
    "irr/builtin/materials/lambertian/singletexture/specializedshader_vertex"
    "irr/builtin/materials/lambertian/singletexture/specializedshader_fragment"
    "irr/builtin/materials/lambertian/no_texture/specializedshader_vertex"
    "irr/builtin/materials/lambertian/no_texture/specializedshader_fragment"
  )
endif()