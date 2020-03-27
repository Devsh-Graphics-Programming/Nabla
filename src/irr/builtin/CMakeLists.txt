function(irr_add_builtin_resource builtin_resource_list)
  # generate files
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/builtinResources.h" "${CMAKE_CURRENT_BINARY_DIR}/builtinResourceData.cpp"
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/builtinHeaderGen.py ${CMAKE_CURRENT_BINARY_DIR}/builtinResources.h ${CMAKE_SOURCE_DIR} ${builtin_resource_list}
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/builtinDataGen.py ${CMAKE_CURRENT_BINARY_DIR}/builtinResourceData.cpp ${CMAKE_SOURCE_DIR} ${builtin_resource_list}
    COMMENT "Generated built-in resources"
    VERBATIM  
  )

  # set up dependency
  add_custom_target(builtin_resources DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/builtinResources.h" "${CMAKE_CURRENT_BINARY_DIR}/builtinResourceData.cpp")
  add_dependencies(Irrlicht builtin_resources)

endfunction()


if(IRR_EMBED_BUILTIN_RESOURCES)
  #fill this list with resource paths relative to top level of IrrlichtBaW's source
  irr_add_builtin_resource()
endif()