include_guard(GLOBAL)

#
# nabla_add_nsc_ide_target(
#   [TARGET <name>]
#   [PACKAGE_ROOT <path>]
#   [MANIFEST_ROOT <path>]
#   [CHANNEL <name>]
# )
#
# Creates a lightweight IDE-only target that exposes the consumed NSC package
# layout as browsable sources. This helper does not create or wrap any build
# rules. It only adds a local target for IDE UX.
#
function(nabla_add_nsc_ide_target)
  cmake_parse_arguments(PARSE_ARGV 0 _NBL_NSC_IDE "" "TARGET;PACKAGE_ROOT;MANIFEST_ROOT;CHANNEL" "")

  if(_NBL_NSC_IDE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Nabla: unexpected arguments for nabla_add_nsc_ide_target: ${_NBL_NSC_IDE_UNPARSED_ARGUMENTS}")
  endif()

  if(_NBL_NSC_IDE_TARGET)
    set(_nbl_nsc_ide_target "${_NBL_NSC_IDE_TARGET}")
  else()
    set(_nbl_nsc_ide_target nsc)
  endif()

  if(TARGET "${_nbl_nsc_ide_target}")
    message(FATAL_ERROR "Nabla: target \"${_nbl_nsc_ide_target}\" already exists")
  endif()

  if(_NBL_NSC_IDE_PACKAGE_ROOT)
    cmake_path(ABSOLUTE_PATH _NBL_NSC_IDE_PACKAGE_ROOT BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" OUTPUT_VARIABLE _nbl_nsc_package_root)
  elseif(DEFINED Nabla_ROOT AND IS_DIRECTORY "${Nabla_ROOT}")
    cmake_path(ABSOLUTE_PATH Nabla_ROOT OUTPUT_VARIABLE _nbl_nsc_package_root)
  else()
    message(FATAL_ERROR "Nabla: nabla_add_nsc_ide_target requires PACKAGE_ROOT <path> or a valid Nabla_ROOT from find_package(Nabla)")
  endif()

  if(NOT IS_DIRECTORY "${_nbl_nsc_package_root}")
    message(FATAL_ERROR "Nabla: PACKAGE_ROOT \"${_nbl_nsc_package_root}\" does not exist")
  endif()

  file(GLOB_RECURSE _nbl_nsc_package_files CONFIGURE_DEPENDS LIST_DIRECTORIES false "${_nbl_nsc_package_root}/*")
  if(_nbl_nsc_package_files)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_nbl_nsc_package_files})
  endif()

  set(_nbl_nsc_ide_sources ${_nbl_nsc_package_files})

  if(_NBL_NSC_IDE_MANIFEST_ROOT)
    cmake_path(ABSOLUTE_PATH _NBL_NSC_IDE_MANIFEST_ROOT BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" OUTPUT_VARIABLE _nbl_nsc_manifest_root)
    if(NOT IS_DIRECTORY "${_nbl_nsc_manifest_root}")
      message(FATAL_ERROR "Nabla: MANIFEST_ROOT \"${_nbl_nsc_manifest_root}\" does not exist")
    endif()

    if(_NBL_NSC_IDE_CHANNEL)
      set(_nbl_nsc_manifest_glob_root "${_nbl_nsc_manifest_root}/${_NBL_NSC_IDE_CHANNEL}")
    else()
      set(_nbl_nsc_manifest_glob_root "${_nbl_nsc_manifest_root}")
    endif()

    file(GLOB_RECURSE _nbl_nsc_manifest_files CONFIGURE_DEPENDS LIST_DIRECTORIES false "${_nbl_nsc_manifest_glob_root}/*.dvc")
    if(_nbl_nsc_manifest_files)
      set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_nbl_nsc_manifest_files})
      list(APPEND _nbl_nsc_ide_sources ${_nbl_nsc_manifest_files})
    endif()
  endif()

  add_library(${_nbl_nsc_ide_target} INTERFACE ${_nbl_nsc_ide_sources})

  source_group(TREE "${_nbl_nsc_package_root}" PREFIX "package" FILES ${_nbl_nsc_package_files})
  if(DEFINED _nbl_nsc_manifest_root AND _nbl_nsc_manifest_files)
    source_group(TREE "${_nbl_nsc_manifest_root}" PREFIX "manifests" FILES ${_nbl_nsc_manifest_files})
  endif()
endfunction()
