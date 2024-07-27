# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

# TODO HAVE .CMAKE WITH ALL OPTIONS REGARDING BUILD

option(_NBL_COMPILE_WITH_MTL_LOADER_ "Compile with MTL Loader" OFF) #default off until Material Compiler 2
option(_NBL_COMPILE_WITH_OBJ_LOADER_ "Compile with OBJ Loader" OFF) #default off until Material Compiler 2
#option(_NBL_COMPILE_WITH_OBJ_WRITER_ "Compile with OBJ Writer" ON) uncomment when writer exists
option(_NBL_COMPILE_WITH_STL_LOADER_ "Compile with STL Loader" OFF) #default off until Material Compiler 2
option(_NBL_COMPILE_WITH_STL_WRITER_ "Compile with STL Writer" ON)
option(_NBL_COMPILE_WITH_PLY_LOADER_ "Compile with PLY Loader" OFF) #default off until Material Compiler 2
option(_NBL_COMPILE_WITH_PLY_WRITER_ "Compile with PLY Writer" ON)
option(_NBL_COMPILE_WITH_JPG_LOADER_ "Compile with JPG Loader" ON)
option(_NBL_COMPILE_WITH_JPG_WRITER_ "Compile with JPG Writer" ON)
option(_NBL_COMPILE_WITH_PNG_LOADER_ "Compile with PNG Loader" ON)
option(_NBL_COMPILE_WITH_PNG_WRITER_ "Compile with PNG Writer" ON)
option(_NBL_COMPILE_WITH_TGA_LOADER_ "Compile with TGA Loader" ON)
option(_NBL_COMPILE_WITH_TGA_WRITER_ "Compile with TGA Writer" ON)
option(_NBL_COMPILE_WITH_OPENEXR_LOADER_ "Compile with OpenEXR Loader" ON)
option(_NBL_COMPILE_WITH_OPENEXR_WRITER_ "Compile with OpenEXR Writer" ON)
option(_NBL_COMPILE_WITH_GLI_LOADER_ "Compile with GLI Loader" ON)
option(_NBL_COMPILE_WITH_GLI_WRITER_ "Compile with GLI Writer" ON)
option(_NBL_COMPILE_WITH_GLTF_LOADER_ "Compile with GLTF Loader" OFF) # TMP OFF COMPILE ERRORS ON V143 ON MASTER
option(_NBL_COMPILE_WITH_GLTF_WRITER_ "Compile with GLTF Writer" OFF) # TMP OFF COMPILE ERRORS ON V143 ON MASTER
set(_NBL_EG_PRFNT_LEVEL 0 CACHE STRING "EasterEgg Profanity Level")

option(NBL_EXPLICIT_MODULE_LOAD_LOG "Enable Runtime logs for external dynamic module loading" OFF)
option(NBL_CPACK_NO_BUILD_DIRECTORY_MODULES "Disable searching for modules in developer's build directory in install mode" OFF)

add_subdirectory(device EXCLUDE_FROM_ALL)

# TODO: ALL BUILD CONFIG OPTIONS GO HERE OR BEFORE ! WITH INCLUDE
set(__NBL_FAST_MATH ${NBL_FAST_MATH})

nbl_generate_conf_files()

file(GLOB_RECURSE NABLA_HEADERS_PUBLIC "${NBL_ROOT_PATH}/include/*.h")
file(GLOB_RECURSE NABLA_HEADERS_PUBLIC "${NBL_ROOT_PATH}/include/*.tcc")
file(GLOB_RECURSE NABLA_HEADERS_PRIVATE "${NBL_ROOT_PATH}/src/nbl/*.h")

file(GLOB_RECURSE NBL_CORE_SOURCES "${NBL_ROOT_PATH}/src/nbl/core/*.cpp")
file(GLOB_RECURSE NBL_SYSTEM_SOURCES "${NBL_ROOT_PATH}/src/nbl/system/*.cpp")
file(GLOB_RECURSE NBL_UI_SOURCES "${NBL_ROOT_PATH}/src/nbl/ui/*.cpp")
file(GLOB_RECURSE NBL_ASSET_SOURCES "${NBL_ROOT_PATH}/src/nbl/asset/*.cpp")
file(GLOB_RECURSE NBL_VIDEO_SOURCES "${NBL_ROOT_PATH}/src/nbl/video/*.cpp")
file(GLOB_RECURSE NBL_SCENE_SOURCES "${NBL_ROOT_PATH}/src/nbl/scene/*.cpp")

# TODO: TO SEPARATE DLLS FOR EACH NAMESPACE

set(NBL_SOURCES_COMMON
	${NBL_CORE_SOURCES}
	${NBL_SYSTEM_SOURCES}
	${NBL_UI_SOURCES}
	${NBL_ASSET_SOURCES}
	${NBL_VIDEO_SOURCES}
	${NBL_SCENE_SOURCES}
)

if(MSVC)
	list(APPEND NBL_SOURCES_COMMON "${PROJECT_SOURCE_DIR}/tools/debug/VisualStudio/DynamicArrayVisualizer.natvis")
endif()

set(NBL_LIBRARY_CREATION_SOURCES 
	${NBL_SOURCES_COMMON}
	${NABLA_HEADERS_PUBLIC}
	${NABLA_HEADERS_PRIVATE}
)

if(NBL_STATIC_BUILD)
	add_library(Nabla STATIC
		${NBL_LIBRARY_CREATION_SOURCES}
	)
else()
	add_library(Nabla SHARED
		${NBL_LIBRARY_CREATION_SOURCES}
	)
endif()

nbl_adjust_flags(TARGET Nabla MAP_RELEASE Release MAP_RELWITHDEBINFO RelWithDebInfo MAP_DEBUG Debug)
nbl_adjust_definitions()
nbl_handle_runtime_lib_properties(Nabla)

target_include_directories(Nabla 
	PUBLIC "${NBL_ROOT_PATH}/include"
	PUBLIC $<TARGET_PROPERTY:config,INCLUDE_DIRECTORIES>
)

target_compile_definitions(Nabla 
	PRIVATE __NBL_BUILDING_NABLA__
	PRIVATE SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
	INTERFACE _NBL_PCH_IGNORE_PRIVATE_HEADERS
)

if(NBL_EXPLICIT_MODULE_LOAD_LOG)
	target_compile_definitions(Nabla PUBLIC NBL_EXPLICIT_MODULE_LOAD_LOG)
endif()

if(NBL_CPACK_NO_BUILD_DIRECTORY_MODULES)
	target_compile_definitions(Nabla PUBLIC NBL_CPACK_NO_BUILD_DIRECTORY_MODULES)
endif()

if(WIN32)
	target_compile_definitions(Nabla PRIVATE VK_USE_PLATFORM_WIN32_KHR)
endif()

if(NOT NBL_STATIC_BUILD)
	set(NBL_ASSEMBLY_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/devshgraphicsprogramming.nabla" CACHE INTERNAL "" FORCE)	
	set_target_properties(Nabla PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${NBL_ASSEMBLY_DIRECTORY})
endif()

add_dependencies(Nabla DeviceHeaders)
target_include_directories(Nabla PRIVATE "${NBL_DEVICE_GEN_INCLUDE_DIR}") # TODO USE GEN EXPRESSION
add_subdirectory(builtin EXCLUDE_FROM_ALL)

glue_source_definitions(Nabla _NBL_SOURCE_DEFINITIONS_)
set(_NBL_DEFINE_FILE_WRAPPER_ ${CMAKE_CURRENT_BINARY_DIR}/include/define.h)
write_source_definitions("${_NBL_DEFINE_FILE_WRAPPER_}" "${_NBL_SOURCE_DEFINITIONS_}")

if (MSVC)
	target_compile_options(Nabla PUBLIC /bigobj)
endif()

if(NBL_PCH)
	target_precompile_headers(Nabla PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/pch.h")
endif()

# extensions
start_tracking_variables_for_propagation_to_parent()
add_subdirectory(ext)
propagate_changed_variables_to_parent_scope()

if(NOT NBL_STATIC_BUILD)
	add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/generated/define.h"
		COMMAND "${CMAKE_COMMAND}" -DNBL_ROOT_PATH:PATH=${NBL_ROOT_PATH} -DNBL_WRAPPER_FILE:FILEPATH=${_NBL_DEFINE_FILE_WRAPPER_} -DNBL_GEN_DIRECTORY:PATH=${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/generated -D_NABLA_DLL_NAME_:STRING=$<TARGET_FILE_NAME:Nabla> -D_DXC_DLL_NAME_:STRING=${DXC_DLL_NAME} -D_NABLA_INSTALL_DIR_:PATH="${CMAKE_INSTALL_PREFIX}" -P ${NBL_ROOT_PATH}/cmake/scripts/nbl/nablaDefines.cmake
		COMMENT "Launching defines.h generation script!"
		VERBATIM
	)
	
	add_custom_target(generate_nabla_defines ALL
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/generated/define.h"
        VERBATIM
    )
	
	add_dependencies(Nabla generate_nabla_defines)
endif()

NBL_HOOK_3RDPARTY(Nabla)

nbl_install_config_header(BuildConfigOptions.h)
nbl_install_headers("${NABLA_HEADERS_PUBLIC}")

if(NOT NBL_STATIC_BUILD)
	nbl_install_file("${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/generated/define.h")
endif()

set_target_properties(Nabla PROPERTIES DEBUG_POSTFIX _debug)
set_target_properties(Nabla PROPERTIES RELWITHDEBINFO_POSTFIX _relwithdebinfo)

nbl_install_lib_spec(Nabla "nbl")

if(NBL_STATIC_BUILD)
	nbl_install_lib("${NBL_3RDPARTY_TARGETS}") # TODO SPLIT THEN INTO NAMESPACE FOR PROGRAMS,LIBRAIES, INTERFACES & use, I want libraries here only
else()
	nbl_install_program_spec(Nabla "nbl")
endif()

nbl_install_program_spec("${DXC_DLL}" "nbl/3rdparty/dxc")