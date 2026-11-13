# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

include(../cmake/common.cmake)

project(Nabla-3rdparty LANGUAGES CXX C)
enable_language(C CXX ASM ASM_NASM)
set(BUILD_SHARED_LIBS OFF)
set(SKIP_INSTALL_ALL ON)
set(BUILD_TESTING OFF)

option(NBL_FORCE_RELEASE_3RDPARTY "Force map 3rdparty's configuration regardless Nabla configuration to Release" OFF)
option(NBL_FORCE_RELWITHDEBINFO_3RDPARTY "Force map 3rdparty's configuration regardless Nabla configuration to RelWithDebInfo" OFF)

# cmake-git-version-tracking
# note that checking if a repository is dirty may cost build time (especially a lot if like us you have a lot of submodules) - by default we run with all checks but if you want to increase build time iterations I recommend to exclude this check
option(GIT_EXCLUDE_IS_DIRTY "Exclude IS_DIRTY from git tracking checks, will increase build time iterations at the expense of the meta information loss" OFF)
find_package(git-version-tracking CONFIG REQUIRED)
NBL_ADD_GIT_TRACKING_META_LIBRARY(nabla "${NBL_ROOT_PATH}")
NBL_ADD_GIT_TRACKING_META_LIBRARY(dxc "${dxc_INCLUDE_DIR}")
nbl_install_file("${CMAKE_CURRENT_BINARY_DIR}/git-version-tracking/nabla_git_info.json")
nbl_install_file("${CMAKE_CURRENT_BINARY_DIR}/git-version-tracking/dxc_git_info.json")

NBL_GENERATE_GIT_TRACKING_META()

# Final gather
set(NBL_3RDPARTY_TARGETS
				${NBL_MSDFGEN_TARGETS}
)

foreach(trgt IN LISTS NBL_3RDPARTY_TARGETS)
		set_property(TARGET ${trgt} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>$<$<BOOL:${NBL_COMPILER_DYNAMIC_RUNTIME}>:DLL>")
		
		get_target_property(NBL_TARGET_TYPE ${trgt} TYPE)
		if(NOT "${NBL_TARGET_TYPE}" STREQUAL "INTERFACE_LIBRARY")
			# maybe explicit global mapping would be better, to discuss
			if(NBL_FORCE_RELEASE_3RDPARTY) # priority over RWDI
				nbl_adjust_flags(TARGET ${trgt} MAP_RELEASE Release MAP_RELWITHDEBINFO Release MAP_DEBUG Release)
			elseif(NBL_FORCE_RELWITHDEBINFO_3RDPARTY)
				nbl_adjust_flags(TARGET ${trgt} MAP_RELEASE RelWithDebInfo MAP_RELWITHDEBINFO RelWithDebInfo MAP_DEBUG RelWithDebInfo)
			else()
				nbl_adjust_flags(TARGET ${trgt} MAP_RELEASE Release MAP_RELWITHDEBINFO RelWithDebInfo MAP_DEBUG Debug)
			endif()
			
			if(MSVC)
				target_compile_definitions(${trgt} PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR) # temporary till nbl_adjust_definitions has nbl_adjust_flags's modes
			endif()
		endif()
endforeach()
nbl_adjust_definitions()

add_custom_target(3rdparty)
add_dependencies(3rdparty ${NBL_3RDPARTY_TARGETS})

NBL_ADJUST_FOLDERS(3rdparty)

nbl_install_file(${CMAKE_CURRENT_BINARY_DIR}/git-version-tracking/git_info.h)

# parent scope exports, must be at the end of the file
set(_NBL_3RDPARTY_TARGETS_ 
	${NBL_3RDPARTY_TARGETS}
PARENT_SCOPE)

set(NBL_MSDFGEN_TARGETS
	${NBL_MSDFGEN_TARGETS}
PARENT_SCOPE)
