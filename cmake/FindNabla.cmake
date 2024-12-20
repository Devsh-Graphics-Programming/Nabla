# Define NBL_CONFIG_ROOT_DIRECTORY
# variable to help the module find 
# Nabla package

if(NOT DEFINED CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_CONFIGURATION_TYPES Release;RelWithDebInfo;Debug)
endif()

if(NOT DEFINED NBL_PACKAGE_STATIC) # turn ON NBL_PACKAGE_STATIC to look for package with STATIC library type, turn off to look for DYNAMIC
	if(${NBL_STATIC_BUILD}) # internal, if called with Nabla's build system it will get detected autoamtically
		set(NBL_PACKAGE_STATIC ON)
	else()
		message(FATAL_ERROR "NBL_PACKAGE_STATIC must be defined!")
	endif()
endif()

if(NBL_PACKAGE_STATIC)
	set(NBL_LIBRARY_TYPE static)
else()
	set(NBL_LIBRARY_TYPE dynamic)
endif()

foreach(X IN LISTS CMAKE_CONFIGURATION_TYPES)
	if(NOT "${X}" STREQUAL "")
		string(TOLOWER "nabla-${NBL_LIBRARY_TYPE}-${X}" _NBL_TARGET_PACKAGE_)
	
		if(DEFINED NBL_CONFIG_ROOT_DIRECTORY)
			file(GLOB_RECURSE _NBL_G_CONFIG_ROOT_DIRECTORY_ "${NBL_CONFIG_ROOT_DIRECTORY}/*/${_NBL_TARGET_PACKAGE_}Config.cmake")
			cmake_path(GET _NBL_G_CONFIG_ROOT_DIRECTORY_ PARENT_PATH _NBL_G_CONFIG_ROOT_DIRECTORY_)
		else()
			unset(_NBL_G_CONFIG_ROOT_DIRECTORY_)
		endif()
	
		find_package(${_NBL_TARGET_PACKAGE_} QUIET
			GLOBAL
			PATHS ${_NBL_G_CONFIG_ROOT_DIRECTORY_}
		)
	endif()
endforeach() 