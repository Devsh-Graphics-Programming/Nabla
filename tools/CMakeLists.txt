add_subdirectory(nsc)
add_subdirectory(xxHash256)

if(NBL_BUILD_IMGUI)
	add_subdirectory(nite EXCLUDE_FROM_ALL)
endif()

NBL_ADJUST_FOLDERS(tools)