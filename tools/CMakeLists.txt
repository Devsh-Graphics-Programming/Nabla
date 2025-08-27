add_subdirectory(nsc)
add_subdirectory(xxHash256)

if(NBL_BUILD_IMGUI)
	add_subdirectory(nite)
endif()

NBL_ADJUST_FOLDERS(tools)