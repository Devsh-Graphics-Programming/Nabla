if(NBL_BUILD_IMGUI)

	include(common RESULT_VARIABLE RES)
	if(NOT RES)
		message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
	endif()

	list(APPEND libs 
		"imgui"
		"${NBL_EXT_IMGUI_UI_LIB}"
	)

	# I'll ask about why it is not working about me adding imGui.cpp manually.
	nbl_create_executable_project(
		""
		"" 
		"${NBL_EXT_IMGUI_UI_INCLUDE_DIRS}" 
		"${libs}"
	)

endif()