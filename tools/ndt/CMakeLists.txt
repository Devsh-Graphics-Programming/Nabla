set(NBL_EXTRA_SOURCES
	# TODO: Cypi list extra sources if you need
)

nbl_create_executable_project("${NBL_EXTRA_SOURCES}" "" "" "" "${NBL_EXECUTABLE_PROJECT_CREATION_PCH_TARGET}")