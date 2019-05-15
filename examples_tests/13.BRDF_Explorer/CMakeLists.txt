
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

set(BRDF_EXPLORER_SOURCES
	"../../ext/CEGUI/CEGUI.cpp"
	"../../ext/CEGUI/CEGUIOpenGLState.cpp"
	"../../ext/CEGUI/CEGUIOpenGLClip.cpp"
	"../../ext/CEGUI/Helpers.cpp"
	"./BRDFExplorerApp.cpp"
)

irr_create_executable_project(
	"${BRDF_EXPLORER_SOURCES}"
	"-DCEGUI_STATIC -DCEGUI_BUILD_STATIC_FACTORY_MODULE"
	${CEGUI_INCLUDE_PATH}
	"${CEGUI_LIBS}"
)
