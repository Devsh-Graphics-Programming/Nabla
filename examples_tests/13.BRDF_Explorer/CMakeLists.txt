
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

irr_create_executable_project(
	../../ext/CEGUI/CEGUI.cpp
	"-DCEGUI_STATIC -DCEGUI_BUILD_STATIC_FACTORY_MODULE"
	${CEGUI_INCLUDE_PATH}
	"${CEGUI_LIBS}"
)
