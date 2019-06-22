
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

update_git_submodule(../media/cegui_alfisko)


add_compile_definitions(CEGUI_STATIC CEGUI_BUILD_STATIC_FACTORY_MODULE)

set(BRDF_EXPLORER_LIBS
	${IRR_EXT_CEGUI_LIB}
	${CEGUI_LIBS}
)


set(EXAMPLE_SOURCES
	BRDFExplorerApp.cpp
	irrTex2ceguiTex.cpp
	getTextureGLname.cpp
)

irr_create_executable_project(
	"${EXAMPLE_SOURCES}"
	""
	"${IRR_EXT_CEGUI_INCLUDE_DIRS}"
	"${BRDF_EXPLORER_LIBS}"
)
