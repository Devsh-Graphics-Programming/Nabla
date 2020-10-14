include(../../cmake/common.cmake)
update_git_submodule(./stb)


set(NBL_EXT_CEGUI_H
  ExtCEGUI.h
	CEGUIOpenGLState.h
	CEGUIOpenGLClip.h
	Helpers.h
)

set(NBL_EXT_CEGUI_SRC
  ExtCEGUI.cpp
	CEGUIOpenGLState.cpp
	CEGUIOpenGLClip.cpp
	Helpers.cpp
)

set(NBL_EXT_CEGUI_EXTERNAL_INCLUDE
  "${CEGUI_INCLUDE_PATH}"
  "${NBL_ROOT_PATH}/3rdparty"
  "${CMAKE_BINARY_DIR}/3rdparty/CEGUI/dependencies/install/include"
  "${NBL_ROOT_PATH}/src/3rdparty/"
)

add_compile_definitions(CEGUI_STATIC CEGUI_BUILD_STATIC_FACTORY_MODULE)

irr_create_ext_library_project(
  CEGUI
  "${NBL_EXT_CEGUI_H}"
  "${NBL_EXT_CEGUI_SRC}"
  "${NBL_EXT_CEGUI_EXTERNAL_INCLUDE}"
  ""
)

add_dependencies(IrrExtCEGUI CEGUIBase-0_Static)
add_dependencies(IrrExtCEGUI CEGUICommonDialogs-0_Static)
add_dependencies(IrrExtCEGUI CEGUICoreWindowRendererSet_Static)
add_dependencies(IrrExtCEGUI CEGUIExpatParser_Static)
add_dependencies(IrrExtCEGUI CEGUIOpenGLRenderer-0_Static)
add_dependencies(IrrExtCEGUI CEGUISTBImageCodec_Static)
