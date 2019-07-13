include(../../cmake/common.cmake)
update_git_submodule(./stb)


set(IRR_EXT_CEGUI_H
  ExtCEGUI.h
	CEGUIOpenGLState.h
	CEGUIOpenGLClip.h
	Helpers.h
)

set(IRR_EXT_CEGUI_SRC
  ExtCEGUI.cpp
	CEGUIOpenGLState.cpp
	CEGUIOpenGLClip.cpp
	Helpers.cpp
)

set(IRR_EXT_CEGUI_EXTERNAL_INCLUDE
  "${CEGUI_INCLUDE_PATH}"
  "${CMAKE_SOURCE_DIR}/3rdparty"
  "${CMAKE_BINARY_DIR}/3rdparty/CEGUI/dependencies/install/include"
)

add_compile_definitions(CEGUI_STATIC CEGUI_BUILD_STATIC_FACTORY_MODULE)

irr_create_ext_library_project(
  CEGUI
  "${IRR_EXT_CEGUI_H}"
  "${IRR_EXT_CEGUI_SRC}"
  "${IRR_EXT_CEGUI_EXTERNAL_INCLUDE}"
  ""
)

add_dependencies(IrrExtCEGUI CEGUIBase-0_Static)
add_dependencies(IrrExtCEGUI CEGUICommonDialogs-0_Static)
add_dependencies(IrrExtCEGUI CEGUICoreWindowRendererSet_Static)
add_dependencies(IrrExtCEGUI CEGUIExpatParser_Static)
add_dependencies(IrrExtCEGUI CEGUIOpenGLRenderer-0_Static)
add_dependencies(IrrExtCEGUI CEGUISTBImageCodec_Static)
