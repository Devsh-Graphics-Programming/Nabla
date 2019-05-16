set(IRR_CEGUI_EXT_H
  CEGUI.h
	CEGUIOpenGLState.h
	CEGUIOpenGLClip.h
	Helpers.h
)

set(IRR_CEGUI_EXT_SRC
  CEGUI.cpp
	CEGUIOpenGLState.cpp
	CEGUIOpenGLClip.cpp
	Helpers.cpp
)

set(IRR_CEGUI_EXT_EXTERNAL_INCLUDE
  "${CEGUI_INCLUDE_PATH}"
  "${CMAKE_SOURCE_DIR}/3rdparty/CEGUI/cegui/dependencies/include"
)

irr_create_library_project(
  IrrCEGUIExt
  "${IRR_CEGUI_EXT_SRC}"
  "-DCEGUI_STATIC -DCEGUI_BUILD_STATIC_FACTORY_MODULE"
  "${IRR_CEGUI_EXT_EXTERNAL_INCLUDE}"
  "${CEGUI_LIBS}"
)

install(
  FILES ${IRR_CEGUI_EXT_H}
  DESTINATION ./include/irr/ext/CEGUI
  CONFIGURATIONS Release
)
install(
  FILES ${IRR_CEGUI_EXT_H}
  DESTINATION ./debug/include/irr/ext/CEGUI
  CONFIGURATIONS Debug
)

set(IRR_CEGUI_EXT_INCLUDE_DIRS
	"${CMAKE_INSTALL_PREFIX}/include/"
  "${CMAKE_INSTALL_PREFIX}/debug/include/"
	"${CMAKE_INSTALL_PREFIX}/include/irr/ext/GUI"
	"${CMAKE_INSTALL_PREFIX}/include/irr/ext/GUI/include"
  "${CMAKE_INSTALL_PREFIX}/debug/include/irr/ext/GUI"
  "${CMAKE_INSTALL_PREFIX}/debug/include/irr/ext/GUI/include"
)
set(IRR_CEGUI_EXT_INCLUDE_DIRS ${IRR_CEGUI_EXT_INCLUDE_DIRS} PARENT_SCOPE)
