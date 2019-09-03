include(../../cmake/common.cmake)
#update_git_submodule(./stb)


set(IRR_EXT_MITSUBA_LOADER_H
	test.h
)

set(IRR_EXT_MITSUBA_LOADER_SRC
	test.cpp
)

set(IRR_EXT_CEGUI_EXTERNAL_INCLUDE
  "${MITSUBA_LOADER_INCLUDE_PATH}"
)

irr_create_ext_library_project(
  MITSUBA_LOADER
  "${IRR_EXT_MITSUBA_LOADER_H}"
  "${IRR_EXT_MITSUBA_LOADER_SRC}"
  "${IRR_EXT_MITSUBA_LOADER_EXTERNAL_INCLUDE}"
  ""
)
