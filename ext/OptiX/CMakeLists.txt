include(../../cmake/common.cmake)


set(IRR_EXT_OPTIX_H
	OptiXManager.h
)

set(IRR_EXT_OPTIX_SRC
	OptiXManager.cpp
)

set(IRR_EXT_OPTIX_EXTERNAL_INCLUDE
	"${IRR_ROOT_PATH}/3rdparty"
	"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
	"${OPTIX_INCLUDE_DIR}"
)

irr_create_ext_library_project(
	OPTIX
	"${IRR_EXT_OPTIX_H}"
	"${IRR_EXT_OPTIX_SRC}"
	"${IRR_EXT_OPTIX_EXTERNAL_INCLUDE}"
	""
)