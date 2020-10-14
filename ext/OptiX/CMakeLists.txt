include(../../cmake/common.cmake)


set(NBL_EXT_OPTIX_H
	SbtRecord.h
	IModule.h
	IContext.h
	IDenoiser.h
	Manager.h
)

set(NBL_EXT_OPTIX_SRC
	Manager.cpp
	IContext.cpp
)

set(NBL_EXT_OPTIX_EXTERNAL_INCLUDE
	"${NBL_ROOT_PATH}/3rdparty"
	"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
	"${OPTIX_INCLUDE_DIR}"
)

irr_create_ext_library_project(
	OPTIX
	"${NBL_EXT_OPTIX_H}"
	"${NBL_EXT_OPTIX_SRC}"
	"${NBL_EXT_OPTIX_EXTERNAL_INCLUDE}"
	""
)