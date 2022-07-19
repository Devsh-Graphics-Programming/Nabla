include(${NBL_ROOT_PATH}/cmake/common.cmake)

set(NBL_EXT_INTERNAL_INCLUDE_DIR "${NBL_ROOT_PATH}/include/nbl/ext/MitsubaLoader")


set(NBL_EXT_MITSUBA_LOADER_H
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/PropertyElement.h
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/IElement.h
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/CMitsubaMetadata.h
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/CElementIntegrator.h
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/CElementSensor.h
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/CElementFilm.h
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/CElementRFilter.h
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/CElementSampler.h
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/CElementTransform.h
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/CElementShape.h
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/CElementBSDF.h
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/CElementTexture.h
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/CElementEmitter.h
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/CElementFactory.h
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/CMitsubaSerializedMetadata.h
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/ParserUtil.h
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/CSerializedLoader.h
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/CMitsubaLoader.h
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/CMitsubaMaterialCompilerFrontend.h
)

set(NBL_EXT_MITSUBA_LOADER_SRC
	PropertyElement.cpp
	CElementIntegrator.cpp
	CElementSensor.cpp
	CElementFilm.cpp
	CElementRFilter.cpp
	CElementSampler.cpp
	CElementTransform.cpp
	CElementShape.cpp
	CElementBSDF.cpp
	CElementTexture.cpp
	CElementEmitter.cpp
	CElementFactory.cpp
	ParserUtil.cpp
	CSerializedLoader.cpp
	CMitsubaLoader.cpp
	CMitsubaMaterialCompilerFrontend.cpp
)

set(NBL_EXT_MITSUBA_LOADER_EXTERNAL_INCLUDE
	"${NBL_ROOT_PATH}/3rdparty"
	"${NBL_ROOT_PATH}/3rdparty/libexpat"
)

nbl_create_ext_library_project(
	MITSUBA_LOADER
	"${NBL_EXT_MITSUBA_LOADER_H}"
	"${NBL_EXT_MITSUBA_LOADER_SRC}"
	"${NBL_EXT_MITSUBA_LOADER_EXTERNAL_INCLUDE}"
	""
    ""
)

add_compile_definitions(XML_STATIC)

add_dependencies(NblExtMITSUBA_LOADER ${MITSUBA_LOADER_DEPENDENTS})
target_link_libraries(NblExtMITSUBA_LOADER INTERFACE ${MITSUBA_LOADER_DEPENDENT_LIBS})