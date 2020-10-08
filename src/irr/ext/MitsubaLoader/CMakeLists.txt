include(${IRR_ROOT_PATH}/cmake/common.cmake)

set(IRR_EXT_INTERNAL_INCLUDE_DIR "${IRR_ROOT_PATH}/include/irr/ext/MitsubaLoader")


set(IRR_EXT_MITSUBA_LOADER_H
	${IRR_EXT_INTERNAL_INCLUDE_DIR}/PropertyElement.h
	${IRR_EXT_INTERNAL_INCLUDE_DIR}/IElement.h
	${IRR_EXT_INTERNAL_INCLUDE_DIR}/CMitsubaMetadata.h
	${IRR_EXT_INTERNAL_INCLUDE_DIR}/CElementIntegrator.h
	${IRR_EXT_INTERNAL_INCLUDE_DIR}/CElementSensor.h
	${IRR_EXT_INTERNAL_INCLUDE_DIR}/CElementFilm.h
	${IRR_EXT_INTERNAL_INCLUDE_DIR}/CElementRFilter.h
	${IRR_EXT_INTERNAL_INCLUDE_DIR}/CElementSampler.h
	${IRR_EXT_INTERNAL_INCLUDE_DIR}/CElementTransform.h
	${IRR_EXT_INTERNAL_INCLUDE_DIR}/CElementShape.h
	${IRR_EXT_INTERNAL_INCLUDE_DIR}/CElementBSDF.h
	${IRR_EXT_INTERNAL_INCLUDE_DIR}/CElementTexture.h
	${IRR_EXT_INTERNAL_INCLUDE_DIR}/CElementEmitter.h
	${IRR_EXT_INTERNAL_INCLUDE_DIR}/CElementFactory.h
	${IRR_EXT_INTERNAL_INCLUDE_DIR}/CGlobalMitsubaMetadata.h
	${IRR_EXT_INTERNAL_INCLUDE_DIR}/ParserUtil.h
	${IRR_EXT_INTERNAL_INCLUDE_DIR}/CSerializedLoader.h
	${IRR_EXT_INTERNAL_INCLUDE_DIR}/CMitsubaLoader.h
	${IRR_EXT_INTERNAL_INCLUDE_DIR}/CMitsubaMaterialCompilerFrontend.h
)

set(IRR_EXT_MITSUBA_LOADER_SRC
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

set(IRR_EXT_MITSUBA_LOADER_EXTERNAL_INCLUDE
	"${IRR_ROOT_PATH}/3rdparty"
	"${IRR_ROOT_PATH}/3rdparty/libexpat"
)

irr_create_ext_library_project(
	MITSUBA_LOADER
	"${IRR_EXT_MITSUBA_LOADER_H}"
	"${IRR_EXT_MITSUBA_LOADER_SRC}"
	"${IRR_EXT_MITSUBA_LOADER_EXTERNAL_INCLUDE}"
	""
)

add_compile_definitions(XML_STATIC)

add_dependencies(IrrExtMITSUBA_LOADER ${MITSUBA_LOADER_DEPENDENTS})
target_link_libraries(IrrExtMITSUBA_LOADER INTERFACE ${MITSUBA_LOADER_DEPENDENT_LIBS})