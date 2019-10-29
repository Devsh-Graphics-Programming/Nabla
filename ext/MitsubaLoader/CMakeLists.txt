include(../../cmake/common.cmake)


set(IRR_EXT_MITSUBA_LOADER_H
	PropertyElement.h
	IElement.h
	CElementIntegrator.h
	CElementSensor.h
	CElementFilm.h
	CElementRFilter.h
	CElementSampler.h
	CElementTransform.h
	CElementShape.h
	CElementBSDF.h
	CElementTexture.h
	CElementEmitter.h
	CElementFactory.h
	CGlobalMitsubaMetadata.h
	ParserUtil.h
	CSerializedLoader.h
	CMitsubaLoader.h
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