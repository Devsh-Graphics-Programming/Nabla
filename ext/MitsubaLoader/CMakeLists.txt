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
	#CElementEmitter.h
	CElementFactory.h
	#CShapeCreator.h
	#Shape.h
	CGlobalMitsubaMetadata.h
	ParserUtil.h
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
	#CElementEmitter.cpp
	CElementFactory.cpp
	CMitsubaLoader.cpp
	#CShapeCreator.cpp
	ParserUtil.cpp
	#Shape.cpp
)

set(IRR_EXT_MITSUBA_LOADER_EXTERNAL_INCLUDE
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