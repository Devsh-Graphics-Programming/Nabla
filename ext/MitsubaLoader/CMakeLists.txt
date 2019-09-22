include(../../cmake/common.cmake)


set(IRR_EXT_MITSUBA_LOADER_H
	CElementEmitter.h
	CElementFactory.h
	CElementFilm.h
	CElementSampler.h
	CElementSensor.h
	CElementTransform.h
	CMitsubaLoader.h
	CShapeCreator.h
	IElement.h
	ParserUtil.h
	PropertyElement.h
	Shape.h
)

set(IRR_EXT_MITSUBA_LOADER_SRC
	#CElementEmitter.cpp
	#CElementFactory.cpp
	#CElementFilm.cpp
	CElementSampler.cpp
	#CElementSensor.cpp
	#CElementTransform.cpp
	CMitsubaLoader.cpp
	#CShapeCreator.cpp
	ParserUtil.cpp
	PropertyElement.cpp
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