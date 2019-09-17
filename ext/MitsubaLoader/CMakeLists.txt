include(../../cmake/common.cmake)


set(IRR_EXT_MITSUBA_LOADER_H
	CElementEmitter.h
	CElementFactory.h
	CElementFilm.h
	CElementSampler.h
	CElementSensor.h
	CElementTransform.h
	CMitsubaLoader.h
	CMitsubaScene.h
	CShapeCreator.h
	IElement.h
	ParserUtil.h
	PropertyElement.h
	Shape.h
)

set(IRR_EXT_MITSUBA_LOADER_SRC
	CElementEmitter.cpp
	CElementFactory.cpp
	CElementFilm.cpp
	CElementSampler.cpp
	CElementSensor.cpp
	CElementTransform.cpp
	CMitsubaLoader.cpp
	CMitsubaScene.cpp
	CShapeCreator.cpp
	ParserUtil.cpp
	PropertyElement.cpp
	Shape.cpp
)

#set(IRR_EXT_MITSUBA_LOADER_EXTERNAL_INCLUDE
#	"${IRR_ROOT_PATH}/3rdparty"
#)

irr_create_ext_library_project(
	MitsubaLoader
	"${IRR_EXT_MITSUBA_LOADER_H}"
	"${IRR_EXT_MITSUBA_LOADER_SRC}"
	""
#	"${IRR_EXT_MITSUBA_LOADER_EXTERNAL_INCLUDE}"
	""
)

#add_dependencies(IrrExtMitsubaLoader xxxxx)
