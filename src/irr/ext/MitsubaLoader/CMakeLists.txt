include(../../../../cmake/common.cmake)


set(IRR_EXT_MITSUBA_LOADER_H
	../../../../include/irr/ext/MitsubaLoader/PropertyElement.h
	../../../../include/irr/ext/MitsubaLoader/IElement.h
	../../../../include/irr/ext/MitsubaLoader/CMitsubaMetadata.h
	../../../../include/irr/ext/MitsubaLoader/CElementIntegrator.h
	../../../../include/irr/ext/MitsubaLoader/CElementSensor.h
	../../../../include/irr/ext/MitsubaLoader/CElementFilm.h
	../../../../include/irr/ext/MitsubaLoader/CElementRFilter.h
	../../../../include/irr/ext/MitsubaLoader/CElementSampler.h
	../../../../include/irr/ext/MitsubaLoader/CElementTransform.h
	../../../../include/irr/ext/MitsubaLoader/CElementShape.h
	../../../../include/irr/ext/MitsubaLoader/CElementBSDF.h
	../../../../include/irr/ext/MitsubaLoader/CElementTexture.h
	../../../../include/irr/ext/MitsubaLoader/CElementEmitter.h
	../../../../include/irr/ext/MitsubaLoader/CElementFactory.h
	../../../../include/irr/ext/MitsubaLoader/CGlobalMitsubaMetadata.h
	../../../../include/irr/ext/MitsubaLoader/ParserUtil.h
	../../../../include/irr/ext/MitsubaLoader/CSerializedLoader.h
	../../../../include/irr/ext/MitsubaLoader/CMitsubaLoader.h
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