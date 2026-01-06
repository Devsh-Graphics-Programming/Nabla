include(${NBL_ROOT_PATH}/cmake/common.cmake)

set(NBL_EXT_INTERNAL_INCLUDE_DIR "${NBL_ROOT_PATH}/include")

set(NBL_EXT_TEXT_RENDERING_H
	# extra headers goes there
	# eg. ${NBL_EXT_INTERNAL_INCLUDE_DIR}/something.hpp
)

set(NBL_EXT_TEXT_RENDERING_SRC
	TextRendering.cpp
)

set(NBL_EXT_TEXT_RENDERING_EXTERNAL_INCLUDE
	"${NBL_ROOT_PATH}/3rdparty"
	"${NBL_EXT_INTERNAL_INCLUDE_DIR}"
)

nbl_create_ext_library_project(
	TEXT_RENDERING
	"${NBL_EXT_TEXT_RENDERING_H}"
	"${NBL_EXT_TEXT_RENDERING_SRC}"
	"${NBL_EXT_TEXT_RENDERING_EXTERNAL_INCLUDE}"
	""
    ""
)

# Freetype
add_dependencies(${LIB_NAME} freetype)
if(NBL_STATIC_BUILD)
	target_link_libraries(${LIB_NAME} INTERFACE freetype)
else()
	target_link_libraries(${LIB_NAME} PUBLIC freetype)
endif()
target_include_directories(${LIB_NAME} PUBLIC $<TARGET_PROPERTY:freetype,INCLUDE_DIRECTORIES>)

# msdfgen
add_dependencies(${LIB_NAME} ${NBL_MSDFGEN_TARGETS})
if(NBL_STATIC_BUILD)
	target_link_libraries(${LIB_NAME} INTERFACE ${NBL_MSDFGEN_TARGETS})
else()
	target_link_libraries(${LIB_NAME} PUBLIC ${NBL_MSDFGEN_TARGETS})
endif()

foreach(NBL_TARGET IN LISTS NBL_MSDFGEN_TARGETS)
	target_include_directories(${LIB_NAME} PUBLIC $<TARGET_PROPERTY:${NBL_TARGET},INCLUDE_DIRECTORIES>)
endforeach()

# TODO: once the extension is used + some shaders then we can make use of NSC to precompile to
# SPIRV as builtin resources library linked to the extension like IMGUI ext does, then the 
# extension would benefit from constexpr pointers to precompiled SPIRV shaders

set(NBL_EXT_TEXT_RENDERING_TARGET ${LIB_NAME} CACHE INTERNAL "Nabla's Text Rendering logical target name")