
get_filename_component(EXECUTABLE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REGEX REPLACE "[0-9]+\." "" EXECUTABLE_NAME ${EXECUTABLE_NAME})

add_executable(${EXECUTABLE_NAME} main.cpp ../../ext/Blur/CBlurPerformer.cpp)
add_dependencies(${EXECUTABLE_NAME} Irrlicht)

target_include_directories(${EXECUTABLE_NAME} PUBLIC ../../include)
target_link_libraries(${EXECUTABLE_NAME} Irrlicht)
set_target_properties(${EXECUTABLE_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "bin"
)
if(MSVC)
	set_target_properties(convert2BAW
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY_DEBUG "bin"
		RUNTIME_OUTPUT_DIRECTORY_RELEASE "bin"
	)
endif()