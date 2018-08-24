
add_executable(convert2BAW main.cpp print.cpp)
add_dependencies(convert2BAW Irrlicht)

target_include_directories(convert2BAW PUBLIC ../../include)
target_link_libraries(convert2BAW Irrlicht)
set_target_properties(convert2BAW
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