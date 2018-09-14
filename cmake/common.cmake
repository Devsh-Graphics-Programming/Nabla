
# Macro creating project for an executable
# Project and target get its name from directory when this macro gets executed (truncating number in the beginning of the name and making all lower case)
# Created because of common cmake code for examples and tools
macro(irr_create_executable_project _EXTRA_SOURCES _EXTRA_OPTIONS)
	get_filename_component(EXECUTABLE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
	string(REGEX REPLACE "^[0-9]+\." "" EXECUTABLE_NAME ${EXECUTABLE_NAME})
	string(TOLOWER ${EXECUTABLE_NAME} EXECUTABLE_NAME)

	project(${EXECUTABLE_NAME})

	add_executable(${EXECUTABLE_NAME} main.cpp ${_EXTRA_SOURCES}) 
	# EXTRA_SOURCES is var containing non-common names of sources (if any such sources, then EXTRA_SOURCES must be set before including this cmake code)
	add_dependencies(${EXECUTABLE_NAME} Irrlicht)

	target_include_directories(${EXECUTABLE_NAME} PUBLIC ../../include)
	target_link_libraries(${EXECUTABLE_NAME} Irrlicht)
	add_compile_options(${_EXTRA_OPTIONS})
	
	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
		add_compile_options(-fsanitize=address -fstack-protector-all)
	
		set(COMMON_LINKER_OPTIONS "-msse3 -mfpmath=sse -fuse-ld=gold")
		set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${COMMON_LINKER_OPTIONS}")
		set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${COMMON_LINKER_OPTIONS} -fstack-protector-strong -fsanitize=address")
	endif()

	irr_adjust_flags() # macro defined in root CMakeLists
	irr_adjust_definitions() # macro defined in root CMakeLists

	set_target_properties(${EXECUTABLE_NAME} PROPERTIES DEBUG_POSTFIX _d)
	set_target_properties(${EXECUTABLE_NAME}
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "bin"
	)
	if(MSVC)
		set_target_properties(${EXECUTABLE_NAME}
			PROPERTIES
			RUNTIME_OUTPUT_DIRECTORY_DEBUG "bin"
			RUNTIME_OUTPUT_DIRECTORY_RELEASE "bin"
		)
	endif()
endmacro()