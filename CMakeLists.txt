cmake_minimum_required(VERSION 3.1)
project(Irrlicht)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11, but is it really needed?

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

SET(CMAKE_CONFIGURATION_TYPES "Debug;Release")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type: Debug or Release" FORCE)
endif()

macro(irr_adjust_flags)
	if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
		add_compile_options( # TODO: check with make VERBOSE=1 that it gets added to all target objects C++ and C
			-std=c++11 #add it to all C++ target objects but not C
			-Wall
			-Wextra
			-Wno-unused-parameter
			-fno-exceptions
			-fstrict-aliasing
			-msse3
			-mfpmath=sse
		)
		if($<CONFIG:Release>) # TODO: check with make VERBOSE=1 that it gets added to all target objects C++ and C
			add_compile_options(
				-O3
				-fexpensive-optimizations
				-fno-omit-frame-pointer
				-fstack-protector-strong
			)
		endif()
		if($<CONFIG:Debug>) # TODO: check with make VERBOSE=1 that it gets added to all target objects C++ and C
			add_compile_options(
				-fno-omit-frame-pointer
				-fstack-protector-strong
			)
		endif()
		if(FAST_MATH) # TODO: check with make VERBOSE=1 that it gets added to all target objects C++ and C
			add_compile_options(
				-ffast-math
			)
		endif()
	elseif(MSVC)
		# debug
		string(REPLACE "/MDd" "/MTd" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
		string(REPLACE "/W3" "/W0" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MP")
		string(REPLACE "/MDd" "/MTd" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
		string(REPLACE "/W3" "/W0" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
		set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MP")
		
		#release
		string(REPLACE "/GS" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}") # for some reason simply replacing /GS -> /GS- doesn't work... so it vanishes here and appears a few lines below!
		string(REPLACE "/O2" "/Ox" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
		string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /GL /MP /Gy- /Zc:wchar_t /Zi /sdl- /GF- /Ot /Qpar- /GS-")
		string(REPLACE "/GS" "/GS-" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
		string(REPLACE "/O2" "/Ox" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
		string(REPLACE "/MD" "/MT" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
		set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /GL /MP /Gy- /Zc:wchar_t /Zi /sdl- /GF- /Ot /Qpar- /GS-")
	else()
		message(WARNING "UNTESTED COMPILER DETECTED, EXPECT WRONG OPTIMIZATION FLAGS! SUBMIT ISSUE ON GITHUB https://github.com/buildaworldnet/IrrlichtBAW/issues")
	endif()
endmacro()

option(FAST_MATH "Enable fast low-precision math" ON)

option(BUILD_EXAMPLES "Enable building examples" ON)

option(BUILD_TOOLS "Enable building tools (just convert2BAW as for now)" ON)

option(NASTY_OPENSSL_WORKAROUND "Do not check this on if you actually want to build Irrlicht; just for cmake scripts testing purposes" OFF)
add_subdirectory(source/Irrlicht) # Irrlicht code

if(BUILD_TOOLS)
	add_subdirectory(tools) # convert2BAW
endif()
if(BUILD_EXAMPLES)
	add_subdirectory(examples_tests)
endif()