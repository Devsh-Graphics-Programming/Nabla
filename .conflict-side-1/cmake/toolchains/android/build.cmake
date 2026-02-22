# Copyright (c) 2022 DevSH Graphics Programming Sp. z O.O.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(DEFINED CMAKE_TOOLCHAIN_FILE)
	if(EXISTS "${CMAKE_TOOLCHAIN_FILE}")
		cmake_path(GET CMAKE_TOOLCHAIN_FILE FILENAME _NBL_CMAKE_TOOLCHAIN_FILENAME_)
	else()
		message(FATAL_ERROR "CMAKE_TOOLCHAIN_FILE as '${CMAKE_TOOLCHAIN_FILE}' is invalid!")
	endif()

	if(${_NBL_CMAKE_TOOLCHAIN_FILENAME_} STREQUAL "android.toolchain.cmake")
		message(STATUS "Found Android toolchain! Validating...")
		
		set(_NBL_SUPPORTED_ABI_ 
			"x86_64"
			# "arm64-v8a" in future
		)
		
		if(DEFINED ANDROID_ABI) # see _NBL_SUPPORTED_ABI_
			set(FOUND 0)
			foreach(CURRENT_ABI IN LISTS _NBL_SUPPORTED_ABI_)
				if("${CURRENT_ABI}" STREQUAL ${ANDROID_ABI})
					message(STATUS "Selecting found ${CURRENT_ABI} ABI!")
					set(FOUND 1)
					break()
				endif()
			endforeach()
			
			if(NOT FOUND)
				message(FATAL_ERROR "Selected ${ANDROID_ABI} isn't appropriate. Supported ABIs: ${_NBL_SUPPORTED_ABI_}!")
			else()
				unset(FOUND)
			endif()
		else()
			message(FATAL_ERROR "ANDROID_ABI must be specified at the very beginning of execution!")
		endif()
		
		if(NOT DEFINED ANDROID_PLATFORM) # android-28
			message(FATAL_ERROR "ANDROID_PLATFORM must be specified at the very beginning of execution!")
		endif()
	
		include(${CMAKE_TOOLCHAIN_FILE})

		if(${ANDROID_NDK_MAJOR} LESS 22) 
			message(FATAL_ERROR "Update your NDK to at least 22. We don't support ${ANDROID_NDK_REVISION} NDK version!")
		endif()
		
		# note that we assume NDK has been installed using Android Studio, so the existing paths should look as followning:
		# ${ANDROID_SDK_ROOT_PATH}/ndk/${ANDROID_NDK_REVISION}/build/cmake/android.toolchain.cmake
		
		set(ANDROID_NDK_ROOT_PATH ${CMAKE_ANDROID_NDK})
		get_filename_component(ANDROID_SDK_ROOT_PATH "${ANDROID_NDK_ROOT_PATH}/../../" ABSOLUTE)
		
		if(EXISTS "${ANDROID_SDK_ROOT_PATH}/build-tools")
			message(STATUS "Building Nabla with ${ANDROID_NDK_REVISION} NDK!")
		endif()
		
		set(NBL_BUILD_ANDROID ON)
		set(_NBL_ANDROID_VALIDATED_ ON)
	endif()
	
	# TODO - another toolchains in future

endif()

if(NBL_BUILD_ANDROID)
	if(NOT DEFINED _NBL_ANDROID_VALIDATED_)
		message(FATAL_ERROR "Android toolchain hasn't been validated properly. Pass an appropriate path to CMAKE_TOOLCHAIN_FILE!")
	endif()
	
	if("${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Linux")
		message(STATUS "Using Linux as a host OS for Android cross-compiling!")
	elseif("${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Windows")
		message(STATUS "Using Windows as a host OS for Android cross-compiling!")
	else()
		message(FATAL_ERROR "Current host OS for Android cross-compiling isn't allowed!")
	endif()
	
	find_package(Java 1.8)
	
	if(DEFINED Java_JAVA_EXECUTABLE)
		message(STATUS "Found Java executable!: ${Java_JAVA_EXECUTABLE}")
	else()
		message(FATAL_ERROR "Could not find java executable for Android build!")
	endif()
	
	string(LENGTH ${Java_JAVA_EXECUTABLE} Java_JAVA_EXECUTABLE_LENGTH)
	if("${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Windows") 
		set(Java_JAVA_EXE_BIN_LENGTH 13)
	else() 
		set(Java_JAVA_EXE_BIN_LENGTH 9)
	endif()
	math(EXPR JAVA_HOME_LENGTH "${Java_JAVA_EXECUTABLE_LENGTH} - ${Java_JAVA_EXE_BIN_LENGTH}" OUTPUT_FORMAT DECIMAL)
	string(SUBSTRING ${Java_JAVA_EXECUTABLE} 0 ${JAVA_HOME_LENGTH} JAVA_HOME)

    message(STATUS "Using JAVA_HOME = ${JAVA_HOME}")

    execute_process(COMMAND ${JAVA_HOME}/bin/java -version
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _output
        ERROR_VARIABLE _output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE)
    
    if(NOT _result AND _output MATCHES "version \"([0-9]+).([0-9]+)")
        message(STATUS "Java in JAVA_HOME is ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
    else()
        message(STATUS "Java in JAVA_HOME is unknown version ${_output} ${_result}")
    endif()

    #option(STRIP_ANDROID_LIBRARY "Strip the resulting android library" OFF)

	string(SUBSTRING
		"${ANDROID_PLATFORM}"
		8  # length of "android-"
		-1 # take remainder
		ANDROID_API_LEVEL
	)
    
    # default to libc++_static as the other options can cause crashes
    if(NOT ANDROID_STL)
        set(ANDROID_STL "c++_static")
    endif()

    # Choose clang if the NDK has both gcc and clang, since gcc sometimes fails
	# iirc api levels > 21 have only clang available
    set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION "clang")

	#
	# android-specific utils and build tools
	#
    set(ANDROID_BUILD_TOOLS_VERSION "" CACHE STRING "Version of Android build-tools to use instead of the default")
	# take most recent version of build tools, if not set explicitely
    if(ANDROID_BUILD_TOOLS_VERSION STREQUAL "")
        file(GLOB __buildTools RELATIVE "${ANDROID_SDK_ROOT_PATH}/build-tools" "${ANDROID_SDK_ROOT_PATH}/build-tools/*")
        list(SORT __buildTools)

        list(GET __buildTools -1 ANDROID_BUILD_TOOLS_VERSION)

        unset(__buildTools)
    endif()
	set(ANDROID_BUILD_TOOLS "${ANDROID_SDK_ROOT_PATH}/build-tools/${ANDROID_BUILD_TOOLS_VERSION}")
	
	set(ANDROID_JAVA_BIN "${JAVA_HOME}/bin")
	set(ANDROID_JAVA_RT_JAR "${JAVA_HOME}/jre/lib/rt.jar")

	set(ANDROID_APK_TARGET_ID "" CACHE STRING "The Target ID to build the APK for like 'android-99', use <android list targets> to choose another one.")
    if(ANDROID_APK_TARGET_ID STREQUAL "")
        # This seems different from the platform we're targetting,
        # default to the latest available that's greater or equal to our target platform
        file(GLOB __platforms RELATIVE "${ANDROID_SDK_ROOT_PATH}/platforms" "${ANDROID_SDK_ROOT_PATH}/platforms/*")
        list(SORT __platforms)

        # In case we don't find one, target the latest platform
        list(GET __platforms -1 ANDROID_APK_TARGET_ID)

        string(REPLACE "android-" "" __targetPlat "${ANDROID_PLATFORM}")

		# TODO we might want to adjust min version in the future
        # We require at least android 23 for Activity.requestPermissions
        if(__targetPlat LESS 23)
            set(__targetPlat 23)
        endif()

        foreach( __plat ${__platforms})
            string(REPLACE "android-" "" __curPlat "${__plat}")

            if(NOT (__curPlat LESS __targetPlat) )
                set(ANDROID_APK_TARGET_ID "android-${__curPlat}")
                break()
            endif()
        endforeach()

        unset(__platforms)
        unset(__targetPlat)
        unset(__curPlat)
    endif()

    message(STATUS "Using android.jar from platform ${ANDROID_APK_TARGET_ID}")
	set(ANDROID_JAR "${ANDROID_SDK_ROOT_PATH}/platforms/${ANDROID_APK_TARGET_ID}/android.jar")
endif()