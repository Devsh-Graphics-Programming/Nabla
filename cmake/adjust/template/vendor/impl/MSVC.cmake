include("${CMAKE_CURRENT_LIST_DIR}/reset.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/frontend/MSVC.cmake")

# vendor template with options fitting for both C and CXX LANGs

if(NBL_REQUEST_SSE_4_2)
	NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} COMPILE_OPTIONS
		/arch:SSE4.2 # https://learn.microsoft.com/en-us/cpp/build/reference/arch-x64?view=msvc-170
) # TODO: (****) should be (?) optional but then adjust 3rdparty options on fail
endif()