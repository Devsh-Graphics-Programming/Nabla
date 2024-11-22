include_guard(GLOBAL)

# https://learn.microsoft.com/en-us/cpp/build/reference/arch-x64?view=msvc-170

# The default instruction set is SSE2 if no /arch option is specified.
if(NBL_REQUEST_SSE_4_2)
	NBL_REQUEST_COMPILE_OPTION_SUPPORT("/arch:SSE4.2")
endif()

# Enables Intel Advanced Vector Extensions 2.
if(NBL_REQUEST_SSE_AXV2)
	NBL_REQUEST_COMPILE_OPTION_SUPPORT("/arch:AVX2")
endif()

# Debug
set(NBL_C_DEBUG_COMPILE_OPTIONS
	/Ob0 /Od /MP${_NBL_JOBS_AMOUNT_} /fp:fast /Zc:wchar_t /INCREMENTAL
)

if(NBL_SANITIZE_ADDRESS)
	list(APPEND NBL_C_DEBUG_COMPILE_OPTIONS /RTC1)
endif()

# Release
set(NBL_C_RELEASE_COMPILE_OPTIONS
	/O2 /Ob2 /DNDEBUG /GL /MP${_NBL_JOBS_AMOUNT_} /Gy- /Zc:wchar_t /sdl- /GF /GS- /fp:fast
)

# RelWithDebInfo
set(NBL_C_RELWITHDEBINFO_COMPILE_OPTIONS
	/O2 /Ob1 /DNDEBUG /GL /Zc:wchar_t /MP${_NBL_JOBS_AMOUNT_} /Gy /sdl- /Oy- /fp:fast
)

if(NBL_SANITIZE_ADDRESS)
	list(APPEND NBL_C_COMPILE_OPTIONS /fsanitize=address)
endif()

# this should also be not part of profile, pasting from old flags-set function temporary
# TODO: use profile

#reason for INCREMENTAL:NO: https://docs.microsoft.com/en-us/cpp/build/reference/ltcg-link-time-code-generation?view=vs-2019 /LTCG is not valid for use with /INCREMENTAL.
set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} /INCREMENTAL:NO /LTCG:incremental")