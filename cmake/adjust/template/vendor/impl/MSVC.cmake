include("${CMAKE_CURRENT_LIST_DIR}/reset.cmake")

# vendor template with options fitting for both C and CXX LANGs

if(NOT DEFINED LANG)
	message(FATAL_ERROR "LANG must be defined!")
endif()

if(NBL_REQUEST_SSE_4_2)
	NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} COMPILE_OPTIONS
		/arch:SSE4.2 # https://learn.microsoft.com/en-us/cpp/build/reference/arch-x64?view=msvc-170
) # TODO: (****) should be (?) optional but then adjust 3rdparty options on fail
endif()

if(NBL_REQUEST_SSE_AVX2)
	NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} COMPILE_OPTIONS
		/arch:AVX2 # https://learn.microsoft.com/en-us/cpp/build/reference/arch-x64?view=msvc-170
) # TODO: (****) should be (?) optional but then adjust 3rdparty options on fail
endif()

NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} COMPILE_OPTIONS 
	/Zc:preprocessor # https://learn.microsoft.com/en-us/cpp/build/reference/zc-preprocessor?view=msvc-170
	/Zc:__cplusplus # https://learn.microsoft.com/en-us/cpp/build/reference/zc-cplusplus?view=msvc-170
	/Zc:wchar_t # https://learn.microsoft.com/en-us/cpp/build/reference/zc-wchar-t-wchar-t-is-native-type?view=msvc-170
	/fp:fast # https://learn.microsoft.com/en-us/cpp/build/reference/fp-specify-floating-point-behavior?view=msvc-170
	/MP${_NBL_JOBS_AMOUNT_} # https://learn.microsoft.com/en-us/cpp/build/reference/mp-build-with-multiple-processes?view=msvc-170
)

if(NBL_SANITIZE_ADDRESS)
	NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} COMPILE_OPTIONS 
		/fsanitize=address # https://learn.microsoft.com/en-us/cpp/build/reference/fsanitize?view=msvc-170
	)

	NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} CONFIG DEBUG COMPILE_OPTIONS
		/RTC1 # https://learn.microsoft.com/en-us/cpp/build/reference/rtc-run-time-error-checks?view=msvc-170
	)
endif()

NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} CONFIG DEBUG COMPILE_OPTIONS
	/Ob0 # https://learn.microsoft.com/en-us/cpp/build/reference/ob-inline-function-expansion?view=msvc-170
	/Od # https://learn.microsoft.com/en-us/cpp/build/reference/od-disable-debug?view=msvc-170
	/Oy- # https://learn.microsoft.com/en-us/cpp/build/reference/oy-frame-pointer-omission?view=msvc-170

	LINK_OPTIONS
		/INCREMENTAL # https://learn.microsoft.com/en-us/cpp/build/reference/incremental-link-incrementally?view=msvc-170
)

NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} CONFIG RELEASE COMPILE_OPTIONS
	/O2 # https://learn.microsoft.com/en-us/cpp/build/reference/o1-o2-minimize-size-maximize-speed?view=msvc-170
	/Ob2 # https://learn.microsoft.com/en-us/cpp/build/reference/ob-inline-function-expansion?view=msvc-170
	/DNDEBUG # https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/assert-macro-assert-wassert?view=msvc-170
	/GL # https://learn.microsoft.com/en-us/cpp/build/reference/gl-whole-program-optimization?view=msvc-170
	/Gy- # https://learn.microsoft.com/en-us/cpp/build/reference/gy-enable-function-level-linking?view=msvc-170
	/sdl- # https://learn.microsoft.com/en-us/cpp/build/reference/sdl-enable-additional-security-checks?view=msvc-170
	/GF # https://learn.microsoft.com/en-us/cpp/build/reference/gf-eliminate-duplicate-strings?view=msvc-170
	/GS- # https://learn.microsoft.com/en-us/cpp/build/reference/gs-buffer-security-check?view=msvc-170

	LINK_OPTIONS
		/INCREMENTAL:NO # https://learn.microsoft.com/en-us/cpp/build/reference/incremental-link-incrementally?view=msvc-170
		/LTCG # https://learn.microsoft.com/en-us/cpp/build/reference/ltcg-link-time-code-generation?view=msvc-170 (note: /GL implies fallback with LTCG)
)

NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} CONFIG RELWITHDEBINFO COMPILE_OPTIONS
	/O2 # https://learn.microsoft.com/en-us/cpp/build/reference/o1-o2-minimize-size-maximize-speed?view=msvc-170
	/Ob1 # https://learn.microsoft.com/en-us/cpp/build/reference/ob-inline-function-expansion?view=msvc-170
	/Oy- # https://learn.microsoft.com/en-us/cpp/build/reference/oy-frame-pointer-omission?view=msvc-170
	/DNDEBUG # https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/assert-macro-assert-wassert?view=msvc-170
	/GL # https://learn.microsoft.com/en-us/cpp/build/reference/gl-whole-program-optimization?view=msvc-170
	/Gy # https://learn.microsoft.com/en-us/cpp/build/reference/gy-enable-function-level-linking?view=msvc-170
	/sdl- # https://learn.microsoft.com/en-us/cpp/build/reference/sdl-enable-additional-security-checks?view=msvc-170

	LINK_OPTIONS
		/INCREMENTAL:NO # https://learn.microsoft.com/en-us/cpp/build/reference/incremental-link-incrementally?view=msvc-170 (note: cannot use /INCREMENTAL with /LTCG:incremental, would cause fallback)
		/LTCG:incremental # https://learn.microsoft.com/en-us/cpp/build/reference/ltcg-link-time-code-generation?view=msvc-170
)