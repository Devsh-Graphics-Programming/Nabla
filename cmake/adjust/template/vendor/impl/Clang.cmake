include("${CMAKE_CURRENT_LIST_DIR}/reset.cmake")

# vendor template with options fitting for both C and CXX LANGs

if(NOT DEFINED LANG)
	message(FATAL_ERROR "LANG must be defined!")
endif()

if(NBL_WITH_COMPILER_CRASH_DIAGNOSTICS)
	NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} COMPILE_OPTIONS
		# use it to make a repro and attach to an issue if you Clang crashes
		# - it outputs preprocessed cpp files with sh script for compilation
		-fcrash-diagnostics=compiler
		-fcrash-diagnostics-dir=${NBL_ROOT_PATH_BINARY}/.crash-report
	)
endif()

NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} COMPILE_OPTIONS
	-Xclang=-fconstexpr-backtrace-limit=696969
	-Xclang=-fconstexpr-depth=696969
	-Xclang=-fconstexpr-steps=696969
	-Xclang=-ftemplate-backtrace-limit=0 # no limit
	-Xclang=-ftemplate-depth=696969
	-Xclang=-fmacro-backtrace-limit=0 # no limit
	-Xclang=-fspell-checking-limit=0 # no limit
	-Xclang=-fcaret-diagnostics-max-lines=0 # no limit

	# latest Clang(CL) 19.1.1 shipped with VS seems to require explicitly features to be listed (simdjson)
	# TODO: Yas, we should first do independent check if host has the flags, if the request fail then 
	# do not promote simdjson to build with HASWELL implementation because those flags + avx2 compose 
	# subset it wants in this case

	################
	# TODO: (****) ->
	-mbmi # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mbmi
	-mlzcnt # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mlzcnt
	-mpclmul # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mpclmul
	################ <-

	-Wextra # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-W-warning
	-maes # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-maes
	-mfpmath=sse # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mfpmath

	# TODO: Yas, eliminate all below
	-fno-strict-aliasing
	-Wno-sequence-point
	-Wno-c++98-compat
	-Wno-c++98-compat-pedantic
	-Wno-padded
	-Wno-unsafe-buffer-usage
	-Wno-switch-enum
	-Wno-error=ignored-attributes
	-Wno-unused-parameter
	-Wno-unused-but-set-parameter
	-Wno-error=unused-function
	-Wno-error=unused-variable
	-Wno-error=unused-parameter
	-Wno-error=ignored-attributes
	-Wno-error=non-pod-varargs
)

if(NBL_REQUEST_SSE_4_2)
	NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} COMPILE_OPTIONS
		-msse4.2 # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang1-msse4.2
) # TODO: (****) optional but then adjust 3rdparty options on fail
endif()

if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT MATCHES MSVC)
	# ClangCL with MSVC frontend (most of the options are compatible but eg /arch:SSE4.2 seems to be not)
	include("${CMAKE_CURRENT_LIST_DIR}/frontend/MSVC.cmake")
	return()
else()
	if(NBL_REQUEST_SSE_AVX2)
		NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} COMPILE_OPTIONS
			-mavx2 # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mavx2
	) # TODO: (****)
	endif()

	if(NBL_SANITIZE_ADDRESS)
		NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} COMPILE_OPTIONS -fsanitize=address)
	endif()

	if(NBL_SANITIZE_THREAD)
		NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} COMPILE_OPTIONS -fsanitize=thread)
	endif()

	NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} CONFIG DEBUG COMPILE_OPTIONS
		-g # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-g
		-mincremental-linker-compatible # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mincremental-linker-compatible
		-Wall # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-W-warning
		-gline-tables-only # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-gline-tables-only
		-Xclang=-fno-inline-functions # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-finline-functions
	)

	NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} CONFIG RELEASE COMPILE_OPTIONS
		-O2 # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-O-arg
		-Xclang=-finline-functions # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-finline-functions
		-mno-incremental-linker-compatible # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mincremental-linker-compatible
		-DNDEBUG
	)

	NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} CONFIG RELWITHDEBINFO COMPILE_OPTIONS
		-g # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-g
		-O1 # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-O-arg
		-Xclang=-finline-functions # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-finline-functions
		-mno-incremental-linker-compatible # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mincremental-linker-compatible
		-DNDEBUG
	)
endif()