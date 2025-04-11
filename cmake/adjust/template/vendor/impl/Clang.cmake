include("${CMAKE_CURRENT_LIST_DIR}/reset.cmake")

# vendor template with options fitting for both C and CXX LANGs

if(NOT DEFINED LANG)
	message(FATAL_ERROR "LANG must be defined!")
endif()

if(NBL_REQUEST_SSE_4_2)
	NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} OPTIONS
		-msse4.2 # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang1-msse4.2
	)
endif()

if(NBL_REQUEST_SSE_AVX2)
	NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG ${LANG} OPTIONS
		-mavx2 # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mavx2
	)
endif()

list(APPEND NBL_${LANG}_COMPILE_OPTIONS
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

if(NBL_SANITIZE_ADDRESS)
	list(APPEND NBL_${LANG}_COMPILE_OPTIONS -fsanitize=address)
endif()

if(NBL_SANITIZE_THREAD)
	list(APPEND NBL_${LANG}_COMPILE_OPTIONS -fsanitize=thread)
endif()

set(NBL_${LANG}_DEBUG_COMPILE_OPTIONS
	-g # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-g
	-mincremental-linker-compatible # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mincremental-linker-compatible
	-fincremental-extensions # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fincremental-extensions
	-Wall # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-W-warning
	-fstack-protector-strong # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fstack-protector-strong
	-gline-tables-only # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-gline-tables-only
	-fno-omit-frame-pointer # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fomit-frame-pointer
	-fno-inline-functions # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-finline-functions
)

set(NBL_${LANG}_RELEASE_COMPILE_OPTIONS
	-O2 # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-O-arg
	-finline-functions # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-finline-functions
	-mno-incremental-linker-compatible # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mincremental-linker-compatible
)

set(NBL_${LANG}_RELWITHDEBINFO_COMPILE_OPTIONS 
	-g # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-g
	-O1 # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-O-arg
	-finline-functions # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-finline-functions
	-mno-incremental-linker-compatible # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mincremental-linker-compatible
	-fno-omit-frame-pointer # https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-fomit-frame-pointer
)