include_guard(GLOBAL)

# Debug
set(NBL_CXX_DEBUG_COMPILE_OPTIONS
	-ggdb3 -Wall -fno-omit-frame-pointer -fstack-protector-strong
)

# Release
set(NBL_CXX_RELEASE_COMPILE_OPTIONS
	-fexpensive-optimizations
)

# RelWithDebInfo
set(NBL_CXX_RELWITHDEBINFO_COMPILE_OPTIONS "")

# Global
list(APPEND NBL_CXX_COMPILE_OPTIONS
	-Wextra
	-fno-strict-aliasing
	-msse4.2
	-mfpmath=sse		
	-Wextra
	-Wno-sequence-point
	-Wno-unused-parameter
	-Wno-unused-but-set-parameter
	-Wno-error=ignored-attributes
	-Wno-error=unused-function
	-Wno-error=unused-variable
	-Wno-error=unused-parameter
	-Wno-error=ignored-attributes
	-Wno-error=non-pod-varargs
	-fno-exceptions
)

if(NBL_SANITIZE_ADDRESS)
	list(APPEND NBL_CXX_COMPILE_OPTIONS -fsanitize=address)
endif()

if(NBL_SANITIZE_THREAD)
	list(APPEND NBL_CXX_COMPILE_OPTIONS -fsanitize=thread)
endif()

# our pervious flags-set function called this, does not affect flags nor configs so I will keep it here temporary
# TODO: move it out from the profile
link_libraries(-fuse-ld=gold)