include_guard(GLOBAL)

function(nbl_adjust_definitions)
	add_compile_definitions(
		PNG_THREAD_UNSAFE_OK
		PNG_NO_MMX_CODE
		PNG_NO_MNG_FEATURES
		_7ZIP_ST
		SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
		BOOST_ALL_NO_LIB
	)
	
	if(MSVC)
		add_compile_definitions(
			_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR
		)
	endif()
	
	if(ANDROID)
		add_compile_definitions(
			NBL_ANDROID_TOOLCHAIN
		)
	endif()

	if(WIN32)
		add_compile_definitions(
			WIN32
			__GNUWIN32__
			_CRT_SECURE_NO_DEPRECATE
			NOMINMAX
		)
	endif()
endfunction()