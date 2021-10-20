include(common)

nbl_create_executable_project("" "" "" "")

# https://github.com/android-ndk/ndk/issues/381
if(NBL_BUILD_ANDROID)
	target_link_options(androidsample PRIVATE -u ANativeActivity_onCreate)
	nbl_android_create_apk(androidsample)
endif()	