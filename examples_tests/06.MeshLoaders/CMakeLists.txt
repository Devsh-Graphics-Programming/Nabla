include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

nbl_create_executable_project("" "" "" "")

if(NBL_BUILD_ANDROID)
	target_link_options(meshloaders PRIVATE -u ANativeActivity_onCreate)
	nbl_android_create_apk(meshloaders)
endif()