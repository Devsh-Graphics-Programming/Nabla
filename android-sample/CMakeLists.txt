include(common)

nbl_create_executable_project("" "" "" "")

# https://github.com/android-ndk/ndk/issues/381
target_link_options(android_sample PRIVATE -u ANativeActivity_onCreate)

nbl_android_create_apk(android_sample)
