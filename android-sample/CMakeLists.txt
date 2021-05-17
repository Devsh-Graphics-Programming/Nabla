project(android-sample-proj)

include(common)

add_library(native_app_glue STATIC
    ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)

add_library(android-sample SHARED main.cpp)

#https://medium.com/androiddevelopers/getting-started-with-c-and-android-native-activities-2213b402ffff
#TODO put native_app_glue to separate static lib

target_link_options(android-sample PRIVATE -u ANativeActivity_onCreate)
target_include_directories(android-sample PRIVATE
    ${ANDROID_NDK}/sources/android/native_app_glue)
target_link_libraries(android-sample PRIVATE
    android
    native_app_glue
    egl # spoof egl from 3rdparty dir
    GLESv1_CM # change to GLESv3 later
    log)

nbl_android_create_apk(android-sample 1 1)