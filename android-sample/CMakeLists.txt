project(android-sample-proj)

add_library(android-sample SHARED main.cpp)

#https://medium.com/androiddevelopers/getting-started-with-c-and-android-native-activities-2213b402ffff
#TODO put native_app_glue to separate static lib

target_include_directories(android-sample PRIVATE
    ${ANDROID_NDK}/sources/android/native_app_glue)
target_link_libraries(android-sample PRIVATE
    android
    egl # spoof egl from 3rdparty dir
    GLESv1_CM
    log)