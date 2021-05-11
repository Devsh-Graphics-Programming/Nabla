export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
export ANDROID_NDK="/home/criss/Android/Sdk/android-ndk-r22b"
export ANDROID_SDK="/home/criss/Android/Sdk"

cmake -DCMAKE_TOOLCHAIN_FILE="/home/criss/Android/Sdk/android-ndk-r22b/build/cmake/android.toolchain.cmake" -DNBL_BUILD_ANDROID=On -DANDROID_ABI=x86_64 -DNBL_BUILD_EXAMPLES=Off -DNBL_BUILD_DOCS=Off -DNBL_BUILD_MITSUBA_LOADER=Off -DNBL_COMPILE_WITH_SDL2=Off -D_NBL_COMPILE_WITH_OPEN_EXR_=Off -S . -B ./build-android
cd ./build-android && make VERBOSE=1
cd ./android-sample && make VERBOSE=1