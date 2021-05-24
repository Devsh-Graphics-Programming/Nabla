export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
export ANDROID_NDK="/home/criss/Android/Sdk/android-ndk-r22b"
export ANDROID_SDK="/home/criss/Android/Sdk"

[ -d "/media/criss/usbdrive/" ] && mkdir -p /media/criss/usbdrive/build-android

cmake -DCMAKE_TOOLCHAIN_FILE="/home/criss/Android/Sdk/android-ndk-r22b/build/cmake/android.toolchain.cmake" -DCMAKE_BUILD_TYPE=Debug -DNBL_BUILD_ANDROID=On -DANDROID_ABI=x86_64 -DNBL_BUILD_EXAMPLES=Off -DNBL_BUILD_DOCS=Off -DNBL_BUILD_MITSUBA_LOADER=Off -DNBL_COMPILE_WITH_SDL2=Off -D_NBL_COMPILE_WITH_OPEN_EXR_=Off -S . -B /media/criss/usbdrive/build-android
cd /media/criss/usbdrive/build-android && make -j 12 VERBOSE=1
cd ./android-sample && make -j 12 VERBOSE=1
