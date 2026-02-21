# Android build

## Supported toolsets

- **[Clang NDK](https://developer.android.com/ndk/guides/other_build_systems)**

## **Additional required dependencies:**

- **[Android Studio](https://developer.android.com/studio)**
- **[JDK 8](https://www.java.com/download/)**

## Build modes

### Vanilla (DEPRICATED NOTES)

Most extensions disabled.

The first step is to install Android Studio and JDK 8. When done, open Android Studio and navigate to **Tools** -> **SDK Manager** -> **System Settings** -> **Android SDK**.

Select *SDK Platforms* and install proper individual SDK components - install Android version with Android API level you will be targeting. Then switch to *SDK Tools* and make sure to install **Android SDK Build-Tools 32** and **NDK (Side by side)** - it's a *requirement*! Also you must make sure that your **JAVA_HOME** enviroment variable is set to proper JDK installation path.

## CMake (DEPRICATED NOTES)

We use **Ninja** generator tools as a generator for building Nabla for Android on both Windows and Linux Host-OSes. *Note that Nabla Android build has been performed and tested so far on Windows as cross compile OS with **Ninja** generator and on Linux as cross compile OS with **Makefile** and **Ninja** generators, but we recommend using **Ninja** for both OSes.* 

Before configuring CMake you must add 2 cache variables:

- **ANDROID_PLATFORM**
- **ANDROID_ABI**

**ANDROID_PLATFORM** is a target API platform that you pass as `android-x` where `x` is your android API level (you can pass 28 for instance). **ANDROID_ABI** is Application Binary Interface and note, that we support only `x86_64` currently. Those 2 cache variables *must be* specified before CMake configuration. Having done it you can specify toolchain file for cross-compiling by passing path to `android.toolchain.cmake`. You can find it in Android Studio's SDK directory in `ndk/<version>/build/cmake/android.toolchain.cmake`. Basically the entire path should look like this one `C:/Users/<your_user>/AppData/Local/AndroidSdk/ndk/<version>/build/cmake/android.toolchain.cmake`. With all of this feel free to generate.

Having Nabla generated you need to enter build directory, launch the terminal and type `cmake --build . --target Nabla -j4 -v` or if you want build android sample example you would type `cmake --build . --target android_sample_apk -j4 -v`. The android sample example produces *.apk* file you can use for debugging and profiling.

**Note:** each example provided by the engine builds as an executable with non-cross builds and with target of a name called `a_target`, in following example above it would be `android_sample`. When building cross-compile for android **to produce the APK file you need to add `_apk` postfix to the `a_target`, because `a_target` gets built then as a library.

#### DEPRECATED: Chrome Book SDK version

In order for the chromebook to work with the apk you build you need to install the right SDK version. Go to **Tools** -> **SDK Manager** -> **System Settings** -> **Android SDK** then select the *SDK Platforms* tab and tick the "Show Packake Details" checkbox in the bottom-right corner. After that select *Android 9.0 (Pie) -> Android SDK Platform 28* and hit "OK".

#### DEPRECATED: Chrome Book upload

To upload generated *.apk* into your ChromeBook you need first to make sure your device is in *developer mode* state. If it is, you can open Android Studio and choose Debug or Profile choosing *.apk* file. Then you will need to connect to your device using **adb** connector. To make use of adb, you need to find path to the executable that is placed in `C:/Users/<your_user>/AppData/Local/AndroidSdk/platform-tools` directory. When found, you can type in Android Studio command line `C:/Users/<your_user>/AppData/Local/AndroidSdk/platform-tools/adb connect <IP of ChromeBook network>`. You can find ChromeBook's IP by entering network settings and choosing current network ChromeBook is connected to. This way the ChromeBook should be listed in available devices and you should be able to upload *.apk* to the machine through debugging app shortcut. Take into account that you won't probably be able to debug in that scenario, but you will be able to upload *.apk* to the device.

#### DEPRECATED: Chrome Book debug

To debug the *.apk* on your chromebook you need to open the source file you want to debug in Android Studio (Either via *File->Open* or Drag&Drop, but be aware that d&d can deadlock your Android Studio 25% of the time so youll need to restart it), then place your breakpoints and hit "Debug" (The bug icon)  in the top right corner.
