#ifdef _NBL_PLATFORM_ANDROID_
#include <sys/param.h>
#include "nbl/ui/CGraphicalApplicationAndroid.h"

namespace nbl::ui
{
system::path CGraphicalApplicationAndroid::getSharedResourcesPath(JNIEnv* env)
{
    // Get File object for the external storage directory.
    jclass classEnvironment = env->FindClass("android/os/Environment");
    jmethodID methodIDgetExternalStorageDirectory = env->GetStaticMethodID(classEnvironment, "getExternalStorageDirectory", "()Ljava/io/File;");  // public static File getExternalStorageDirectory ()
    jobject objectFile = env->CallStaticObjectMethod(classEnvironment, methodIDgetExternalStorageDirectory);

    // Call method on File object to retrieve String object.
    jclass classFile = env->GetObjectClass(objectFile);
    jmethodID methodIDgetAbsolutePath = env->GetMethodID(classFile, "getAbsolutePath", "()Ljava/lang/String;");
    jstring stringPath = (jstring)env->CallObjectMethod(objectFile, methodIDgetAbsolutePath);

    // Extract a C string from the String object, and chdir() to it.
    const char* sharedPath = env->GetStringUTFChars(stringPath, NULL);
    return system::path(sharedPath);
}
}

#endif