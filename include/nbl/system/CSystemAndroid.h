#ifndef _NBL_SYSTEM_C_SYSTEM_ANDROID_H_INCLUDED_
#define _NBL_SYSTEM_C_SYSTEM_ANDROID_H_INCLUDED_


#include "nbl/system/ISystem.h"


namespace nbl::system
{
#ifdef _NBL_PLATFORM_ANDROID_
#include "nbl/system/ISystemPOSIX.h"

struct ANativeActivity;
struct JNIEnv;

class CSystemAndroid final : public ISystemPOSIX
{
		ANativeActivity* m_nativeActivity;
		JNIEnv* m_jniEnv;
	public:
		CSystemAndroid(ANativeActivity* activity, JNIEnv* jni, const system::path& APKResourcesPath);

		//
		SystemInfo getSystemInfo() const override;
};

#endif
}

#endif