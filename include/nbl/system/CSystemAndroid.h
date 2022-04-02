#ifndef _NBL_SYSTEM_C_SYSTEM_ANDROID_H_INCLUDED_
#define _NBL_SYSTEM_C_SYSTEM_ANDROID_H_INCLUDED_


#include "nbl/system/ISystem.h"

#ifdef _NBL_PLATFORM_ANDROID_
#include "nbl/system/ISystemPOSIX.h"

#include <jni.h>


struct ANativeActivity;

namespace nbl::system
{

class CSystemAndroid final : public ISystemPOSIX
{
	public:
		CSystemAndroid(ANativeActivity* activity, JNIEnv* jni, const path& APKResourcesPath);

		//
		SystemInfo getSystemInfo() const override;

	protected:
		ANativeActivity* m_nativeActivity;
		JNIEnv* m_jniEnv;
};

}
#endif

#endif