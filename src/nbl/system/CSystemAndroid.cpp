#include "nbl/system/CSystemAndroid.h"

using namespace nbl::system;

#ifdef _NBL_PLATFORM_ANDROID_
#include "nbl/system/CAPKResourcesArchive.h"

#include <android_native_app_glue.h>
#include <sys/system_properties.h>
#include <sys/ioctl.h>
#include <linux/fb.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

CSystemAndroid::CSystemAndroid(ANativeActivity* activity, JNIEnv* jni, const path& APKResourcesPath) :
	ISystemPOSIX(), m_nativeActivity(activity), m_jniEnv(jni)
{
	m_cachedArchiveFiles.insert(APKResourcesPath,core::make_smart_refctd_ptr<CAPKResourcesArchive>(
		path(APKResourcesPath),
		nullptr, // for now no logger
		m_nativeActivity,
		m_jniEnv
	));
}

ISystem::SystemInfo CSystemAndroid::getSystemInfo() const
{
	SystemInfo info;
	// TODO: hardcoded
	info.cpuFrequency = 1100;
	info.totalMemory = 4ull<<30ull;
	info.availableMemory = 2ull<<30ull;

	struct fb_var_screeninfo fb_var;
	int fd = open("/dev/graphics/fb0", O_RDONLY);
	ioctl(fd, FBIOGET_VSCREENINFO, &fb_var);
	close(fd);
	info.desktopResX = fb_var.width;
	info.desktopResY = fb_var.height;

	char sdk_ver_str[92];
	__system_property_get("ro.build.version.sdk", sdk_ver_str);
	info.OSFullName = std::string("Android ") + sdk_ver_str;

	return info;
}

#endif