#include "nbl/system/CSystemAndroid.h"

using namespace nbl::system;

#ifdef _NBL_PLATFORM_ANDROID_
#include "nbl/system/CAPKResourcesArchive.h"

#include <android_native_app_glue.h>
#include <jni.h>
#include <sys/system_properties.h>
#include <sys/ioctl.h>
#include <linux/fb.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

CSystemAndroid::CSystemAndroid(ANativeActivity* activity, JNIEnv* jni, const system::path& APKResourcesPath) :
	ISystemPOSIX(), m_nativeActivity(activity), m_jniEnv(jni)
{
	addArchiveLoader(core::make_smart_refctd_ptr<CArchiveLoaderTar>(core::smart_refctd_ptr<ISystem>(this), nullptr));
	m_cachedArchiveFiles.insert(APKResourcesPath,core::make_smart_refctd_ptr<CAPKResourcesArchive>(
		core::smart_refctd_ptr<ISystem>(this),
		nullptr, 
		activity->assetManager,
		nativeActivity,
		env
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