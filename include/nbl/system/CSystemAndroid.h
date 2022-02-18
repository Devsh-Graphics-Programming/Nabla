#ifndef _NBL_SYSTEM_C_SYSTEM_ANDROID_H_INCLUDED_
#define _NBL_SYSTEM_C_SYSTEM_ANDROID_H_INCLUDED_
#ifdef _NBL_PLATFORM_ANDROID_
#include "nbl/system/ISystem.h"
#include "nbl/system/CAPKResourcesArchive.h"
#include <android_native_app_glue.h>
#include <jni.h>
#include <sys/system_properties.h>
#include <sys/ioctl.h>
#include <linux/fb.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
namespace nbl::system
{
	class CSystemAndroid final : public ISystem
	{
		ANativeActivity* nativeActivity = nullptr;
		core::smart_refctd_ptr<IFileArchive> androidAssetArchive = nullptr;
		JNIEnv* env;
	public:
		CSystemAndroid(core::smart_refctd_ptr<ISystemCaller>&& caller, ANativeActivity* activity, JNIEnv* jni, const system::path& APKResourcesPath) :
			ISystem(std::move(caller)), nativeActivity(activity), env(jni)
		{
			auto archive = core::make_smart_refctd_ptr<CAPKResourcesArchive>(
				core::smart_refctd_ptr<ISystem>(this),
				nullptr, 
				activity->assetManager,
				nativeActivity,
				env
			);
			m_cachedArchiveFiles.insert(APKResourcesPath, std::move(archive));

		}
		SystemInfo getSystemInfo() const override
		{
			SystemInfo info;
			char sdk_ver_str[92];
			__system_property_get("ro.build.version.sdk", sdk_ver_str);
			info.OSFullName = std::string("Android ") + sdk_ver_str;
			// TODO: hardcoded
			info.cpuFrequency = 1100;

			struct fb_var_screeninfo fb_var;
			int fd = open("/dev/graphics/fb0", O_RDONLY);
			ioctl(fd, FBIOGET_VSCREENINFO, &fb_var);
			close(fd);
			info.desktopResX = fb_var.width;
			info.desktopResY = fb_var.height;
			return info;
		}
	};
}
#endif
#endif