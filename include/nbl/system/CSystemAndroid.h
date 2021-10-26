#ifndef _NBL_SYSTEM_C_SYSTEM_ANDROID_H_INCLUDED_
#define _NBL_SYSTEM_C_SYSTEM_ANDROID_H_INCLUDED_
#ifdef _NBL_PLATFORM_ANDROID_
#include "nbl/system/ISystem.h"
#include "nbl/system/CAPKResourcesArchive.h"
#include <android_native_app_glue.h>

namespace nbl::system
{
	class CSystemAndroid final : public ISystem
	{
		ANativeActivity* nativeActivity = nullptr;
		core::smart_refctd_ptr<IFileArchive> androidAssetArchive = nullptr;
	public:
		CSystemAndroid(core::smart_refctd_ptr<ISystemCaller>&& caller, ANativeActivity* activity, const system::path& APKResourcesPath) :
			ISystem(std::move(caller)), nativeActivity(activity)
		{
			auto archive = core::make_smart_refctd_ptr<CAPKResourcesArchive>(
				core::smart_refctd_ptr<ISystem>(this),
				nullptr, 
				activity->assetManager
			);
			m_cachedArchiveFiles.insert(APKResourcesPath, std::move(archive));

		}
	};
}
#endif
#endif