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
		AAssetManager* assetManager = nullptr;
		core::smart_refctd_ptr<IFileArchive> androidAssetArchive = nullptr;
	public:
		CSystemAndroid(core::smart_refctd_ptr<ISystemCaller>&& caller, ANativeActivity* activity) :
			ISystem(std::move(caller)), nativeActivity(activity), assetManager(activity->assetManager)
		{
			androidAssetArchive = make_smart_refctd_ptr<CAPKResourcesArchive>(
				core::smart_refctd_ptr<ISystem>(this),
				nullptr, 
				activity->assetManager
			);
		}
		
		core::smart_refctd_ptr<IFile> openFileOpt_impl(const system::path& filename, core::bitflag<IFile::E_CREATE_FLAGS> flags) override
		{
			IFileArchive::SOpenFileParams params{ filename, filename, "" };
			auto asset = androidAssetArchive->readFile(params);
			return asset;
		}
	};
}
#endif
#endif