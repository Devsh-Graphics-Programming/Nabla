#ifndef _NBL_SYSTEM_C_SYSTEM_ANDROID_H_INCLUDED_
#define _NBL_SYSTEM_C_SYSTEM_ANDROID_H_INCLUDED_
#include "nbl/system/ISystem.h"
#ifdef _NBL_PLATFORM_ANDROID_
namespace nbl::system
{
	class CSystemAndroid : public ISystem
	{
		ANativeActivity* nativeActivity = nullptr;
		AAssetManager assetManaget = nullptr;
		core::smart_refctd_ptr<IFileArchive> androidAssetArchive = nullptr;
	public:
		CSystemAndroid(ANativeActivity* activity, core::smart_refctd_ptr<IFileArchive>&& aar) : 
			nativeActivity(activity), assetManager(activity->assetManager), androidAssetArchive(aar)
		{}

	};
}
#endif
#endif