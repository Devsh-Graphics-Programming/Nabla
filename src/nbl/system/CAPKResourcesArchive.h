#ifndef	_NBL_SYSTEM_C_APK_RESOURCES_ARCHIVE_LOADER_H_INCLUDED_
#define	_NBL_SYSTEM_C_APK_RESOURCES_ARCHIVE_LOADER_H_INCLUDED_


#include "nbl/system/IFileArchive.h"


#ifdef _NBL_PLATFORM_ANDROID_
struct AAssetManager;
struct JNIEnv;
struct ANativeActivity;

namespace nbl::system
{

class CAPKResourcesArchive final : public CFileArchive
{
	public:
		static core::smart_refctd_ptr<CAPKResourcesArchive> create();

	protected:
		CAPKResourcesArchive(
			path&& _path, system::logger_opt_smart_ptr&& logger,
			AAssetManager* _mgr, ANativeActivity* act, JNIEnv* jni
		) : CFileArchive(std::move(_path),std::move(logger),std::move(_items)),
			m_mgr(_mgr), m_activity(act), m_jniEnv(jni)
		{}
		
		std::pair<void*, size_t> getFileBuffer(const IFileArchive::SListEntry* item) override;
		

		AAssetManager* m_mgr;
		JNIEnv* m_jniEnv;
		ANativeActivity* m_activity;
};

}
#endif

#endif