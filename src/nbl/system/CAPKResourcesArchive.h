#ifndef	_NBL_SYSTEM_C_APK_RESOURCES_ARCHIVE_LOADER_H_INCLUDED_
#define	_NBL_SYSTEM_C_APK_RESOURCES_ARCHIVE_LOADER_H_INCLUDED_


#include "nbl/system/CFileArchive.h"

#ifdef _NBL_PLATFORM_ANDROID_
#include <jni.h>


struct AAssetManager;
struct ANativeActivity;

namespace nbl::system
{

class CAPKResourcesArchive final : public CFileArchive
{
	public:
		CAPKResourcesArchive(const path& _path, system::logger_opt_smart_ptr&& logger, ANativeActivity* act, JNIEnv* jniEnv);
		
	protected:
		static core::vector<SListEntry> computeItems(const std::string& asset_path);

		file_buffer_t getFileBuffer(const IFileArchive::SListEntry* item) override;
		

		AAssetManager* m_mgr;
		ANativeActivity* m_activity;
};

}
#endif

#endif