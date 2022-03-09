#include "nbl/system/CAPKResourcesArchive.h"

using namespace nbl;
using namespace nbl::system;

#ifdef _NBL_PLATFORM_ANDROID_
#include <android/native_activity.h>
#include <android/asset_manager.h>

CAPKResourcesArchive::CAPKResourcesArchive(const path& _path, system::logger_opt_smart_ptr&& logger, ANativeActivity* activity, JNIEnv* jniEnv)
	: CFileArchive(path(_path),std::move(logger),computeItems(_path.string(),activity,jniEnv)), m_mgr(activity->assetManager)
{
}

core::vector<IFileArchive::SListEntry> CAPKResourcesArchive::computeItems(const std::string& asset_path, ANativeActivity* activity, JNIEnv* jniEnv)
{
	auto context_object = activity->clazz;
	auto getAssets_method = env->GetMethodID(env->GetObjectClass(context_object), "getAssets", "()Landroid/content/res/AssetManager;");
	auto assetManager_object = env->CallObjectMethod(context_object,getAssets_method);
	auto list_method = env->GetMethodID(env->GetObjectClass(assetManager_object), "list", "(Ljava/lang/String;)[Ljava/lang/String;");

	jstring path_object = env->NewStringUTF(asset_path.c_str());

	auto files_object = (jobjectArray)env->CallObjectMethod(assetManager_object, list_method, path_object);

	env->DeleteLocalRef(path_object);

	auto length = env->GetArrayLength(files_object);
	
	core::vector<IFileArchive::SListEntry> result;
	for (decltype(length) i=0; i<length; i++)
	{
		jstring jstr = (jstring)env->GetObjectArrayElement(files_object,i);

		const char* filename = env->GetStringUTFChars(jstr,nullptr);
		if (filename != nullptr)
		{
			auto& item = result.emplace_back();
			item.pathRelativeToArchive = filename;
			{
				AAsset* asset = AAssetManager_open(activity->assetManager,filename,AASSET_MODE_STREAMING);
				item.size = AAsset_getLength(asset);
				AAsset_close(asset);
			}
			item.offset = 0xdeadbeefu;
			item.ID = i;
			item.allocatorType = EAT_APK_ALLOCATOR;
				
			env->ReleaseStringUTFChars(jstr,filename);
		}

		env->DeleteLocalRef(jstr);
	}
}

CFileArchive::file_buffer_t CAPKResourcesArchive::getFileBuffer(const IFileArchive::SListEntry* item)
{
	AAsset* asset = AAssetManager_open(m_mgr,item->pathRelativeToArchive.string().c_str(),AASSET_MODE_BUFFER);
	return {const_cast<void*>(AAsset_getBuffer(asset)),static_cast<size_t>(AAsset_getLength(asset)),asset};
}

#endif