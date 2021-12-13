#ifndef	_NBL_SYSTEM_C_APK_RESOURCES_ARCHIVE_LOADER_H_INCLUDED_
#define	_NBL_SYSTEM_C_APK_RESOURCES_ARCHIVE_LOADER_H_INCLUDED_
#ifdef _NBL_PLATFORM_ANDROID_
#include "nbl/system/IFileArchive.h"
#include <jni.h>

namespace nbl::system
{
class CAPKResourcesArchive : public IFileArchive
{
	using base_t = IFileArchive;
	AAssetManager* mgr;
	JNIEnv* env;
	ANativeActivity* activity;
public:
	CAPKResourcesArchive(core::smart_refctd_ptr<ISystem>&& system, system::logger_opt_smart_ptr&& logger, AAssetManager* _mgr, ANativeActivity* act, JNIEnv* jni) :
		base_t(nullptr, std::move(system), std::move(logger)), mgr(_mgr), activity(act), env(jni)
	{
		auto assets = listAssets("");
		uint32_t index = 0;
		for (auto& a : assets)
		{
			addItem(a, index, 0, EAT_VIRTUAL_ALLOC);
			index++;
		}
		setFlagsVectorSize(m_files.size());
	}
	core::smart_refctd_ptr<IFile> readFile_impl(const SOpenFileParams& params) override
	{
		auto filename = params.filename;
		AAsset* asset = AAssetManager_open(mgr, filename.c_str(), AASSET_MODE_BUFFER);
		if (asset == nullptr) return nullptr;
		const void* buffer = AAsset_getBuffer(asset);
		size_t assetSize = AAsset_getLength(asset);
		auto fileView = make_smart_refctd_ptr <CFileView<VirtualAllocator>>(core::smart_refctd_ptr(m_system), params.absolutePath, IFile::ECF_READ, assetSize);

		fileView->write_impl(buffer, 0, assetSize);
		AAsset_close(asset);
		return fileView;
	}
	std::vector<std::string> listAssets(const char* asset_path)
	{
		std::vector<std::string> result;

		auto context_object = activity->clazz;
		auto getAssets_method = env->GetMethodID(env->GetObjectClass(context_object), "getAssets", "()Landroid/content/res/AssetManager;");
		auto assetManager_object = env->CallObjectMethod(context_object, getAssets_method);
		auto list_method = env->GetMethodID(env->GetObjectClass(assetManager_object), "list", "(Ljava/lang/String;)[Ljava/lang/String;");

		jstring path_object = env->NewStringUTF(asset_path);

		auto files_object = (jobjectArray)env->CallObjectMethod(assetManager_object, list_method, path_object);

		env->DeleteLocalRef(path_object);

		auto length = env->GetArrayLength(files_object);

		for (int i = 0; i < length; i++)
		{
			jstring jstr = (jstring)env->GetObjectArrayElement(files_object, i);

			const char* filename = env->GetStringUTFChars(jstr, nullptr);

			if (filename != nullptr)
			{
				if (std::string(asset_path) = "")
				{
					result.push_back(filename);
				}
				else
				{
					result.push_back((std::filesystem::path(asset_path) / filename).generic_string());
				}
				env->ReleaseStringUTFChars(jstr, filename);
			}

			env->DeleteLocalRef(jstr);
		}

		return result;
}

};
}


#endif
#endif