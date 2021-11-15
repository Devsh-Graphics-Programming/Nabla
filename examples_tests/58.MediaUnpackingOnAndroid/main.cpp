#include <string>
#include <filesystem>

#include <jni.h>
#include <android_native_app_glue.h>
#include <android/sensor.h>
#include <android/log.h>
using namespace std::filesystem;

JNIEnv* env = nullptr;
ANativeActivity* activity;
static std::vector<std::string> listAssets(const char* asset_path)
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
			result.push_back(filename);
			env->ReleaseStringUTFChars(jstr, filename);
		}

		env->DeleteLocalRef(jstr);
	}

	return result;
}

bool copyFileToDest(const path& src, const path& dest, AAssetManager* mgr)
{
	AAsset* asset = AAssetManager_open(mgr, src.string().c_str(), AASSET_MODE_BUFFER);
	off_t fileSize = AAsset_getLength(asset);
	const void* data = AAsset_getBuffer(asset);

	FILE* file = fopen(dest.string().c_str(), "w");
	if (file != nullptr)
	{
		fwrite(data, 1, fileSize, file);
		fclose(file);
	}
	return file != nullptr;;
}
bool copyDirToDest(const path& src, const path& dest, AAssetManager* mgr)
{	
	create_directories(dest);

	const char* filename = (const char*)NULL;
	auto assets = listAssets(src.string().c_str());
	for(auto& asset : assets)
	{
		if (std::filesystem::path(asset).extension() == "")
			copyDirToDest(src / asset, dest / path(asset).filename(), mgr);
		else
			copyFileToDest(src / asset, dest / path(asset).filename(), mgr);
	}
	return true;
}

void android_main(android_app* app)
{
	activity = app->activity;
	const std::string mediaFolder = "media";

	app->activity->vm->AttachCurrentThread(&env, nullptr);

	path sharedPath = path(app->activity->externalDataPath) / mediaFolder;
	//create_directories(sharedPath);

	auto mgr = app->activity->assetManager;
	copyDirToDest("", sharedPath, mgr);
}