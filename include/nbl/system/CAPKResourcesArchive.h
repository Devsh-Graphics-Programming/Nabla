#ifndef	_NBL_SYSTEM_C_APK_RESOURCES_ARCHIVE_LOADER_H_INCLUDED_
#define	_NBL_SYSTEM_C_APK_RESOURCES_ARCHIVE_LOADER_H_INCLUDED_
#ifdef _NBL_PLATFORM_ANDROID_
#include "nbl/system/IArchiveLoader.h"

namespace nbl::system
{
class CAPKResourcesArchive : public IFileArchive
{
	using base_t = IFileArchive;
	AAssetManager* mgr;

public:
	CAPKResourcesArchive(core::smart_refctd_ptr<IFile>&& file, core::smart_refctd_ptr<ISystem>&& system, system::logger_opt_smart_ptr&& logger, AAssetManager* _mgr) :
		base_t(std::move(file), std::move(system), std::move(logger)), mgr(_mgr)
	{

	}
	core::smart_refctd_ptr<IFile> readFile_impl(const SOpenFileParams& params) override
	{
		auto filename = params.filename;
		AAsset* asset = AAssetManager_open(mgr, filename.c_str(), AASSET_MODE_BUFFER);
		if (asset == nullptr) return nullptr;
		const void* buffer = AAsset_getBuffer(asset);
		size_t assetSize = AAsset_getLength(asset);
		auto fileView = make_smart_refctd_ptr < CFileView<CNullAllocator>(core::smart_refctd_ptr(system), params.fullName, IFile::ECF_READ, const_cast<void*>(buffer), assetSize);
		return fileView;
	}
};
}


#endif
#endif