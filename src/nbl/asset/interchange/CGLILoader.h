// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_IMAGE_LOADER_GLI__
#define __NBL_ASSET_C_IMAGE_LOADER_GLI__

#include "BuildConfigOptions.h"

#ifdef _NBL_COMPILE_WITH_GLI_LOADER_

#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/interchange/IAssetLoader.h"

namespace nbl
{
namespace asset
{

//! Texture loader capable of loading in .ktx, .dds and .kmg file extensions
class CGLILoader final : public asset::IAssetLoader
{
	protected:
		virtual ~CGLILoader() {}

	public:
		explicit CGLILoader() = default;

		bool isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const override;

		const char** getAssociatedFileExtensions() const override
		{
			static const char* extensions[]{ "ktx", "dds", "kmg", nullptr };
			return extensions;
		}

		uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_IMAGE_VIEW; }

		asset::SAssetBundle loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

	private:

		static inline bool doesItHaveFaces(const IImageView<ICPUImage>::E_TYPE& type)
		{
			switch (type)
			{
				case ICPUImageView::ET_CUBE_MAP: return true;
				case ICPUImageView::ET_CUBE_MAP_ARRAY: return true;
				default: return false;
			}
		}
		static inline bool doesItHaveLayers(const IImageView<ICPUImage>::E_TYPE& type)
		{
			switch (type)
			{
				case ICPUImageView::ET_1D_ARRAY: return true;
				case ICPUImageView::ET_2D_ARRAY: return true;
				case ICPUImageView::ET_CUBE_MAP_ARRAY: return true;
				default: return false;
			}
		}		
};

}
}

#endif // _NBL_COMPILE_WITH_GLI_LOADER_
#endif
