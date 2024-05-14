// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_C_HLSL_LOADER_H_INCLUDED_
#define _NBL_ASSET_C_HLSL_LOADER_H_INCLUDED_

#include <algorithm>

#include "nbl/asset/interchange/IAssetLoader.h"
#include <nbl/system/ISystem.h>

namespace nbl::asset
{

//!  Surface Loader for PNG files
class CHLSLLoader final : public asset::IAssetLoader
{
	public:
		CHLSLLoader() = default;

		bool isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger = nullptr) const override
		{
			return true;
		}

		const char** getAssociatedFileExtensions() const override
		{
			static const char* ext[]{ "hlsl", nullptr };
			return ext;
		}

		uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_SHADER; }

		asset::SAssetBundle loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;
};

} // namespace nbl::asset

#endif

