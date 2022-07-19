// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_C_SPIR_V_LOADER_H_INCLUDED_
#define _NBL_ASSET_C_SPIR_V_LOADER_H_INCLUDED_

#include "nbl/asset/interchange/IAssetLoader.h"

namespace nbl::asset
{

class CSPVLoader final : public asset::IAssetLoader
{
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t SPV_MAGIC_NUMBER = 0x07230203u;
	public:
		CSPVLoader() = default;
		inline bool isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr) const override
		{
			uint32_t magicNumber = 0u;

			system::IFile::success_t success;
			_file->read(success, &magicNumber, 0, sizeof magicNumber);
			return success && magicNumber==SPV_MAGIC_NUMBER;
		}

		const char** getAssociatedFileExtensions() const override
		{
			static const char* ext[]{ "spv", nullptr };
			return ext;
		}

		inline uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_SHADER; }

		asset::SAssetBundle loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;
};

} // namespace nbl::asset

#endif

