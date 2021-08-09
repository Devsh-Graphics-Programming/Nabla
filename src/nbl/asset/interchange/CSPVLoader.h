// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_SPIR_V_LOADER_H_INCLUDED__
#define __NBL_ASSET_C_SPIR_V_LOADER_H_INCLUDED__

#include "nbl/asset/interchange/IAssetLoader.h"

namespace nbl::asset
{

class CSPVLoader final : public asset::IAssetLoader
{
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t SPV_MAGIC_NUMBER = 0x07230203u;
	public:
		CSPVLoader() = default;
		bool isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr) const override
		{
			uint32_t magicNumber = 0u;

			
			system::future<size_t> future;
			_file->read(future, &magicNumber, 0, sizeof magicNumber);
			future.get();
			return magicNumber==SPV_MAGIC_NUMBER;
		}

		const char** getAssociatedFileExtensions() const override
		{
			static const char* ext[]{ "spv", nullptr };
			return ext;
		}

		uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_SHADER; }

		asset::SAssetBundle loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;
};

} // namespace nbl::asset

#endif

