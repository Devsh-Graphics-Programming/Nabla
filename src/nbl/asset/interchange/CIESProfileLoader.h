// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef __NBL_ASSET_C_IES_PROFILE_LOADER_H_INCLUDED__
#define __NBL_ASSET_C_IES_PROFILE_LOADER_H_INCLUDED__

#include "nbl/asset/ICPUImage.h"
#include "nbl/asset/ICPUShader.h"

#include "nbl/asset/IAssetManager.h"

#include "nbl/asset/interchange/IAssetLoader.h"
#include "nbl/asset/utils/CIESProfileParser.h"
#include "nbl/asset/metadata/CIESProfileMetadata.h"

namespace nbl::asset 
{
class CIESProfileLoader final : public asset::IAssetLoader 
{
	public:
		//! Check if the file might be loaded by this class
		/**
			Check might look into the file.
			\param file File handle to check.
			\return True if file seems to be loadable. 
		*/

		bool isALoadableFileFormat(io::IReadFile *_file) const override 
		{
			const size_t begginingOfFile = _file->getPos();
			_file->seek(0ull);
			std::string versionBuffer(0x45, ' ');
			_file->read(versionBuffer.data(), versionBuffer.size());
			_file->seek(begginingOfFile);

			for (const auto& it : CIESProfileParser::VALID_SIGNATURES)
				if (versionBuffer.find(it.data()) != std::string::npos)
					return true;
			
			return false;
		}

		//! Returns an array of string literals terminated by nullptr
		const char **getAssociatedFileExtensions() const override 
		{
			static const char *extensions[]{"ies", nullptr};
			return extensions;
		}

		//! Returns the assets loaded by the loader
		/** 
			Bits of the returned value correspond to each IAsset::E_TYPE
			enumeration member, and the return value cannot be 0. 
		*/
		uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_IMAGE_VIEW; }

		//! Loads an asset from an opened file, returns nullptr in case of failure.
		asset::SAssetBundle
		loadAsset(io::IReadFile* _file,
		const asset::IAssetLoader::SAssetLoadParams& _params,
		asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr,
		uint32_t _hierarchyLevel = 0u) override;
};
} // namespace nbl::asset
#endif // __NBL_ASSET_C_IES_PROFILE_LOADER_H_INCLUDED__
