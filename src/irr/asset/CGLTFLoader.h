// Copyright (C) 2020 AnastaZIuk
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifdef _IRR_COMPILE_WITH_GLTF_LOADER_

#include "irr/asset/ICPUImageView.h"
#include "irr/asset/IAssetLoader.h"

namespace irr
{
	namespace asset
	{
		//! glTF Loader capable of loading .gltf files
		/*
			glTF bridges the gap between 3D content creation tools and modern 3D applications 
			by providing an efficient, extensible, interoperable format for the transmission and loading of 3D content.
		*/

		class CGLTFLoader final : public asset::IAssetLoader
		{
			protected:
				virtual ~CGLTFLoader() {}

			public:
				CGLTFLoader() {}

				bool isALoadableFileFormat(io::IReadFile* _file) const override;

				const char** getAssociatedFileExtensions() const override
				{
					static const char* extensions[]{ "gltf", nullptr };
					return extensions;
				}

				uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MESH; }

				asset::SAssetBundle loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;
		};
	}
}

#endif // _IRR_COMPILE_WITH_GLTF_LOADER_
