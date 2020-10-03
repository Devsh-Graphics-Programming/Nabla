// Copyright (C) 2020 AnastaZIuk
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CGLTFLoader.h"

#ifdef _IRR_COMPILE_WITH_GLTF_LOADER_

#include "simdjson/singleheader/simdjson.h"
#include "os.h"

namespace irr
{
	namespace asset
	{
		bool CGLTFLoader::isALoadableFileFormat(io::IReadFile* _file) const
		{
			// TODO: change it when implementing .glb
			return true;
		}

		asset::SAssetBundle CGLTFLoader::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
		{
			simdjson::dom::parser parser;
			simdjson::dom::element tweets = parser.load(_file->getFileName().c_str());

			return {};
		}
	}
}

#endif // _IRR_COMPILE_WITH_GLTF_LOADER_
