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
		/*
			Each glTF asset must have an asset property. 
			In fact, it's the only required top-level property
			for JSON to be a valid glTF.
		*/

		bool CGLTFLoader::isALoadableFileFormat(io::IReadFile* _file) const
		{
			simdjson::dom::parser parser;
			simdjson::dom::object tweets = parser.load(_file->getFileName().c_str());
			simdjson::dom::element element;

			if (tweets.at_key("asset").get(element) == simdjson::error_code::SUCCESS)
				if (element.at_key("version").get(element) == simdjson::error_code::SUCCESS)
					return true;

			return false;
		}

		asset::SAssetBundle CGLTFLoader::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
		{
			simdjson::dom::parser parser;
			simdjson::dom::object tweets = parser.load(_file->getFileName().c_str());
			simdjson::dom::element element;

			for (auto& [key, value] : tweets)
			{
				if (key == "asset")
				{
					tweets.at_key("asset").at_key("version").get(element);
					header.version = std::stoi(element.get_string().value().data());
				}
			}

			return {};
		}
	}
}

#endif // _IRR_COMPILE_WITH_GLTF_LOADER_
