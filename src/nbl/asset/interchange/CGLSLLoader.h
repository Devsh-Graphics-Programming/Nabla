// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_GLSL_LOADER_H_INCLUDED__
#define __NBL_ASSET_C_GLSL_LOADER_H_INCLUDED__

#include <algorithm>

#include "nbl/asset/interchange/IAssetLoader.h"
#include <nbl/system/ISystem.h>

namespace nbl
{
namespace asset
{

//!  Surface Loader for PNG files
class CGLSLLoader final : public asset::IAssetLoader
{
	public:
		CGLSLLoader() = default;
		bool isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger = nullptr) const override
		{
			char tmp[10] = { 0 };
			char* end = tmp+sizeof(tmp);
			auto filesize = _file->getSize();
			size_t readPos = 0;
			while (readPos+sizeof(tmp)<filesize)
			{
				system::future<size_t> future;
				_file->read(future, tmp, readPos, sizeof tmp);
				size_t count = future.get();
				
				if (strncmp(tmp,"#version ",9u)==0)
					return true;
				auto found = std::find(tmp,end,'#');
				if (found==end || found==tmp)
				{
					readPos += count;
					continue;
				}
				readPos += found - end;
			}

			return false;
		}

		const char** getAssociatedFileExtensions() const override
		{
			static const char* ext[]{ "vert","tesc","tese","geom","frag","comp", nullptr };
			return ext;
		}

		uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_SPECIALIZED_SHADER; }

		asset::SAssetBundle loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;
};

} // namespace asset
} // namespace nbl

#endif

