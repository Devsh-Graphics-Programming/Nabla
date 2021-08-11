// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "CBufferLoaderBIN.h"

namespace nbl
{
	namespace asset
	{
		asset::SAssetBundle CBufferLoaderBIN::loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
		{
			if (!_file)
				return {};

			SContext ctx(_file->getSize());
			ctx.file = _file;
			system::future<size_t> read_future;
			_file->read(read_future, ctx.sourceCodeBuffer->getPointer(), 0, _file->getSize());
			read_future.get();

			return SAssetBundle(nullptr,{std::move(ctx.sourceCodeBuffer)});
		}

		bool CBufferLoaderBIN::isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const
		{
			return true; // validation if needed
		}
	}
}