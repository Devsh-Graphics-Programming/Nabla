// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/core/declarations.h"

#include "nbl/asset/ICPUShader.h"

#include "CSPVLoader.h"

using namespace nbl;
using namespace nbl::io;
using namespace nbl::asset;

// load in the image data
SAssetBundle CSPVLoader::loadAsset(IReadFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	if (!_file)
        return {};

	const size_t prevPos = _file->getPos();
	_file->seek(0u);
	
	auto buffer = core::make_smart_refctd_ptr<ICPUBuffer>(_file->getSize());
	_file->read(buffer->getPointer(),_file->getSize());
	_file->seek(prevPos);

	if (reinterpret_cast<uint32_t*>(buffer->getPointer())[0]!=SPV_MAGIC_NUMBER)
		return {};

    return SAssetBundle(nullptr,{core::make_smart_refctd_ptr<ICPUShader>(std::move(buffer))});
}
