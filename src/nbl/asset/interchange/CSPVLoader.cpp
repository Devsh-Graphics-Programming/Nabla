// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/core/declarations.h"

#include "nbl/asset/ICPUShader.h"

#include "CSPVLoader.h"

using namespace nbl;
using namespace nbl::asset;

// load in the image data
SAssetBundle CSPVLoader::loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	if (!_file)
        return {};

	
	auto buffer = core::make_smart_refctd_ptr<ICPUBuffer>(_file->getSize());
	
	system::future<size_t> future;
	_file->read(future, buffer->getPointer(), 0, _file->getSize());
	future.get();

	if (reinterpret_cast<uint32_t*>(buffer->getPointer())[0]!=SPV_MAGIC_NUMBER)
		return {};

	// Todo(achal): Need to get shader stage and file path hint here
    return SAssetBundle(nullptr,{core::make_smart_refctd_ptr<ICPUShader>(std::move(buffer), IShader::ESS_UNKNOWN, "????")});
}
