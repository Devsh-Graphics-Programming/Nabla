// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/core/declarations.h"

#include "nbl/asset/IShader.h"
#include "nbl/asset/interchange/CSPVLoader.h"

using namespace nbl;
using namespace nbl::asset;

// load in the image data
SAssetBundle CSPVLoader::loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	if (!_file)
        return {};
	
	auto buffer = ICPUBuffer::create({ _file->getSize() });
	
	system::IFile::success_t success;
	_file->read(success, buffer->getPointer(), 0, _file->getSize());
    if (!success)
        return {};

	if (reinterpret_cast<uint32_t*>(buffer->getPointer())[0]!=SPV_MAGIC_NUMBER)
		return {};

    buffer->setContentHash(buffer->computeContentHash());
    return SAssetBundle(nullptr,{core::make_smart_refctd_ptr<IShader>(std::move(buffer),asset::IShader::E_CONTENT_TYPE::ECT_SPIRV,_file->getFileName().string())});
}
