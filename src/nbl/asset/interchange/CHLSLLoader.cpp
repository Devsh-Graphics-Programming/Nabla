// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/asset.h"
#include "nbl/asset/interchange/CHLSLLoader.h"

using namespace nbl;
using namespace nbl::asset;

// load in the image data
SAssetBundle CHLSLLoader::loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	if (!_file)
        return {};

	const auto len = _file->getSize();
	auto source = ICPUBuffer::create({ len+1 });

	system::IFile::success_t success;
	_file->read(success, source->getPointer(), 0, len);
	if (!success)
		return {};
	// make sure put string end terminator
	reinterpret_cast<char*>(source->getPointer())[len] = 0;


	const auto filename = _file->getFileName();
	source->setContentHash(source->computeContentHash());
	return SAssetBundle(nullptr,{core::make_smart_refctd_ptr<IShader>(std::move(source), IShader::E_CONTENT_TYPE::ECT_HLSL, filename.string())});
} 
