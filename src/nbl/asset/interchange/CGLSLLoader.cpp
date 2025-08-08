// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/asset.h"
#include "nbl/asset/interchange/CGLSLLoader.h"

using namespace nbl;
using namespace nbl::asset;

// load in the image data
SAssetBundle CGLSLLoader::loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	if (!_file)
        return {};

	auto len = _file->getSize();
	void* source = _NBL_ALIGNED_MALLOC(len+1u,_NBL_SIMD_ALIGNMENT);

	system::IFile::success_t success;
	_file->read(success, source, 0, len);
	if (!success)
		return {};
	reinterpret_cast<char*>(source)[len] = 0;

	const auto filename = _file->getFileName();
	auto shader = core::make_smart_refctd_ptr<IShader>(reinterpret_cast<char*>(source), IShader::E_CONTENT_TYPE::ECT_GLSL, filename.string());
	{
		auto backingBuffer = shader->getContent();
		const_cast<ICPUBuffer*>(backingBuffer)->setContentHash(backingBuffer->computeContentHash());
	}
	_NBL_ALIGNED_FREE(source);
	return SAssetBundle(nullptr,{ std::move(shader) });
} 
