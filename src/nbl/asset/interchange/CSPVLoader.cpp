// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/core/core.h"

#include "nbl/asset/ICPUShader.h"

#include "CSPVLoader.h"

using namespace nbl;
using namespace nbl::io;
using namespace nbl::asset;

// load in the image data
SAssetBundle CSPVLoader::loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	if (!_file)
        return {};

	
	auto buffer = core::make_smart_refctd_ptr<ICPUBuffer>(_file->getSize());
	
	system::ISystem::future_t<uint32_t> future;
	m_system->readFile(future, _file, buffer->getPointer(), 0, _file->getSize());
	future.get();

	if (reinterpret_cast<uint32_t*>(buffer->getPointer())[0]!=SPV_MAGIC_NUMBER)
		return {};

    return SAssetBundle(nullptr,{core::make_smart_refctd_ptr<ICPUShader>(std::move(buffer))});
}
