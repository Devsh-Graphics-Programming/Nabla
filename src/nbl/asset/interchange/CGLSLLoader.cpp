// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/asset.h"
#include "CGLSLLoader.h"

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
	//! TODO: Actually invoke the GLSL compiler to decode our type from any `#pragma`s
	std::filesystem::path extension = filename.extension();


	core::unordered_map<std::string,IShader::E_SHADER_STAGE> typeFromExt =	{	
																							{".vert",IShader::ESS_VERTEX},
																							{".tesc",IShader::ESS_TESSELLATION_CONTROL},
																							{".tese",IShader::ESS_TESSELLATION_EVALUATION},
																							{".geom",IShader::ESS_GEOMETRY},
																							{".frag",IShader::ESS_FRAGMENT},
																							{".comp",IShader::ESS_COMPUTE}
																						};
	auto found = typeFromExt.find(extension.string());
	if (found == typeFromExt.end())
	{
		_NBL_ALIGNED_FREE(source);
		return {};
	}

	auto shader = core::make_smart_refctd_ptr<ICPUShader>(reinterpret_cast<char*>(source), found->second, IShader::E_CONTENT_TYPE::ECT_GLSL, filename.string());
	_NBL_ALIGNED_FREE(source);

	return SAssetBundle(nullptr,{ std::move(shader) });
} 
