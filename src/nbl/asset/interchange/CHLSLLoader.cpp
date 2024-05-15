// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/asset.h"
#include "CHLSLLoader.h"

using namespace nbl;
using namespace nbl::asset;

// load in the image data
SAssetBundle CHLSLLoader::loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
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
	auto filenameEnding = filename.filename().string();

	core::unordered_map<std::string,IShader::E_SHADER_STAGE> typeFromExt =	{	
		{".vert.hlsl",IShader::ESS_VERTEX},
		{".tesc.hlsl",IShader::ESS_TESSELLATION_CONTROL},
		{".tese.hlsl",IShader::ESS_TESSELLATION_EVALUATION},
		{".geom.hlsl",IShader::ESS_GEOMETRY},
		{".frag.hlsl",IShader::ESS_FRAGMENT},
		{".comp.hlsl",IShader::ESS_COMPUTE},
		{".mesh.hlsl",IShader::ESS_MESH},
		{".task.hlsl",IShader::ESS_TASK},
	};
	auto shaderStage = IShader::ESS_UNKNOWN;
	for (auto& it : typeFromExt) {
		if (filenameEnding.size() <= it.first.size()) continue;
		auto stringPart = filenameEnding.substr(filenameEnding.size() - it.first.size());
		if (stringPart  == it.first)
		{
			shaderStage = it.second;
			break;
		}
	}

	// TODO: allocate the source as an ICPUBuffer right away!
	auto shader = core::make_smart_refctd_ptr<ICPUShader>(reinterpret_cast<char*>(source), shaderStage, IShader::E_CONTENT_TYPE::ECT_HLSL, filename.string());
	_NBL_ALIGNED_FREE(source);

	return SAssetBundle(nullptr,{std::move(shader)});
} 
