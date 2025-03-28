// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/asset.h"
#include "nbl/asset/interchange/CHLSLLoader.h"
#include "nbl/asset/metadata/CHLSLMetadata.h"

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
	auto filenameEnding = filename.filename().string();

	core::unordered_map<std::string,IShader::E_SHADER_STAGE> typeFromExt =
	{
		{".vert.hlsl",IShader::E_SHADER_STAGE::ESS_VERTEX},
		{".tesc.hlsl",IShader::E_SHADER_STAGE::ESS_TESSELLATION_CONTROL},
		{".tese.hlsl",IShader::E_SHADER_STAGE::ESS_TESSELLATION_EVALUATION},
		{".geom.hlsl",IShader::E_SHADER_STAGE::ESS_GEOMETRY},
		{".frag.hlsl",IShader::E_SHADER_STAGE::ESS_FRAGMENT},
		{".comp.hlsl",IShader::E_SHADER_STAGE::ESS_COMPUTE},
		{".mesh.hlsl",IShader::E_SHADER_STAGE::ESS_MESH},
		{".task.hlsl",IShader::E_SHADER_STAGE::ESS_TASK},
		{".rgen.hlsl",IShader::E_SHADER_STAGE::ESS_RAYGEN},
		{".rahit.hlsl",IShader::E_SHADER_STAGE::ESS_ANY_HIT},
		{".rchit.hlsl",IShader::E_SHADER_STAGE::ESS_CLOSEST_HIT},
		{".rmiss.hlsl",IShader::E_SHADER_STAGE::ESS_MISS},
		{".rint.hlsl",IShader::E_SHADER_STAGE::ESS_INTERSECTION},
		{".rcall.hlsl",IShader::E_SHADER_STAGE::ESS_CALLABLE},
	};
	auto shaderStage = IShader::E_SHADER_STAGE::ESS_UNKNOWN;
	for (auto& it : typeFromExt)
	{
		if (filenameEnding.size() <= it.first.size()) continue;
		auto stringPart = filenameEnding.substr(filenameEnding.size() - it.first.size());
		if (stringPart  == it.first)
		{
			shaderStage = it.second;
			break;
		}
	}

	source->setContentHash(source->computeContentHash());
	return SAssetBundle(core::make_smart_refctd_ptr<CHLSLMetadata>(shaderStage), {core::make_smart_refctd_ptr<IShader>(std::move(source), IShader::E_CONTENT_TYPE::ECT_HLSL, filename.string())});
} 
