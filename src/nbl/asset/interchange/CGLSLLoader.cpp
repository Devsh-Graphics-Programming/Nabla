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
	system::future<size_t> future;
	_file->read(future, source, 0, len);
	future.get();
	reinterpret_cast<char*>(source)[len] = 0;



	auto shader = core::make_smart_refctd_ptr<ICPUShader>(reinterpret_cast<char*>(source));
	_NBL_ALIGNED_FREE(source);

	const auto filename = _file->getFileName();
	//! TODO: Actually invoke the GLSL compiler to decode our type from any `#pragma`s
	std::filesystem::path extension = filename.extension();

	core::unordered_map<std::string,ISpecializedShader::E_SHADER_STAGE> typeFromExt =	{	
																							{".vert",ISpecializedShader::ESS_VERTEX},
																							{".tesc",ISpecializedShader::ESS_TESSELATION_CONTROL},
																							{".tese",ISpecializedShader::ESS_TESSELATION_EVALUATION},
																							{".geom",ISpecializedShader::ESS_GEOMETRY},
																							{".frag",ISpecializedShader::ESS_FRAGMENT},
																							{".comp",ISpecializedShader::ESS_COMPUTE}
																						};
	auto found = typeFromExt.find(extension.string());
	if (found==typeFromExt.end())
		return {};

	return SAssetBundle(nullptr,{ core::make_smart_refctd_ptr<ICPUSpecializedShader>(std::move(shader),ISpecializedShader::SInfo({},nullptr,"main",found->second,filename.string())) });
} 
