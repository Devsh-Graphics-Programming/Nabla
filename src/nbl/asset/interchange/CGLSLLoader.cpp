// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/asset.h"
#include "CGLSLLoader.h"

using namespace nbl;
using namespace nbl::io;
using namespace nbl::asset;

// load in the image data
SAssetBundle CGLSLLoader::loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	if (!_file)
        return {};

	const size_t prevPos = _file->getPos();
	_file->seek(0u);

	auto len = _file->getSize();
	void* source = _NBL_ALIGNED_MALLOC(len+1u,_NBL_SIMD_ALIGNMENT);
	_file->read(source,len);
	reinterpret_cast<char*>(source)[len] = 0;

	_file->seek(prevPos);


	auto shader = core::make_smart_refctd_ptr<ICPUShader>(reinterpret_cast<char*>(source));
	_NBL_ALIGNED_FREE(source);

	const std::string filename = _file->getFileName().c_str();
	//! TODO: Actually invoke the GLSL compiler to decode our type from any `#pragma`s
	io::path extension;
	core::getFileNameExtension(extension,filename.c_str());

	core::unordered_map<std::string,ISpecializedShader::E_SHADER_STAGE> typeFromExt =	{	
																							{".vert",ISpecializedShader::ESS_VERTEX},
																							{".tesc",ISpecializedShader::ESS_TESSELATION_CONTROL},
																							{".tese",ISpecializedShader::ESS_TESSELATION_EVALUATION},
																							{".geom",ISpecializedShader::ESS_GEOMETRY},
																							{".frag",ISpecializedShader::ESS_FRAGMENT},
																							{".comp",ISpecializedShader::ESS_COMPUTE}
																						};
	auto found = typeFromExt.find(extension.c_str());
	if (found==typeFromExt.end())
		return {};

	return SAssetBundle(nullptr,{ core::make_smart_refctd_ptr<ICPUSpecializedShader>(std::move(shader),ISpecializedShader::SInfo({},nullptr,"main",found->second,filename)) });
} 
