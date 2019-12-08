#include "irr/core/core.h"
#include "CGLSLLoader.h"

using namespace irr;
using namespace irr::io;
using namespace irr::asset;

// load in the image data
SAssetBundle CGLSLLoader::loadAsset(IReadFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	if (!_file)
        return {};

	const size_t prevPos = _file->getPos();
	_file->seek(0u);

	auto len = _file->getSize();
	void* source = _IRR_ALIGNED_MALLOC(len+1u,_IRR_SIMD_ALIGNMENT);
	_file->read(source,len);
	reinterpret_cast<char*>(source)[len] = 0;

	_file->seek(prevPos);


	auto shader = core::make_smart_refctd_ptr<ICPUShader>(reinterpret_cast<char*>(source));
	_IRR_ALIGNED_FREE(source);

	//! TODO: Actually invoke the GLSL compiler to decode our type from any `#pragma`s
	io::path extension;
	core::getFileNameExtension(extension,_file->getFileName());

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

	return SAssetBundle{ core::make_smart_refctd_ptr<ICPUSpecializedShader>(std::move(shader),ISpecializedShader::SInfo({},nullptr,"main",found->second)) };
} 