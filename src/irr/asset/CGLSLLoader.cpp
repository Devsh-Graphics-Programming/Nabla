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

	core::unordered_map<std::string,E_SHADER_STAGE> typeFromExt =	{	
																		{".vert",ESS_VERTEX},
																		{".tesc",ESS_TESSELATION_CONTROL},
																		{".tese",ESS_TESSELATION_EVALUATION},
																		{".geom",ESS_GEOMETRY},
																		{".frag",ESS_FRAGMENT},
																		{".comp",ESS_COMPUTE}
																	};
	auto found = typeFromExt.find(extension.c_str());
	if (found==typeFromExt.end())
		return {};

	return SAssetBundle{core::make_smart_refctd_ptr<ICPUSpecializedShader>(std::move(shader),core::make_smart_refctd_ptr<ISpecializationInfo>(core::vector<SSpecializationMapEntry>(),nullptr,"main",found->second))};
} 