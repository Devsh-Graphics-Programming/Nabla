#include "..\..\..\..\include\nbl\ext\MitsubaLoader\CMitsubaLoader.h"
// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "os.h"

#include <cwchar>

#include "nbl/ext/MitsubaLoader/CMitsubaLoader.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"

#include "nbl/asset/utils/CDerivativeMapCreator.h"

#include "nbl/ext/MitsubaLoader/CMitsubaSerializedMetadata.h"
#include "nbl/ext/MitsubaLoader/CGLSLMitsubaLoaderBuiltinIncludeLoader.h"


#if defined(_NBL_DEBUG) || defined(_NBL_RELWITHDEBINFO)
#	define DEBUG_MITSUBA_LOADER
#endif

namespace nbl
{
using namespace asset;

namespace ext
{
namespace MitsubaLoader
{

_NBL_STATIC_INLINE_CONSTEXPR const char* DUMMY_VERTEX_SHADER =
R"(#version 430 core

layout (location = 0) in vec3 vPosition;
layout (location = 2) in vec2 vUV;
layout (location = 3) in vec3 vNormal;

layout (location = 0) out vec3 WorldPos;
layout (location = 1) flat out uint InstanceIndex;
layout (location = 2) out vec3 Normal;
layout (location = 3) out vec2 UV;

#include <nbl/builtin/glsl/utils/common.glsl>
#include <nbl/builtin/glsl/utils/transform.glsl>

#ifndef _NBL_VERT_SET1_BINDINGS_DEFINED_
#define _NBL_VERT_SET1_BINDINGS_DEFINED_
layout (set = 1, binding = 0, row_major, std140) uniform UBO {
    nbl_glsl_SBasicViewParameters params;
} CamData;
#endif //_NBL_VERT_SET1_BINDINGS_DEFINED_

#include <nbl/builtin/glsl/ext/MitsubaLoader/instance_data_struct.glsl>

layout (set = 0, binding = 5, row_major, std430) readonly restrict buffer InstDataBuffer {
	nbl_glsl_ext_Mitsuba_Loader_instance_data_t data[];
} InstData;

void main()
{
	mat4x3 tform = InstData.data[gl_InstanceIndex].tform;
	mat4 mvp = nbl_glsl_pseudoMul4x4with4x3(CamData.params.MVP, tform);
	gl_Position = nbl_glsl_pseudoMul4x4with3x1(mvp, vPosition);
	WorldPos = nbl_glsl_pseudoMul3x4with3x1(tform, vPosition);
	mat3 normalMat = mat3(InstData.data[gl_InstanceIndex].normalMatrixRow0,InstData.data[gl_InstanceIndex].normalMatrixRow1,InstData.data[gl_InstanceIndex].normalMatrixRow2);
	Normal = transpose(normalMat)*normalize(vNormal);
	UV = vUV;
	InstanceIndex = gl_InstanceIndex;
}

)";

_NBL_STATIC_INLINE_CONSTEXPR const char* FRAGMENT_SHADER_PROLOGUE =
R"(#version 430 core
#extension GL_EXT_shader_integer_mix : require
)";
_NBL_STATIC_INLINE_CONSTEXPR const char* FRAGMENT_SHADER_INPUT_OUTPUT =
R"(
layout (location = 0) in vec3 WorldPos;
layout (location = 1) flat in uint InstanceIndex;
layout (location = 2) in vec3 Normal;
layout (location = 3) in vec2 UV;

layout (location = 0) out vec4 OutColor;
)";
_NBL_STATIC_INLINE_CONSTEXPR const char* FRAGMENT_SHADER_DEFINITIONS =
R"(
#include <nbl/builtin/glsl/utils/common.glsl>

layout (set = 1, binding = 0, row_major, std140) uniform UBO {
    nbl_glsl_SBasicViewParameters params;
} CamData;

vec3 nbl_glsl_MC_getNormalizedWorldSpaceV()
{
	vec3 campos = nbl_glsl_SBasicViewParameters_GetEyePos(CamData.params.NormalMatAndEyePos);
	return normalize(campos - WorldPos);
}
vec3 nbl_glsl_MC_getNormalizedWorldSpaceN()
{
	return normalize(Normal);
}
#ifdef TEX_PREFETCH_STREAM
mat2x3 nbl_glsl_perturbNormal_dPdSomething() {return mat2x3(dFdx(WorldPos),dFdy(WorldPos));}
mat2 nbl_glsl_perturbNormal_dUVdSomething()
{
    return mat2(dFdx(UV),dFdy(UV));
}
#endif
#define _NBL_USER_PROVIDED_MATERIAL_COMPILER_GLSL_BACKEND_FUNCTIONS_
)";
_NBL_STATIC_INLINE_CONSTEXPR const char* FRAGMENT_SHADER_IMPL = R"(
#include <nbl/builtin/glsl/format/decode.glsl>

#ifndef _NBL_BSDF_COS_EVAL_DEFINED_
#define _NBL_BSDF_COS_EVAL_DEFINED_
// Spectrum can be exchanged to a float for monochrome
#define Spectrum vec3
//! This is the function that evaluates the BSDF for specific view and observer direction
// params can be either BSDFIsotropicParams or BSDFAnisotropicParams
nbl_glsl_MC_precomputed_t precomp;
nbl_glsl_MC_oriented_material_t material;
Spectrum nbl_bsdf_cos_eval(in nbl_glsl_LightSample _sample, in nbl_glsl_AnisotropicViewSurfaceInteraction inter)
{
	nbl_glsl_MC_instr_stream_t eis = nbl_glsl_MC_oriented_material_t_getEvalStream(material);

	return nbl_glsl_MC_runEvalStream(precomp, eis, _sample.L);
}
#endif

#ifndef _NBL_COMPUTE_LIGHTING_DEFINED_
#define _NBL_COMPUTE_LIGHTING_DEFINED_
vec3 nbl_computeLighting(inout nbl_glsl_IsotropicViewSurfaceInteraction out_interaction)
{
	vec3 campos = nbl_glsl_MC_getCamPos();
	out_interaction = nbl_glsl_calcSurfaceInteraction(campos,WorldPos,normalize(Normal));

	nbl_glsl_LightSample _sample = nbl_glsl_createLightSample(precomp.V,1.0,precomp.N);
	return nbl_glsl_MC_oriented_material_t_getEmissive(material)+nbl_bsdf_cos_eval(_sample,out_interaction)/dot(interaction.V.dir,interaction.V.dir);
}
#endif

#ifndef _NBL_FRAG_MAIN_DEFINED_
#define _NBL_FRAG_MAIN_DEFINED_
void main()
{
	mat2 dUV = mat2(dFdx(UV),dFdy(UV));

	// "The sign of this computation is negated when the value of GL_CLIP_ORIGIN (the clip volume origin, set with glClipControl) is GL_UPPER_LEFT."
	const bool front = bool((InstData.data[InstanceIndex].determinantSignBit^mix(~0u,0u,gl_FrontFacing))&0x80000000u);
	precomp = nbl_glsl_MC_precomputeData(front);
	material = nbl_glsl_MC_material_data_t_getOriented(InstData.data[InstanceIndex].material,precomp.frontface);
#ifdef TEX_PREFETCH_STREAM
	nbl_glsl_MC_runTexPrefetchStream(nbl_glsl_MC_oriented_material_t_getTexPrefetchStream(material), UV, dUV);
#endif
#ifdef NORM_PRECOMP_STREAM
	nbl_glsl_MC_runNormalPrecompStream(nbl_glsl_MC_oriented_material_t_getNormalPrecompStream(material), precomp);
#endif


	nbl_glsl_AnisotropicViewSurfaceInteraction inter;
	vec3 color = nbl_computeLighting(inter);

	OutColor = vec4(color, 1.0);
}
#endif
)";

_NBL_STATIC_INLINE_CONSTEXPR const char* VERTEX_SHADER_CACHE_KEY = "nbl/builtin/specialized_shader/loaders/mitsuba_xml/default";

_NBL_STATIC_INLINE_CONSTEXPR uint32_t PAGE_TAB_TEX_BINDING = 0u;
_NBL_STATIC_INLINE_CONSTEXPR uint32_t PHYS_PAGE_VIEWS_BINDING = 1u;
_NBL_STATIC_INLINE_CONSTEXPR uint32_t PRECOMPUTED_VT_DATA_BINDING = 2u;
_NBL_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BUF_BINDING = 3u;
_NBL_STATIC_INLINE_CONSTEXPR uint32_t BSDF_BUF_BINDING = 4u;
_NBL_STATIC_INLINE_CONSTEXPR uint32_t INSTANCE_DATA_BINDING = 5u;
_NBL_STATIC_INLINE_CONSTEXPR uint32_t PREFETCH_INSTR_BUF_BINDING = 6u;
_NBL_STATIC_INLINE_CONSTEXPR uint32_t DS0_BINDING_COUNT_WO_VT = 5u;

template <typename AssetT>
static void insertAssetIntoCache(core::smart_refctd_ptr<AssetT>& asset, const char* path, IAssetManager* _assetMgr) // TODO: @Crisspl this is duplicate code
{
	asset::SAssetBundle bundle(nullptr,{ asset });
	_assetMgr->changeAssetKey(bundle, path);
	_assetMgr->insertAssetIntoCache(bundle);
}
// @Crisspl TODO this needs to use the IAssetLoaderOverride instead
template<typename AssetType, IAsset::E_TYPE assetType>
static auto getBuiltinAsset(const char* _key, IAssetManager* _assetMgr) -> std::enable_if_t<std::is_base_of_v<asset::IAsset, AssetType>, core::smart_refctd_ptr<AssetType>>
{
	size_t storageSz = 1ull;
	asset::SAssetBundle bundle;
	const IAsset::E_TYPE types[]{ assetType, static_cast<IAsset::E_TYPE>(0u) };

	_assetMgr->findAssets(storageSz, &bundle, _key, types);
	auto assets = bundle.getContents();
	if (assets.empty())
		return nullptr;
	//assert(!assets.empty());

	return core::smart_refctd_ptr_static_cast<AssetType>(assets.begin()[0]);
}

static core::smart_refctd_ptr<asset::ICPUSpecializedShader> createSpecShader(const char* _glsl, asset::ISpecializedShader::E_SHADER_STAGE _stage)
{
	auto shader = core::make_smart_refctd_ptr<asset::ICPUShader>(_glsl);
	asset::ICPUSpecializedShader::SInfo info(nullptr, nullptr, "main", _stage);
	auto specd = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(shader), std::move(info));

	return specd;
}
static core::smart_refctd_ptr<asset::ICPUSpecializedShader> createAndCacheVertexShader(asset::IAssetManager* _manager, const char* _glsl)
{
	auto vs = createSpecShader(_glsl, asset::ISpecializedShader::ESS_VERTEX);

	insertAssetIntoCache(vs, VERTEX_SHADER_CACHE_KEY, _manager);

	return vs;
}
static core::smart_refctd_ptr<asset::ICPUSpecializedShader> createFragmentShader(const asset::material_compiler::CMaterialCompilerGLSLBackendCommon::result_t& _mcRes, size_t _VTstorageViewCount)
{
	std::string source = 
		FRAGMENT_SHADER_PROLOGUE +
		_mcRes.fragmentShaderSource_declarations +
		FRAGMENT_SHADER_INPUT_OUTPUT +
		"#include <nbl/builtin/glsl/ext/MitsubaLoader/material_compiler_compatibility.glsl/" + std::to_string(_VTstorageViewCount) + ">" +
		FRAGMENT_SHADER_DEFINITIONS +
		_mcRes.fragmentShaderSource +
		FRAGMENT_SHADER_IMPL;

	return createSpecShader(source.c_str(), asset::ISpecializedShader::ESS_FRAGMENT);
}
static core::smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline> createPipeline(core::smart_refctd_ptr<asset::ICPUPipelineLayout>&& _layout, core::smart_refctd_ptr<asset::ICPUSpecializedShader>&& _vertshader, core::smart_refctd_ptr<asset::ICPUSpecializedShader>&& _fragshader)
{
	auto vs = std::move(_vertshader);
	auto fs = std::move(_fragshader);
	asset::ICPUSpecializedShader* shaders[2]{ vs.get(), fs.get() };

	SRasterizationParams rasterParams;
	rasterParams.faceCullingMode = asset::EFCM_NONE;
	rasterParams.frontFaceIsCCW = 1;
	auto pipeline = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(
		std::move(_layout),
		shaders, shaders+2,
		//all the params will be overriden with those loaded with meshes
		SVertexInputParams(),
		SBlendParams(),
		SPrimitiveAssemblyParams(),
		rasterParams
	);

	return pipeline;
}

static core::smart_refctd_ptr<asset::ICPUImageView> createImageView(core::smart_refctd_ptr<asset::ICPUImage>&& _img) // TODO: this should seriously be a utility somewhere
{
	const auto& iparams = _img->getCreationParameters();

	asset::ICPUImageView::SCreationParams params;
	params.format = iparams.format;
	params.subresourceRange.baseArrayLayer = 0u;
	params.subresourceRange.layerCount = iparams.arrayLayers;
	assert(params.subresourceRange.layerCount == 1u);
	params.subresourceRange.baseMipLevel = 0u;
	params.subresourceRange.levelCount = iparams.mipLevels;
	params.viewType = asset::IImageView<asset::ICPUImage>::ET_2D;
	params.flags = static_cast<asset::IImageView<asset::ICPUImage>::E_CREATE_FLAGS>(0);
	params.image = std::move(_img);

	return asset::ICPUImageView::create(std::move(params));
}
static core::smart_refctd_ptr<asset::ICPUImage> createDerivMap(asset::ICPUImage* _heightMap, asset::ICPUSampler* _smplr)
{
	const auto& sp = _smplr->getParams();

	return asset::CDerivativeMapCreator::createDerivativeMapFromHeightMap(
			_heightMap,
			static_cast<asset::ICPUSampler::E_TEXTURE_CLAMP>(sp.TextureWrapU),
			static_cast<asset::ICPUSampler::E_TEXTURE_CLAMP>(sp.TextureWrapV),
			static_cast<asset::ICPUSampler::E_TEXTURE_BORDER_COLOR>(sp.BorderColor)
	);
}
static core::smart_refctd_ptr<asset::ICPUImage> createBlendWeightImage(const asset::ICPUImage* _img)
{
	auto outParams = _img->getCreationParameters();
	assert(asset::getFormatChannelCount(outParams.format) == 1u);

	auto get3ChannelFormat = [](uint32_t bytesPerChannel) -> asset::E_FORMAT {
		switch (bytesPerChannel)
		{
		case 1u:
			return asset::EF_R8G8B8_UNORM;
		case 2u:
			return asset::EF_R16G16B16_SFLOAT;
		case 4u:
			return asset::EF_R32G32B32_SFLOAT;
		case 8u:
			return asset::EF_R64G64B64_SFLOAT;
		default:
			return asset::EF_UNKNOWN;
		}
	};

	asset::ICPUImage::SBufferCopy region;
	const uint32_t bytesPerChannel = (getBytesPerPixel(outParams.format) * core::rational(1, getFormatChannelCount(outParams.format))).getIntegerApprox();
	outParams.format = get3ChannelFormat(outParams.format);
	const size_t texelBytesz = asset::getTexelOrBlockBytesize(outParams.format);
	region.bufferRowLength = asset::IImageAssetHandlerBase::calcPitchInBlocks(outParams.extent.width, texelBytesz);
	auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(texelBytesz * region.bufferRowLength * outParams.extent.height);
	region.imageOffset = { 0,0,0 };
	region.imageExtent = outParams.extent;
	region.imageSubresource.baseArrayLayer = 0u;
	region.imageSubresource.layerCount = 1u;
	region.imageSubresource.mipLevel = 0u;
	region.bufferImageHeight = 0u;
	region.bufferOffset = 0u;
	auto outImg = asset::ICPUImage::create(std::move(outParams));
	outImg->setBufferAndRegions(std::move(buffer), core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy>>(1ull, region));

	asset::ICPUImageView::SComponentMapping swizzle;
	constexpr auto RED = asset::ICPUImageView::SComponentMapping::ES_R;
	swizzle = { RED, RED, RED, RED };

	using convert_filter_t = asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN>;
	convert_filter_t::state_type conv;
	conv.swizzle = swizzle;
	conv.extent = outParams.extent;
	conv.layerCount = 1u;
	conv.inMipLevel = 0u;
	conv.outMipLevel = 0u;
	conv.inBaseLayer = 0u;
	conv.outBaseLayer = 0u;
	conv.inOffset = { 0u,0u,0u };
	conv.outOffset = { 0u,0u,0u };
	conv.inImage = _img;
	conv.outImage = outImg.get();

	if (!convert_filter_t::execute(std::execution::par_unseq,&conv))
	{
		os::Printer::log("Mitsuba XML Loader: blend weight texture creation failed!", ELL_ERROR);
		_NBL_DEBUG_BREAK_IF(true);
	}

	return outImg;
}

core::smart_refctd_ptr<asset::ICPUPipelineLayout> CMitsubaLoader::createPipelineLayout(asset::IAssetManager* _manager, asset::ICPUVirtualTexture* _vt)
{
	core::smart_refctd_ptr<ICPUDescriptorSetLayout> ds0layout;
	{
		auto sizes = _vt->getDSlayoutBindings(nullptr, nullptr);
		auto bindings = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUDescriptorSetLayout::SBinding>>(sizes.first + DS0_BINDING_COUNT_WO_VT);
		auto samplers = core::make_refctd_dynamic_array< core::smart_refctd_dynamic_array<core::smart_refctd_ptr<asset::ICPUSampler>>>(sizes.second);

		_vt->getDSlayoutBindings(bindings->data(), samplers->data(), PAGE_TAB_TEX_BINDING, PHYS_PAGE_VIEWS_BINDING);
		auto* b = bindings->data() + (bindings->size() - DS0_BINDING_COUNT_WO_VT);
		b[0].binding = PRECOMPUTED_VT_DATA_BINDING;
		b[0].count = 1u;
		b[0].samplers = nullptr;
		b[0].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
		b[0].type = asset::EDT_STORAGE_BUFFER;

		b[1].binding = INSTR_BUF_BINDING;
		b[1].count = 1u;
		b[1].samplers = nullptr;
		b[1].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
		b[1].type = asset::EDT_STORAGE_BUFFER;

		b[2].binding = BSDF_BUF_BINDING;
		b[2].count = 1u;
		b[2].samplers = nullptr;
		b[2].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
		b[2].type = asset::EDT_STORAGE_BUFFER;

		b[3].binding = INSTANCE_DATA_BINDING;
		b[3].count = 1u;
		b[3].samplers = nullptr;
		b[3].stageFlags = static_cast<asset::ISpecializedShader::E_SHADER_STAGE>(asset::ISpecializedShader::ESS_FRAGMENT | asset::ISpecializedShader::ESS_VERTEX);
		b[3].type = asset::EDT_STORAGE_BUFFER;

		b[4].binding = PREFETCH_INSTR_BUF_BINDING;
		b[4].count = 1u;
		b[4].samplers = nullptr;
		b[4].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
		b[4].type = asset::EDT_STORAGE_BUFFER;

		ds0layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bindings->data(), bindings->data() + bindings->size());
	}
	auto ds1layout = getBuiltinAsset<ICPUDescriptorSetLayout, IAsset::ET_DESCRIPTOR_SET_LAYOUT>("nbl/builtin/descriptor_set_layout/basic_view_parameters", _manager);

	return core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(nullptr, nullptr, std::move(ds0layout), std::move(ds1layout), nullptr, nullptr);
}

CMitsubaLoader::CMitsubaLoader(asset::IAssetManager* _manager, io::IFileSystem* _fs) : asset::IRenderpassIndependentPipelineLoader(_manager), m_filesystem(_fs)
{
#ifdef _NBL_DEBUG
	setDebugName("CMitsubaLoader");
#endif
}

void CMitsubaLoader::initialize()
{
	IRenderpassIndependentPipelineLoader::initialize();

	auto* glslc = m_assetMgr->getGLSLCompiler();

	glslc->getIncludeHandler()->addBuiltinIncludeLoader(core::make_smart_refctd_ptr<CGLSLMitsubaLoaderBuiltinIncludeLoader>(m_filesystem));
}

bool CMitsubaLoader::isALoadableFileFormat(io::IReadFile* _file) const
{
	constexpr uint32_t stackSize = 16u*1024u;
	char tempBuff[stackSize+1];
	tempBuff[stackSize] = 0;

	static const char* stringsToFind[] = { "<?xml", "version", "scene"};
	static const wchar_t* stringsToFindW[] = { L"<?xml", L"version", L"scene"};
	constexpr uint32_t maxStringSize = 8u; // "version\0"
	static_assert(stackSize > 2u*maxStringSize, "WTF?");

	const size_t prevPos = _file->getPos();
	const auto fileSize = _file->getSize();
	if (fileSize < maxStringSize)
		return false;

	_file->seek(0);
	_file->read(tempBuff, 3u);
	bool utf16 = false;
	if (tempBuff[0]==0xEFu && tempBuff[1]==0xBBu && tempBuff[2]==0xBFu)
		utf16 = false;
	else if (reinterpret_cast<uint16_t*>(tempBuff)[0]==0xFEFFu)
	{
		utf16 = true;
		_file->seek(2);
	}
	else
		_file->seek(0);
	while (true)
	{
		auto pos = _file->getPos();
		if (pos >= fileSize)
			break;
		if (pos > maxStringSize)
			_file->seek(_file->getPos()-maxStringSize);
		_file->read(tempBuff,stackSize);
		for (auto i=0u; i<sizeof(stringsToFind)/sizeof(const char*); i++)
		if (utf16 ? (wcsstr(reinterpret_cast<wchar_t*>(tempBuff),stringsToFindW[i])!=nullptr):(strstr(tempBuff, stringsToFind[i])!=nullptr))
		{
			_file->seek(prevPos);
			return true;
		}
	}
	_file->seek(prevPos);
	return false;
}

const char** CMitsubaLoader::getAssociatedFileExtensions() const
{
	static const char* ext[]{ "xml", nullptr };
	return ext;
}

asset::SAssetBundle CMitsubaLoader::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	ParserManager parserManager(m_assetMgr->getFileSystem(),_override);
	if (!parserManager.parse(_file))
		return {};

	if (_params.loaderFlags & IAssetLoader::ELPF_LOAD_METADATA_ONLY)
	{
		auto emptyMesh = core::make_smart_refctd_ptr<asset::ICPUMesh>();
		return SAssetBundle(std::move(parserManager.m_metadata),{ std::move(emptyMesh) });
	}
	else
	{
		//
		auto currentDir = io::IFileSystem::getFileDir(_file->getFileName()) + "/";
		SContext ctx(
			m_assetMgr->getGeometryCreator(),
			m_assetMgr->getMeshManipulator(),
			asset::IAssetLoader::SAssetLoadContext{ 
				asset::IAssetLoader::SAssetLoadParams(_params.decryptionKeyLen, _params.decryptionKey, _params.cacheFlags, currentDir.c_str()),
				_file
			},
			_override,
			parserManager.m_metadata.get()
		);
		if (!getBuiltinAsset<asset::ICPUPipelineLayout, asset::IAsset::ET_SPECIALIZED_SHADER>(VERTEX_SHADER_CACHE_KEY, m_assetMgr))
		{
			createAndCacheVertexShader(m_assetMgr, DUMMY_VERTEX_SHADER);
		}

		core::map<core::smart_refctd_ptr<asset::ICPUMesh>,std::pair<std::string,CElementShape::Type>> meshes;
		for (auto& shapepair : parserManager.shapegroups)
		{
			auto* shapedef = shapepair.first;
			if (shapedef->type == CElementShape::Type::SHAPEGROUP)
				continue;

			auto lowermeshes = getMesh(ctx, _hierarchyLevel, shapedef);
			for (auto& mesh : lowermeshes)
			{
				if (!mesh)
					continue;

				auto found = meshes.find(mesh);
				if (found == meshes.end())
					meshes.emplace(std::move(mesh),std::pair<std::string,CElementShape::Type>(std::move(shapepair.second),shapedef->type));
			}
		}

		parserManager.m_metadata->reserveMeshStorage(meshes.size(),ctx.mapMesh2instanceData.size());
		for (auto& mesh : meshes)
		{
			auto instances_rng = ctx.mapMesh2instanceData.equal_range(mesh.first.get());
			assert(instances_rng.first!=instances_rng.second);

			const uint32_t instanceCount = parserManager.m_metadata->addMeshMeta(mesh.first.get(),std::move(mesh.second.first),mesh.second.second,instances_rng.first,instances_rng.second);
			for (auto mb : mesh.first.get()->getMeshBuffers())
				mb->setInstanceCount(instanceCount);
		}

		auto compResult = ctx.backend.compile(&ctx.backend_ctx, ctx.ir.get());
		ctx.backend_ctx.vt.commitAll();
		auto pipelineLayout = createPipelineLayout(m_assetMgr, ctx.backend_ctx.vt.vt.get());
		auto fragShader = createFragmentShader(compResult, ctx.backend_ctx.vt.vt->getFloatViews().size());
		auto ds0 = createDS0(ctx, pipelineLayout.get(), compResult, meshes.begin(), meshes.end());
		auto basePipeline = createPipeline(
			std::move(pipelineLayout),
			getBuiltinAsset<asset::ICPUSpecializedShader, asset::IAsset::ET_SPECIALIZED_SHADER>(VERTEX_SHADER_CACHE_KEY, m_assetMgr),
			std::move(fragShader)
		);
		ctx.meta->m_global.m_materialCompilerGLSL_declarations = compResult.fragmentShaderSource_declarations;
		ctx.meta->m_global.m_materialCompilerGLSL_source = compResult.fragmentShaderSource;
		ctx.meta->m_global.m_ds0 = ds0;

		auto meshSmartPtrArray = core::make_refctd_dynamic_array<SAssetBundle::contents_container_t>(meshes.size());
		auto meshSmartPtrArrayIt = meshSmartPtrArray->begin();
		for (const auto& mesh_ : meshes)
		{
			for (auto mb : mesh_.first.get()->getMeshBuffers())
			{
				const auto* prevPipeline = mb->getPipeline();
				SContext::SPipelineCacheKey cacheKey;
				cacheKey.vtxParams = prevPipeline->getVertexInputParams();
				cacheKey.primParams = prevPipeline->getPrimitiveAssemblyParams();
				auto found = ctx.pipelineCache.find(cacheKey);
				core::smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline> pipeline;
				if (found != ctx.pipelineCache.end())
				{
					pipeline = found->second;
				}
				else
				{
					pipeline = core::smart_refctd_ptr_static_cast<ICPURenderpassIndependentPipeline>(//shallow copy because we're only going to override parameter structs
						basePipeline->clone(0u)
						);
					pipeline->getVertexInputParams() = cacheKey.vtxParams;
					pipeline->getPrimitiveAssemblyParams() = cacheKey.primParams;
					ctx.pipelineCache.insert({ cacheKey, pipeline });
				}

				mb->setPipeline(core::smart_refctd_ptr(pipeline));
			}
			*(meshSmartPtrArrayIt++) = std::move(mesh_.first);
		}

		parserManager.m_metadata->reservePplnStorage(ctx.pipelineCache.size(),core::smart_refctd_ptr(IRenderpassIndependentPipelineLoader::m_basicViewParamsSemantics));
		for (auto& ppln : ctx.pipelineCache)
			parserManager.m_metadata->addPplnMeta(ppln.second.get(),core::smart_refctd_ptr(ds0));

		return asset::SAssetBundle(std::move(parserManager.m_metadata),std::move(meshSmartPtrArray));
	}
}

core::vector<SContext::shape_ass_type> CMitsubaLoader::getMesh(SContext& ctx, uint32_t hierarchyLevel, CElementShape* shape)
{
	if (!shape)
		return {};

	if (shape->type!=CElementShape::Type::INSTANCE)
		return {loadBasicShape(ctx, hierarchyLevel, shape, core::matrix3x4SIMD())};
	else
	{
		core::matrix3x4SIMD relTform = shape->getAbsoluteTransform();
		// get group reference
		const CElementShape* parent = shape->instance.parent;
		if (!parent)
			return {};
		assert(parent->type==CElementShape::Type::SHAPEGROUP);
		const CElementShape::ShapeGroup* shapegroup = &parent->shapegroup;
		
		return loadShapeGroup(ctx, hierarchyLevel, shapegroup, relTform);
	}
}

core::vector<SContext::shape_ass_type> CMitsubaLoader::loadShapeGroup(SContext& ctx, uint32_t hierarchyLevel, const CElementShape::ShapeGroup* shapegroup, const core::matrix3x4SIMD& relTform)
{
	// @Crisspl why no group cache?
	// find group
	//auto found = ctx.groupCache.find(shapegroup);
	//if (found != ctx.groupCache.end())
	//	return found->second;

	const auto children = shapegroup->children;

	core::vector<SContext::shape_ass_type> meshes;
	for (auto i=0u; i<shapegroup->childCount; i++)
	{
		auto child = children[i];
		if (!child)
			continue;

		assert(child->type!=CElementShape::Type::INSTANCE);
		if (child->type != CElementShape::Type::SHAPEGROUP) {
			auto lowermesh = loadBasicShape(ctx, hierarchyLevel, child, relTform);
			meshes.push_back(std::move(lowermesh));
		}
		else {
			auto lowermeshes = loadShapeGroup(ctx, hierarchyLevel, &child->shapegroup, relTform);
			meshes.insert(meshes.begin(), std::make_move_iterator(lowermeshes.begin()), std::make_move_iterator(lowermeshes.end()));
		}
	}

	//ctx.groupCache.insert({shapegroup,meshes});
	return meshes;
}

static core::smart_refctd_ptr<ICPUMesh> createMeshFromGeomCreatorReturnType(IGeometryCreator::return_type&& _data, asset::IAssetManager* _manager)
{
	//creating pipeline just to forward vtx and primitive params
	auto pipeline = core::make_smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>(
		nullptr, nullptr, nullptr, //no layout nor shaders
		_data.inputParams, 
		asset::SBlendParams(),
		_data.assemblyParams,
		asset::SRasterizationParams()
		);

	auto mb = core::make_smart_refctd_ptr<ICPUMeshBuffer>(
		nullptr, nullptr,
		_data.bindings, std::move(_data.indexBuffer)
	);
	mb->setIndexCount(_data.indexCount);
	mb->setIndexType(_data.indexType);
	mb->setBoundingBox(_data.bbox);
	mb->setPipeline(std::move(pipeline));
	constexpr auto NORMAL_ATTRIBUTE = 3;
	mb->setNormalAttributeIx(NORMAL_ATTRIBUTE);

	auto mesh = core::make_smart_refctd_ptr<ICPUMesh>();
	mesh->getMeshBufferVector().push_back(std::move(mb));

	return mesh;
}

SContext::shape_ass_type CMitsubaLoader::loadBasicShape(SContext& ctx, uint32_t hierarchyLevel, CElementShape* shape, const core::matrix3x4SIMD& relTform)
{
	constexpr uint32_t UV_ATTRIB_ID = 2u;

	auto addInstance = [shape,&ctx,&relTform,this](SContext::shape_ass_type& mesh)
	{
		auto bsdf = getBSDFtreeTraversal(ctx, shape->bsdf);
		core::matrix3x4SIMD tform = core::concatenateBFollowedByA(relTform, shape->getAbsoluteTransform());
		SContext::SInstanceData instance(
			tform,
			bsdf,
#if defined(_NBL_DEBUG) || defined(_NBL_RELWITHDEBINFO)
			shape->bsdf ? shape->bsdf->id:"",
#endif
			shape->obtainEmitter(),
			CElementEmitter{} // TODO: does enabling a twosided BRDF make the emitter twosided?
		);
		ctx.mapMesh2instanceData.insert({ mesh.get(), instance });
	};

	auto found = ctx.shapeCache.find(shape);
	if (found != ctx.shapeCache.end()) {
		addInstance(found->second);

		return found->second;
	}

	auto loadModel = [&](const ext::MitsubaLoader::SPropertyElementData& filename, int64_t index=-1) -> core::smart_refctd_ptr<asset::ICPUMesh>
	{
		assert(filename.type==ext::MitsubaLoader::SPropertyElementData::Type::STRING);
		auto loadParams = ctx.inner.params;
		loadParams.loaderFlags = static_cast<IAssetLoader::E_LOADER_PARAMETER_FLAGS>(loadParams.loaderFlags | IAssetLoader::ELPF_RIGHT_HANDED_MESHES);
		auto retval = interm_getAssetInHierarchy(m_assetMgr, filename.svalue, loadParams, hierarchyLevel/*+ICPUScene::MESH_HIERARCHY_LEVELS_BELOW*/, ctx.override_);
		if (retval.getAssetType()!=asset::IAsset::ET_MESH)
			return nullptr;
		auto contentRange = retval.getContents();
		auto serializedMeta = retval.getMetadata()->selfCast<CMitsubaSerializedMetadata>();
		//
		uint32_t actualIndex = 0;
		if (index>=0ll && serializedMeta)
		for (auto it=contentRange.begin(); it!=contentRange.end(); it++)
		{
			auto meshMeta = static_cast<const CMitsubaSerializedMetadata::CMesh*>(serializedMeta->getAssetSpecificMetadata(IAsset::castDown<ICPUMesh>(*it).get()));
			if (meshMeta->m_id!=static_cast<uint32_t>(index))
				continue;
			actualIndex = it-contentRange.begin();
			break;
		}
		//
		if (contentRange.begin()+actualIndex < contentRange.end())
		{
			auto asset = contentRange.begin()[actualIndex];
			if (!asset)
				return nullptr;
			return core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(asset);
		}
		else
			return nullptr;
	};

	core::smart_refctd_ptr<asset::ICPUMesh> mesh,newMesh;
	bool flipNormals = false;
	bool faceNormals = false;
	float maxSmoothAngle = NAN;
	switch (shape->type)
	{
		case CElementShape::Type::CUBE:
		{
			auto cubeData = ctx.creator->createCubeMesh(core::vector3df(2.f));

			mesh = createMeshFromGeomCreatorReturnType(ctx.creator->createCubeMesh(core::vector3df(2.f)), m_assetMgr);
			flipNormals = flipNormals!=shape->cube.flipNormals;
		}
			break;
		case CElementShape::Type::SPHERE:
			mesh = createMeshFromGeomCreatorReturnType(ctx.creator->createSphereMesh(1.f,64u,64u), m_assetMgr);
			flipNormals = flipNormals!=shape->sphere.flipNormals;
			{
				core::matrix3x4SIMD tform;
				tform.setScale(core::vectorSIMDf(shape->sphere.radius,shape->sphere.radius,shape->sphere.radius));
				tform.setTranslation(shape->sphere.center);
				shape->transform.matrix = core::concatenateBFollowedByA(shape->transform.matrix,core::matrix4SIMD(tform));
			}
			break;
		case CElementShape::Type::CYLINDER:
			{
				auto diff = shape->cylinder.p0-shape->cylinder.p1;
				mesh = createMeshFromGeomCreatorReturnType(ctx.creator->createCylinderMesh(1.f, 1.f, 64), m_assetMgr);
				core::vectorSIMDf up(0.f);
				float maxDot = diff[0];
				uint32_t index = 0u;
				for (auto i = 1u; i < 3u; i++)
					if (diff[i] < maxDot)
					{
						maxDot = diff[i];
						index = i;
					}
				up[index] = 1.f;
				core::matrix3x4SIMD tform;
				// mesh is left haded so transforming by LH matrix is fine (I hope but lets check later on)
				core::matrix3x4SIMD::buildCameraLookAtMatrixLH(shape->cylinder.p0,shape->cylinder.p1,up).getInverse(tform);
				core::matrix3x4SIMD scale;
				scale.setScale(core::vectorSIMDf(shape->cylinder.radius,shape->cylinder.radius,core::length(diff).x));
				shape->transform.matrix = core::concatenateBFollowedByA(shape->transform.matrix,core::matrix4SIMD(core::concatenateBFollowedByA(tform,scale)));
			}
			flipNormals = flipNormals!=shape->cylinder.flipNormals;
			break;
		case CElementShape::Type::RECTANGLE:
			mesh = createMeshFromGeomCreatorReturnType(ctx.creator->createRectangleMesh(core::vector2df_SIMD(1.f,1.f)), m_assetMgr);
			flipNormals = flipNormals!=shape->rectangle.flipNormals;
			break;
		case CElementShape::Type::DISK:
			mesh = createMeshFromGeomCreatorReturnType(ctx.creator->createDiskMesh(1.f,64u), m_assetMgr);
			flipNormals = flipNormals!=shape->disk.flipNormals;
			break;
		case CElementShape::Type::OBJ:
			mesh = loadModel(shape->obj.filename);
			flipNormals = flipNormals!=shape->obj.flipNormals;
			faceNormals = shape->obj.faceNormals;
			maxSmoothAngle = shape->obj.maxSmoothAngle;
			if (mesh && shape->obj.flipTexCoords)
			{
				newMesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh> (mesh->clone(1u));
				for (auto& meshbuffer : mesh->getMeshBufferVector())
				{
					auto binding = meshbuffer->getVertexBufferBindings()[UV_ATTRIB_ID];
					if (binding.buffer)
					{
						binding.buffer = core::smart_refctd_ptr_static_cast<ICPUBuffer>(binding.buffer->clone(0u));
						meshbuffer->setVertexBufferBinding(std::move(binding),UV_ATTRIB_ID);
						core::vectorSIMDf uv;
						for (uint32_t i=0u; meshbuffer->getAttribute(uv,UV_ATTRIB_ID,i); i++)
						{
							uv.y = -uv.y;
							meshbuffer->setAttribute(uv,UV_ATTRIB_ID,i);
						}
					}
				}
			}
			// collapse parameter gets ignored
			break;
		case CElementShape::Type::PLY:
			_NBL_DEBUG_BREAK_IF(true); // this code has never been tested
			mesh = loadModel(shape->ply.filename);
			flipNormals = flipNormals!=shape->ply.flipNormals;
			faceNormals = shape->ply.faceNormals;
			maxSmoothAngle = shape->ply.maxSmoothAngle;
			if (mesh && shape->ply.srgb)
			{
				uint32_t totalVertexCount = 0u;
				for (auto meshbuffer : mesh->getMeshBuffers())
					totalVertexCount += IMeshManipulator::upperBoundVertexID(meshbuffer);
				if (totalVertexCount)
				{
					constexpr uint32_t hidefRGBSize = 4u;
					auto newRGBbuff = core::make_smart_refctd_ptr<asset::ICPUBuffer>(hidefRGBSize*totalVertexCount);
					newMesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(mesh->clone(1u));
					constexpr uint32_t COLOR_ATTR = 1u;
					constexpr uint32_t COLOR_BUF_BINDING = 15u;
					uint32_t* newRGB = reinterpret_cast<uint32_t*>(newRGBbuff->getPointer());
					uint32_t offset = 0u;
					for (auto& meshbuffer : mesh->getMeshBufferVector())
					{
						core::vectorSIMDf rgb;
						for (uint32_t i=0u; meshbuffer->getAttribute(rgb,COLOR_ATTR,i); i++,offset++)
						{
							for (auto i=0; i<3u; i++)
								rgb[i] = core::srgb2lin(rgb[i]);
							ICPUMeshBuffer::setAttribute(rgb,newRGB+offset,asset::EF_A2B10G10R10_UNORM_PACK32);
						}
						auto newPipeline = core::smart_refctd_ptr_static_cast<ICPURenderpassIndependentPipeline>(meshbuffer->getPipeline()->clone(0u));
						auto& vtxParams = newPipeline->getVertexInputParams();
						vtxParams.attributes[COLOR_ATTR].format = EF_A2B10G10R10_UNORM_PACK32;
						vtxParams.attributes[COLOR_ATTR].relativeOffset = 0u;
						vtxParams.attributes[COLOR_ATTR].binding = COLOR_BUF_BINDING;
						vtxParams.bindings[COLOR_BUF_BINDING].inputRate = EVIR_PER_VERTEX;
						vtxParams.bindings[COLOR_BUF_BINDING].stride = hidefRGBSize;
						vtxParams.enabledBindingFlags |= (1u<<COLOR_BUF_BINDING);
						meshbuffer->setPipeline(std::move(newPipeline));
						meshbuffer->setVertexBufferBinding({offset*hidefRGBSize,core::smart_refctd_ptr(newRGBbuff)},COLOR_BUF_BINDING);
					}
				}
			}
			break;
		case CElementShape::Type::SERIALIZED:
			mesh = loadModel(shape->serialized.filename,shape->serialized.shapeIndex);
			flipNormals = flipNormals!=shape->serialized.flipNormals;
			faceNormals = shape->serialized.faceNormals;
			maxSmoothAngle = shape->serialized.maxSmoothAngle;
			break;
		case CElementShape::Type::SHAPEGROUP:
			[[fallthrough]];
		case CElementShape::Type::INSTANCE:
			assert(false);
			break;
		default:
			_NBL_DEBUG_BREAK_IF(true);
			break;
	}
	//
	if (!mesh)
		return nullptr;

	// mesh including meshbuffers needs to be cloned because instance counts and base instances will be changed
	if (!newMesh)
		newMesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(mesh->clone(1u));
	// flip normals if necessary
	if (flipNormals)
	{
		for (auto& meshbuffer : mesh->getMeshBufferVector())
		{
			auto binding = meshbuffer->getIndexBufferBinding();
			binding.buffer = core::smart_refctd_ptr_static_cast<ICPUBuffer>(binding.buffer->clone(0u));
			meshbuffer->setIndexBufferBinding(std::move(binding));
			ctx.manipulator->flipSurfaces(meshbuffer.get());
		}
	}
	// recompute normalis if necessary
	if (faceNormals || !std::isnan(maxSmoothAngle))
	for (auto& meshbuffer : mesh->getMeshBufferVector())
	{
		const float smoothAngleCos = cos(core::radians(maxSmoothAngle));

		// TODO: make these mesh manipulator functions const-correct
		auto newMeshBuffer = ctx.manipulator->createMeshBufferUniquePrimitives(meshbuffer.get());
		ctx.manipulator->filterInvalidTriangles(newMeshBuffer.get());
		ctx.manipulator->calculateSmoothNormals(newMeshBuffer.get(), false, 0.f, newMeshBuffer->getNormalAttributeIx(),
			[&](const asset::IMeshManipulator::SSNGVertexData& a, const asset::IMeshManipulator::SSNGVertexData& b, asset::ICPUMeshBuffer* buffer)
			{
				if (faceNormals)
					return a.indexOffset == b.indexOffset;
				else
					return core::dot(a.parentTriangleFaceNormal, b.parentTriangleFaceNormal).x >= smoothAngleCos;
			});
		meshbuffer = std::move(newMeshBuffer);
	}
	IMeshManipulator::recalculateBoundingBox(newMesh.get());
	mesh = std::move(newMesh);

	addInstance(mesh);
	// cache and return
	ctx.shapeCache.insert({ shape,mesh });
	return mesh;
}

SContext::tex_ass_type CMitsubaLoader::cacheTexture(SContext& ctx, uint32_t hierarchyLevel, const CElementTexture* tex, bool _restore)
{
	if (!tex)
		return {};

	ICPUImageView::SCreationParams viewParams;
	viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0);
	viewParams.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);
	viewParams.subresourceRange.baseArrayLayer = 0u;
	viewParams.subresourceRange.layerCount = 1u;
	viewParams.subresourceRange.baseMipLevel = 0u;
	viewParams.viewType = IImageView<ICPUImage>::ET_2D;
	ICPUSampler::SParams samplerParams;
	samplerParams.AnisotropicFilter = core::max(core::findMSB(uint32_t(tex->bitmap.maxAnisotropy)),1);
	samplerParams.LodBias = 0.f;
	samplerParams.TextureWrapW = ISampler::ETC_REPEAT;
	samplerParams.BorderColor = ISampler::ETBC_FLOAT_OPAQUE_BLACK;
	samplerParams.CompareEnable = false;
	samplerParams.CompareFunc = ISampler::ECO_NEVER;
	samplerParams.MaxLod = 10000.f;
	samplerParams.MinLod = 0.f;

	switch (tex->type)
	{
		case CElementTexture::Type::BITMAP:
		{
				std::string cacheKey = ctx.imageViewCacheKey(tex->bitmap.filename.svalue);
				switch (tex->bitmap.channel)
				{
				case CElementTexture::Bitmap::CHANNEL::R:
					cacheKey += "?r";
					break;
				case CElementTexture::Bitmap::CHANNEL::G:
					cacheKey += "?g";
					break;
				case CElementTexture::Bitmap::CHANNEL::B:
					cacheKey += "?b";
					break;
				case CElementTexture::Bitmap::CHANNEL::A:
					cacheKey += "?a";
					break;
				default:
					break;
				}

				core::smart_refctd_ptr<asset::ICPUImageView> view;
				{
					asset::SAssetBundle viewBundle;
					const asset::IAsset::E_TYPE types[]{ asset::IAsset::ET_IMAGE_VIEW, static_cast<asset::IAsset::E_TYPE>(0) };
					viewBundle = ctx.override_->findCachedAsset(cacheKey, types, ctx.inner, 0u);

					auto contents = viewBundle.getContents();
					if (!contents.empty())
						view = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(contents.begin()[0]);
					if (view)
					{
						auto& image = view->getCreationParameters().image;
						if (_restore && image->isADummyObjectForCache())
						{
							auto loadParams = ctx.inner.params;
							loadParams.restoreLevels = std::max(loadParams.restoreLevels, hierarchyLevel + 2u);
							// this will restore the image being kept by found `view`
							auto bundle = interm_getAssetInHierarchy(m_assetMgr, tex->bitmap.filename.svalue, loadParams, hierarchyLevel, ctx.override_);
							if (bundle.getContents().empty() || image->isADummyObjectForCache())
							{
								// if for some reason restore failed, force recreating whole view
								auto removeBundle = asset::SAssetBundle(nullptr, { view });
								m_assetMgr->removeAssetFromCache(removeBundle);
								view = nullptr;
							}
						}
					}
				}

				core::smart_refctd_ptr<asset::ICPUImage> img;
				if (!view)
				{
					const uint32_t restoreLevels = _restore ? 2u : 0u;
					auto loadParams = ctx.inner.params;
					loadParams.restoreLevels = std::max(loadParams.restoreLevels, hierarchyLevel + restoreLevels);
					asset::SAssetBundle imgBundle = interm_getAssetInHierarchy(m_assetMgr,tex->bitmap.filename.svalue,loadParams,hierarchyLevel,ctx.override_);
					auto contentRange = imgBundle.getContents();
					if (contentRange.begin() < contentRange.end())
					{
						auto asset = contentRange.begin()[0];
						if (asset && asset->getAssetType() == asset::IAsset::ET_IMAGE)
						{
							img = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(asset);

							switch (tex->bitmap.channel)
							{
								// no GL_R8_SRGB support yet
								case CElementTexture::Bitmap::CHANNEL::R:
									{
									constexpr auto RED = ICPUImageView::SComponentMapping::ES_R;
									viewParams.components = {RED,RED,RED,RED};
									}
									break;
								case CElementTexture::Bitmap::CHANNEL::G:
									{
									constexpr auto GREEN = ICPUImageView::SComponentMapping::ES_G;
									viewParams.components = {GREEN,GREEN,GREEN,GREEN};
									}
									break;
								case CElementTexture::Bitmap::CHANNEL::B:
									{
									constexpr auto BLUE = ICPUImageView::SComponentMapping::ES_B;
									viewParams.components = {BLUE,BLUE,BLUE,BLUE};
									}
									break;
								case CElementTexture::Bitmap::CHANNEL::A:
									{
									constexpr auto ALPHA = ICPUImageView::SComponentMapping::ES_A;
									viewParams.components = {ALPHA,ALPHA,ALPHA,ALPHA};
									}
									break;
								/* special conversions needed to CIE space
								case CElementTexture::Bitmap::CHANNEL::X:
								case CElementTexture::Bitmap::CHANNEL::Y:
								case CElementTexture::Bitmap::CHANNEL::Z:*/
								case CElementTexture::Bitmap::CHANNEL::INVALID:
									[[fallthrough]];
								default:
									break;
							}
							viewParams.subresourceRange.levelCount = img->getCreationParameters().mipLevels;
							viewParams.format = img->getCreationParameters().format;
							viewParams.image = std::move(img);
							//! TODO: this stuff (custom shader sampling code?)
							_NBL_DEBUG_BREAK_IF(tex->bitmap.uoffset != 0.f);
							_NBL_DEBUG_BREAK_IF(tex->bitmap.voffset != 0.f);
							_NBL_DEBUG_BREAK_IF(tex->bitmap.uscale != 1.f);
							_NBL_DEBUG_BREAK_IF(tex->bitmap.vscale != 1.f);
						}

						//in case of <channel>, extract one channel
						if (viewParams.components.g != asset::ICPUImageView::SComponentMapping::ES_G)
						{
							auto get1ChannelFormat = [](uint32_t bytesPerChannel) -> asset::E_FORMAT {
								switch (bytesPerChannel)
								{
								case 1u:
									return asset::EF_R8_UNORM;
								case 2u:
									return asset::EF_R16_SFLOAT;
								case 4u:
									return asset::EF_R32_SFLOAT;
								case 8u:
									return asset::EF_R64_SFLOAT;
								default:
									return asset::EF_UNKNOWN;
								}
							};

							auto outParams = viewParams.image->getCreationParameters();
							asset::ICPUImage::SBufferCopy region;
							const uint32_t bytesPerChannel = (getBytesPerPixel(outParams.format) * core::rational(1, getFormatChannelCount(outParams.format))).getIntegerApprox();
							outParams.format = get1ChannelFormat(bytesPerChannel);
							const size_t texelBytesz = asset::getTexelOrBlockBytesize(outParams.format);
							region.bufferRowLength = asset::IImageAssetHandlerBase::calcPitchInBlocks(outParams.extent.width, texelBytesz);
							auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(texelBytesz * region.bufferRowLength * outParams.extent.height);
							region.imageOffset = { 0,0,0 };
							region.imageExtent = outParams.extent;
							region.imageSubresource.baseArrayLayer = 0u;
							region.imageSubresource.layerCount = 1u;
							region.imageSubresource.mipLevel = 0u;
							region.bufferImageHeight = 0u;
							region.bufferOffset = 0u;
							auto outImg = asset::ICPUImage::create(std::move(outParams));
							outImg->setBufferAndRegions(std::move(buffer), core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy>>(1ull, region));

							using convert_filter_t = asset::CSwizzleAndConvertImageFilter<asset::EF_UNKNOWN, asset::EF_UNKNOWN>;
							convert_filter_t::state_type conv;
							conv.swizzle = viewParams.components;
							conv.extent = viewParams.image->getCreationParameters().extent;
							conv.layerCount = 1u;
							conv.inMipLevel = 0u;
							conv.outMipLevel = 0u;
							conv.inBaseLayer = 0u;
							conv.outBaseLayer = 0u;
							conv.inOffset = { 0u,0u,0u };
							conv.outOffset = { 0u,0u,0u };
							conv.inImage = viewParams.image.get();
							conv.outImage = outImg.get();

							viewParams.components = asset::ICPUImageView::SComponentMapping{};
							if (!convert_filter_t::execute(std::execution::par_unseq,&conv))
								_NBL_DEBUG_BREAK_IF(true);
							viewParams.format = outImg->getCreationParameters().format;
							viewParams.image = std::move(outImg);
						}

						view = ICPUImageView::create(std::move(viewParams));
						asset::SAssetBundle viewBundle(nullptr,{ view });
						ctx.override_->insertAssetIntoCache(std::move(viewBundle), cacheKey, ctx.inner, hierarchyLevel);
					}

					// adjust gamma on pixels (painful and long process)
					if (!std::isnan(tex->bitmap.gamma))
					{
						_NBL_DEBUG_BREAK_IF(true); // TODO : use an image filter!
					}
				}

				const std::string samplerCacheKey = ctx.samplerCacheKey(tex);
				switch (tex->bitmap.filterType)
				{
					case CElementTexture::Bitmap::FILTER_TYPE::EWA:
						[[fallthrough]]; // we dont support this fancy stuff
					case CElementTexture::Bitmap::FILTER_TYPE::TRILINEAR:
						samplerParams.MinFilter = ISampler::ETF_LINEAR;
						samplerParams.MaxFilter = ISampler::ETF_LINEAR;
						samplerParams.MipmapMode = ISampler::ESMM_LINEAR;
						break;
					default:
						samplerParams.MinFilter = ISampler::ETF_NEAREST;
						samplerParams.MaxFilter = ISampler::ETF_NEAREST;
						samplerParams.MipmapMode = ISampler::ESMM_NEAREST;
						break;
				}
				auto getWrapMode = [&samplerCacheKey](CElementTexture::Bitmap::WRAP_MODE mode)
				{
					switch (mode)
					{
						case CElementTexture::Bitmap::WRAP_MODE::CLAMP:
							return ISampler::ETC_CLAMP_TO_EDGE;
							break;
						case CElementTexture::Bitmap::WRAP_MODE::MIRROR:
							return ISampler::ETC_MIRROR;
							break;
						case CElementTexture::Bitmap::WRAP_MODE::ONE:
							_NBL_DEBUG_BREAK_IF(true); // TODO : replace whole texture?
							break;
						case CElementTexture::Bitmap::WRAP_MODE::ZERO:
							_NBL_DEBUG_BREAK_IF(true); // TODO : replace whole texture?
							break;
						default:
							break;
					}
					return ISampler::ETC_REPEAT;
				};
				samplerParams.TextureWrapU = getWrapMode(tex->bitmap.wrapModeU);
				samplerParams.TextureWrapV = getWrapMode(tex->bitmap.wrapModeV);

				core::smart_refctd_ptr<ICPUSampler> sampler;
				{
					const asset::IAsset::E_TYPE types[]{ asset::IAsset::ET_SAMPLER, static_cast<asset::IAsset::E_TYPE>(0) };
					auto samplerBundle = ctx.override_->findCachedAsset(samplerCacheKey, types, ctx.inner, 0u);
					auto contents = samplerBundle.getContents();
					if (!contents.empty())
						sampler = core::smart_refctd_ptr_static_cast<asset::ICPUSampler>(contents.begin()[0]);
				}

				if (!view)
					sampler = nullptr;
				else if (!sampler)
				{
					sampler = core::make_smart_refctd_ptr<ICPUSampler>(samplerParams);
					SAssetBundle samplerBundle(nullptr,{ sampler });
					ctx.override_->insertAssetIntoCache(std::move(samplerBundle), samplerCacheKey, ctx.inner, hierarchyLevel);
				}

				SContext::tex_ass_type tex_ass(std::move(view), std::move(sampler));

				return tex_ass;
		}
			break;
		case CElementTexture::Type::SCALE:
		{
			return cacheTexture(ctx,hierarchyLevel,tex->scale.texture);
		}
			break;
		default:
			_NBL_DEBUG_BREAK_IF(true);
			return SContext::tex_ass_type{nullptr,nullptr};
			break;
	}
}

auto CMitsubaLoader::getBSDFtreeTraversal(SContext& ctx, const CElementBSDF* bsdf) -> SContext::bsdf_type
{
	if (!bsdf)
		return {nullptr,nullptr};

	auto found = ctx.instrStreamCache.find(bsdf);
	if (found!=ctx.instrStreamCache.end())
		return found->second;
	auto retval = genBSDFtreeTraversal(ctx, bsdf);
	ctx.instrStreamCache.insert({bsdf,retval});
	return retval;
}

auto CMitsubaLoader::genBSDFtreeTraversal(SContext& ctx, const CElementBSDF* _bsdf) -> SContext::bsdf_type
{
	{
		auto cachePropertyTexture = [&](const auto& const_or_tex, SContext::tex_ass_type& out_img) {
			bool is_tex = (const_or_tex.value.type == SPropertyElementData::INVALID);
			if (const_or_tex.value.type == SPropertyElementData::INVALID)
				out_img = cacheTexture(ctx, 0u, const_or_tex.texture);

			return is_tex;
		};
		auto unrollScales = [](CElementTexture* tex)
		{
			while (tex->type == CElementTexture::SCALE)
				tex = tex->scale.texture;
			return tex;
		};

		core::stack<const CElementBSDF*> stack;
		stack.push(_bsdf);

		while (!stack.empty())
		{
			auto* bsdf = stack.top();
			stack.pop();
			switch (bsdf->type)
			{
			case CElementBSDF::COATING:
			case CElementBSDF::ROUGHCOATING:
			case CElementBSDF::BUMPMAP:
			case CElementBSDF::BLEND_BSDF:
			case CElementBSDF::MIXTURE_BSDF:
			case CElementBSDF::MASK:
			case CElementBSDF::TWO_SIDED:
				for (uint32_t i = 0u; i < bsdf->meta_common.childCount; ++i)
					stack.push(bsdf->meta_common.bsdf[i]);
			default: break;
			}

			SContext::tex_ass_type tex;
			switch (bsdf->type)
			{
			case CElementBSDF::DIFFUSE:
			case CElementBSDF::ROUGHDIFFUSE:
				cachePropertyTexture(bsdf->diffuse.reflectance, tex);
				cachePropertyTexture(bsdf->diffuse.alpha, tex);
				break;
			case CElementBSDF::DIFFUSE_TRANSMITTER:
				cachePropertyTexture(bsdf->difftrans.transmittance, tex);
				break;
			case CElementBSDF::DIELECTRIC:
			case CElementBSDF::THINDIELECTRIC:
			case CElementBSDF::ROUGHDIELECTRIC:
				cachePropertyTexture(bsdf->dielectric.alphaU, tex);
				if (bsdf->dielectric.distribution == CElementBSDF::RoughSpecularBase::ASHIKHMIN_SHIRLEY)
					cachePropertyTexture(bsdf->dielectric.alphaV, tex);
				break;
			case CElementBSDF::CONDUCTOR:
				cachePropertyTexture(bsdf->conductor.alphaU, tex);
				if (bsdf->conductor.distribution == CElementBSDF::RoughSpecularBase::ASHIKHMIN_SHIRLEY)
					cachePropertyTexture(bsdf->conductor.alphaV, tex);
				break;
			case CElementBSDF::PLASTIC:
			case CElementBSDF::ROUGHPLASTIC:
				cachePropertyTexture(bsdf->plastic.diffuseReflectance, tex);
				cachePropertyTexture(bsdf->plastic.alphaU, tex);
				if (bsdf->plastic.distribution == CElementBSDF::RoughSpecularBase::ASHIKHMIN_SHIRLEY)
					cachePropertyTexture(bsdf->plastic.alphaV, tex);
				break;
			case CElementBSDF::BUMPMAP:
			{
				using namespace std::string_literals;

				auto* bumpmap_element = unrollScales(bsdf->bumpmap.texture);
				auto bm = cacheTexture(ctx, 0u, bumpmap_element);
				// TODO check and restore if dummy (image and sampler)
				auto bumpmap = std::get<0>(bm)->getCreationParameters().image;
				auto sampler = std::get<1>(bm);
				const std::string key = ctx.derivMapCacheKey(bumpmap_element);

				if (!getBuiltinAsset<asset::ICPUImage, asset::IAsset::ET_IMAGE>(key.c_str(), m_assetMgr))
				{
					auto derivmap = createDerivMap(bumpmap.get(), sampler.get());
					asset::SAssetBundle imgBundle(nullptr,{ derivmap });
					ctx.override_->insertAssetIntoCache(std::move(imgBundle), key, ctx.inner, 0u);
					auto derivmap_view = createImageView(std::move(derivmap));
					asset::SAssetBundle viewBundle(nullptr,{ derivmap_view });
					ctx.override_->insertAssetIntoCache(std::move(viewBundle), ctx.imageViewCacheKey(key), ctx.inner, 0u);
				}
			}
				break;
			case CElementBSDF::BLEND_BSDF:
				if (cachePropertyTexture(bsdf->blendbsdf.weight, tex))
				{
					auto* weight_element = unrollScales(bsdf->blendbsdf.weight.texture);
					const std::string key = ctx.blendWeightImageCacheKey(weight_element);

					if (!getBuiltinAsset<asset::ICPUImage, asset::IAsset::ET_IMAGE>(key.c_str(), m_assetMgr))
					{
						auto img = std::get<0>(tex)->getCreationParameters().image;

						auto blendweight = createBlendWeightImage(img.get());
						asset::SAssetBundle imgBundle(nullptr,{ blendweight });
						ctx.override_->insertAssetIntoCache(std::move(imgBundle), key, ctx.inner, 0u);
						auto blendweight_view = createImageView(std::move(blendweight));
						asset::SAssetBundle viewBundle(nullptr,{ blendweight_view });
						ctx.override_->insertAssetIntoCache(std::move(viewBundle), ctx.imageViewCacheKey(key), ctx.inner, 0u);
					}
				}
				break;
			case CElementBSDF::MASK:
				cachePropertyTexture(bsdf->mask.opacity, tex);
				break;
			default: break;
			}
		}
	}

	return ctx.frontend.compileToIRTree(ctx.ir.get(), _bsdf);
}



// TODO: this function shouldn't really exist because the backend should produce this directly @Crisspl
asset::material_compiler::oriented_material_t impl_backendToGLSLStream(const core::vectorSIMDf& emissive, const asset::material_compiler::CMaterialCompilerGLSLBackendCommon::result_t::instr_streams_t& streams)
{
	asset::material_compiler::oriented_material_t orientedMaterial;
	orientedMaterial.emissive = core::rgb32f_to_rgb19e7(emissive.pointer);
	orientedMaterial.prefetch_offset = streams.prefetch_offset;
	orientedMaterial.prefetch_count = streams.tex_prefetch_count;
	orientedMaterial.instr_offset = streams.offset;
	orientedMaterial.rem_pdf_count = streams.rem_and_pdf_count;
	orientedMaterial.nprecomp_count = streams.norm_precomp_count;
	orientedMaterial.genchoice_count = streams.gen_choice_count;
	return orientedMaterial;
}


// Also sets instance data buffer offset into meshbuffers' base instance
template<typename Iter>
inline core::smart_refctd_ptr<asset::ICPUDescriptorSet> CMitsubaLoader::createDS0(const SContext& _ctx, asset::ICPUPipelineLayout* _layout, const asset::material_compiler::CMaterialCompilerGLSLBackendCommon::result_t& _compResult, Iter meshBegin, Iter meshEnd)
{
	auto* ds0layout = _layout->getDescriptorSetLayout(0u);

	auto ds0 = core::make_smart_refctd_ptr<ICPUDescriptorSet>(core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(ds0layout));
	{
		auto count = _ctx.backend_ctx.vt.vt->getDescriptorSetWrites(nullptr, nullptr, nullptr);

		auto writes = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUDescriptorSet::SWriteDescriptorSet>>(count.first);
		auto info = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUDescriptorSet::SDescriptorInfo>>(count.second);

		_ctx.backend_ctx.vt.vt->getDescriptorSetWrites(writes->data(), info->data(), ds0.get());

		for (const auto& w : (*writes))
		{
			auto descRng = ds0->getDescriptors(w.binding);
			for (uint32_t i = 0u; i < w.count; ++i)
				descRng.begin()[w.arrayElement+i].assign(w.info[i], w.descriptorType);
		}
	}
	auto d = ds0->getDescriptors(PRECOMPUTED_VT_DATA_BINDING).begin();
	{
		auto precompDataBuf = core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(asset::ICPUVirtualTexture::SPrecomputedData));
		memcpy(precompDataBuf->getPointer(), &_ctx.backend_ctx.vt.vt->getPrecomputedData(), precompDataBuf->getSize());

		d->buffer.offset = 0u;
		d->buffer.size = precompDataBuf->getSize();
		d->desc = std::move(precompDataBuf);
	}
	d = ds0->getDescriptors(INSTR_BUF_BINDING).begin();
	{
		auto instrbuf = core::make_smart_refctd_ptr<ICPUBuffer>(_compResult.instructions.size()*sizeof(decltype(_compResult.instructions)::value_type));
		memcpy(instrbuf->getPointer(), _compResult.instructions.data(), instrbuf->getSize());

		d->buffer.offset = 0u;
		d->buffer.size = instrbuf->getSize();
		d->desc = std::move(instrbuf);
	}
	d = ds0->getDescriptors(BSDF_BUF_BINDING).begin();
	{
		auto bsdfbuf = core::make_smart_refctd_ptr<ICPUBuffer>(_compResult.bsdfData.size()*sizeof(decltype(_compResult.bsdfData)::value_type));
		memcpy(bsdfbuf->getPointer(), _compResult.bsdfData.data(), bsdfbuf->getSize());

		d->buffer.offset = 0u;
		d->buffer.size = bsdfbuf->getSize();
		d->desc = std::move(bsdfbuf);
	}
	d = ds0->getDescriptors(PREFETCH_INSTR_BUF_BINDING).begin();
	{
		const size_t sz = _compResult.prefetch_stream.size()*sizeof(decltype(_compResult.prefetch_stream)::value_type);
		
		constexpr size_t MIN_SSBO_SZ = 128ull; //prefetch stream won't be generated if no textures are used, so make sure we're not creating 0-size buffer
		auto prefetch_instr_buf = core::make_smart_refctd_ptr<ICPUBuffer>(std::max(MIN_SSBO_SZ, sz));
		memcpy(prefetch_instr_buf->getPointer(), _compResult.prefetch_stream.data(), sz);

		d->buffer.offset = 0u;
		d->buffer.size = prefetch_instr_buf->getSize();
		d->desc = std::move(prefetch_instr_buf);
	}

#ifdef DEBUG_MITSUBA_LOADER
	std::ofstream ofile("log.txt");
#endif
	core::vector<nbl_glsl_ext_Mitsuba_Loader_instance_data_t> instanceData;
	for (auto it=meshBegin; it != meshEnd; ++it)
	{		
		auto mesh = it->first.get();

		core::vectorSIMDf emissive;
		for (auto& mb : mesh->getMeshBuffers())
			mb->setBaseInstance(instanceData.size());
		auto* meshMeta = _ctx.meta->getAssetSpecificMetadata(mesh);
		auto baseInstanceDataIt = meshMeta->m_instances.begin();
		for (const auto& inst : meshMeta->m_instanceAuxData)
		{
			nbl_glsl_ext_Mitsuba_Loader_instance_data_t instData;

			instData.tform = baseInstanceDataIt->worldTform;
			instData.tform.getSub3x3InverseTranspose(reinterpret_cast<core::matrix3x4SIMD&>(instData.normalMatrixRow0));
			reinterpret_cast<float&>(instData.determinantSignBit) = instData.tform.getPseudoDeterminant().x;
			instData.determinantSignBit &= 0x80000000;

			auto bsdf = inst.bsdf;
			auto bsdf_front = bsdf.front;
			auto bsdf_back  = bsdf.back;
			auto streams_it = _compResult.streams.find(bsdf_front);
			{
				_NBL_DEBUG_BREAK_IF(streams_it == _compResult.streams.end());
				const auto& streams = streams_it->second;

#ifdef DEBUG_MITSUBA_LOADER
				//@Crisspl TODO No way how to fix it for reporting the `inst.bsdf_id`
				//os::Printer::log("Debug print front BSDF with id = ", inst.bsdf_id, ELL_INFORMATION);
				//ofile << "Debug print front BSDF with id = " << inst.bsdf_id << std::endl;
				//_ctx.backend.debugPrint(ofile, streams, _compResult, &_ctx.backend_ctx);
#endif
				const auto emissive = inst.frontEmitter.type==CElementEmitter::AREA ? inst.frontEmitter.area.radiance:core::vectorSIMDf(0.f);
				instData.material.front = impl_backendToGLSLStream(emissive,streams);
			}
			streams_it = _compResult.streams.find(bsdf_back);
			{
				_NBL_DEBUG_BREAK_IF(streams_it == _compResult.streams.end());
				const auto& streams = streams_it->second;

#ifdef DEBUG_MITSUBA_LOADER
				//@Crisspl TODO No way how to fix it for reporting the `inst.bsdf_id`
				//os::Printer::log("Debug print back BSDF with id = ", inst.bsdf_id, ELL_INFORMATION);
				//ofile << "Debug print back BSDF with id = " << inst.bsdf_id << std::endl;
				//_ctx.backend.debugPrint(ofile, streams, _compResult, &_ctx.backend_ctx);
#endif
				const auto emissive = inst.backEmitter.type==CElementEmitter::AREA ? inst.backEmitter.area.radiance:core::vectorSIMDf(0.f);
				instData.material.back = impl_backendToGLSLStream(emissive,streams);
			}

			instanceData.push_back(instData);
			baseInstanceDataIt++;
		}
	}
#ifdef DEBUG_MITSUBA_LOADER
	ofile.close();
#endif
	d = ds0->getDescriptors(INSTANCE_DATA_BINDING).begin();
	{
		auto instDataBuf = core::make_smart_refctd_ptr<ICPUBuffer>(instanceData.size()*sizeof(nbl_glsl_ext_Mitsuba_Loader_instance_data_t));
		memcpy(instDataBuf->getPointer(), instanceData.data(), instDataBuf->getSize());

		d->buffer.offset = 0u;
		d->buffer.size = instDataBuf->getSize();
		d->desc = std::move(instDataBuf);
	}

	return ds0;
}

using namespace std::string_literals;

SContext::SContext(
	const asset::IGeometryCreator* _geomCreator,
	const asset::IMeshManipulator* _manipulator,
	const asset::IAssetLoader::SAssetLoadContext& _ctx,
	asset::IAssetLoader::IAssetLoaderOverride* _override,
	CMitsubaMetadata* _metadata
) : creator(_geomCreator), manipulator(_manipulator), inner(_ctx), override_(_override), meta(_metadata),
	ir(core::make_smart_refctd_ptr<asset::material_compiler::IR>()), frontend(this),
	samplerCacheKeyBase(inner.mainFile->getFileName().c_str() + "?sampler"s)
{
	backend_ctx.vt.vt = core::make_smart_refctd_ptr<asset::ICPUVirtualTexture>(
		[](asset::E_FORMAT_CLASS) -> uint32_t { return VT_PHYSICAL_PAGE_TEX_TILES_PER_DIM_LOG2; }, // 16x16 tiles per layer for all dynamically created storages
		VT_PAGE_SZ_LOG2, 
		VT_PAGE_PADDING, 
		VT_MAX_ALLOCATABLE_TEX_SZ_LOG2
	);
	meta->m_global.m_VT = backend_ctx.vt.vt;
}

}
}
}