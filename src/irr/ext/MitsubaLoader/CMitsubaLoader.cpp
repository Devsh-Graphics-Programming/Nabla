#include "os.h"

#include <cwchar>
#include <irr/asset/filters/CSwizzleAndConvertImageFilter.h>

#include "irr/ext/MitsubaLoader/CMitsubaLoader.h"
#include "irr/ext/MitsubaLoader/ParserUtil.h"

namespace irr
{
using namespace asset;

namespace ext
{
namespace MitsubaLoader
{

_IRR_STATIC_INLINE_CONSTEXPR const char* DUMMY_VERTEX_SHADER =
R"(#version 430 core

layout (location = 0) in vec3 vPosition;
layout (location = 2) in vec2 vUV;
layout (location = 3) in vec3 vNormal;

layout (location = 0) out vec3 WorldPos;
layout (location = 1) flat out uint InstanceIndex;
layout (location = 2) out vec3 Normal;
layout (location = 3) out vec2 UV;

#include <irr/builtin/glsl/utils/common.glsl>
#include <irr/builtin/glsl/utils/transform.glsl>

layout (push_constant) uniform Block {
    uint instDataOffset;
} PC;

#ifndef _IRR_VERT_SET1_BINDINGS_DEFINED_
#define _IRR_VERT_SET1_BINDINGS_DEFINED_
layout (set = 1, binding = 0, row_major, std140) uniform UBO {
    irr_glsl_SBasicViewParameters params;
} CamData;
#endif //_IRR_VERT_SET1_BINDINGS_DEFINED_

struct InstanceData
{
	mat4x3 tform;
	vec3 normalMatrixRow0;
	uint bsdf_instrOffset;
	vec3 normalMatrixRow1;
	uint bsdf_instrCount;
	vec3 normalMatrixRow2;
	uint _padding;//not needed
	uvec2 prefetch_instrStream;
	uvec2 nprecomp_instrStream;
	uvec2 genchoice_instrStream;
	uvec2 emissive;
};
layout (set = 0, binding = 5, row_major, std430) readonly restrict buffer InstDataBuffer {
	InstanceData data[];
} InstData;

void main()
{
	uint instIx = PC.instDataOffset+gl_InstanceIndex;
	mat4x3 tform = InstData.data[instIx].tform;
	mat4 mvp = irr_glsl_pseudoMul4x4with4x3(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier(CamData.params.MVP), tform);
	gl_Position = irr_glsl_pseudoMul4x4with3x1(mvp, vPosition);
	WorldPos = irr_glsl_pseudoMul3x4with3x1(tform, vPosition);
	//InstrOffsetCount = uvec2(InstData.data[instIx].instrOffset,InstData.data[instIx].instrCount);
	mat3 normalMat = mat3(InstData.data[instIx].normalMatrixRow0,InstData.data[instIx].normalMatrixRow1,InstData.data[instIx].normalMatrixRow2);
	Normal = transpose(normalMat)*normalize(vNormal);
	UV = vUV;
	//Emissive = InstData.data[instIx].emissive;
	InstanceIndex = instIx;
}

)";

_IRR_STATIC_INLINE_CONSTEXPR const char* FRAGMENT_SHADER_PROLOGUE =
R"(#version 430 core
)";
_IRR_STATIC_INLINE_CONSTEXPR const char* FRAGMENT_SHADER_DEFINITIONS =
R"(
layout (location = 0) in vec3 WorldPos;
layout (location = 1) flat in uint InstanceIndex;
layout (location = 2) in vec3 Normal;
layout (location = 3) in vec2 UV;

layout (location = 0) out vec4 OutColor;

#define _IRR_VT_DESCRIPTOR_SET 0
#define _IRR_VT_PAGE_TABLE_BINDING 0

#define _IRR_VT_FLOAT_VIEWS_BINDING 1 
#define _IRR_VT_FLOAT_VIEWS_COUNT 5
#define _IRR_VT_FLOAT_VIEWS

#define _IRR_VT_INT_VIEWS_BINDING 2
#define _IRR_VT_INT_VIEWS_COUNT 0
#define _IRR_VT_INT_VIEWS

#define _IRR_VT_UINT_VIEWS_BINDING 3
#define _IRR_VT_UINT_VIEWS_COUNT 0
#define _IRR_VT_UINT_VIEWS
#include <irr/builtin/glsl/virtual_texturing/descriptors.glsl>

layout (set = 0, binding = 2, std430) restrict readonly buffer PrecomputedStuffSSBO
{
    uint pgtab_sz_log2;
    float vtex_sz_rcp;
    float phys_pg_tex_sz_rcp[_IRR_VT_MAX_PAGE_TABLE_LAYERS];
    uint layer_to_sampler_ix[_IRR_VT_MAX_PAGE_TABLE_LAYERS];
} precomputed;

layout (set = 0, binding = 3, std430) restrict readonly buffer INSTR_BUF
{
	instr_t data[];
} instr_buf;
layout (set = 0, binding = 4, std430) restrict readonly buffer BSDF_BUF
{
	bsdf_data_t data[];
} bsdf_buf;

uint irr_glsl_VT_layer2pid(in uint layer)
{
    return precomputed.layer_to_sampler_ix[layer];
}
uint irr_glsl_VT_getPgTabSzLog2()
{
    return precomputed.pgtab_sz_log2;
}
float irr_glsl_VT_getPhysPgTexSzRcp(in uint layer)
{
    return precomputed.phys_pg_tex_sz_rcp[layer];
}
float irr_glsl_VT_getVTexSzRcp()
{
    return precomputed.vtex_sz_rcp;
}
#define _IRR_USER_PROVIDED_VIRTUAL_TEXTURING_FUNCTIONS_

#include <irr/builtin/glsl/virtual_texturing/functions.glsl/7/8>

#include <irr/builtin/glsl/utils/common.glsl>

layout (set = 1, binding = 0, row_major, std140) uniform UBO {
    irr_glsl_SBasicViewParameters params;
} CamData;

struct InstanceData
{
	mat4x3 tform;
	vec3 normalMatrixRow0;
	uint bsdf_instrOffset;
	vec3 normalMatrixRow1;
	uint bsdf_instrCount;
	vec3 normalMatrixRow2;
	uint _padding;//not needed
	uvec2 prefetch_instrStream;
	uvec2 nprecomp_instrStream;
	uvec2 genchoice_instrStream;
	uvec2 emissive;
};
layout (set = 0, binding = 5, row_major, std430) readonly restrict buffer InstDataBuffer {
	InstanceData data[];
} InstData;

vec3 irr_glsl_MC_getCamPos()
{
	vec3 campos = irr_glsl_SBasicViewParameters_GetEyePos(CamData.params.NormalMatAndEyePos);
	return campos;
}
instr_t irr_glsl_MC_fetchInstr(in uint ix)
{
	return instr_buf.data[ix];
}
bsdf_data_t irr_glsl_MC_fetchBSDFData(in uint ix)
{
	return bsdf_buf.data[ix];
}
#define _IRR_USER_PROVIDED_MATERIAL_COMPILER_GLSL_BACKEND_FUNCTIONS_
)";
_IRR_STATIC_INLINE_CONSTEXPR const char* FRAGMENT_SHADER_IMPL = R"(
#include <irr/builtin/glsl/format/decode.glsl>

instr_stream_t getEvalStream()
{
	instr_stream_t stream;
	stream.offset = InstData.data[InstanceIndex].bsdf_instrOffset;
	stream.count = InstData.data[InstanceIndex].bsdf_instrCount;

	return stream;
}
//rem'n'pdf and eval use the same instruction stream
instr_stream_t getRemAndPdfStream()
{
	return getEvalStream();
}
instr_stream_t getGenChoiceStream()
{
	instr_stream_t stream;
	uvec2 s = InstData.data[InstanceIndex].genchoice_instrStream;
	stream.offset = s.x;
	stream.count =  s.y;

	return stream;
}
instr_stream_t getTexPrefetchStream()
{
	uvec2 s = InstData.data[InstanceIndex].prefetch_instrStream;
	instr_stream_t stream;
	stream.offset = s.x;
	stream.count = s.y;

	return stream;
}
instr_stream_t getNormalPrecompStream()
{
	uvec2 s = InstData.data[InstanceIndex].nprecomp_instrStream;
	instr_stream_t stream;
	stream.offset = s.x;
	stream.count = s.y;

	return stream;
}

#ifndef _IRR_BSDF_COS_EVAL_DEFINED_
#define _IRR_BSDF_COS_EVAL_DEFINED_
// Spectrum can be exchanged to a float for monochrome
#define Spectrum vec3
//! This is the function that evaluates the BSDF for specific view and observer direction
// params can be either BSDFIsotropicParams or BSDFAnisotropicParams
Spectrum irr_bsdf_cos_eval(in irr_glsl_BSDFIsotropicParams params, in irr_glsl_IsotropicViewSurfaceInteraction inter, in mat2 dUV)
{
	instr_stream_t eval_instrStream = getEvalStream();
	
	return runEvalStream(eval_instrStream, params.L);
}
#endif

#ifndef _IRR_COMPUTE_LIGHTING_DEFINED_
#define _IRR_COMPUTE_LIGHTING_DEFINED_
vec3 irr_computeLighting(inout irr_glsl_IsotropicViewSurfaceInteraction out_interaction, in mat2 dUV)
{
	vec3 emissive = irr_glsl_decodeRGB19E7(InstData.data[InstanceIndex].emissive);

	vec3 campos = irr_glsl_MC_getCamPos();
	irr_glsl_BSDFIsotropicParams params;
	params.L = campos-WorldPos;
	out_interaction = irr_glsl_calcFragmentShaderSurfaceInteraction(campos, WorldPos, normalize(Normal));

	return irr_bsdf_cos_eval(params, out_interaction, dUV)*1000.0/dot(params.L,params.L) + emissive;
}
#endif

void main()
{
	mat2 dUV = mat2(dFdx(UV),dFdy(UV));

	InstanceData instData = InstData.data[InstanceIndex];
#ifdef TEX_PREFETCH_STREAM
	runTexPrefetchStream(getTexPrefetchStream(), dUV);
#endif
#ifdef NORM_PRECOMP_STREAM
	runNormalPrecompStream(getNormalPrecompStream(), dUV);
#endif

	irr_glsl_IsotropicViewSurfaceInteraction inter;
	vec3 color = irr_computeLighting(inter, dUV);

	OutColor = vec4(color, 1.0);
}
)";

_IRR_STATIC_INLINE_CONSTEXPR const char* PIPELINE_LAYOUT_CACHE_KEY = "irr/builtin/pipeline_layout/loaders/mitsuba_xml/default";
_IRR_STATIC_INLINE_CONSTEXPR const char* VERTEX_SHADER_CACHE_KEY = "irr/builtin/specialized_shader/loaders/mitsuba_xml/default";

_IRR_STATIC_INLINE_CONSTEXPR uint32_t PAGE_TAB_TEX_BINDING = 0u;
_IRR_STATIC_INLINE_CONSTEXPR uint32_t PHYS_PAGE_VIEWS_BINDING = 1u;
_IRR_STATIC_INLINE_CONSTEXPR uint32_t PRECOMPUTED_VT_DATA_BINDING = 2u;
_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTR_BUF_BINDING = 3u;
_IRR_STATIC_INLINE_CONSTEXPR uint32_t BSDF_BUF_BINDING = 4u;
_IRR_STATIC_INLINE_CONSTEXPR uint32_t INSTANCE_DATA_BINDING = 5u;
_IRR_STATIC_INLINE_CONSTEXPR uint32_t DS0_BINDING_COUNT_WO_VT = 4u;

template <typename AssetT>
static void insertAssetIntoCache(core::smart_refctd_ptr<AssetT>& asset, const char* path, IAssetManager* _assetMgr)
{
	asset::SAssetBundle bundle({ asset });
	_assetMgr->changeAssetKey(bundle, path);
	_assetMgr->insertAssetIntoCache(bundle);
}
template<typename AssetType, IAsset::E_TYPE assetType>
static core::smart_refctd_ptr<AssetType> getBuiltinAsset(const char* _key, IAssetManager* _assetMgr)
{
	size_t storageSz = 1ull;
	asset::SAssetBundle bundle;
	const IAsset::E_TYPE types[]{ assetType, static_cast<IAsset::E_TYPE>(0u) };

	_assetMgr->findAssets(storageSz, &bundle, _key, types);
	if (bundle.isEmpty())
		return nullptr;
	auto assets = bundle.getContents();
	//assert(!assets.empty());

	return core::smart_refctd_ptr_static_cast<AssetType>(assets.begin()[0]);
}

static core::smart_refctd_ptr<asset::ICPUPipelineLayout> createAndCachePipelineLayout(asset::IAssetManager* _manager, asset::ICPUVirtualTexture* _vt)
{
	SPushConstantRange pcrng;
	pcrng.offset = 0u;
	pcrng.size = sizeof(uint32_t);//instance data offset
	pcrng.stageFlags = static_cast<asset::ISpecializedShader::E_SHADER_STAGE>(asset::ISpecializedShader::ESS_FRAGMENT | asset::ISpecializedShader::ESS_VERTEX);

	core::smart_refctd_ptr<ICPUDescriptorSetLayout> ds0layout;
	{
		auto sizes = _vt->getDSlayoutBindings(nullptr, nullptr);
		auto bindings = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUDescriptorSetLayout::SBinding>>(sizes.first+DS0_BINDING_COUNT_WO_VT);
		auto samplers = core::make_refctd_dynamic_array< core::smart_refctd_dynamic_array<core::smart_refctd_ptr<asset::ICPUSampler>>>(sizes.second);

		_vt->getDSlayoutBindings(bindings->data(), samplers->data(), PAGE_TAB_TEX_BINDING, PHYS_PAGE_VIEWS_BINDING);
		auto* b = bindings->data()+(bindings->size()-DS0_BINDING_COUNT_WO_VT);
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

		ds0layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bindings->data(), bindings->data()+bindings->size());
	}
	auto ds1layout = getBuiltinAsset<ICPUDescriptorSetLayout, IAsset::ET_DESCRIPTOR_SET_LAYOUT>("irr/builtin/descriptor_set_layout/basic_view_parameters", _manager);

	auto layout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(&pcrng, &pcrng+1, std::move(ds0layout), std::move(ds1layout), nullptr, nullptr);
	insertAssetIntoCache(layout, PIPELINE_LAYOUT_CACHE_KEY, _manager);

	return layout;
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
static core::smart_refctd_ptr<asset::ICPUSpecializedShader> createFragmentShader(const asset::material_compiler::CMaterialCompilerGLSLBackendCommon::result_t& _mcRes)
{
	std::string source = 
		FRAGMENT_SHADER_PROLOGUE +
		_mcRes.fragmentShaderSource_declarations +
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

CMitsubaLoader::CMitsubaLoader(asset::IAssetManager* _manager) : asset::IAssetLoader(), m_manager(_manager)
{
#ifdef _IRR_DEBUG
	setDebugName("CMitsubaLoader");
#endif
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
	ParserManager parserManager(m_manager->getFileSystem(),_override);
	if (!parserManager.parse(_file))
		return {};

	if (_params.loaderFlags & IAssetLoader::ELPF_LOAD_METADATA_ONLY)
	{
		auto emptyMesh = core::make_smart_refctd_ptr<asset::CCPUMesh>();
		m_manager->setAssetMetadata(emptyMesh.get(), core::make_smart_refctd_ptr<ext::MitsubaLoader::CMitsubaMetadata>(parserManager.m_globalMetadata));
		return SAssetBundle({ std::move(emptyMesh) });
	}
	else
	{
		//
		auto currentDir = io::IFileSystem::getFileDir(_file->getFileName()) + "/";
		SContext ctx(
			m_manager->getGeometryCreator(),
			m_manager->getMeshManipulator(),
			asset::IAssetLoader::SAssetLoadParams(_params.decryptionKeyLen, _params.decryptionKey, _params.cacheFlags, currentDir.c_str()),
			_override,
			parserManager.m_globalMetadata.get()
		);
		if (!getBuiltinAsset<asset::ICPUPipelineLayout, asset::IAsset::ET_PIPELINE_LAYOUT>(PIPELINE_LAYOUT_CACHE_KEY, m_manager))
		{
			createAndCachePipelineLayout(m_manager, ctx.backend_ctx.vt.get());
			createAndCacheVertexShader(m_manager, DUMMY_VERTEX_SHADER);
		}

		core::set<core::smart_refctd_ptr<asset::ICPUMesh>> meshes;

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
				{
					auto metadata = core::make_smart_refctd_ptr<IMeshMetadata>(
						core::smart_refctd_ptr(parserManager.m_globalMetadata),
						std::move(shapepair.second),
						shapedef
						);
					m_manager->setAssetMetadata(mesh.get(), std::move(metadata));
					meshes.insert(std::move(mesh));
				}
			}
		}

		//insert all instances into metadata
		for (auto& mesh_ : meshes)
		{
			auto* mesh = mesh_.get();
			auto* meshmeta = static_cast<IMeshMetadata*>(mesh->getMetadata());
			auto instances_rng = ctx.mapMesh2instanceData.equal_range(mesh);
			assert(instances_rng.first!=instances_rng.second);
			for (auto it = instances_rng.first; it!=instances_rng.second; ++it) {
				const auto& inst = it->second;
				meshmeta->instances.push_back({inst.tform, inst.bsdf, inst.bsdf_id, inst.emitter});
			}
			if (meshmeta->instances.size() > 1ull)
				printf("");

			for (uint32_t i = 0u; i < mesh->getMeshBufferCount(); ++i)
				mesh->getMeshBuffer(i)->setInstanceCount(meshmeta->instances.size());
		}

		auto compResult = ctx.backend.compile(&ctx.backend_ctx, ctx.ir.get());
		auto pipeline_metadata = createPipelineMetadata(createDS0(ctx, compResult, meshes.begin(), meshes.end()), getBuiltinAsset<ICPUPipelineLayout, IAsset::ET_PIPELINE_LAYOUT>(PIPELINE_LAYOUT_CACHE_KEY, m_manager).get());
		auto fragShader = createFragmentShader(compResult);
		auto basePipeline = createPipeline(
			getBuiltinAsset<asset::ICPUPipelineLayout, asset::IAsset::ET_PIPELINE_LAYOUT>(PIPELINE_LAYOUT_CACHE_KEY, m_manager),
			getBuiltinAsset<asset::ICPUSpecializedShader, asset::IAsset::ET_SPECIALIZED_SHADER>(VERTEX_SHADER_CACHE_KEY, m_manager),
			std::move(fragShader)
		);
		for (auto& mesh : meshes)
		{
			for (uint32_t i = 0u; i < mesh->getMeshBufferCount(); ++i)
			{
				asset::ICPUMeshBuffer* mb = mesh->getMeshBuffer(i);
				auto* prevPipeline = mb->getPipeline();
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
					pipeline = core::smart_refctd_ptr_static_cast<ICPURenderpassIndependentPipeline>(//shallow copy because we're going to override parameter structs
						basePipeline->clone(0u)
						);
					pipeline->getVertexInputParams() = cacheKey.vtxParams;
					pipeline->getPrimitiveAssemblyParams() = cacheKey.primParams;
					ctx.pipelineCache.insert({ cacheKey, pipeline });
				}

				mb->setPipeline(core::smart_refctd_ptr(pipeline));
				if (!pipeline->getMetadata())
					m_manager->setAssetMetadata(pipeline.get(), core::smart_refctd_ptr(pipeline_metadata));
			}
		}

		return { meshes };
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

	auto mesh = core::make_smart_refctd_ptr<CCPUMesh>();
	mesh->addMeshBuffer(std::move(mb));

	return mesh;
}

SContext::shape_ass_type CMitsubaLoader::loadBasicShape(SContext& ctx, uint32_t hierarchyLevel, CElementShape* shape, const core::matrix3x4SIMD& relTform)
{
	constexpr uint32_t UV_ATTRIB_ID = 2U;

	auto addInstance = [shape,&ctx,&relTform,this](SContext::shape_ass_type& mesh) {
		assert(shape->bsdf);
		auto bsdf = getBSDFtreeTraversal(ctx, shape->bsdf);
		core::matrix3x4SIMD tform = core::concatenateBFollowedByA(relTform, shape->getAbsoluteTransform());
		SContext::SInstanceData instance{ tform, bsdf, shape->bsdf->id, shape->obtainEmitter() };
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
		auto loadParams = ctx.params;
		loadParams.loaderFlags = static_cast<IAssetLoader::E_LOADER_PARAMETER_FLAGS>(loadParams.loaderFlags | IAssetLoader::ELPF_RIGHT_HANDED_MESHES);
		auto retval = interm_getAssetInHierarchy(m_manager, filename.svalue, loadParams, hierarchyLevel/*+ICPUSCene::MESH_HIERARCHY_LEVELS_BELOW*/, ctx.override);
		auto contentRange = retval.getContents();
		//
		uint32_t actualIndex = 0;
		if (index>=0ll)
		for (auto it=contentRange.begin(); it!=contentRange.end(); it++)
		{
			auto meta = it->get()->getMetadata();
			if (!meta || core::strcmpi(meta->getLoaderName(),ext::MitsubaLoader::CSerializedMetadata::LoaderName))
				continue;
			auto serializedMeta = static_cast<CSerializedMetadata*>(meta);
			if (serializedMeta->id!=static_cast<uint32_t>(index))
				continue;
			actualIndex = it-contentRange.begin();
			break;
		}
		//
		if (contentRange.begin()+actualIndex < contentRange.end())
		{
			auto asset = contentRange.begin()[actualIndex];
			if (asset && asset->getAssetType()==asset::IAsset::ET_MESH)
			{
				// make a (shallow) copy because the mesh will get mutilated and abused for metadata
				auto mesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(asset);
				auto copy = core::make_smart_refctd_ptr<asset::CCPUMesh>();
				for (auto j=0u; j<mesh->getMeshBufferCount(); j++)
					copy->addMeshBuffer(core::smart_refctd_ptr<asset::ICPUMeshBuffer>(mesh->getMeshBuffer(j)));
				copy->recalculateBoundingBox();
				m_manager->setAssetMetadata(copy.get(),core::smart_refctd_ptr<asset::IAssetMetadata>(mesh->getMetadata()));
				return copy;
			}
			else
				return nullptr;
		}
		else
			return nullptr;
	};

	core::smart_refctd_ptr<asset::ICPUMesh> mesh;
	bool flipNormals = false;
	bool faceNormals = false;
	float maxSmoothAngle = NAN;
	switch (shape->type)
	{
		case CElementShape::Type::CUBE:
		{
			auto cubeData = ctx.creator->createCubeMesh(core::vector3df(2.f));

			mesh = createMeshFromGeomCreatorReturnType(ctx.creator->createCubeMesh(core::vector3df(2.f)), m_manager);
			flipNormals = flipNormals!=shape->cube.flipNormals;
		}
			break;
		case CElementShape::Type::SPHERE:
			mesh = createMeshFromGeomCreatorReturnType(ctx.creator->createSphereMesh(1.f,64u,64u), m_manager);
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
				mesh = createMeshFromGeomCreatorReturnType(ctx.creator->createCylinderMesh(1.f, 1.f, 64), m_manager);
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
			mesh = createMeshFromGeomCreatorReturnType(ctx.creator->createRectangleMesh(core::vector2df_SIMD(1.f,1.f)), m_manager);
			flipNormals = flipNormals!=shape->rectangle.flipNormals;
			break;
		case CElementShape::Type::DISK:
			mesh = createMeshFromGeomCreatorReturnType(ctx.creator->createDiskMesh(1.f,64u), m_manager);
			flipNormals = flipNormals!=shape->disk.flipNormals;
			break;
		case CElementShape::Type::OBJ:
			mesh = loadModel(shape->obj.filename);
			flipNormals = flipNormals==shape->obj.flipNormals;
			faceNormals = shape->obj.faceNormals;
			maxSmoothAngle = shape->obj.maxSmoothAngle;
			if (mesh && shape->obj.flipTexCoords)
			{
				for (auto i = 0u; i < mesh->getMeshBufferCount(); i++)
				{
					auto meshbuffer = mesh->getMeshBuffer(i);
					core::vectorSIMDf uv;
					for (uint32_t i=0u; meshbuffer->getAttribute(uv, UV_ATTRIB_ID, i); i++)
					{
						uv.y = -uv.y;
						meshbuffer->setAttribute(uv, UV_ATTRIB_ID, i);
					}
				}
			}
			// collapse parameter gets ignored
			break;
		case CElementShape::Type::PLY:
			_IRR_DEBUG_BREAK_IF(true); // this code has never been tested
			mesh = loadModel(shape->ply.filename);
			mesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(mesh->clone(~0u));//clone everything
			flipNormals = flipNormals!=shape->ply.flipNormals;
			faceNormals = shape->ply.faceNormals;
			maxSmoothAngle = shape->ply.maxSmoothAngle;
			if (mesh && shape->ply.srgb)
			{
				uint32_t totalVertexCount = 0u;
				for (auto i = 0u; i < mesh->getMeshBufferCount(); i++)
					totalVertexCount += mesh->getMeshBuffer(i)->calcVertexCount();
				if (totalVertexCount)
				{
					constexpr uint32_t hidefRGBSize = 4u;
					auto newRGB = core::make_smart_refctd_ptr<asset::ICPUBuffer>(hidefRGBSize*totalVertexCount);
					uint32_t* it = reinterpret_cast<uint32_t*>(newRGB->getPointer());
					for (auto i = 0u; i < mesh->getMeshBufferCount(); i++)
					{
						auto meshbuffer = mesh->getMeshBuffer(i);
						uint32_t offset = reinterpret_cast<uint8_t*>(it)-reinterpret_cast<uint8_t*>(newRGB->getPointer());
						core::vectorSIMDf rgb;
						for (uint32_t i=0u; meshbuffer->getAttribute(rgb, 1u, i); i++,it++)
						{
							for (auto i=0; i<3u; i++)
								rgb[i] = core::srgb2lin(rgb[i]);
							meshbuffer->setAttribute(rgb,it,asset::EF_A2B10G10R10_UNORM_PACK32);
						}
						constexpr uint32_t COLOR_BUF_BINDING = 15u;
						auto& vtxParams = meshbuffer->getPipeline()->getVertexInputParams();
						vtxParams.attributes[1].format = EF_A2B10G10R10_UNORM_PACK32;
						vtxParams.attributes[1].relativeOffset = 0u;
						vtxParams.attributes[1].binding = COLOR_BUF_BINDING;
						vtxParams.bindings[COLOR_BUF_BINDING].inputRate = EVIR_PER_VERTEX;
						vtxParams.bindings[COLOR_BUF_BINDING].stride = hidefRGBSize;
						vtxParams.enabledBindingFlags |= (1u<<COLOR_BUF_BINDING);
						meshbuffer->setVertexBufferBinding({0ull,core::smart_refctd_ptr(newRGB)}, COLOR_BUF_BINDING);
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
			_IRR_FALLTHROUGH;
		case CElementShape::Type::INSTANCE:
			assert(false);
			break;
		default:
			_IRR_DEBUG_BREAK_IF(true);
			break;
	}
	//
	if (!mesh)
		return nullptr;

	// flip normals if necessary
	if (flipNormals)
	for (auto i=0u; i<mesh->getMeshBufferCount(); i++)
		ctx.manipulator->flipSurfaces(mesh->getMeshBuffer(i));
	// flip normals if necessary
//#define CRISS_FIX_THIS
#ifdef CRISS_FIX_THIS
	if (faceNormals || !std::isnan(maxSmoothAngle))
	{
		auto newMesh = core::make_smart_refctd_ptr<asset::CCPUMesh>();
		float smoothAngleCos = cos(core::radians(maxSmoothAngle));
		for (auto i=0u; i<mesh->getMeshBufferCount(); i++)
		{
			ctx.manipulator->filterInvalidTriangles(mesh->getMeshBuffer(i));
			auto newMeshBuffer = ctx.manipulator->createMeshBufferUniquePrimitives(mesh->getMeshBuffer(i));
			ctx.manipulator->calculateSmoothNormals(newMeshBuffer.get(), false, 0.f, newMeshBuffer->getNormalAttributeIx(),
				[&](const asset::IMeshManipulator::SSNGVertexData& a, const asset::IMeshManipulator::SSNGVertexData& b, asset::ICPUMeshBuffer* buffer)
				{
					if (faceNormals)
						return a.indexOffset==b.indexOffset;
					else
						return core::dot(a.parentTriangleFaceNormal, b.parentTriangleFaceNormal).x >= smoothAngleCos;
				});

			asset::IMeshManipulator::SErrorMetric metrics[16];
			metrics[3].method = asset::IMeshManipulator::EEM_ANGLES;
			newMeshBuffer = ctx.manipulator->createOptimizedMeshBuffer(newMeshBuffer.get(),metrics);

			newMesh->addMeshBuffer(std::move(newMeshBuffer));
		}
		newMesh->recalculateBoundingBox();
		m_manager->setAssetMetadata(newMesh.get(), core::smart_refctd_ptr<asset::IAssetMetadata>(mesh->getMetadata()));
		mesh = std::move(newMesh);
	}
#endif

	addInstance(mesh);
	// cache and return
	ctx.shapeCache.insert({ shape,mesh });
	return mesh;
}

SContext::tex_ass_type CMitsubaLoader::getTexture(SContext& ctx, uint32_t hierarchyLevel, const CElementTexture* tex)
{
	if (!tex)
		return {};

	auto found = ctx.textureCache.find(tex);
	if (found != ctx.textureCache.end())
		return found->second;

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
				auto retval = interm_getAssetInHierarchy(m_manager,tex->bitmap.filename.svalue,ctx.params,hierarchyLevel,ctx.override);
				auto contentRange = retval.getContents();
				if (contentRange.begin() < contentRange.end())
				{
					auto asset = contentRange.begin()[0];
					if (asset && asset->getAssetType() == asset::IAsset::ET_IMAGE)
					{
						auto texture = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(asset);

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
								_IRR_FALLTHROUGH;
							default:
								break;
						}
						viewParams.subresourceRange.levelCount = texture->getCreationParameters().mipLevels;
						viewParams.format = texture->getCreationParameters().format;
						viewParams.image = std::move(texture);
						//! TODO: this stuff (custom shader sampling code?)
						_IRR_DEBUG_BREAK_IF(tex->bitmap.uoffset != 0.f);
						_IRR_DEBUG_BREAK_IF(tex->bitmap.voffset != 0.f);
						_IRR_DEBUG_BREAK_IF(tex->bitmap.uscale != 1.f);
						_IRR_DEBUG_BREAK_IF(tex->bitmap.vscale != 1.f);
					}
				}
				// adjust gamma on pixels (painful and long process)
				if (!std::isnan(tex->bitmap.gamma))
				{
					_IRR_DEBUG_BREAK_IF(true); // TODO
				}
				switch (tex->bitmap.filterType)
				{
					case CElementTexture::Bitmap::FILTER_TYPE::EWA:
						_IRR_FALLTHROUGH; // we dont support this fancy stuff
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
				auto getWrapMode = [](CElementTexture::Bitmap::WRAP_MODE mode)
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
							_IRR_DEBUG_BREAK_IF(true); // TODO : replace whole texture?
							break;
						case CElementTexture::Bitmap::WRAP_MODE::ZERO:
							_IRR_DEBUG_BREAK_IF(true); // TODO : replace whole texture?
							break;
						default:
							return ISampler::ETC_REPEAT;
							break;
					}
				};
				samplerParams.TextureWrapU = getWrapMode(tex->bitmap.wrapModeU);
				samplerParams.TextureWrapV = getWrapMode(tex->bitmap.wrapModeV);

				//in case of <channel>, extract one channel
				if (viewParams.components.g!=asset::ICPUImageView::SComponentMapping::ES_G)
				{
					auto get1ChannelFormat = [](asset::E_FORMAT f) -> asset::E_FORMAT {
						const uint32_t bytesPerChannel = (getBytesPerPixel(f) * core::rational(1, getFormatChannelCount(f))).getIntegerApprox();
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
					outParams.format = get1ChannelFormat(outParams.format);
					const size_t texelBytesz = asset::getTexelOrBlockBytesize(outParams.format);
					region.bufferRowLength = asset::IImageAssetHandlerBase::calcPitchInBlocks(outParams.extent.width, texelBytesz);
					auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(texelBytesz*region.bufferRowLength*outParams.extent.height);
					region.imageOffset = {0,0,0};
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
					conv.inOffset = {0u,0u,0u};
					conv.outOffset = {0u,0u,0u};
					conv.inImage = viewParams.image.get();
					conv.outImage = outImg.get();

					viewParams.components = asset::ICPUImageView::SComponentMapping{};
					if (!convert_filter_t::execute(&conv))
						_IRR_DEBUG_BREAK_IF(true);
					viewParams.format = outImg->getCreationParameters().format;
					viewParams.image = std::move(outImg);
				}

				auto view = ICPUImageView::create(std::move(viewParams));
				core::smart_refctd_ptr<ICPUSampler> sampler = view ? core::make_smart_refctd_ptr<ICPUSampler>(samplerParams) : nullptr;

				SContext::tex_ass_type tex_ass(std::move(view), std::move(sampler), 1.f);
				ctx.textureCache.insert({ tex,tex_ass });

				return tex_ass;
		}
			break;
		case CElementTexture::Type::SCALE:
		{
			auto retval = getTexture(ctx,hierarchyLevel,tex->scale.texture);
			std::get<float>(retval) *= tex->scale.scale;
			ctx.textureCache[tex] = retval;

			return retval;
		}
			break;
		default:
			_IRR_DEBUG_BREAK_IF(true);
			return SContext::tex_ass_type{nullptr,nullptr,0.f};
			break;
	}
}

auto CMitsubaLoader::getBSDFtreeTraversal(SContext& ctx, const CElementBSDF* bsdf) -> SContext::bsdf_type
{
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
		auto cacheTexture = [&](const auto& const_or_tex) {
			if (const_or_tex.value.type == SPropertyElementData::INVALID)
				getTexture(ctx, 0u, const_or_tex.texture);
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

			switch (bsdf->type)
			{
			case CElementBSDF::DIFFUSE:
			case CElementBSDF::ROUGHDIFFUSE:
				cacheTexture(bsdf->diffuse.reflectance);
				cacheTexture(bsdf->diffuse.alpha);
				break;
			case CElementBSDF::DIFFUSE_TRANSMITTER:
				cacheTexture(bsdf->difftrans.transmittance);
				break;
			case CElementBSDF::DIELECTRIC:
			case CElementBSDF::THINDIELECTRIC:
			case CElementBSDF::ROUGHDIELECTRIC:
				cacheTexture(bsdf->dielectric.alphaU);
				if (bsdf->dielectric.distribution == CElementBSDF::RoughSpecularBase::ASHIKHMIN_SHIRLEY)
					cacheTexture(bsdf->dielectric.alphaV);
				break;
			case CElementBSDF::CONDUCTOR:
				cacheTexture(bsdf->conductor.alphaU);
				if (bsdf->conductor.distribution == CElementBSDF::RoughSpecularBase::ASHIKHMIN_SHIRLEY)
					cacheTexture(bsdf->conductor.alphaV);
				break;
			case CElementBSDF::PLASTIC:
			case CElementBSDF::ROUGHPLASTIC:
				cacheTexture(bsdf->plastic.diffuseReflectance);
				cacheTexture(bsdf->plastic.alphaU);
				if (bsdf->plastic.distribution == CElementBSDF::RoughSpecularBase::ASHIKHMIN_SHIRLEY)
					cacheTexture(bsdf->plastic.alphaV);
				break;
			case CElementBSDF::BUMPMAP:
				getTexture(ctx, 0u, bsdf->bumpmap.texture);
				break;
			case CElementBSDF::BLEND_BSDF:
				cacheTexture(bsdf->blendbsdf.weight);
				break;
			case CElementBSDF::MASK:
				cacheTexture(bsdf->mask.opacity);
				break;
			default: break;
			}
		}
	}

	return ctx.frontend.compileToIRTree(ctx.ir.get(), _bsdf);
}

// Also sets instance data buffer offset into meshbuffers' push constants
template<typename Iter>
inline core::smart_refctd_ptr<asset::ICPUDescriptorSet> CMitsubaLoader::createDS0(const SContext& _ctx, const asset::material_compiler::CMaterialCompilerGLSLBackendCommon::result_t& _compResult, Iter meshBegin, Iter meshEnd)
{
	auto pplnLayout = getBuiltinAsset<ICPUPipelineLayout,IAsset::ET_PIPELINE_LAYOUT>(PIPELINE_LAYOUT_CACHE_KEY, m_manager);
	auto* ds0layout = pplnLayout->getDescriptorSetLayout(0u);

	auto ds0 = core::make_smart_refctd_ptr<ICPUDescriptorSet>(core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(ds0layout));
	{
		auto count = _ctx.backend_ctx.vt->getDescriptorSetWrites(nullptr, nullptr, nullptr);

		auto writes = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUDescriptorSet::SWriteDescriptorSet>>(count.first);
		auto info = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUDescriptorSet::SDescriptorInfo>>(count.second);

		_ctx.backend_ctx.vt->getDescriptorSetWrites(writes->data(), info->data(), ds0.get());

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
		memcpy(precompDataBuf->getPointer(), &_ctx.backend_ctx.vt->getPrecomputedData(), precompDataBuf->getSize());

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

	std::ofstream ofile("log.txt");
	core::vector<SInstanceData> instanceData;
	for (auto it = meshBegin; it != meshEnd; ++it)
	{
		auto& mesh = *it;
		auto* meta = static_cast<const IMeshMetadata*>(mesh->getMetadata());
		
		core::vectorSIMDf emissive;
		uint32_t instDataOffset = instanceData.size();
		for (const auto& inst : meta->getInstances()) {
			emissive = inst.emitter.type==CElementEmitter::AREA ? inst.emitter.area.radiance : core::vectorSIMDf(0.f);

			auto bsdf = inst.bsdf;
			auto streams_it = _compResult.streams.find(bsdf);
			_IRR_DEBUG_BREAK_IF(streams_it==_compResult.streams.end());
			const auto& streams = streams_it->second;

			os::Printer::log("Debug print BSDF with id = ", inst.bsdf_id, ELL_INFORMATION);
			ofile << "Debug print BSDF with id = " << inst.bsdf_id << std::endl;
			_ctx.backend.debugPrint(ofile, streams, _compResult, &_ctx.backend_ctx);

			SInstanceData instData;
			instData.tform = inst.tform;
			instData.tform.getSub3x3InverseTranspose(instData.normalMatrix);
			instData.emissive = core::rgb32f_to_rgb19e7(emissive.pointer);
			core::floatBitsToUint(instData.normalMatrix(0u, 3u)) = streams.rem_and_pdf.first;
			core::floatBitsToUint(instData.normalMatrix(1u, 3u)) = streams.rem_and_pdf.count;
			instData.prefetch_instrStream = {streams.tex_prefetch.first, streams.tex_prefetch.count};
			instData.nprecomp_instrStream = {streams.norm_precomp.first, streams.norm_precomp.count};
			instData.genchoice_instrStream = {streams.gen_choice.first, streams.gen_choice.count};

			instanceData.push_back(instData);
		}
		for (uint32_t i = 0u; i < mesh->getMeshBufferCount(); ++i)
		{
			auto* mb = mesh->getMeshBuffer(i);
			reinterpret_cast<uint32_t*>(mb->getPushConstantsDataPtr())[0] = instDataOffset;
		}
	}
	d = ds0->getDescriptors(INSTANCE_DATA_BINDING).begin();
	{
		auto instDataBuf = core::make_smart_refctd_ptr<ICPUBuffer>(instanceData.size()*sizeof(SInstanceData));
		memcpy(instDataBuf->getPointer(), instanceData.data(), instDataBuf->getSize());

		d->buffer.offset = 0u;
		d->buffer.size = instDataBuf->getSize();
		d->desc = std::move(instDataBuf);
	}

	return ds0;
}

core::smart_refctd_ptr<CMitsubaPipelineMetadata> CMitsubaLoader::createPipelineMetadata(core::smart_refctd_ptr<ICPUDescriptorSet>&& _ds0, const ICPUPipelineLayout* _layout)
{
	constexpr size_t DS1_METADATA_ENTRY_CNT = 3ull;
	core::smart_refctd_dynamic_array<IPipelineMetadata::ShaderInputSemantic> inputs = core::make_refctd_dynamic_array<decltype(inputs)>(DS1_METADATA_ENTRY_CNT);
	{
		const ICPUDescriptorSetLayout* ds1layout = _layout->getDescriptorSetLayout(1u);

		constexpr IPipelineMetadata::E_COMMON_SHADER_INPUT types[DS1_METADATA_ENTRY_CNT]{ IPipelineMetadata::ECSI_WORLD_VIEW_PROJ, IPipelineMetadata::ECSI_WORLD_VIEW, IPipelineMetadata::ECSI_WORLD_VIEW_INVERSE_TRANSPOSE };
		constexpr uint32_t sizes[DS1_METADATA_ENTRY_CNT]{ sizeof(SBasicViewParameters::MVP), sizeof(SBasicViewParameters::MV), sizeof(SBasicViewParameters::NormalMat) };
		constexpr uint32_t relOffsets[DS1_METADATA_ENTRY_CNT]{ offsetof(SBasicViewParameters,MVP), offsetof(SBasicViewParameters,MV), offsetof(SBasicViewParameters,NormalMat) };
		for (uint32_t i = 0u; i < DS1_METADATA_ENTRY_CNT; ++i)
		{
			auto& semantic = (*inputs)[i];
			semantic.type = types[i];
			semantic.descriptorSection.type = IPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER;
			semantic.descriptorSection.uniformBufferObject.binding = ds1layout->getBindings().begin()[0].binding;
			semantic.descriptorSection.uniformBufferObject.set = 1u;
			semantic.descriptorSection.uniformBufferObject.relByteoffset = relOffsets[i];
			semantic.descriptorSection.uniformBufferObject.bytesize = sizes[i];
			semantic.descriptorSection.shaderAccessFlags = ICPUSpecializedShader::ESS_VERTEX;
		}
	}

	return core::make_smart_refctd_ptr<CMitsubaPipelineMetadata>(std::move(_ds0), std::move(inputs));
}

SContext::SContext(
	const asset::IGeometryCreator* _geomCreator,
	const asset::IMeshManipulator* _manipulator,
	const asset::IAssetLoader::SAssetLoadParams& _params,
	asset::IAssetLoader::IAssetLoaderOverride* _override,
	CGlobalMitsubaMetadata* _metadata
) : creator(_geomCreator), manipulator(_manipulator), params(_params), override(_override), globalMeta(_metadata),
	ir(core::make_smart_refctd_ptr<asset::material_compiler::IR>()), frontend(this)
{
	//TODO (maybe) dynamically decide which of those are needed OR just wait until IVirtualTexture does it on itself (dynamically creates resident storages)
	constexpr asset::E_FORMAT formats[]{ asset::EF_R8_UNORM, asset::EF_R8G8_UNORM, asset::EF_R8G8B8_SRGB, asset::EF_R8G8B8A8_SRGB
#ifdef DERIV_MAP_FLOAT32
		, asset::EF_R32G32_SFLOAT
#endif
	};
	constexpr size_t formatCount = sizeof(formats)/sizeof(*formats);
	std::array<asset::ICPUVirtualTexture::ICPUVTResidentStorage::SCreationParams, formatCount> storage;
	storage[0].formatClass = asset::EFC_8_BIT;
	storage[0].layerCount = VT_PHYSICAL_PAGE_TEX_LAYERS;
	storage[0].tilesPerDim_log2 = VT_PHYSICAL_PAGE_TEX_TILES_PER_DIM_LOG2;
	storage[0].formatCount = 1u;
	storage[0].formats = formats;

	storage[1].formatClass = asset::EFC_16_BIT;
	storage[1].layerCount = VT_PHYSICAL_PAGE_TEX_LAYERS;
	storage[1].tilesPerDim_log2 = VT_PHYSICAL_PAGE_TEX_TILES_PER_DIM_LOG2;
	storage[1].formatCount = 1u;
	storage[1].formats = formats+1;

	storage[2].formatClass = asset::EFC_24_BIT;
	storage[2].layerCount = VT_PHYSICAL_PAGE_TEX_LAYERS;
	storage[2].tilesPerDim_log2 = VT_PHYSICAL_PAGE_TEX_TILES_PER_DIM_LOG2;
	storage[2].formatCount = 1u;
	storage[2].formats = formats+2;

	storage[3].formatClass = asset::EFC_32_BIT;
	storage[3].layerCount = VT_PHYSICAL_PAGE_TEX_LAYERS;
	storage[3].tilesPerDim_log2 = VT_PHYSICAL_PAGE_TEX_TILES_PER_DIM_LOG2;
	storage[3].formatCount = 1u;
	storage[3].formats = formats+3;

#ifdef DERIV_MAP_FLOAT32
	storage[4].formatClass = asset::EFC_64_BIT;
	storage[4].layerCount = 2u;
	storage[4].tilesPerDim_log2 = VT_PHYSICAL_PAGE_TEX_TILES_PER_DIM_LOG2;
	storage[4].formatCount = 1u;
	storage[4].formats = formats+4;
#endif

	backend_ctx.vt = core::make_smart_refctd_ptr<asset::ICPUVirtualTexture>(storage.data(), storage.size(), VT_PAGE_SZ_LOG2, VT_PAGE_TABLE_LAYERS, VT_PAGE_PADDING, VT_MAX_ALLOCATABLE_TEX_SZ_LOG2);

	globalMeta->VT = backend_ctx.vt;
}

}
}
}