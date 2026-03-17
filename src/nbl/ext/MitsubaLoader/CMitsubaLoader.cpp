// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "nbl/builtin/hlsl/math/linalg/basic.hlsl"
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"

#include "nbl/ext/MitsubaLoader/CMitsubaLoader.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"
#include "nbl/ext/MitsubaLoader/CMitsubaSerializedMetadata.h"

#include <cwchar>

//#include "nbl/asset/utils/CDerivativeMapCreator.h"



#if defined(_NBL_DEBUG) || defined(_NBL_RELWITHDEBINFO)
#	define DEBUG_MITSUBA_LOADER
#endif

namespace nbl
{
using namespace nbl::asset;
using namespace nbl::hlsl;

namespace ext::MitsubaLoader
{

#if 0 // old material compiler
_NBL_STATIC_INLINE_CONSTEXPR const char* FRAGMENT_SHADER_IMPL = R"(
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
// TODO: move to IAssetLoader
static core::smart_refctd_ptr<asset::ICPUImage> createDerivMap(SContext& ctx, asset::ICPUImage* _heightMap, const ICPUSampler::SParams& _samplerParams, bool fromNormalMap)
{
	core::smart_refctd_ptr<asset::ICPUImage> derivmap_img;
	float scale;
	if (fromNormalMap)
		derivmap_img = asset::CDerivativeMapCreator::createDerivativeMapFromNormalMap<true>(_heightMap,&scale);
	else
	{
		derivmap_img = asset::CDerivativeMapCreator::createDerivativeMapFromHeightMap<true>(
			_heightMap,
			static_cast<asset::ICPUSampler::E_TEXTURE_CLAMP>(_samplerParams.TextureWrapU),
			static_cast<asset::ICPUSampler::E_TEXTURE_CLAMP>(_samplerParams.TextureWrapV),
			static_cast<asset::ICPUSampler::E_TEXTURE_BORDER_COLOR>(_samplerParams.BorderColor),
			&scale
		);
	}

	if (!derivmap_img)
		return nullptr;

	ctx.derivMapCache.insert({derivmap_img,scale});

	return derivmap_img;
}
#endif

constexpr auto LoggerError = system::ILogger::ELL_ERROR;

bool CMitsubaLoader::isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const
{
	constexpr uint32_t stackSize = 16u<<10u;
	char tempBuff[stackSize+1];
	tempBuff[stackSize] = 0;

	static const char* stringsToFind[] = { "<?xml", "version", "scene"};
	static const wchar_t* stringsToFindW[] = { L"<?xml", L"version", L"scene"};
	constexpr uint32_t maxStringSize = 8u; // "version\0"
	static_assert(stackSize>2u*maxStringSize);

	const auto fileSize = _file->getSize();
	if (fileSize<maxStringSize)
		return false;

	size_t pos = 0;
	bool utf16 = false;
	{
		system::IFile::success_t success;
		_file->read(success,tempBuff,pos,3);
		if (!success)
			return false;
		if (tempBuff[0] == 0xEFu && tempBuff[1] == 0xBBu && tempBuff[2] == 0xBFu)
			utf16 = false;
		else if (reinterpret_cast<uint16_t*>(tempBuff)[0] == 0xFEFFu)
		{
			utf16 = true;
			pos = 2;
		}
		else
			pos = 0;
	}

	while (pos<fileSize)
	{
		if (pos>maxStringSize)
			pos -= maxStringSize;
		system::ISystem::future_t<size_t> bytesRead;
		_file->read(bytesRead,tempBuff,pos,stackSize);
		if (!bytesRead.wait())
			return false;
		tempBuff[bytesRead.copy()] = '\0';
		// TODO: should we require all 3 are found?
		for (auto i=0u; i<sizeof(stringsToFind)/sizeof(const char*); i++)
		if (utf16 ? (wcsstr(reinterpret_cast<wchar_t*>(tempBuff),stringsToFindW[i])!=nullptr):(strstr(tempBuff,stringsToFind[i])!=nullptr))
			return true;
	}
	return false;
}


// TODO: make configurable
constexpr bool PrintMaterialDot3 = true;
system::path DebugDir("D:\\work\\Nabla-master\\examples_tests\\15_MitsubaLoader\\bin");
//
void SContext::writeDot3File(system::ISystem* system, const system::path& filepath, frontend_ir_t::SDotPrinter& printer)
{
	using namespace nbl::system;
	core::smart_refctd_ptr<IFile> file = {};
	{
		ISystem::future_t<core::smart_refctd_ptr<IFile>> future;
		system->createFile(future,filepath,IFileBase::E_CREATE_FLAGS::ECF_WRITE);
		if (future.wait())
			future.acquire().move_into(file);
	}
	if (!file)
	{
		inner.params.logger.log("Failed to Open \"%s\" for writing",LoggerError,filepath.c_str());
		return;
	}
	auto str = printer();
	// file write does not take an internal copy of pointer given, need to keep source alive till end
	system::IFile::success_t succ;
	file->write(succ,str.c_str(),0,str.size());
	succ.getBytesProcessed();
}

SAssetBundle CMitsubaLoader::loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	auto result = m_parser.parse(_file,{.logger=_params.logger,.system=m_system.get(),._override=_override});
	if (!result)
		return {};

	if (_params.loaderFlags&IAssetLoader::ELPF_LOAD_METADATA_ONLY)
	{
		return SAssetBundle(std::move(result.metadata),{ICPUScene::create(nullptr)});
	}
	else
	{
		// TODO: need to catch exception and return if failed
		if constexpr (PrintMaterialDot3)
		{
			m_system->deleteDirectory(DebugDir/"material_frontend");
			m_system->createDirectory(DebugDir/"material_frontend");
			m_system->createDirectory(DebugDir/"material_frontend/emitters");
			m_system->createDirectory(DebugDir/"material_frontend/bsdfs");
		}

		SContext ctx(
			IAssetLoader::SAssetLoadContext{ 
				IAssetLoader::SAssetLoadParams(_params.decryptionKeyLen,_params.decryptionKey,_params.cacheFlags,_params.loaderFlags,_params.logger,_file->getFileName().parent_path()),
				_file
			},
			_override,
			result.metadata.get()
		);
		ctx.interm_getAssetInHierarchy = [&](const char* filename, const uint16_t hierarchyOffset)->SAssetBundle
		{
			return this->interm_getAssetInHierarchy(filename,ctx.inner.params,_hierarchyLevel+hierarchyOffset,ctx.override_);
		};
		ctx.interm_getImageViewInHierarchy = [&](const char* filename, const uint16_t hierarchyOffset)->SAssetBundle
		{
			return this->interm_getImageViewInHierarchy<std::string>(filename,ctx.inner.params,_hierarchyLevel+hierarchyOffset,ctx.override_);
		};
		//
		ctx.scene->m_ambientLight = result.ambient;


		// TODO: abstract/move away since many loaders will need to do this
		core::unordered_map<const ICPUGeometryCollection*,core::smart_refctd_ptr<const ICPUMorphTargets>> morphTargetCache;
		auto createMorphTargets = [&_params,&morphTargetCache](core::smart_refctd_ptr<const ICPUGeometryCollection>&& collection)->core::smart_refctd_ptr<const ICPUMorphTargets>
		{
			auto found = morphTargetCache.find(collection.get());
			if (found!=morphTargetCache.end())
				return found->second;
			auto targets = core::make_smart_refctd_ptr<ICPUMorphTargets>();
			if (targets)
			{
				morphTargetCache[collection.get()] = targets;
				targets->getTargets()->push_back({.geoCollection=core::smart_refctd_ptr<ICPUGeometryCollection>(const_cast<ICPUGeometryCollection*>(collection.get()))});
			}
			return targets;
		};

		//
		auto& instances = ctx.scene->getInstances();
		instances.reserve(result.shapegroups.size());
		auto addToScene = [&](const CElementShape* shape, core::smart_refctd_ptr<const ICPUGeometryCollection>&& collection)->void
		{
			if (!collection)
			{
				_params.logger.log("Failed to load a ICPUGeometryCollection for Shape with id %s",LoggerError,shape->id.c_str());
				return;
			}
			assert(shape);
			auto targets = createMorphTargets(std::move(collection));
			if (!targets)
			{
				_params.logger.log("Failed to create ICPUMorphTargets for Shape with id %s",LoggerError,shape->id.c_str());
				return;
			}
			const auto index = instances.size();
			instances.resize(index+1,true);
			instances.getMorphTargets()[index] = core::smart_refctd_ptr<ICPUMorphTargets>(const_cast<ICPUMorphTargets*>(targets.get()));
			if (shape->transform.matrix[3]!=float32_t4(0,0,0,1))
				_params.logger.log("Shape with id %s has Non-Affine transformation matrix, last row is not 0,0,0,1!",LoggerError,shape->id.c_str());
			instances.getInitialTransforms()[index] = shape->getTransform();
			//
			const core::string debugName = shape->id.empty() ? std::format("0x{:x}",ptrdiff_t(shape)):shape->id;
			ctx.getMaterial(shape->bsdf,shape->obtainEmitter(),debugName,PrintMaterialDot3 ? m_system.get():nullptr);
			// TODO: compile the material into the true IR and push it into the instances
		};

		// first go over all actually used shapes which are not shapegroups (regular shapes and instances)
		for (auto& shapepair : result.shapegroups)
		{
			auto* shapedef = shapepair.element;
			// this should be filtered out while parsing and we should just assert it
			if (shapedef->type==CElementShape::Type::SHAPEGROUP)
				continue;

			if (shapedef->type!=CElementShape::Type::INSTANCE)
				addToScene(shapedef,ctx.loadBasicShape(shapedef));
			else // mitsuba is weird and lists instances under a shapegroup instead of having instances reference the shapegroup
			{
				// get group reference
				const CElementShape* parent = shapedef->instance.parent;
				if (!parent) // we should probably assert this
					continue;
				addToScene(shapedef,ctx.loadShapeGroup(parent));
			}
		}
		result.shapegroups.clear();

		if constexpr (PrintMaterialDot3)
			ctx.writeFrontendForestDot3(m_system.get(),DebugDir/"material_frontend/forest.dot");

#if 0
		// TODO: put IR and stuff in metadata so that we can recompile the materials after load
		auto compResult = ctx.backend.compile(&ctx.backend_ctx, ctx.ir.get(), decltype(ctx.backend)::EGST_PRESENT_WITH_AOV_EXTRACTION);
		ctx.backend_ctx.vt.commitAll();
		auto fragShader = createFragmentShader(compResult, ctx.backend_ctx.vt.getCPUVirtualTexture()->getFloatViews().size());
		auto ds0 = createDS0(ctx, pipelineLayout.get(), compResult, meshes.begin(), meshes.end());
		auto basePipeline = createPipeline(
			std::move(pipelineLayout),
			getBuiltinAsset<asset::ICPUSpecializedShader, asset::IAsset::ET_SPECIALIZED_SHADER>(VERTEX_SHADER_CACHE_KEY, m_assetMgr),
			std::move(fragShader)
		);
		ctx.meta->m_global.m_materialCompilerGLSL_declarations = compResult.fragmentShaderSource_declarations;
		ctx.meta->m_global.m_materialCompilerGLSL_source = compResult.fragmentShaderSource;
		ctx.meta->m_global.m_ds0 = ds0;

		ctx.meta->reserveDerivMapStorage(ctx.derivMapCache.size());
		for (auto& derivMap : ctx.derivMapCache)
		{
			ctx.meta->addDerivMapMeta(derivMap.first.get(), derivMap.second);
		}
#endif
		for (const auto& emitter : result.emitters)
		{
			if(emitter.element->type == ext::MitsubaLoader::CElementEmitter::Type::ENVMAP)
			{
				const auto& envmap = emitter.element->envmap;
				SAssetBundle envmapBundle = interm_getImageViewInHierarchy<std::string>(envmap.filename,ctx.inner.params,_hierarchyLevel,ctx.override_);
				auto contentRange = envmapBundle.getContents();
				if (contentRange.empty())
				{
					_params.logger.log("Could not load Envnmap image from path: %s",LoggerError,envmap.filename);
					continue;
				}
				ctx.scene->addEnvLight(ICPUScene::EEnvLightType::SphereMap,core::smart_refctd_ptr_static_cast<const asset::ICPUImageView>(contentRange[0]));
			}
		}

		ctx.transferMetadata();
		return SAssetBundle(std::move(result.metadata),{std::move(ctx.scene)});
	}
}

auto SContext::getMaterial(
	const CElementBSDF* bsdf, const CElementEmitter* frontFaceEmitter, const core::string& debugName, system::ISystem* debugFileWriter
) -> material_t
{
	// cache the BSDF part
	auto foundBSDF = bsdfCache.find(bsdf);
	if (foundBSDF ==bsdfCache.end())
		foundBSDF = bsdfCache.insert({bsdf,genMaterial(bsdf,debugFileWriter)}).first;

	// if we have no emitters, we can reuse existing BSDF subtree
	if (!frontFaceEmitter)
		return foundBSDF->second;
	assert(frontFaceEmitter->type==CElementEmitter::AREA);

	auto foundEmitter = emitterCache.find(frontFaceEmitter);
	if (foundEmitter==emitterCache.end())
		foundEmitter = emitterCache.insert({frontFaceEmitter,genEmitter(frontFaceEmitter,debugFileWriter)}).first;

	auto& frontPool = frontIR->getObjectPool();

	// A new root node gets made for every {bsdf,emitter} combo
	// TODO: cache this if memory/hash-load ever becomes a problem
	assert(frontFaceEmitter->type==CElementEmitter::AREA);
	const auto rootH = frontPool.emplace<frontend_ir_t::CLayer>();
	auto* const root = frontPool.deref(rootH);
	root->debugInfo = frontPool.emplace<frontend_ir_t::CDebugInfo>(debugName);
	if (auto* const original=frontPool.deref(foundBSDF->second); original)
	{
		// only front-face on top layer get changed, Mistuba XML only allows for unobscured/uncoated emission
		{
			// TODO replace with utility
			auto combinerH = frontPool.emplace<frontend_ir_t::CAdd>();
			auto* const combiner = frontPool.deref(combinerH);
			combiner->lhs = foundEmitter->second._const_cast();
			combiner->rhs = original->brdfTop;
			root->brdfTop = combinerH;
		}
		// rest stays the same
		root->btdf = original->btdf;
		root->brdfBottom = original->brdfBottom;
		root->coated = original->coated;
	}
	else
		root->brdfTop = foundEmitter->second._const_cast();
	const bool success = frontIR->addMaterial(rootH,inner.params.logger);
	
	auto logger = inner.params.logger;
	if (!success)
	{
		logger.log("Failed to add Material for %s",LoggerError,debugName.c_str());
		return {};
	}
	else if (debugFileWriter)
	{
		const frontend_ir_t::typed_pointer_type<const frontend_ir_t::CLayer> constRootH = rootH;
		frontend_ir_t::SDotPrinter printer = {frontIR.get(),{&constRootH,1}};
		writeDot3File(debugFileWriter,DebugDir/"material_frontend"/(debugName+".dot"),printer);
	}

	return rootH;
}


using parameter_t = asset::material_compiler3::CFrontendIR::SParameter;
parameter_t SContext::getTexture(const CElementTexture* const rootTex, hlsl::float32_t2x3* outUvTransform)
{
	parameter_t retval = {};
	// unroll scale
	const CElementTexture* tex = rootTex;
	for (retval.scale=1.f; tex && tex->type==CElementTexture::Type::SCALE; tex=tex->scale.texture)
		retval.scale *= tex->scale.scale;
	if (tex)
	{
		assert(tex->type==CElementTexture::Type::BITMAP);
		const auto& bitmap = tex->bitmap;
		SAssetBundle viewBundle = interm_getImageViewInHierarchy(tex->bitmap.filename,/*ICPUScene::MATERIAL_IMAGES_HIERARCHY_LEVELS_BELOW*/1);
		if (auto contents=viewBundle.getContents(); !contents.empty())
		{
			retval.view = IAsset::castDown<const ICPUImageView>(*contents.begin());
			if (bitmap.channel!=CElementTexture::Bitmap::CHANNEL::INVALID)
				retval.viewChannel = bitmap.channel-CElementTexture::Bitmap::CHANNEL::R;
			// get sampler parameters
			using tex_clamp_e = asset::ISampler::E_TEXTURE_CLAMP;
			auto getWrapMode = [](CElementTexture::Bitmap::WRAP_MODE mode)
			{
				switch (mode)
				{
					case CElementTexture::Bitmap::WRAP_MODE::CLAMP:
						return tex_clamp_e::ETC_CLAMP_TO_EDGE;
						break;
					case CElementTexture::Bitmap::WRAP_MODE::MIRROR:
						return tex_clamp_e::ETC_MIRROR;
						break;
					case CElementTexture::Bitmap::WRAP_MODE::ONE:
						assert(false); // TODO : replace whole texture?
						break;
					case CElementTexture::Bitmap::WRAP_MODE::ZERO:
						assert(false); // TODO : replace whole texture?
						break;
					default:
						break;
				}
				return tex_clamp_e::ETC_REPEAT;
			};
			auto& params = retval.sampler;
			params.TextureWrapU = getWrapMode(bitmap.wrapModeU);
			params.TextureWrapV = getWrapMode(bitmap.wrapModeV);
			switch (bitmap.filterType)
			{
				case CElementTexture::Bitmap::FILTER_TYPE::EWA:
					[[fallthrough]]; // we dont support this fancy stuff
				case CElementTexture::Bitmap::FILTER_TYPE::TRILINEAR:
					params.MinFilter = ISampler::ETF_LINEAR;
					params.MaxFilter = ISampler::ETF_LINEAR;
					params.MipmapMode = ISampler::ESMM_LINEAR;
					break;
				default:
					params.MinFilter = ISampler::ETF_NEAREST;
					params.MaxFilter = ISampler::ETF_NEAREST;
					params.MipmapMode = ISampler::ESMM_NEAREST;
					break;
			}
			params.AnisotropicFilter = core::max(hlsl::findMSB<uint32_t>(bitmap.maxAnisotropy),1u);
			// TODO: embed the gamma in the material compiler Frontend
			// or adjust gamma on pixels (painful and long process)
			//assert(std::isnan(bitmap.gamma));
			auto& transform = *outUvTransform;
			transform[0][0] = bitmap.uscale;
			transform[0][2] = bitmap.uoffset;
			transform[1][1] = bitmap.vscale;
			transform[1][2] = bitmap.voffset;
		}
		else
			inner.params.logger.log("Failed to load bitmap texture for %p with id %s",LoggerError,tex,tex ? tex->id.c_str():"");
	}
	else
		inner.params.logger.log("Failed to unroll texture scale for %p with id %s",LoggerError,rootTex,rootTex ? rootTex->id.c_str():"");
	if (!retval.view) // set a clear error value
		retval.scale = std::numeric_limits<decltype(retval.scale)>::signaling_NaN();
	return retval;
}


template<int N>
void getParameters(const std::span<parameter_t,N> out, const hlsl::vector<float32_t,N> value)
{
	for (auto c=0; c<N; c++)
		out[c].scale = value[c];
}
void getParameters(const std::span<parameter_t> out, const float32_t value)
{
	for (auto it=out.begin(); it!=out.end(); it++)
		it->scale = value;
}
hlsl::float32_t2x3 SContext::getParameters(const std::span<parameter_t,3> out, const CElementTexture::SpectrumOrTexture& src)
{
	auto retval = hlsl::math::linalg::diagonal<hlsl::float32_t2x3>(0.f);
	if (src.texture)
	{
		const auto param = getTexture(src.texture,&retval);
		for (auto c=0; c<out.size(); c++)
		{
			out[c].scale = param.scale;
			out[c].viewChannel = param.viewChannel+c;
			out[c].view = param.view;
			out[c].sampler = param.sampler;
		}
	}
	else
	switch (src.value.type)
	{
		case SPropertyElementData::Type::FLOAT:
			MitsubaLoader::getParameters({out.data(),out.size()},src.value.fvalue);
			break;
		case SPropertyElementData::Type::SRGB: [[fallthrough]]; // already linearized when parsed!
		case SPropertyElementData::Type::SPECTRUM: [[fallthrough]]; // we're not spectral but we convert <spectrum> tags to RGB approximately
		case SPropertyElementData::Type::RGB:
			MitsubaLoader::getParameters<3>(out,src.value.vvalue.xyz);
			break;
		default:
		assert(false);
			break;
	}
	return retval;
}
hlsl::float32_t2x3 SContext::getParameters(const std::span<parameter_t> out, const CElementTexture::FloatOrTexture& src)
{
	auto retval = hlsl::math::linalg::diagonal<hlsl::float32_t2x3>(0.f);
    if (src.texture)
	{
		const auto param = getTexture(src.texture,&retval);
		for (auto c=0; c<out.size(); c++)
		{
			out[c].scale = param.scale;
			out[c].viewChannel = param.viewChannel+c;
			out[c].view = param.view;
			out[c].sampler = param.sampler;
		}
	}
    else
		MitsubaLoader::getParameters(out,src.value);
	return retval;
}

using spectral_var_t = asset::material_compiler3::CFrontendIR::CSpectralVariable;
auto SContext::genEmitter(const CElementEmitter* _emitter, system::ISystem* debugFileWriter) -> frontend_emitter_t
{
	auto& frontPool = frontIR->getObjectPool();
	const auto handle = frontPool.emplace<frontend_ir_t::CMul>();
	auto* const mul = frontPool.deref(handle);
	// debug info first
	const core::string debugName = _emitter->id.empty() ? std::format("0x{:x}",ptrdiff_t(_emitter)):_emitter->id;
	mul->debugInfo = frontPool.emplace<frontend_ir_t::CDebugInfo>(debugName);
	// emitter
	{
		const auto emitterH = frontPool.emplace<frontend_ir_t::CEmitter>();
		// unit emission
		mul->lhs = emitterH;
		// but with a profile
		if (const auto* const inProfile=_emitter->area.emissionProfile; inProfile)
		{
			auto* const emitter = frontPool.deref(emitterH);
			auto found = profileCache.find(inProfile);
			if (found==profileCache.end())
				found = profileCache.insert({inProfile,genProfile(inProfile)}).first;
			emitter->profile = found->second;
			emitter->profileTransform = hlsl::math::linalg::truncate<3,3>(_emitter->transform.matrix);
		}
	}
	{
		spectral_var_t::SCreationParams<3> params = {};
		// if you wanted a textured emitter, this would be the place to do it
		MitsubaLoader::getParameters<3>(params.knots.params,_emitter->area.radiance);
		params.getSemantics() = spectral_var_t::Semantics::Fixed3_SRGB;
		mul->rhs = frontPool.emplace<spectral_var_t>(std::move(params));
	}

	if (debugFileWriter)
	{
		frontend_ir_t::SDotPrinter printer = {frontIR.get()};
		printer.exprStack.push(handle);
		writeDot3File(debugFileWriter,DebugDir/"material_frontend/emitters"/(debugName+".dot"),printer);
	}
	return handle;
}

auto SContext::genProfile(const CElementEmissionProfile* profile) -> frontend_ir_t::SParameter
{
	frontend_ir_t::SParameter retval = {};
	// load it!
	using namespace nbl::asset;
	const CIESProfileMetadata* iesMeta = nullptr;
	{
		auto assetLoaded = interm_getAssetInHierarchy(profile->filename,/*ICPUScene::EMITTER_PROFILE_HIERARCHY_LEVELS_BELOW*/1);
		if (auto* const meta=assetLoaded.getMetadata(); meta)
			iesMeta = meta->selfCast<const CIESProfileMetadata>();
		if (!iesMeta)
		{
			inner.params.logger.log("Could Load Emission Profile from \"%s\" or its not an IES profile!",LoggerError,profile->filename);
			return retval;
		}
		assert(assetLoaded.getAssetType()==IAsset::ET_IMAGE_VIEW);
		retval.view = IAsset::castDown<const ICPUImageView>(*assetLoaded.getContents().begin());
	}
	// continue
	retval.viewChannel = 0;
	const float maxIntesity = iesMeta->profile.getMaxCandelaValue();
    // note that IES texel intensity value is already divided by max 
	switch (profile->normalization)
	{
		// already normalized to max
		case CElementEmissionProfile::EN_UNIT_MAX:
			retval.scale = 1.f; // essentially `maxIntesity/maxIntesity`
			break;
		case CElementEmissionProfile::EN_UNIT_AVERAGE_OVER_IMPLIED_DOMAIN:
			retval.scale = maxIntesity/iesMeta->profile.getAvgEmmision(false);
			break;
		case CElementEmissionProfile::EN_UNIT_AVERAGE_OVER_FULL_DOMAIN:
			retval.scale = maxIntesity/iesMeta->profile.getAvgEmmision(true);
			break;
		default:
			retval.scale = maxIntesity;
			break;
	}
	return retval;
}

// TODO: include source debug information / location, e.g. XML path, line and column in the nodes
auto SContext::genMaterial(const CElementBSDF* bsdf, system::ISystem* debugFileWriter) -> frontend_material_t
{
	auto& frontPool = frontIR->getObjectPool();
	auto logger = inner.params.logger;
	const core::string debugName = bsdf->id.empty() ? std::format("0x{:x}",ptrdiff_t(bsdf)):bsdf->id;
	
	auto createFactorNode = [&](const CElementTexture::SpectrumOrTexture& factor, const ECommonDebug debug)->auto
	{
		spectral_var_t::SCreationParams<3> params = {};
		params.knots.uvTransform = getParameters(params.knots.params,factor);
		params.getSemantics() = spectral_var_t::Semantics::Fixed3_SRGB;
		const auto factorH = frontPool.emplace<spectral_var_t>(std::move(params));
		frontPool.deref(factorH)->debugInfo = commonDebugNames[uint16_t(debug)]._const_cast();
		return factorH;
	};

	struct SDerivativeMap
	{
		// TODO: derivative map SParameter[2]
	};
	// TODO: take `SParameter[2]` for the derivative maps
	auto createCookTorrance = [&](const CElementBSDF::RoughSpecularBase* base, const frontend_ir_t::typed_pointer_type<frontend_ir_t::CFresnel> fresnelH, const CElementTexture::SpectrumOrTexture& specularReflectance)->auto
	{
		const auto mulH = frontPool.emplace<frontend_ir_t::CMul>();
		const auto factorH = createFactorNode(specularReflectance,ECommonDebug::MitsubaExtraFactor);
		{
			auto* mul = frontPool.deref(mulH);
			const auto ctH = frontPool.emplace<frontend_ir_t::CCookTorrance>();
			{
				using ndf_e = frontend_ir_t::CCookTorrance::NDF;
				constexpr ndf_e ndfMap[4] = {
					ndf_e::Beckmann,
					ndf_e::GGX,
					ndf_e::Beckmann, // Phong can be mapped to Beckmann
					ndf_e::Beckmann // Ahskhmin is Ani Beckmann last I remember
				};
				auto* const ct = frontPool.deref(ctH);
				auto roughness = ct->ndParams.getRougness();
				if (base)
				{
					// ct->orientedRealEta gets set in the fresnel part
					ct->ndf = ndfMap[base->distribution];
					// this function sets both roughnesses to same alpha
					ct->ndParams.uvTransform = getParameters(roughness,base->alpha);
					if (base->alphaV.texture || !hlsl::isnan(base->alphaV.value))
					{
						// TODO: check if UV transform is the same and warn if not
						getParameters({roughness.data()+1,1},base->alphaV);
					}
					// TODO: derivative maps
				}
				else
					roughness[0].scale = roughness[1].scale = 0.f;
				// can only be set to monochrome 
				if (const auto& orientedRealEta=frontPool.deref(fresnelH)->orientedRealEta; frontPool.deref(orientedRealEta)->getKnotCount()==1)
					ct->orientedRealEta = orientedRealEta;
			}
			// cook torrance goes in most lhs side
			mul->lhs = ctH;
			mul->rhs = fresnelH;
		}
		return frontIR->createMul(mulH,factorH);
	};
	auto createOrenNayar = [&](const CElementTexture::SpectrumOrTexture& albedo, const CElementTexture::FloatOrTexture& alphaU, const CElementTexture::FloatOrTexture& alphaV)->auto
	{
		const auto orenNayarH = frontPool.emplace<frontend_ir_t::COrenNayar>();
		const auto factorH = createFactorNode(albedo,ECommonDebug::Albedo);
		{
			auto* orenNayar = frontPool.deref(orenNayarH);
			// TODO: factor this out between Oren-Nayar and Cook Torrance
			auto roughness = orenNayar->ndParams.getRougness();
			orenNayar->ndParams.uvTransform = getParameters(roughness,alphaU);
			if (alphaV.texture || !hlsl::isnan(alphaV.value))
			{
				// TODO: check if UV transform is the same and warn if not
				getParameters({roughness.data()+1,1},alphaV);
			}
			// TODO: derivative maps
		}
		return frontIR->createMul(orenNayarH,factorH);
	};

	auto fillCoatingLayer = [&]<typename T>(frontend_ir_t::CLayer* layer, const T& element, const bool rough, const frontend_ir_t::typed_pointer_type<frontend_ir_t::CBeer> extinctionH={})->void
	{
		const auto fresnelH = frontIR->createConstantMonochromeRealFresnel(element.intIOR/element.extIOR);
		const auto dielectricH = createCookTorrance(rough ? &element:nullptr,fresnelH,element.specularReflectance);
		layer->brdfTop = dielectricH;
		const auto transH = frontPool.emplace<frontend_ir_t::CMul>();
		{
			auto* const trans = frontPool.deref(transH);
			if (extinctionH)
				trans->lhs = frontIR->createMul(deltaTransmission._const_cast(),extinctionH);
			else
				trans->lhs = deltaTransmission._const_cast();
			const auto factorH = createFactorNode(element.specularTransmittance,ECommonDebug::MitsubaExtraFactor);
			trans->rhs = frontIR->createMul(fresnelH,factorH);
		}
		layer->btdf = transH;
		// identical BRDF on the bottom, to have correct multiscatter
		layer->brdfBottom = dielectricH;
	};

	struct SEntry
	{
		inline bool operator==(const SEntry& other) const {return bsdf==other.bsdf;}

		const CElementBSDF* bsdf;
		// SDerivativeMap derivMap;
	};
	struct HashEntry
	{
		inline size_t operator()(const SEntry& entry) const {return std::hash<const void*>()(entry.bsdf);}
	};
	core::unordered_map<SEntry,material_t,HashEntry> localCache;
	localCache.reserve(16);
	// the layer returned will never have a bottom BRDF
	auto createMistubaLeaf = [&](const SEntry& entry)->frontend_ir_t::typed_pointer_type<frontend_ir_t::CLayer>
	{
		const CElementBSDF* _bsdf = entry.bsdf;
		auto retval = frontPool.emplace<frontend_ir_t::CLayer>();
		auto* leaf = frontPool.deref(retval);
		switch (_bsdf->type)
		{
			case CElementBSDF::DIFFUSE: [[fallthrough]];
			case CElementBSDF::ROUGHDIFFUSE:
			{
				const auto roughDiffuseH = frontPool.emplace<frontend_ir_t::CMul>();
				{
					auto* mul = frontPool.deref(roughDiffuseH);
					{
						const auto orenNayarH = frontPool.emplace<frontend_ir_t::COrenNayar>();
						auto* orenNayar = frontPool.deref(orenNayarH);
						auto roughness = orenNayar->ndParams.getRougness();
						if (_bsdf->type==CElementBSDF::ROUGHDIFFUSE)
						{
							// we only support isotropic Oren-Nayar
							orenNayar->ndParams.uvTransform = getParameters(roughness,_bsdf->diffuse.alpha);
						}
						else
							roughness[0].scale = roughness[1].scale = 0.f;
						mul->lhs = orenNayarH;
					}
					{
						spectral_var_t::SCreationParams<3> params = {};
						params.knots.uvTransform = getParameters(params.knots.params,_bsdf->diffuse.reflectance);
						params.getSemantics() = spectral_var_t::Semantics::Fixed3_SRGB;
						const auto albedoH = frontPool.emplace<spectral_var_t>(std::move(params));
						frontPool.deref(albedoH)->debugInfo = commonDebugNames[uint16_t(ECommonDebug::Albedo)]._const_cast();
						mul->rhs = albedoH;
					}
				}
				leaf->brdfTop = roughDiffuseH;
				break;
			}
			case CElementBSDF::DIELECTRIC: [[fallthrough]];
			case CElementBSDF::THINDIELECTRIC: [[fallthrough]];
			case CElementBSDF::ROUGHDIELECTRIC: [[fallthrough]];
			case CElementBSDF::CONDUCTOR: [[fallthrough]];
			case CElementBSDF::ROUGHCONDUCTOR:
			{
				const bool isConductor = _bsdf->type==CElementBSDF::CONDUCTOR || _bsdf->type==CElementBSDF::ROUGHCONDUCTOR;
				// figure out the rough base to use
				const CElementBSDF::RoughSpecularBase* rough = nullptr;
				switch (_bsdf->type)
				{
					case CElementBSDF::THINDIELECTRIC: [[fallthrough]];
					case CElementBSDF::ROUGHDIELECTRIC:
						rough = &_bsdf->dielectric;
						break;
					case CElementBSDF::ROUGHCONDUCTOR:
						rough = &_bsdf->conductor;
						break;
				}
				// the fresnels
				frontend_ir_t::typed_pointer_type<frontend_ir_t::CFresnel> fresnelH;
				if (isConductor)
				{
					fresnelH = frontPool.emplace<frontend_ir_t::CFresnel>();
					auto* const fresnel = frontPool.deref(fresnelH);
					const float extEta = _bsdf->conductor.extEta;
					{
						spectral_var_t::SCreationParams<3> params = {};
						const hlsl::float32_t3 eta = _bsdf->conductor.eta.vvalue.xyz;
						MitsubaLoader::getParameters<3>(params.knots.params,eta/extEta);
						params.getSemantics() = spectral_var_t::Semantics::Fixed3_SRGB;
						fresnel->orientedRealEta = frontPool.emplace<spectral_var_t>(std::move(params));
					}
					{
						spectral_var_t::SCreationParams<3> params = {};
						const hlsl::float32_t3 k = _bsdf->conductor.k.vvalue.xyz;
						MitsubaLoader::getParameters<3>(params.knots.params,k/extEta);
						params.getSemantics() = spectral_var_t::Semantics::Fixed3_SRGB;
						fresnel->orientedImagEta = frontPool.emplace<spectral_var_t>(std::move(params));
					}
				}
				else
					fresnelH = frontIR->createConstantMonochromeRealFresnel(_bsdf->dielectric.intIOR/_bsdf->dielectric.extIOR);
				const auto brdfH = createCookTorrance(rough,fresnelH,isConductor ? _bsdf->conductor.specularReflectance:_bsdf->dielectric.specularReflectance);
				leaf->brdfTop = brdfH;
				if (!isConductor)
				{
					// want a specularTransmittance instead of SpecularReflectance factor
					const auto factorH = createFactorNode(_bsdf->dielectric.specularTransmittance,ECommonDebug::MitsubaExtraFactor);
					frontend_ir_t::typed_pointer_type<frontend_ir_t::IExprNode> btdfH;
					{
						const auto* const brdf = frontPool.deref(brdfH);
						// make the trans node refraction-less for thin dielectric and apply thin scattering correction to the transmissive fresnel
						if (_bsdf->type==CElementBSDF::THINDIELECTRIC)
						{
							const auto correctedTransmissionH = frontPool.emplace<frontend_ir_t::CMul>();
							{
								auto* mul = frontPool.deref(correctedTransmissionH);
								// apply energy conserving correction
								const auto thinInfiniteScatterH = frontPool.emplace<frontend_ir_t::CThinInfiniteScatterCorrection>();
								{
									auto* thinInfiniteScatter = frontPool.deref(thinInfiniteScatterH);
									thinInfiniteScatter->reflectanceTop = fresnelH;
									thinInfiniteScatter->reflectanceBottom = fresnelH;
									// TODO: extinction
								}
								mul->lhs = deltaTransmission._const_cast();
								mul->rhs = thinInfiniteScatterH;
							}
							// We need to attach the other glass interface as another layer because the Etas need to be reciprocated because the interactions are flipped
							leaf->coated = frontIR->reverse(retval);
							// we're rethreading the fresnel here, but needed to be careful to not apply thindielectric scatter correction and volumetric absorption / Mitsuba extra factor twice
							btdfH = correctedTransmissionH;
						}
						else
						{
							// beautiful thing we can reuse the same BxDF nodes for a transmission function without touching Etas or anything!
							btdfH = brdf->lhs;
							leaf->brdfBottom = brdfH;
						}
					}
					leaf->btdf = frontIR->createMul(btdfH,factorH);
				}
				break;
			}
			case CElementBSDF::PLASTIC: [[fallthrough]];
			case CElementBSDF::ROUGHPLASTIC:
			{
				fillCoatingLayer(leaf,_bsdf->plastic,_bsdf->type==CElementBSDF::ROUGHPLASTIC); // shall we plug albedo back inside as an extinction factor !?
				const auto substrateH = frontPool.emplace<frontend_ir_t::CLayer>();
				{
					// TODO: `_bsdf->plastic.nonlinear` to let the backend know the multiscatter should match provided albedo? Basically provided albedo is not the color at every bounce but after all bounces
					auto* const substrate = frontPool.deref(substrateH);
					substrate->brdfTop = createOrenNayar(_bsdf->plastic.diffuseReflectance,_bsdf->plastic.alphaU,_bsdf->plastic.alphaV);
				}
				leaf->coated = substrateH;
				break;
			}
			case CElementBSDF::PHONG:
				logger.log("Failed to Create a Phong BxDF for Material for %s, Phong is Unsupported",LoggerError,debugName.c_str());
				retval = unsupportedPhong._const_cast();
				break;
			case CElementBSDF::WARD:
				logger.log("Failed to Create a Ward BxDF for Material for %s, Ward is Unsupported",LoggerError,debugName.c_str());
				retval = unsupportedWard._const_cast();
				break;
			case CElementBSDF::DIFFUSE_TRANSMITTER:
			{
				const auto diffTransH = frontPool.emplace<frontend_ir_t::CMul>();
				{
					auto* mul = frontPool.deref(diffTransH);
					const CElementTexture::FloatOrTexture constZero = {0.f};
					mul->lhs = createOrenNayar(_bsdf->difftrans.transmittance,constZero,constZero);
					// normalize the Oren Nayar over the full sphere
					{
						spectral_var_t::SCreationParams<1> params = {};
						params.knots.params[0].scale = 0.5f;
						mul->rhs = frontPool.emplace<frontend_ir_t::CSpectralVariable>(std::move(params));
					}
				}
				leaf->brdfTop = diffTransH;
				leaf->btdf = diffTransH;
				// By default, all non-transmissive scattering models in Mitsuba are one-sided => all transmissive are two sided
				leaf->brdfBottom = diffTransH;
				break;
			}
			default:
				assert(false); // we shouldn't get this case here
				retval = errorMaterial._const_cast();
				break;
		}
		leaf = frontPool.deref(retval);
		assert(leaf->brdfTop);
		return retval;
	};

	// Post-order Depth First Traversal (create children first, then create parent)
	struct SStackEntry
	{
		SEntry immutable;
		bool visited = false;
	};
	core::vector<SStackEntry> stack;
	stack.reserve(128);
	stack.emplace_back() = {.immutable={.bsdf=bsdf}};
	//
	frontend_ir_t::typed_pointer_type<frontend_ir_t::CLayer> rootH = {};
	while (!stack.empty())
	{
		auto& entry = stack.back();
		const auto* const _bsdf = entry.immutable.bsdf;
		assert(_bsdf);
		// we only do post-dfs for non-leafs
		if (_bsdf->isMeta() && !entry.visited)
		{
			if (_bsdf->isMeta())
			{
				const auto& meta_common = _bsdf->meta_common;
				assert(meta_common.childCount);
				switch(_bsdf->type)
				{
					case CElementBSDF::COATING: [[fallthrough]];
					case CElementBSDF::ROUGHCOATING:
						assert(meta_common.childCount==1);
						break;
					case CElementBSDF::BUMPMAP: [[fallthrough]];
					case CElementBSDF::NORMALMAP:
						assert(meta_common.childCount==1);
						// TODO : create the derivative map and cache it
						break;
					case CElementBSDF::MASK:
						assert(meta_common.childCount==1);
						break;
					case CElementBSDF::MIXTURE_BSDF:
						break;
					case CElementBSDF::BLEND_BSDF:
						assert(meta_common.childCount==2);
						break;
					case CElementBSDF::TWO_SIDED:
						assert(meta_common.childCount<=2);
						break;
					default:
						assert(false); // we shouldn't get this case here
						break;
				}
				// TODO : make sure child gets pushed with derivative map info
				for (decltype(meta_common.childCount) i=0; i<meta_common.childCount; i++)
					stack.emplace_back() = {.immutable={.bsdf=meta_common.bsdf[i]}};
			}
			entry.visited = true;
		}
		else
		{
			material_t newMaterialH = {};
			if (_bsdf->isMeta())
			{
				const auto childCount = _bsdf->meta_common.childCount;
				auto getChildFromCache = [&](const CElementBSDF* child)->frontend_ir_t::typed_pointer_type<frontend_ir_t::CLayer>
				{
					return localCache[{.bsdf=child/*, TODO: copy the current normalmap stuff from entry or self if self is bump map*/}]._const_cast();
				};
				switch(_bsdf->type)
				{
					case CElementBSDF::COATING: [[fallthrough]];
					case CElementBSDF::ROUGHCOATING:
					{
						const auto coatingH = frontPool.emplace<frontend_ir_t::CLayer>();
						auto* const coating = frontPool.deref(coatingH);
						// the top layer
						const auto& sigmaA = _bsdf->coating.sigmaA;
						const bool hasExtinction = sigmaA.texture||sigmaA.value.type==SPropertyElementData::FLOAT&&sigmaA.value.fvalue!=0.f;
						const auto beerH = hasExtinction ? frontPool.emplace<frontend_ir_t::CBeer>():frontend_ir_t::typed_pointer_type<frontend_ir_t::CBeer>{};
						if (auto* const beer=frontPool.deref(beerH); beer)
						{
							spectral_var_t::SCreationParams<3> params = {};
							params.knots.uvTransform = getParameters(params.knots.params,sigmaA);
							params.getSemantics() = spectral_var_t::Semantics::Fixed3_SRGB;
							beer->perpTransmittance = frontPool.emplace<spectral_var_t>(std::move(params));
						}
						fillCoatingLayer(coating,_bsdf->coating,_bsdf->type==CElementBSDF::ROUGHCOATING,beerH);
						// attach the nested as layer
						coating->coated = getChildFromCache(_bsdf->mask.bsdf[0]);
						newMaterialH = coatingH;
						break;
					}
					case CElementBSDF::MASK:
					{
						const auto maskH = frontPool.emplace<frontend_ir_t::CLayer>();
						auto* const mask = frontPool.deref(maskH);
						//
						const auto nestedH = getChildFromCache(_bsdf->mask.bsdf[0]);
						const auto* const nested = frontPool.deref(nestedH);
						assert(nested && nested->brdfTop);
						const auto opacityH = createFactorNode(_bsdf->mask.opacity,ECommonDebug::Opacity);
						mask->brdfTop = frontIR->createMul(nested->brdfTop,opacityH);
						{
							mask->btdf = frontIR->createAdd(
								frontIR->createMul(deltaTransmission._const_cast(),frontIR->createComplement(opacityH)),
								frontIR->createMul(nested->btdf,opacityH)
							);
						}
						mask->brdfBottom = frontIR->createMul(nested->brdfBottom,opacityH);
						newMaterialH = maskH;
						break;
					} 
					case CElementBSDF::BUMPMAP: [[fallthrough]];
					case CElementBSDF::NORMALMAP:
					{
						// we basically ignore and skip because derivative map already applied
						newMaterialH = getChildFromCache(_bsdf->mask.bsdf[0]);
						break;
					}
					case CElementBSDF::MIXTURE_BSDF:
					{
						assert(false); // unimplemented
						break;
					}
					case CElementBSDF::BLEND_BSDF:
					{
						const auto blendH = frontPool.emplace<frontend_ir_t::CLayer>();
						auto* const blend = frontPool.deref(blendH);
						const auto tH = createFactorNode(_bsdf->blendbsdf.weight,ECommonDebug::Weight);
						const auto tComplementH = frontIR->createComplement(tH);
						const auto loH = getChildFromCache(_bsdf->blendbsdf.bsdf[0]);
						const auto hiH = getChildFromCache(_bsdf->blendbsdf.bsdf[1]);
						const auto* const lo = frontPool.deref(loH);
						const auto* const hi = frontPool.deref(hiH);
						// I don't actually need to check if the child Expressions are non-empty, the CFrontendIR utilities nicely carry through NOOPs
						blend->brdfTop = frontIR->createAdd(frontIR->createMul(lo->brdfTop,tComplementH),frontIR->createMul(hi->brdfTop,tH));
						blend->btdf = frontIR->createAdd(frontIR->createMul(lo->btdf,tComplementH),frontIR->createMul(hi->btdf,tH));
						blend->brdfBottom = frontIR->createAdd(frontIR->createMul(lo->brdfBottom,tComplementH),frontIR->createMul(hi->brdfBottom,tH));
						newMaterialH = blendH;
						break;
					}
					case CElementBSDF::TWO_SIDED:
					{
						const auto origFrontH = getChildFromCache(_bsdf->twosided.bsdf[0]);
						const auto chosenBackH = childCount!=1 ? getChildFromCache(_bsdf->twosided.bsdf[1]):origFrontH;
						// Mitsuba does a mad thing where it will pick the BSDF to use based on NdotV which would normally break the required reciprocity of a BxDF
						// but then it saves the day by disallowing transmission on the combination two BxDFs it layers together. Lets do the same.
						if (const bool firstIsTransmissive=frontIR->transmissive(origFrontH); firstIsTransmissive && (childCount==1 || frontIR->transmissive(chosenBackH)))
						{
							logger.log("Mitsuba <twosided> cannot be used to glue two transmissive BxDF Layer Stacks together!",LoggerError,debugName.c_str());
							break;
						}
						// this is the stack we attach to the back, but we need to reverse it
						const auto backH = frontIR->reverse(chosenBackH);
						if (!backH)
						{
							logger.log("Failed to reverse and reciprocate the BxDF Layer Stack!",LoggerError,debugName.c_str());
							break;
						}
						// we need to make a copy because we'll be changing the last layer
						const auto combinedH = frontIR->copyLayers(origFrontH);
						{
							auto* lastInFront = frontPool.deref(combinedH);
							// scroll to the end
							while (lastInFront->coated)
								lastInFront = frontPool.deref(lastInFront->coated);
							// attach the back stack
							lastInFront->coated = backH;
						}
						newMaterialH = combinedH;
						break;
					}
					default:
						assert(false); // we shouldn't get this case here
						break;
				}
			}
			else
				newMaterialH = createMistubaLeaf(entry.immutable);
			if (!newMaterialH)
				newMaterialH = errorMaterial;
			localCache[entry.immutable] = newMaterialH;
			stack.pop_back();
			if (stack.empty())
				rootH = newMaterialH._const_cast();
		}
	}
	if (!rootH)
		return errorMaterial;

	// add debug info
	auto* const root = frontPool.deref(rootH);
	root->debugInfo = frontPool.emplace<frontend_ir_t::CDebugInfo>(debugName);

	const bool success = frontIR->addMaterial(rootH,inner.params.logger);
	if (!success)
	{
		logger.log("Failed to add Material for %s",LoggerError,debugName.c_str());
		return errorMaterial;
	}
	else if (debugFileWriter)
	{
		const frontend_ir_t::typed_pointer_type<const frontend_ir_t::CLayer> constRootH = rootH;
		frontend_ir_t::SDotPrinter printer = {frontIR.get(),{&constRootH,1}};
		writeDot3File(debugFileWriter,DebugDir/"material_frontend/bsdfs"/(debugName+".dot"),printer);
	}
	return rootH;
}

#if 0
// Also sets instance data buffer offset into meshbuffers' base instance
template<typename Iter>
inline core::smart_refctd_ptr<asset::ICPUDescriptorSet> CMitsubaLoader::createDS0(const SContext& _ctx, asset::ICPUPipelineLayout* _layout, const asset::material_compiler::CMaterialCompilerGLSLBackendCommon::result_t& _compResult, Iter meshBegin, Iter meshEnd)
{

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

			const auto& bsdf = inst.bsdf;
			auto bsdf_front = bsdf.front;
			auto bsdf_back  = bsdf.back;
			auto material_it = _compResult.materials.find(bsdf_front);
			{
				const asset::material_compiler::oriented_material_t* material = 
					(material_it != _compResult.materials.end()) ? &material_it->second : nullptr;

				if (material) {
#ifdef DEBUG_MITSUBA_LOADER
				//os::Printer::log("Debug print front BSDF with id = ", std::to_string(&bsdf), ELL_INFORMATION);
				
					ofile << "Debug print front BSDF with id = " << &bsdf << std::endl;
					_ctx.backend.debugPrint(ofile, *material, _compResult, &_ctx.backend_ctx);
			
#endif
					instData.material.front = *material;
				}
			}
			material_it = _compResult.materials.find(bsdf_back);
			{
				const asset::material_compiler::oriented_material_t* material = 
					(material_it != _compResult.materials.end()) ? &material_it->second : nullptr;

				if (material)
				{
#ifdef DEBUG_MITSUBA_LOADER
				//os::Printer::log("Debug print back BSDF with id = ", std::to_string(&bsdf), ELL_INFORMATION);
					ofile << "Debug print back BSDF with id = " << &bsdf << std::endl;
					_ctx.backend.debugPrint(ofile, *material, _compResult, &_ctx.backend_ctx);
#endif

					instData.material.back = *material;
				}
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
#endif

using namespace std::string_literals;

SContext::SContext(
	const asset::IAssetLoader::SAssetLoadContext& _ctx,
	asset::IAssetLoader::IAssetLoaderOverride* _override,
	CMitsubaMetadata* _metadata
) : inner(_ctx), override_(_override), meta(_metadata)
//,ir(core::make_smart_refctd_ptr<asset::material_compiler::IR>()), frontend(this)
{
	auto materialPool = material_compiler3::CTrueIR::create({.composed={.blockSizeKBLog2=4}});
	scene = ICPUScene::create(core::smart_refctd_ptr(materialPool)); // TODO: feed it max shapes per group
	//
	{
		frontIR = frontend_ir_t::create({.composed={.blockSizeKBLog2=4}});
		auto& frontPool = frontIR->getObjectPool();
		{
			constexpr frontend_material_t BlackHoleBxDF = {};
			// Can't have an empty material
			//{
			//	auto handle = frontPool.emplace<frontend_ir_t::CLayer>();
			//	frontPool.deref(handle)->debugInfo = frontPool.emplace<frontend_ir_t::CDebugInfo>("VantaBlackHole");
			//	blackHoleBxDF = handle;
			//}
			//const bool success = frontIR->addMaterial(blackHoleBxDF,inner.params.logger);
			//assert(success);
			bsdfCache.insert({nullptr,BlackHoleBxDF});
		}
		//
		{
			deltaTransmission = frontPool.emplace<frontend_ir_t::CDeltaTransmission>();
			const auto mulH = frontPool.emplace<frontend_ir_t::CMul>();
			{
				auto* const mul = frontPool.deref(mulH);
				mul->lhs = frontPool.emplace<frontend_ir_t::COrenNayar>();
				spectral_var_t::SCreationParams<3> params = {};
				MitsubaLoader::getParameters<3>(params.knots.params,{1.f,0.f,1.f});
				params.getSemantics() = spectral_var_t::Semantics::Fixed3_SRGB;
				mul->rhs = frontPool.emplace<spectral_var_t>(std::move(params));
			}
			errorBRDF = mulH;
			auto constructUnsupported = [&](const std::string_view debugName)->auto
			{
				const auto rootH = frontPool.emplace<frontend_ir_t::CLayer>();
				auto* const root = frontPool.deref(rootH);
				root->brdfTop = errorBRDF._const_cast();
				root->debugInfo = frontPool.emplace<frontend_ir_t::CDebugInfo>(debugName);
				return rootH;
			};
			errorMaterial = constructUnsupported("ERROR Layer");
			unsupportedPhong = constructUnsupported("UNSUPPORTED Phong");
			unsupportedWard = constructUnsupported("UNSUPPORTED Ward");

		}
		// debug names
		{
#define ADD_DEBUG_NODE(NAME) commonDebugNames[uint16_t(ECommonDebug::NAME)] = frontPool.emplace<frontend_ir_t::CDebugInfo>(#NAME)
			ADD_DEBUG_NODE(Albedo);
			ADD_DEBUG_NODE(Weight);
			ADD_DEBUG_NODE(Opacity);
			ADD_DEBUG_NODE(MitsubaExtraFactor);
#undef ADD_DEBUG_NODE
		}
	}
}

auto SContext::loadShapeGroup(const CElementShape* shape) -> SContext::shape_ass_type
{
	assert(shape->type==CElementShape::Type::SHAPEGROUP);
	const auto* const shapegroup = &shape->shapegroup;
	auto found = groupCache.find(shapegroup);
	if (found!=groupCache.end())
		return found->second.collection;
	
	auto collection = core::make_smart_refctd_ptr<ICPUGeometryCollection>();
	if (!collection)
		inner.params.logger.log("Failed to create an ICPUGeometryCollection for Shape Group",LoggerError);
	else
	{
		auto* geometries = collection->getGeometries();
		const auto children = shapegroup->children;
		for (auto i=0u; i<shapegroup->childCount; i++)
		{
			auto child = children[i];
			if (!child)
				continue;
			// shape groups cannot contain instances
			assert(child->type!=CElementShape::Type::INSTANCE);

			shape_ass_type nestedCollection;
			if (child->type!=CElementShape::Type::SHAPEGROUP)
				nestedCollection = loadBasicShape(child);
			else
				nestedCollection = loadShapeGroup(child);
			if (!nestedCollection)
				continue;

			// note that we flatten geometry collections, different children are their own collections we turn them into one mega-collection
			const auto& nestedGeometries = nestedCollection->getGeometries();
			// thankfully because SHAPEGROUPS are not allowed to have transforms we don't need to rack them up
			//if (newRef.hasTransform())
			//	newRef.transform = hlsl::mul(thisTransform,newRef.transform);
			//else
			//	newRef.transform = thisTransform;
			geometries->insert(geometries->end(),nestedGeometries.begin(),nestedGeometries.end());
		}
		CMitsubaMetadata::SGeometryCollectionMetaPair pair = {.collection=collection};
		pair.meta.m_id = shape->id;
		pair.meta.type = shape->type;
		groupCache.insert({shapegroup,std::move(pair)});
	}
	return collection;
}

auto SContext::loadBasicShape(const CElementShape* shape) -> SContext::shape_ass_type
{
	auto found = shapeCache.find(shape);
	if (found!=shapeCache.end())
		return found->second.collection;

	auto collection = core::make_smart_refctd_ptr<ICPUGeometryCollection>();
	if (!collection)
	{
		inner.params.logger.log("Failed to create an ICPUGeometryCollection non-Instanced Shape with id %s",LoggerError,shape->id.c_str());
		return nullptr;
	}
	// the geometry reference transform shall only contain an exceptional and optional relative transform like to make Builtin shapes like cubes, spheres, etc. of different sizes
	// the whole shape (which is a geometry collection) has its own transform
	auto* pGeometries = collection->getGeometries();
	auto addGeometry = [pGeometries](ICPUGeometryCollection::SGeometryReference&& ref)->void
	{
		if (ref)
			pGeometries->push_back(std::move(ref));
	};


	auto loadModel = [&](const char* filename, int64_t index=-1) -> void
	{
		auto retval = interm_getAssetInHierarchy(filename,/*ICPUScene::GEOMETRY_COLLECTION_HIERARCHY_LEVELS_BELOW*/1);
		auto contentRange = retval.getContents();
		if (contentRange.empty())
		{
			inner.params.logger.log("Could Not Load Shape : %s",LoggerError,filename);
			return;
		}
		
		// we used to load with the IAssetLoader::ELPF_RIGHT_HANDED_MESHES flag, this means flipping the mesh x-axis
		auto transform = math::linalg::diagonal<float32_t3x4>(1.f);
		transform[0][0] = -1.f;

		//
		auto addCollectionGeometries = [&](const ICPUGeometryCollection* col)->void
		{
			if (col)
			for (auto ref : col->getGeometries())
			{
				if (ref.hasTransform())
					ref.transform = math::linalg::promoted_mul(ref.transform,transform);
				else
					ref.transform = transform;
				addGeometry(std::move(ref));
			}
		};

		// take first target and replace the collection
		auto addFirstTargetGeometries = [&](const ICPUMorphTargets* morph)->void
		{
			if (const auto& targets=morph->getTargets(); !targets.empty())
				addCollectionGeometries(targets.front().geoCollection.get());
		};

		switch (retval.getAssetType())
		{
			case IAsset::ET_GEOMETRY:
			{
				// only add one geometry, if we meant to add a whole collection, the file would load a collection
				const IGeometry<ICPUBuffer>* geo = nullptr;
				auto serializedMeta = retval.getMetadata()->selfCast<CMitsubaSerializedMetadata>();
				for (auto it=contentRange.begin(); it!=contentRange.end(); it++)
				{
					geo = IAsset::castDown<const ICPUPolygonGeometry>(*it).get();
					assert(geo);
					if (!serializedMeta || index<0ll || index>numeric_limits<uint32_t>::max) // not Misuba serialized or shape index not specialized
						break;
					auto* const meta = serializedMeta->getAssetSpecificMetadata(static_cast<const ICPUPolygonGeometry*>(geo));
					assert(meta);
					auto* const polygonMeta = static_cast<const CMitsubaSerializedMetadata::CPolygonGeometry*>(meta);
					if (polygonMeta->m_id==static_cast<uint32_t>(index))
						break;
				}
				if (auto* const mg=const_cast<IGeometry<ICPUBuffer>*>(geo); mg)
					addGeometry({.transform=transform,.geometry=core::smart_refctd_ptr<IGeometry<ICPUBuffer>>(mg)});
				break;
			}
			case IAsset::ET_GEOMETRY_COLLECTION:
			{
				// only add the first collection's geometries
				addCollectionGeometries(IAsset::castDown<const ICPUGeometryCollection>(contentRange[0]).get());
				break;
			}
			case IAsset::ET_MORPH_TARGETS:
			{
				addFirstTargetGeometries(IAsset::castDown<const ICPUMorphTargets>(contentRange[0]).get());
				break;
			}
			case IAsset::ET_SCENE:
			{
				// flatten the scene into a single instance, this is path for OBJ loading
				const auto& instances = IAsset::castDown<const ICPUScene>(contentRange[0])->getInstances();
				const auto instanceTforms = instances.getInitialTransforms();
				for (auto i=0u; i<instances.size(); i++)
				{
					auto* const targets = instances.getMorphTargets()[i].get();
					const auto oldGeoBegin = pGeometries->size();
					addFirstTargetGeometries(targets);
					if (!instanceTforms.empty())
					for (auto geoIx=oldGeoBegin; geoIx<pGeometries->size(); geoIx++)
					{
						auto& ref = pGeometries->operator[](geoIx);
						ref.transform = math::linalg::promoted_mul(instanceTforms[i],ref.transform);
					}
					// NOTE: also need to preserve/forward the materials somehow (need to chape the `shape_ass_type` to have a default Material Binding Table)
				}
				break;
			}
			default:
				inner.params.logger.log("Loaded an Asset but it didn't contain any geometry, was %s",LoggerError,system::to_string(retval.getAssetType()));
				break;
		}
	};

	bool flipNormals = false;
	bool faceNormals = false;
	float maxSmoothAngle = bit_cast<float>(numeric_limits<float>::quiet_NaN);
	auto* const creator = override_->getGeometryCreator();
	switch (shape->type)
	{
		// TODO: cache the simple geos to not spam new objects ?
		// FAR TODO: create some special non-poly geometries for procedural raycasts?
		case CElementShape::Type::CUBE:
		{
			flipNormals = flipNormals!=shape->cube.flipNormals;
			addGeometry({.geometry=creator->createCube(promote<float32_t3>(2.f))});
			break;
		}
		case CElementShape::Type::SPHERE:
			flipNormals = flipNormals!=shape->sphere.flipNormals;
			{
				auto tform = math::linalg::diagonal<float32_t3x4>(shape->sphere.radius);
				math::linalg::setTranslation(tform,shape->sphere.center);
				addGeometry({.transform=tform,.geometry=creator->createSphere(1.f,64u,64u)});
			}
			break;
		case CElementShape::Type::CYLINDER:
			flipNormals = flipNormals!=shape->cylinder.flipNormals;
			{
				// start off as transpose, so rows are columns
				float32_t4x3 extra;
				extra[2] = shape->cylinder.p1 - shape->cylinder.p0;
				extra[3] = shape->cylinder.p0;
				math::frisvad(normalize(extra[2]),extra[0],extra[1]);
				for (auto i=0u; i<2u; i++)
				{
					assert(length(extra[i])==1.f);
					extra[i] *= shape->cylinder.radius;
				}
				addGeometry({.transform=transpose(extra),.geometry=creator->createCylinder(1.f,1.f,64u)});
			}
			break;
		case CElementShape::Type::RECTANGLE:
			flipNormals = flipNormals!=shape->cylinder.flipNormals;
			addGeometry({.geometry=creator->createRectangle(promote<float32_t2>(1.f))});
			break;
		case CElementShape::Type::DISK:
			flipNormals = flipNormals!=shape->cylinder.flipNormals;
			addGeometry({.geometry=creator->createDisk(1.f,64)});
			break;
		case CElementShape::Type::OBJ:
			assert(false);
#if 0 // TODO: Arek
			mesh = loadModel(shape->obj.filename);
			flipNormals = flipNormals!=shape->obj.flipNormals;
			faceNormals = shape->obj.faceNormals;
			maxSmoothAngle = shape->obj.maxSmoothAngle;
			if (!pGeometries->empty() && shape->obj.flipTexCoords)
			{
				_NBL_DEBUG_BREAK_IF(true);
				// TODO: find the UV attribute, it doesn't help we don't name them
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
#endif
			// collapse parameter gets ignored
			break;
		case CElementShape::Type::PLY:
			_NBL_DEBUG_BREAK_IF(true); // this code has never been tested
			loadModel(shape->ply.filename);
			flipNormals = flipNormals!=shape->ply.flipNormals;
			faceNormals = shape->ply.faceNormals;
			maxSmoothAngle = shape->ply.maxSmoothAngle;
			if (shape->ply.srgb)
			for (auto& ref : *pGeometries)
			{
				// TODO: find the color attribute  (it doesn't help we don't name them, just slap them in vectors)
				// TODO: clone geometry
				// TODO: change the color aux attribute's format from UNORM8 to SRGB
			}
			break;
		case CElementShape::Type::SERIALIZED:
			loadModel(shape->serialized.filename,shape->serialized.shapeIndex);
			flipNormals = flipNormals!=shape->serialized.flipNormals;
			faceNormals = shape->serialized.faceNormals;
			maxSmoothAngle = shape->serialized.maxSmoothAngle;
			break;
		case CElementShape::Type::SHAPEGROUP:
			[[fallthrough]];
		case CElementShape::Type::INSTANCE:
			assert(false); // this shouldn't happen, our parser code shouldn't reach here
			break;
		default:
//			_NBL_DEBUG_BREAK_IF(true);
			break;
	}
	// handle fail
	if (pGeometries->empty())
	{
		inner.params.logger.log("Failed to Load/Create Basic non-Instanced Shape with id %s",LoggerError,shape->id.c_str());
		return nullptr;
	}

	// recompute and flip normals if necessary
	if (faceNormals || !std::isnan(maxSmoothAngle))
	{
		for (auto& ref : *pGeometries)
		{
			const float smoothAngleCos = cos(radians(maxSmoothAngle));

			auto* const polyGeo = static_cast<ICPUPolygonGeometry*>(ref.geometry.get());
			ref.geometry = CPolygonGeometryManipulator::createSmoothVertexNormal(
				CPolygonGeometryManipulator::createUnweldedList(polyGeo,flipNormals,false).get(),false,0.f, // TODO: maybe enable welding based on `!faceNormals` later
				[faceNormals,smoothAngleCos](const CPolygonGeometryManipulator::SSNGVertexData& v0, const CPolygonGeometryManipulator::SSNGVertexData& v1, const ICPUPolygonGeometry* buffer)
				{ 
					if (faceNormals)
						return v0.index==v1.index;
					else
						return dot(v0.weightedNormal,v1.weightedNormal)*rsqrt(dot(v0.weightedNormal,v0.weightedNormal)*dot(v1.weightedNormal,v1.weightedNormal)) >= smoothAngleCos;
				},
				true // rewelding or initial unweld mess with all vertex attributes and index buffers, so recompute every hash
			);
		}
	}
	else if (flipNormals)
	{
		for (auto& ref : *pGeometries)
		{
			auto* const polyGeo = static_cast<ICPUPolygonGeometry*>(ref.geometry.get());
			auto flippedGeo = CPolygonGeometryManipulator::createTriangleListIndexing(polyGeo,true,false);
			CGeometryManipulator::recomputeContentHash(flippedGeo->getIndexView());
			// TODO: don't we also need to flip the normal buffer values? changing the winding doesn't help because the normals weren't recomputed !
			ref.geometry = std::move(flippedGeo);
		}
	}

	// cache and return
	CMitsubaMetadata::SGeometryCollectionMetaPair pair = {.collection=collection};
	pair.meta.m_id = shape->id;
	pair.meta.type = shape->type;
	shapeCache.insert({shape,std::move(pair)});
	return collection;
}

}
}