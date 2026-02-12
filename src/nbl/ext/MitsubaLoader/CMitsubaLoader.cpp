// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include <cwchar>

#include "nbl/ext/MitsubaLoader/CMitsubaLoader.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"

#if 0
#include "nbl/asset/utils/CDerivativeMapCreator.h"

#include "nbl/ext/MitsubaLoader/CMitsubaSerializedMetadata.h"
#endif


#if defined(_NBL_DEBUG) || defined(_NBL_RELWITHDEBINFO)
#	define DEBUG_MITSUBA_LOADER
#endif

namespace nbl
{
using namespace asset;

namespace ext::MitsubaLoader
{

#if 0 // old material compiler
_NBL_STATIC_INLINE_CONSTEXPR const char* FRAGMENT_SHADER_DEFINITIONS =
R"(
vec3 nbl_glsl_MC_getNormalizedWorldSpaceV()
{
	vec3 campos = ....;
	return normalize(campos - WorldPos);
}
vec3 nbl_glsl_MC_getNormalizedWorldSpaceN()
{
	return normalize(Normal);
}

mat2x3 nbl_glsl_perturbNormal_dPdSomething()
{
	return mat2x3(dFdx(WorldPos),dFdy(WorldPos));
}
mat2 nbl_glsl_perturbNormal_dUVdSomething()
{
    return mat2(dFdx(UV),dFdy(UV));
}
)";
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
static core::smart_refctd_ptr<asset::ICPUImage> createSingleChannelImage(const asset::ICPUImage* _img, const asset::ICPUImageView::SComponentMapping::E_SWIZZLE srcChannel)
{
	// deprecated will be expressed in Material Compiler Frontend AST as a swizzle
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
		SContext ctx(
//			m_assetMgr->getGeometryCreator(),
//			m_assetMgr->getMeshManipulator(),
			IAssetLoader::SAssetLoadContext{ 
				IAssetLoader::SAssetLoadParams(_params.decryptionKeyLen,_params.decryptionKey,_params.cacheFlags,_params.loaderFlags,_params.logger,_file->getFileName().parent_path()),
				_file
			},
			_override,
			result.metadata.get()
		);
		//
		ctx.scene->m_ambientLight = result.ambient;


		// TODO: abstract/move away since many loaders will need to do this
		core::unordered_map<const ICPUGeometryCollection*,core::smart_refctd_ptr<ICPUMorphTargets>> morphTargetCache;
		auto createMorphTargets = [&_params,&morphTargetCache](core::smart_refctd_ptr<ICPUGeometryCollection>&& collection)->core::smart_refctd_ptr<ICPUMorphTargets>
		{
			auto found = morphTargetCache.find(collection.get());
			if (found!=morphTargetCache.end())
				return found->second;
			auto targets = core::make_smart_refctd_ptr<ICPUMorphTargets>();
			if (targets)
			{
				morphTargetCache[collection.get()] = targets;
				targets->getTargets()->push_back({.geoCollection=std::move(collection)});
			}
			return targets;
		};

		//
		auto& instances = ctx.scene->getInstances();
		instances.reserve(result.shapegroups.size());
		auto addToScene = [&](const CElementShape* shape, core::smart_refctd_ptr<ICPUGeometryCollection>&& collection)->void
		{
			assert(shape && collection);
			auto targets = createMorphTargets(std::move(collection));
			if (!targets)
			{
				_params.logger.log("Failed to create ICPUMorphTargets for Shape with id %s",LoggerError,shape->id.c_str());
				return;
			}
			const auto index = instances.size();
			instances.resize(index+1);
			instances.getMorphTargets()[index] = std::move(targets);
			// TODO: add materials (incl emission) to the instances
			/*
				auto emitter = shape->obtainEmitter();
				auto bsdf = getBSDFtreeTraversal(ctx, shape->bsdf, &emitter, getAbsoluteTransform());

				SContext::SInstanceData instance(
					tform,
					bsdf,
		#if defined(_NBL_DEBUG) || defined(_NBL_RELWITHDEBINFO)
					shape->bsdf ? shape->bsdf->id : "",
		#endif
					emitter,
					CElementEmitter{} // no backface emission
				);
			*/
			instances.getInitialTransforms()[index] = shape->getTransform();
		};

		// first go over all actually used shapes which are not shapegroups (regular shapes and instances)
		for (auto& shapepair : result.shapegroups)
		{
			auto* shapedef = shapepair.element;
			// this should be filtered out while parsing and we should just assert it
			if (shapedef->type==CElementShape::Type::SHAPEGROUP)
				continue;

			if (shapedef->type!=CElementShape::Type::INSTANCE)
			{
				auto geometry = ctx.loadBasicShape(_hierarchyLevel,shapedef);
				if (!geometry)
					continue;
				auto collection = core::make_smart_refctd_ptr<ICPUGeometryCollection>();
				if (!collection)
				{
					_params.logger.log("Failed to create an ICPUGeometryCollection non-Instanced Shape with id %s",LoggerError,shapedef->id.c_str());
					continue;
				}
				// we don't put a transform on the geometry, because we want the transform on the instance
				collection->getGeometries()->push_back({.geometry=std::move(geometry)});
				addToScene(shapedef,std::move(collection));
			}
			else // mitsuba is weird and lists instances under a shapegroup instead of having instances reference the shapegroup
			{
				// get group reference
				const CElementShape* parent = shapedef->instance.parent;
				if (!parent) // we should probably assert this
					continue;
				assert(parent->type==CElementShape::Type::SHAPEGROUP);
				auto collection = ctx.loadShapeGroup(_hierarchyLevel,&parent->shapegroup);
				addToScene(shapedef,std::move(collection));
			}
		}
		result.shapegroups.clear();

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
		return asset::SAssetBundle(std::move(result.metadata),{std::move(ctx.scene)});
	}
}

#if 0
void CMitsubaLoader::cacheEmissionProfile(SContext& ctx, const CElementEmissionProfile* profile)
{
	if (!profile)
		return;

	auto params = ctx.inner.params;
	params.loaderFlags = asset::IAssetLoader::ELPF_LOAD_METADATA_ONLY;

	auto assetLoaded = interm_getAssetInHierarchy( profile->filename, params, 0u, ctx.override_);

	if (!assetLoaded.getMetadata())
	{
		os::Printer::log("[ERROR] Could Not Find Emission Profile: " + profile->filename, ELL_ERROR);
		return;
	}
}

void CMitsubaLoader::cacheTexture(SContext& ctx, uint32_t hierarchyLevel, const CElementTexture* tex, const CMitsubaMaterialCompilerFrontend::E_IMAGE_VIEW_SEMANTIC semantic)
{
	if (!tex)
		return;

	switch (tex->type)
	{
		case CElementTexture::Type::BITMAP:
			{
				// get sampler parameters
				const auto samplerParams = ctx.computeSamplerParameters(tex->bitmap);
				
				asset::SAssetBundle viewBundle = interm_getImageViewInHierarchy(tex->bitmap.filename.svalue,ctx.inner,hierarchyLevel,ctx.override_);
				// TODO: embed the gamma in the material compiler Frontend
				// adjust gamma on pixels (painful and long process)
				if (!std::isnan(tex->bitmap.gamma))
				{
					_NBL_DEBUG_BREAK_IF(true);
				}
				{
					//! TODO: this stuff (custom shader sampling code?)
					_NBL_DEBUG_BREAK_IF(tex->bitmap.uoffset != 0.f);
					_NBL_DEBUG_BREAK_IF(tex->bitmap.voffset != 0.f);
					_NBL_DEBUG_BREAK_IF(tex->bitmap.uscale != 1.f);
					_NBL_DEBUG_BREAK_IF(tex->bitmap.vscale != 1.f);
				}
			}
			break;
		case CElementTexture::Type::SCALE:
			// get to to the linked list end
			cacheTexture(ctx,hierarchyLevel,tex->scale.texture,semantic);
			break;
		default:
			_NBL_DEBUG_BREAK_IF(true);
			break;
	}
}

auto CMitsubaLoader::getBSDFtreeTraversal(SContext& ctx, const CElementBSDF* bsdf, const CElementEmitter* emitter, core::matrix4SIMD tform) -> SContext::bsdf_type
{
	if (!bsdf)
	{
		static auto blackBSDF = []() -> auto
		{
			CElementBSDF retval("nullptr BSDF");
			retval.type = CElementBSDF::Type::DIFFUSE,
			retval.diffuse.reflectance = 0.f;
			retval.diffuse.alpha = 0.f;
			return retval;
		}();
		bsdf = &blackBSDF;
	}

	auto found = ctx.instrStreamCache.find(bsdf);
	if (found == ctx.instrStreamCache.end())
		found = ctx.instrStreamCache.insert({ bsdf,genBSDFtreeTraversal(ctx, bsdf) }).first;
	auto compiled_bsdf = found->second;

	// TODO cache the IR Node
	CMitsubaMaterialCompilerFrontend::EmitterNode* emitterIRNode = nullptr;
	if (emitter->type == CElementEmitter::AREA)
	{
		cacheEmissionProfile(ctx,emitter->area.emissionProfile);
		emitterIRNode = ctx.frontend.createEmitterNode(ctx.ir.get(),emitter,tform);
	}

	// A new root node gets made for every {bsdf,emitter} combo
	using node_t = asset::material_compiler::IR::INode;
	auto createNewRootNode = [&ctx,emitterIRNode](node_t* realBxDFRoot, node_t* emitter=nullptr) -> node_t*
	{
		// TODO: cache the combo!
		auto newRoot = ctx.ir->allocNode<asset::material_compiler::IR::CRootNode>();
		if (emitter)
			newRoot->children = node_t::createChildrenArray(realBxDFRoot,emitter);
		else
			newRoot->children = node_t::createChildrenArray(realBxDFRoot);
		ctx.ir->addRootNode(newRoot);
		return newRoot;
	};

	return { createNewRootNode(compiled_bsdf.front,emitterIRNode), createNewRootNode(compiled_bsdf.back)};
}

auto CMitsubaLoader::genBSDFtreeTraversal(SContext& ctx, const CElementBSDF* _bsdf) -> SContext::bsdf_type
{
	{
		auto cachePropertyTexture = [&](const auto& const_or_tex, const CMitsubaMaterialCompilerFrontend::E_IMAGE_VIEW_SEMANTIC semantic=CMitsubaMaterialCompilerFrontend::EIVS_IDENTITIY) -> void
		{
			if (const_or_tex.value.type==SPropertyElementData::INVALID)
				cacheTexture(ctx,0u,const_or_tex.texture,semantic);
		};

		core::stack<const CElementBSDF*> stack;
		stack.push(_bsdf);

		while (!stack.empty())
		{
			auto* bsdf = stack.top();
			stack.pop();
			//
			switch (bsdf->type)
			{
				case CElementBSDF::COATING:
					for (uint32_t i = 0u; i < bsdf->coating.childCount; ++i)
						stack.push(bsdf->coating.bsdf[i]);
					break;
				case CElementBSDF::ROUGHCOATING:
				case CElementBSDF::BUMPMAP:
				case CElementBSDF::BLEND_BSDF:
				case CElementBSDF::MIXTURE_BSDF:
				case CElementBSDF::MASK:
				case CElementBSDF::TWO_SIDED:
					for (uint32_t i = 0u; i < bsdf->meta_common.childCount; ++i)
						stack.push(bsdf->meta_common.bsdf[i]);
				default:
					break;
			}
			//
			switch (bsdf->type)
			{
				case CElementBSDF::DIFFUSE:
				case CElementBSDF::ROUGHDIFFUSE:
					cachePropertyTexture(bsdf->diffuse.reflectance);
					cachePropertyTexture(bsdf->diffuse.alpha);
					break;
				case CElementBSDF::DIFFUSE_TRANSMITTER:
					cachePropertyTexture(bsdf->difftrans.transmittance);
					break;
				case CElementBSDF::DIELECTRIC:
				case CElementBSDF::THINDIELECTRIC:
				case CElementBSDF::ROUGHDIELECTRIC:
					cachePropertyTexture(bsdf->dielectric.alphaU);
					if (bsdf->dielectric.distribution == CElementBSDF::RoughSpecularBase::ASHIKHMIN_SHIRLEY)
						cachePropertyTexture(bsdf->dielectric.alphaV);
					break;
				case CElementBSDF::CONDUCTOR:
				case CElementBSDF::ROUGHCONDUCTOR:
					cachePropertyTexture(bsdf->conductor.alphaU);
					if (bsdf->conductor.distribution == CElementBSDF::RoughSpecularBase::ASHIKHMIN_SHIRLEY)
						cachePropertyTexture(bsdf->conductor.alphaV);
					break;
				case CElementBSDF::PLASTIC:
				case CElementBSDF::ROUGHPLASTIC:
					cachePropertyTexture(bsdf->plastic.diffuseReflectance);
					cachePropertyTexture(bsdf->plastic.alphaU);
					if (bsdf->plastic.distribution == CElementBSDF::RoughSpecularBase::ASHIKHMIN_SHIRLEY)
						cachePropertyTexture(bsdf->plastic.alphaV);
					break;
				case CElementBSDF::BUMPMAP:
					cacheTexture(ctx,0u,bsdf->bumpmap.texture,bsdf->bumpmap.wasNormal ? CMitsubaMaterialCompilerFrontend::EIVS_NORMAL_MAP:CMitsubaMaterialCompilerFrontend::EIVS_BUMP_MAP);
					break;
				case CElementBSDF::BLEND_BSDF:
					cachePropertyTexture(bsdf->blendbsdf.weight,CMitsubaMaterialCompilerFrontend::EIVS_BLEND_WEIGHT);
					break;
				case CElementBSDF::MASK:
					cachePropertyTexture(bsdf->mask.opacity,CMitsubaMaterialCompilerFrontend::EIVS_BLEND_WEIGHT);
					break;
				default: break;
			}
		}
	}

	return ctx.frontend.compileToIRTree(ctx.ir.get(), _bsdf);
}


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
//	const asset::IGeometryCreator* _geomCreator,
//	const asset::IMeshManipulator* _manipulator,
	const asset::IAssetLoader::SAssetLoadContext& _ctx,
	asset::IAssetLoader::IAssetLoaderOverride* _override,
	CMitsubaMetadata* _metadata
) : /*creator(_geomCreator), manipulator(_manipulator),*/ inner(_ctx), override_(_override), meta(_metadata)
//,ir(core::make_smart_refctd_ptr<asset::material_compiler::IR>()), frontend(this)
{
	auto materialPool = material_compiler3::CTrueIR::create();
	scene = ICPUScene::create(core::smart_refctd_ptr(materialPool)); // TODO: feed it max shapes per group
	frontIR = material_compiler3::CFrontendIR::create();
}

auto SContext::loadShapeGroup(const uint32_t hierarchyLevel, const CElementShape::ShapeGroup* shapegroup) -> SContext::group_ass_type
{
	auto found = groupCache.find(shapegroup);
	if (found!=groupCache.end())
		return found->second;
	
	auto collection = core::make_smart_refctd_ptr<ICPUGeometryCollection>();
	if (!collection)
		inner.params.logger.log("Failed to create an ICPUGeometryCollection for Shape Group",system::ILogger::ELL_ERROR);
	else
	{
		auto* geometries = collection->getGeometries();
		const auto children = shapegroup->children;
		for (auto i=0u; i<shapegroup->childCount; i++)
		{
			auto child = children[i];
			if (!child)
				continue;

			assert(child->type!=CElementShape::Type::INSTANCE);
			if (child->type!=CElementShape::Type::SHAPEGROUP)
			{
				auto geometry = loadBasicShape(hierarchyLevel,child);
				if (geometry)
					geometries->push_back({.transform=child->getTransform(),.geometry=std::move(geometry)});
			}
			else
			{
				auto nestedCollection = loadShapeGroup(hierarchyLevel,&child->shapegroup);
				if (!nestedCollection)
					continue;
				auto* nestedGeometries = nestedCollection->getGeometries();
				for (auto& ref : *nestedGeometries)
				{
					auto& newRef = geometries->emplace_back(std::move(ref));
					// thankfully because SHAPEGROUPS are not allowed to have transforms we don't need to rack them up
					//if (newRef.hasTransform())
					//	newRef.transform = hlsl::mul(thisTransform,newRef.transform);
					//else
					//	newRef.transform = thisTransform;
				}
			}
		}
		groupCache.insert({shapegroup,collection});
	}
	return collection;
}

#if 0
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
#endif

auto SContext::loadBasicShape(const uint32_t hierarchyLevel, const CElementShape* shape) -> SContext::shape_ass_type
{
	auto found = shapeCache.find(shape);
	if (found!=shapeCache.end())
		return found->second.geom;

	core::smart_refctd_ptr<asset::ICPUPolygonGeometry> geo;
	auto exiter = core::makeRAIIExiter<>([&]()->void
		{
			if (geo)
				return;
			this->inner.params.logger.log("Failed to Load/Create Basic non-Instanced Shape with id %s",system::ILogger::ELL_ERROR,shape->id.c_str());
		}
	);

#if 0
	constexpr uint32_t UV_ATTRIB_ID = 2u;



	auto loadModel = [&](const ext::MitsubaLoader::SPropertyElementData& filename, int64_t index=-1) -> core::smart_refctd_ptr<asset::ICPUMesh>
	{
		assert(filename.type==ext::MitsubaLoader::SPropertyElementData::Type::STRING);
		auto loadParams = ctx.inner.params;
		loadParams.loaderFlags = static_cast<IAssetLoader::E_LOADER_PARAMETER_FLAGS>(loadParams.loaderFlags | IAssetLoader::ELPF_RIGHT_HANDED_MESHES);
		auto retval = interm_getAssetInHierarchy( filename.svalue, loadParams, hierarchyLevel/*+ICPUScene::MESH_HIERARCHY_LEVELS_BELOW*/, ctx.override_);
		if (retval.getContents().empty())
		{
			os::Printer::log(std::string("[ERROR] Could Not Find Mesh: ") + filename.svalue, ELL_ERROR);
			return nullptr;
		}
		if (retval.getAssetType()!=asset::IAsset::ET_MESH)
		{
			os::Printer::log("[ERROR] Loaded an Asset but it wasn't a mesh, was E_ASSET_TYPE " + std::to_string(retval.getAssetType()), ELL_ERROR);
			return nullptr;
		}
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
#endif
	bool flipNormals = false;
	bool faceNormals = false;
	float maxSmoothAngle = hlsl::bit_cast<float>(hlsl::numeric_limits<float>::quiet_NaN);
	switch (shape->type)
	{
#if 0
		case CElementShape::Type::CUBE:
		{
			auto cubeData = ctx.creator->createCubeMesh(core::vector3df(2.f));

			mesh = createMeshFromGeomCreatorReturnType(ctx.creator->createCubeMesh(core::vector3df(2.f)), m_assetMgr);
			flipNormals = flipNormals!=shape->cube.flipNormals;
			break;
		}
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
#endif
#if 0
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
#endif
		case CElementShape::Type::SHAPEGROUP:
			[[fallthrough]];
		case CElementShape::Type::INSTANCE:
			assert(false);
			break;
		default:
//			_NBL_DEBUG_BREAK_IF(true);
			break;
	}
	//
	if (geo)
	{
#if 0
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
#endif
		// cache and return
		CMitsubaMetadata::SGeometryMetaPair geoMeta = {.geom=std::move(geo)};
		geoMeta.meta.m_id = shape->id;
		geoMeta.meta.type = shape->type;
		shapeCache.insert({shape,std::move(geoMeta)});
	}
	return geo;
}

}
}