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
			if (shape->transform.matrix[3]!=float32_t4(0,0,0,1))
				_params.logger.log("Shape with id %s has Non-Affine transformation matrix, last row is not 0,0,0,1!",system::ILogger::ELL_ERROR,shape->id.c_str());
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
	const asset::IAssetLoader::SAssetLoadContext& _ctx,
	asset::IAssetLoader::IAssetLoaderOverride* _override,
	CMitsubaMetadata* _metadata
) : inner(_ctx), override_(_override), meta(_metadata)
//,ir(core::make_smart_refctd_ptr<asset::material_compiler::IR>()), frontend(this)
{
	auto materialPool = material_compiler3::CTrueIR::create();
	scene = ICPUScene::create(core::smart_refctd_ptr(materialPool)); // TODO: feed it max shapes per group
	frontIR = material_compiler3::CFrontendIR::create();
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
		inner.params.logger.log("Failed to Load/Create Basic non-Instanced Shape with id %s",system::ILogger::ELL_ERROR,shape->id.c_str());
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