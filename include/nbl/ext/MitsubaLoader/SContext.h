// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_C_MITSUBA_LOADER_CONTEXT_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_C_MITSUBA_LOADER_CONTEXT_H_INCLUDED_


#include "nbl/asset/ICPUPolygonGeometry.h"
//#include "nbl/asset/utils/IGeometryCreator.h"
#include "nbl/asset/interchange/CIESProfileLoader.h"

//#include "nbl/ext/MitsubaLoader/CMitsubaMaterialCompilerFrontend.h"
//#include "nbl/ext/MitsubaLoader/CElementShape.h"

namespace nbl::ext::MitsubaLoader
{

struct SContext
{
	public:
		SContext(
//			const asset::IGeometryCreator* _geomCreator,
//			const asset::IMeshManipulator* _manipulator,
			const asset::IAssetLoader::SAssetLoadContext& _params,
			asset::IAssetLoader::IAssetLoaderOverride* _override,
//			CMitsubaMetadata* _metadata
		);

//		const asset::IGeometryCreator* creator;
//		const asset::IMeshManipulator* manipulator;
		const asset::IAssetLoader::SAssetLoadContext inner;
		asset::IAssetLoader::IAssetLoaderOverride* override_;
//		CMitsubaMetadata* meta;

#if 0
		//
		using group_ass_type = core::vector<core::smart_refctd_ptr<asset::ICPUMesh>>;
		//core::map<const CElementShape::ShapeGroup*, group_ass_type> groupCache;
		//
		using shape_ass_type = core::smart_refctd_ptr<asset::ICPUMesh>;
		core::map<const CElementShape*, shape_ass_type> shapeCache;
		//image, sampler
		using tex_ass_type = std::tuple<core::smart_refctd_ptr<asset::ICPUImageView>,core::smart_refctd_ptr<asset::ICPUSampler>>;
		//image, scale
		core::map<core::smart_refctd_ptr<asset::ICPUImage>,float> derivMapCache;

		//
		static std::string imageViewCacheKey(const CElementTexture::Bitmap& bitmap, const CMitsubaMaterialCompilerFrontend::E_IMAGE_VIEW_SEMANTIC semantic)
		{
			std::string key = bitmap.filename.svalue;
			switch (bitmap.channel)
			{
				case CElementTexture::Bitmap::CHANNEL::R:
					key += "?rrrr";
					break;
				case CElementTexture::Bitmap::CHANNEL::G:
					key += "?gggg";
					break;
				case CElementTexture::Bitmap::CHANNEL::B:
					key += "?bbbb";
					break;
				case CElementTexture::Bitmap::CHANNEL::A:
					key += "?aaaa";
					break;
				default:
					break;
			}
			switch (semantic)
			{
				case CMitsubaMaterialCompilerFrontend::EIVS_BLEND_WEIGHT:
					key += "?blend";
					break;
				case CMitsubaMaterialCompilerFrontend::EIVS_NORMAL_MAP:
					key += "?deriv?n";
					break;
				case CMitsubaMaterialCompilerFrontend::EIVS_BUMP_MAP:
					key += "?deriv?h";
					{
						static const char* wrap[5]
						{
							"?repeat",
							"?mirror",
							"?clamp",
							"?zero",
							"?one"
						};
						key += wrap[bitmap.wrapModeU];
						key += wrap[bitmap.wrapModeV];
					}
					break;
				default:
					break;
			}
			key += "?view";
			return key;
		}

		static asset::ISampler::SParams emissionProfileSamplerParams(const CElementEmissionProfile* profile, const asset::CIESProfileMetadata& meta)
		{
			return {
				asset::ISampler::ETC_REPEAT,
				asset::ISampler::ETC_REPEAT,
				asset::ISampler::ETC_REPEAT,
				asset::ISampler::ETBC_INT_OPAQUE_BLACK,
				asset::ISampler::ETF_LINEAR,
				asset::ISampler::ETF_LINEAR,
				asset::ISampler::ETF_LINEAR,
				0u, false, asset::ECO_ALWAYS
			};
		}

		static auto computeSamplerParameters(const CElementTexture::Bitmap& bitmap)
		{
			asset::ICPUSampler::SParams params;
			auto getWrapMode = [](CElementTexture::Bitmap::WRAP_MODE mode)
			{
				switch (mode)
				{
					case CElementTexture::Bitmap::WRAP_MODE::CLAMP:
						return asset::ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE;
						break;
					case CElementTexture::Bitmap::WRAP_MODE::MIRROR:
						return asset::ISampler::E_TEXTURE_CLAMP::ETC_MIRROR;
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
				return asset::ISampler::E_TEXTURE_CLAMP::ETC_REPEAT;
			};
			params.TextureWrapU = getWrapMode(bitmap.wrapModeU);
			params.TextureWrapV = getWrapMode(bitmap.wrapModeV);
			params.TextureWrapW = asset::ISampler::E_TEXTURE_CLAMP::ETC_REPEAT;
			params.BorderColor = asset::ISampler::ETBC_FLOAT_OPAQUE_BLACK;
			switch (bitmap.filterType)
			{
				case CElementTexture::Bitmap::FILTER_TYPE::EWA:
					[[fallthrough]]; // we dont support this fancy stuff
				case CElementTexture::Bitmap::FILTER_TYPE::TRILINEAR:
					params.MinFilter = asset::ISampler::ETF_LINEAR;
					params.MaxFilter = asset::ISampler::ETF_LINEAR;
					params.MipmapMode = asset::ISampler::ESMM_LINEAR;
					break;
				default:
					params.MinFilter = asset::ISampler::ETF_NEAREST;
					params.MaxFilter = asset::ISampler::ETF_NEAREST;
					params.MipmapMode = asset::ISampler::ESMM_NEAREST;
					break;
			}
			params.AnisotropicFilter = core::max(hlsl::findMSB<uint32_t>(bitmap.maxAnisotropy),1u);
			params.CompareEnable = false;
			params.CompareFunc = asset::ISampler::ECO_NEVER;
			params.LodBias = 0.f;
			params.MaxLod = 10000.f;
			params.MinLod = 0.f;
			return params;
		}

		//index of root node in IR
		using bsdf_type = const CMitsubaMaterialCompilerFrontend::front_and_back_t;
		//caches instr buffer instr-wise offset (.first) and instruction count (.second) for each bsdf node
		core::unordered_map<const CElementBSDF*, bsdf_type> instrStreamCache;

		struct SInstanceData
		{
			SInstanceData(core::matrix3x4SIMD _tform, SContext::bsdf_type _bsdf, const std::string& _id, const CElementEmitter& _emitterFront, const CElementEmitter& _emitterBack) :
				tform(_tform), bsdf(_bsdf),
#if defined(_NBL_DEBUG) || defined(_NBL_RELWITHDEBINFO)
				bsdf_id(_id),
#endif
				emitter{_emitterFront, _emitterBack}
			{}

			core::matrix3x4SIMD tform;
			SContext::bsdf_type bsdf;
#if defined(_NBL_DEBUG) || defined(_NBL_RELWITHDEBINFO)
			std::string bsdf_id;
#endif
			struct {
				// type is invalid if not used
				CElementEmitter front;
				CElementEmitter back;
			} emitter;
		};
		core::unordered_multimap<const shape_ass_type::pointee*, SInstanceData> mapMesh2instanceData;

		core::unordered_map<SPipelineCacheKey, core::smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>, SPipelineCacheKey::hash> pipelineCache;
#endif
		//material compiler
		core::smart_refctd_ptr<asset::material_compiler::IR> ir;
		CMitsubaMaterialCompilerFrontend frontend;

	private:
};

}
#endif