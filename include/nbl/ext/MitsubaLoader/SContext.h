// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_C_MITSUBA_LOADER_CONTEXT_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_C_MITSUBA_LOADER_CONTEXT_H_INCLUDED_


#include "nbl/asset/ICPUPolygonGeometry.h"
#include "nbl/asset/interchange/CIESProfileLoader.h"

#include "nbl/ext/MitsubaLoader/CMitsubaMetadata.h"
//#include "nbl/ext/MitsubaLoader/CMitsubaMaterialCompilerFrontend.h"


namespace nbl::ext::MitsubaLoader
{
class CMitsubaLoader;

struct SContext final
{
	public:
		using interm_getAssetInHierarchy_t = asset::SAssetBundle(const char*, const uint16_t);

		SContext(
			const asset::IAssetLoader::SAssetLoadContext& _params,
			asset::IAssetLoader::IAssetLoaderOverride* _override,
			CMitsubaMetadata* _metadata
		);

		using shape_ass_type = core::smart_refctd_ptr<const asset::ICPUGeometryCollection>;
		shape_ass_type loadBasicShape(const CElementShape* shape);
		// the `shape` will have to be `Type::SHAPEGROUP`
		shape_ass_type loadShapeGroup(const CElementShape* shape);

		inline void transferMetadata()
		{
			meta->setGeometryCollectionMeta(std::move(shapeCache));
			meta->setGeometryCollectionMeta(std::move(groupCache));
		}

		const asset::IAssetLoader::SAssetLoadContext inner;
		asset::IAssetLoader::IAssetLoaderOverride* override_;
		std::function<interm_getAssetInHierarchy_t> interm_getAssetInHierarchy;
		CMitsubaMetadata* meta;
		core::smart_refctd_ptr<asset::ICPUScene> scene;

	private:
		//
		core::unordered_map<const CElementShape*,CMitsubaMetadata::SGeometryCollectionMetaPair> shapeCache;
		//
		core::unordered_map<const CElementShape::ShapeGroup*,CMitsubaMetadata::SGeometryCollectionMetaPair> groupCache;

#if 0 // stuff that belongs in the Material Compiler backend
		//image, sampler
		using tex_ass_type = std::tuple<core::smart_refctd_ptr<asset::ICPUImageView>,core::smart_refctd_ptr<asset::ICPUSampler>>;
		//image, scale 
		core::map<core::smart_refctd_ptr<asset::ICPUImage>,float> derivMapCache;

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
#endif
		core::smart_refctd_ptr<asset::material_compiler3::CFrontendIR> frontIR;
};

}
#endif