// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __C_MITSUBA_LOADER_CONTEXT_H_INCLUDED__
#define __C_MITSUBA_LOADER_CONTEXT_H_INCLUDED__


#include "nbl/asset/ICPUMesh.h"
#include "nbl/asset/utils/IGeometryCreator.h"
#include "nbl/asset/material_compiler/CMaterialCompilerGLSLRasterBackend.h"

#include "../../ext/MitsubaLoader/CMitsubaMaterialCompilerFrontend.h"
#include "../../ext/MitsubaLoader/CElementShape.h"

namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{

	struct SContext
	{
		SContext(
			const asset::IGeometryCreator* _geomCreator,
			const asset::IMeshManipulator* _manipulator,
			const asset::IAssetLoader::SAssetLoadContext& _params,
			asset::IAssetLoader::IAssetLoaderOverride* _override,
			CMitsubaMetadata* _metadata
		);

		const asset::IGeometryCreator* creator;
		const asset::IMeshManipulator* manipulator;
		const asset::IAssetLoader::SAssetLoadContext inner;
		asset::IAssetLoader::IAssetLoaderOverride* override_;
		CMitsubaMetadata* meta;

		_NBL_STATIC_INLINE_CONSTEXPR uint32_t VT_PAGE_SZ_LOG2 = 7u;//128
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t VT_PHYSICAL_PAGE_TEX_TILES_PER_DIM_LOG2 = 4u;//16
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t VT_PAGE_PADDING = 8u;
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t VT_MAX_ALLOCATABLE_TEX_SZ_LOG2 = 12u;//4096

		//
		using group_ass_type = core::vector<core::smart_refctd_ptr<asset::ICPUMesh>>;
		//core::map<const CElementShape::ShapeGroup*, group_ass_type> groupCache;
		//
		using shape_ass_type = core::smart_refctd_ptr<asset::ICPUMesh>;
		core::map<const CElementShape*, shape_ass_type> shapeCache;
		//image, sampler
		using tex_ass_type = std::tuple<core::smart_refctd_ptr<asset::ICPUImageView>, core::smart_refctd_ptr<asset::ICPUSampler>>;
		//image, scale
		core::map<core::smart_refctd_ptr<asset::ICPUImage>, float> derivMapCache;

		static std::string blendWeightImageCacheKey(const CElementTexture* bitmap)
		{
			using namespace std::string_literals;
			return bitmap->bitmap.filename.svalue + "?blend"s;
		}

		static std::string imageViewCacheKey(const std::string& imageCacheKey)
		{
			return imageCacheKey + "?view";
		}

		static std::string derivMapCacheKey(const CElementTexture* bitmap, bool wasNormal)
		{
			using namespace std::string_literals;
			static const char* wrap[5]
			{
				"?repeat",
				"?mirror",
				"?clamp",
				"?zero",
				"?one"
			};

			std::string key = bitmap->bitmap.filename.svalue + "?deriv"s;
			key += wasNormal ? "?n"s:"?h"s;
			key += wrap[bitmap->bitmap.wrapModeU];
			key += wrap[bitmap->bitmap.wrapModeV];

			return key;
		}

		static std::string derivMapViewCacheKey(const CElementTexture* bitmap, bool wasNormal)
		{
			return imageViewCacheKey(derivMapCacheKey(bitmap,wasNormal));
		}

		static std::string blendWeightViewCacheKey(const CElementTexture* bitmap)
		{
			return imageViewCacheKey(blendWeightImageCacheKey(bitmap));
		}

		static std::string samplerCacheKey(const std::string& base, const CElementTexture* tex)
		{
			std::string samplerCacheKey = base;

			switch (tex->bitmap.filterType)
			{
			case CElementTexture::Bitmap::FILTER_TYPE::EWA:
				[[fallthrough]]; //not supported
			case CElementTexture::Bitmap::FILTER_TYPE::TRILINEAR:
				samplerCacheKey += "?trilinear";
				break;
			default:
				samplerCacheKey += "?nearest";
				break;
			}

			auto perWrapMode = [](CElementTexture::Bitmap::WRAP_MODE mode)
			{
				switch (mode)
				{
				case CElementTexture::Bitmap::WRAP_MODE::CLAMP:
					return "?clamp";
				case CElementTexture::Bitmap::WRAP_MODE::MIRROR:
					return "?clamp";
				case CElementTexture::Bitmap::WRAP_MODE::ONE:
					return "?one";
				case CElementTexture::Bitmap::WRAP_MODE::ZERO:
					return "?zero";
				default:
					return "?repeat";
				}
			};
			samplerCacheKey += perWrapMode(tex->bitmap.wrapModeU);
			samplerCacheKey += perWrapMode(tex->bitmap.wrapModeV);

			return samplerCacheKey;
		}
		std::string samplerCacheKey(const CElementTexture* tex) const
		{
			return samplerCacheKey(samplerCacheKeyBase, tex);
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

		struct SPipelineCacheKey
		{
			asset::SVertexInputParams vtxParams;
			asset::SPrimitiveAssemblyParams primParams;

			inline bool operator==(const SPipelineCacheKey& rhs) const
			{
				return memcmp(&vtxParams, &rhs.vtxParams, sizeof(vtxParams)) == 0 && memcmp(&primParams, &rhs.primParams, sizeof(primParams)) == 0;
			}

			struct hash
			{
				inline size_t operator()(const SPipelineCacheKey& k) const
				{
					constexpr size_t BYTESZ = sizeof(k.vtxParams) + sizeof(k.primParams);
					uint8_t mem[BYTESZ]{};
					uint8_t* ptr = mem;
					memcpy(ptr, &k.vtxParams, sizeof(k.vtxParams));
					ptr += sizeof(k.vtxParams);
					memcpy(ptr, &k.primParams, sizeof(k.primParams));
					ptr += sizeof(k.primParams);

					return std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(mem), BYTESZ));
				}
			};
		};
		core::unordered_map<SPipelineCacheKey, core::smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>, SPipelineCacheKey::hash> pipelineCache;

		//material compiler
		core::smart_refctd_ptr<asset::material_compiler::IR> ir;
		CMitsubaMaterialCompilerFrontend frontend;
		asset::material_compiler::CMaterialCompilerGLSLRasterBackend::SContext backend_ctx;
		asset::material_compiler::CMaterialCompilerGLSLRasterBackend backend;

		const std::string samplerCacheKeyBase;
	};

}}}

#endif