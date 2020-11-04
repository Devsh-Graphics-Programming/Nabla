#ifndef __C_MITSUBA_LOADER_CONTEXT_H_INCLUDED__
#define __C_MITSUBA_LOADER_CONTEXT_H_INCLUDED__


#include <irr/asset/material_compiler/CMaterialCompilerGLSLRasterBackend.h>
#include <irr/asset/ICPUMesh.h>
#include <irr/asset/IGeometryCreator.h>
#include "../../ext/MitsubaLoader/CMitsubaMaterialCompilerFrontend.h"
#include "../../ext/MitsubaLoader/CElementShape.h"

namespace irr
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
			CGlobalMitsubaMetadata* _metadata
		);

		const asset::IGeometryCreator* creator;
		const asset::IMeshManipulator* manipulator;
		const asset::IAssetLoader::SAssetLoadContext inner;
		asset::IAssetLoader::IAssetLoaderOverride* override_;
		CGlobalMitsubaMetadata* globalMeta;

		_IRR_STATIC_INLINE_CONSTEXPR uint32_t VT_PAGE_TABLE_LAYERS = 64u;
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t VT_PAGE_SZ_LOG2 = 7u;//128
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t VT_PHYSICAL_PAGE_TEX_TILES_PER_DIM_LOG2 = 4u;//16
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t VT_PHYSICAL_PAGE_TEX_LAYERS = 20u;
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t VT_PAGE_PADDING = 8u;
		_IRR_STATIC_INLINE_CONSTEXPR uint32_t VT_MAX_ALLOCATABLE_TEX_SZ_LOG2 = 12u;//4096

		//
		using group_ass_type = core::vector<core::smart_refctd_ptr<asset::ICPUMesh>>;
		//core::map<const CElementShape::ShapeGroup*, group_ass_type> groupCache;
		//
		using shape_ass_type = core::smart_refctd_ptr<asset::ICPUMesh>;
		core::map<const CElementShape*, shape_ass_type> shapeCache;
		//image, sampler
		using tex_ass_type = std::tuple<core::smart_refctd_ptr<asset::ICPUImageView>, core::smart_refctd_ptr<asset::ICPUSampler>>;

		static std::string imageViewCacheKey(const std::string& imageCacheKey)
		{
			return imageCacheKey + "?view";
		}

		static std::string derivMapCacheKey(const CElementTexture* bitmap)
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
			key += wrap[bitmap->bitmap.wrapModeU];
			key += wrap[bitmap->bitmap.wrapModeV];

			return key;
		}

		static std::string derivMapViewCacheKey(const CElementTexture* bitmap)
		{
			return imageViewCacheKey(derivMapCacheKey(bitmap));
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
		using bsdf_type = const asset::material_compiler::IR::INode*;
		//caches instr buffer instr-wise offset (.first) and instruction count (.second) for each bsdf node
		core::unordered_map<const CElementBSDF*, bsdf_type> instrStreamCache;

		struct SInstanceData
		{
			SInstanceData(core::matrix3x4SIMD _tform, SContext::bsdf_type _bsdf, const std::string& _id, const CElementEmitter& _emitter) : 
				tform(_tform), bsdf(_bsdf),
#if defined(_IRR_DEBUG) || defined(_IRR_RELWITHDEBINFO)
				bsdf_id(_id),
#endif
				emitter(_emitter)
			{}

			core::matrix3x4SIMD tform;
			SContext::bsdf_type bsdf;
#if defined(_IRR_DEBUG) || defined(_IRR_RELWITHDEBINFO)
			std::string bsdf_id;
#endif
			CElementEmitter emitter; // type is invalid if not used
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