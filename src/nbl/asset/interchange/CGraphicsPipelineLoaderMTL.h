// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_GRAPHICS_PIPELINE_LOADER_MTL_H_INCLUDED_
#define _NBL_ASSET_C_GRAPHICS_PIPELINE_LOADER_MTL_H_INCLUDED_


#include "nbl/asset/interchange/IGeometryLoader.h"
#include "nbl/asset/metadata/CMTLMetadata.h"


namespace nbl::asset
{	

class CGraphicsPipelineLoaderMTL final : public IAssetLoader // TODO: Material Asset and Loader
{
#if 0
        struct SMtl
        {
            CMTLMetadata::CRenderpassIndependentPipeline::SMaterialParameters params;
            std::string name;
            //paths to image files, note that they're relative to the mtl file
            std::string maps[CMTLMetadata::CRenderpassIndependentPipeline::EMP_COUNT];
            //-clamp
            uint32_t clamp;
            static_assert(sizeof(clamp) * 8ull >= CMTLMetadata::CRenderpassIndependentPipeline::EMP_COUNT, "SMtl::clamp is too small!");

            inline bool isClampToBorder(CMTLMetadata::CRenderpassIndependentPipeline::E_MAP_TYPE m) const { return (clamp >> m) & 1u; }
        };
#endif
        struct SContext
        {
            SContext(const IAssetLoader::SAssetLoadContext& _innerCtx, uint32_t _topHierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override)
                : inner(_innerCtx), topHierarchyLevel(_topHierarchyLevel), loaderOverride(_override) {}

            IAssetLoader::SAssetLoadContext inner;
            uint32_t topHierarchyLevel;
            IAssetLoader::IAssetLoaderOverride* loaderOverride;
        };

	public:
        CGraphicsPipelineLoaderMTL(IAssetManager* _am, core::smart_refctd_ptr<system::ISystem>&& sys);

		bool isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger=nullptr) const override;

		const char** getAssociatedFileExtensions() const override
		{
			static const char* extensions[]{ "mtl", nullptr };
			return extensions;
		}

//		uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MATERIAL; }

		asset::SAssetBundle loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel = 0u) override;

    private:
#if 0
        core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> makePipelineFromMtl(SContext& ctx, const SMtl& _mtl, bool hasUV);
        core::vector<SMtl> readMaterials(system::IFile* _file, const system::logger_opt_ptr logger) const;
        const char* readTexture(const char* _bufPtr, const char* const _bufEnd, SMtl* _currMaterial, const char* _mapType) const;

        using images_set_t = std::array<core::smart_refctd_ptr<ICPUImage>, CMTLMetadata::CRenderpassIndependentPipeline::EMP_COUNT>;
        using image_views_set_t = std::array<core::smart_refctd_ptr<ICPUImageView>, CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSX + 1u>;
        image_views_set_t loadImages(const std::string& relDir, SMtl& _mtl, SContext& _ctx);
        core::smart_refctd_ptr<ICPUDescriptorSet> makeDescSet(image_views_set_t&& _views, ICPUDescriptorSetLayout* _dsLayout, SContext& _ctx);
#endif

        core::smart_refctd_ptr<system::ISystem> m_system;

};

}

#endif
