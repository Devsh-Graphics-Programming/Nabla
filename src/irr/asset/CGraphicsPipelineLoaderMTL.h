#ifndef __IRR_C_GRAPHICS_PIPELINE_LOADER_MTL_H_INCLUDED__
#define __IRR_C_GRAPHICS_PIPELINE_LOADER_MTL_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "irr/asset/IAssetLoader.h"
#include "irr/asset/CMTLPipelineMetadata.h"

namespace irr
{
namespace asset
{	
	class CGraphicsPipelineLoaderMTL final : public asset::IAssetLoader
	{
        using SMtl = CMTLPipelineMetadata::SMtl;

        struct SContext
        {
            SContext(const IAssetLoader::SAssetLoadContext& _innerCtx, uint32_t _topHierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override)
                : inner(_innerCtx), topHierarchyLevel(_topHierarchyLevel), loaderOverride(_override) {}

            IAssetLoader::SAssetLoadContext inner;
            uint32_t topHierarchyLevel;
            IAssetLoader::IAssetLoaderOverride* loaderOverride;
        };

	public:
        CGraphicsPipelineLoaderMTL(IAssetManager* _am);

		bool isALoadableFileFormat(io::IReadFile* _file) const override;

		const char** getAssociatedFileExtensions() const override
		{
			static const char* extensions[]{ "mtl", nullptr };
			return extensions;
		}

		uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_RENDERPASS_INDEPENDENT_PIPELINE; }

		asset::SAssetBundle loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

    private:
        core::smart_refctd_ptr<ICPUPipelineLayout> makePipelineLayoutFromMtl(const CMTLPipelineMetadata::SMtl& _mtl);
        core::vector<SMtl> readMaterials(io::IReadFile* _file) const;
        const char* readTexture(const char* _bufPtr, const char* const _bufEnd, SMtl* _currMaterial, const char* _mapType) const;

        std::pair<core::smart_refctd_ptr<ICPUSpecializedShader>, core::smart_refctd_ptr<ICPUSpecializedShader>> getShaders(bool _hasUV);

        using images_set_t = std::array<core::smart_refctd_ptr<ICPUImage>, CMTLPipelineMetadata::SMtl::EMP_COUNT>;
        using image_views_set_t = std::array<core::smart_refctd_ptr<ICPUImageView>, CMTLPipelineMetadata::SMtl::EMP_REFL_POSX + 1u>;
        image_views_set_t loadImages(const char* _relDir, const CMTLPipelineMetadata::SMtl& _mtl, SContext& _ctx);
        core::smart_refctd_ptr<ICPUDescriptorSet> makeDescSet(image_views_set_t&& _views, ICPUDescriptorSetLayout* _dsLayout);

        IAssetManager* m_assetMgr;
	};
}
}

#endif
