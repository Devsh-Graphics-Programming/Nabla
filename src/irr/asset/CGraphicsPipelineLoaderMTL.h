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
        struct SMtl
        {
            CMTLPipelineMetadata::SMTLMaterialParameters params;
            std::string name;
            //paths to image files, note that they're relative to the mtl file
            std::string maps[CMTLPipelineMetadata::EMP_COUNT];
            //-clamp
            uint32_t clamp;
            static_assert(sizeof(clamp) * 8ull >= CMTLPipelineMetadata::EMP_COUNT, "SMtl::clamp is too small!");
        };

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
        core::smart_refctd_ptr<ICPUPipelineLayout> makePipelineLayoutFromMtl(const SMtl& _mtl, bool _noDS3);
        core::vector<SMtl> readMaterials(io::IReadFile* _file) const;
        const char* readTexture(const char* _bufPtr, const char* const _bufEnd, SMtl* _currMaterial, const char* _mapType) const;

        std::pair<core::smart_refctd_ptr<ICPUSpecializedShader>, core::smart_refctd_ptr<ICPUSpecializedShader>> getShaders(bool _hasUV);

        using images_set_t = std::array<core::smart_refctd_ptr<ICPUImage>, CMTLPipelineMetadata::EMP_COUNT>;
        using image_views_set_t = std::array<core::smart_refctd_ptr<ICPUImageView>, CMTLPipelineMetadata::EMP_REFL_POSX + 1u>;
        image_views_set_t loadImages(const char* _relDir, const SMtl& _mtl, SContext& _ctx);
        core::smart_refctd_ptr<ICPUDescriptorSet> makeDescSet(image_views_set_t&& _views, ICPUDescriptorSetLayout* _dsLayout);

        IAssetManager* m_assetMgr;
	};
}
}

#endif
