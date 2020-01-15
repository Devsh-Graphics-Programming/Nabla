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

        IAssetManager* m_assetMgr;
	};
}
}

#endif
