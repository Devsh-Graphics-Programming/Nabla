// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_GLTF_METADATA_H_INCLUDED__
#define __NBL_ASSET_C_GLTF_METADATA_H_INCLUDED__

#include "nbl/asset/metadata/IAssetMetadata.h"
#include "nbl/asset/metadata/CGLTFPipelineMetadata.h"

namespace nbl
{
namespace asset
{

class CGLTFMetadata final : public IAssetMetadata
{
    public:
        
        CGLTFMetadata(uint32_t pipelineCount) : IAssetMetadata(), m_metaStorage(createContainer<asset::CGLTFPipelineMetadata>(pipelineCount)) {}

        _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CGLTFLoader";
        const char* getLoaderName() const override { return LoaderName; }

        inline const asset::CGLTFPipelineMetadata* getAssetSpecificMetadata(const ICPURenderpassIndependentPipeline* asset) const
        {
            const auto found = IAssetMetadata::getAssetSpecificMetadata(asset);
            return static_cast<const asset::CGLTFPipelineMetadata*>(found);
        }

        using INSTANCE_ID = uint32_t;

        struct Scene
        {
            std::vector<uint32_t> instanceIDs;
        };

        struct Instance
        {
            const ICPUSkeleton* skeleton;
            asset::SBufferBinding<asset::ICPUBuffer> skinTranslationTable;
            const ICPUMesh* mesh;
            ICPUSkeleton::joint_id_t attachedToNode; // the node with the `mesh` and `skin` parameters
        };

        std::vector<Scene> scenes;
        uint32_t defaultSceneID = 0xffFFffFFu;

        core::vector<core::smart_refctd_ptr<ICPUSkeleton>> skeletons;
        core::vector<Instance> instances;

    private:
        meta_container_t<asset::CGLTFPipelineMetadata> m_metaStorage;

        friend class CGLTFLoader;
        inline void placeMeta(uint32_t offset, const ICPURenderpassIndependentPipeline* pipeline, const asset::CGLTFPipelineMetadata& _meta)
        {
            auto& meta = m_metaStorage->operator[](offset);

            meta.m_inputSemantics = _meta.m_inputSemantics;

            IAssetMetadata::insertAssetSpecificMetadata(pipeline, &meta);
        }
};

}
}

#endif // __NBL_ASSET_C_GLTF_METADATA_H_INCLUDED__
