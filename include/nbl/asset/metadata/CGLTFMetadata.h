// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_C_GLTF_METADATA_H_INCLUDED_
#define _NBL_ASSET_C_GLTF_METADATA_H_INCLUDED_

#include "nbl/asset/metadata/IAssetMetadata.h"
#include "nbl/asset/metadata/CGLTFPipelineMetadata.h"

namespace nbl::asset
{

class NBL_API CGLTFMetadata final : public IAssetMetadata
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


        struct Instance
        {
            const ICPUSkeleton* skeleton;
            asset::SBufferBinding<const asset::ICPUBuffer> skinTranslationTable;
            const ICPUMesh* mesh;
            ICPUSkeleton::joint_id_t attachedToNode; // the node with the `mesh` and `skin` parameters
        };
        core::vector<Instance> instances;

        core::vector<core::smart_refctd_ptr<ICPUSkeleton>> skeletons;

        struct Scene
        {
            core::vector<uint32_t> instanceIDs;
        };
        core::vector<Scene> scenes;
        uint32_t defaultSceneID = 0xffFFffFFu;

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

#endif // _NBL_ASSET_C_GLTF_METADATA_H_INCLUDED_
