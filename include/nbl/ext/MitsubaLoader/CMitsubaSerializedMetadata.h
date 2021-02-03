// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_C_MITSUBA_SERIALIZED_PIPELINE_METADATA_H_INCLUDED__
#define __NBL_C_MITSUBA_SERIALIZED_PIPELINE_METADATA_H_INCLUDED__

#include "nbl/asset/ICPURenderpassIndependentPipeline.h"
#include "nbl/asset/ICPUMesh.h"
#include "nbl/asset/metadata/IAssetMetadata.h"

namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{

class CMitsubaSerializedMetadata final : public asset::IAssetMetadata
{
    public:
        class CRenderpassIndependentPipeline : public asset::IRenderpassIndependentPipelineMetadata
        {
            public:
                using IRenderpassIndependentPipelineMetadata::IRenderpassIndependentPipelineMetadata;
        };
        class CMesh : public asset::IMeshMetadata
        {
            public:
                inline CMesh() : IMeshMetadata() {}
                template<typename... Args>
                inline CMesh(std::string&& _name, const uint32_t _id, Args&&... args) : m_name(std::move(_name)), m_id(_id), IMeshMetadata(std::forward<Args>(args)...) {}

                std::string m_name;
                uint32_t m_id;
        };

        CMitsubaSerializedMetadata(const uint32_t meshBound) :
            m_pipelineStorage(IAssetMetadata::createContainer<CRenderpassIndependentPipeline>(meshBound)), m_meshStorage(IAssetMetadata::createContainer<CMesh>(meshBound))
        {
        }

        _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "ext::MitsubaLoader::CSerializedLoader";
        const char* getLoaderName() const override { return LoaderName; }

    private:
        meta_container_t<CRenderpassIndependentPipeline> m_pipelineStorage;
        meta_container_t<CMesh> m_meshStorage;

        friend class CSerializedLoader;
        inline void placeMeta(uint32_t offset, const ICPURenderpassIndependentPipeline* ppln, CRenderpassIndependentPipeline&& pplnMeta, const ICPUMesh* mesh, CMesh&& meshMeta)
        {
            auto& pplnMetaRef = m_pipelineStorage->operator[](offset) = std::move(pplnMeta);
            auto& meshMetaRef = m_meshStorage->operator[](offset) = std::move(meshMeta);
            IAssetMetadata::insertAssetSpecificMetadata(ppln,&pplnMetaRef);
            IAssetMetadata::insertAssetSpecificMetadata(mesh,&meshMetaRef);
        }
};

}
}
}

#endif