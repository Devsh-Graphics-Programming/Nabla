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
                CRenderpassIndependentPipeline(CRenderpassIndependentPipeline&& _other) : CRenderpassIndependentPipeline()
                {
                    CRenderpassIndependentPipeline::operator=(std::move(_other));
                }
                template<typename... Args>
                CRenderpassIndependentPipeline(Args&&... args) : IRenderpassIndependentPipelineMetadata(std::forward<Args>(args)...) {}

                inline CRenderpassIndependentPipeline& operator=(CRenderpassIndependentPipeline&& other)
                {
                    IRenderpassIndependentPipelineMetadata::operator=(std::move(other));
                    return *this;
                }
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

        CMitsubaSerializedMetadata(const uint32_t meshBound, core::smart_refctd_dynamic_array<asset::IRenderpassIndependentPipelineMetadata::ShaderInputSemantic>&& _semanticStorage) :
            m_pipelineStorage(asset::IAssetMetadata::createContainer<CRenderpassIndependentPipeline>(meshBound)), m_meshStorage(asset::IAssetMetadata::createContainer<CMesh>(meshBound)), m_semanticStorage(std::move(_semanticStorage))
        {
        }

        _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "ext::MitsubaLoader::CSerializedLoader";
        const char* getLoaderName() const override { return LoaderName; }

        //!
        inline const CRenderpassIndependentPipeline* getAssetSpecificMetadata(const asset::ICPURenderpassIndependentPipeline* asset) const
        {
            const auto found = IAssetMetadata::getAssetSpecificMetadata(asset);
            return static_cast<const CRenderpassIndependentPipeline*>(found);
        }
        inline const CMesh* getAssetSpecificMetadata(const asset::ICPUMesh* asset) const
        {
            const auto found = IAssetMetadata::getAssetSpecificMetadata(asset);
            return static_cast<const CMesh*>(found);
        }

    private:
        meta_container_t<CRenderpassIndependentPipeline> m_pipelineStorage;
        meta_container_t<CMesh> m_meshStorage;
        core::smart_refctd_dynamic_array<asset::IRenderpassIndependentPipelineMetadata::ShaderInputSemantic> m_semanticStorage;

        friend class CSerializedLoader;
        inline void placeMeta(uint32_t offset, const asset::ICPURenderpassIndependentPipeline* ppln, const asset::ICPUMesh* mesh, CMesh&& meshMeta)
        {
            auto& pplnMetaRef = m_pipelineStorage->operator[](offset) = CRenderpassIndependentPipeline(core::SRange<const asset::IRenderpassIndependentPipelineMetadata::ShaderInputSemantic>(m_semanticStorage->begin(),m_semanticStorage->end()));
            auto& meshMetaRef = m_meshStorage->operator[](offset) = std::move(meshMeta);
            IAssetMetadata::insertAssetSpecificMetadata(ppln,&pplnMetaRef);
            IAssetMetadata::insertAssetSpecificMetadata(mesh,&meshMetaRef);
        }
};

}
}
}

#endif