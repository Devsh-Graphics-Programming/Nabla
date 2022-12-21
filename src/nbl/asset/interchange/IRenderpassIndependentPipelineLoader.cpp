// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/interchange/IRenderpassIndependentPipelineLoader.h"
#include "nbl/asset/asset_utils.h"

using namespace nbl;
using namespace asset;

IRenderpassIndependentPipelineLoader::~IRenderpassIndependentPipelineLoader()
{
}

void IRenderpassIndependentPipelineLoader::initialize()
{
    auto dfltOver = IAssetLoaderOverride(m_assetMgr);
    const IAssetLoader::SAssetLoadContext fakeCtx(IAssetLoader::SAssetLoadParams{},nullptr);

    // find ds1 layout
    auto ds1layout = dfltOver.findDefaultAsset<ICPUDescriptorSetLayout>("nbl/builtin/descriptor_set_layout/basic_view_parameters",fakeCtx,0u).first;

    // create common metadata part
    {
        constexpr size_t DS1_METADATA_ENTRY_CNT = 3ull;
        m_basicViewParamsSemantics = core::make_refctd_dynamic_array<decltype(m_basicViewParamsSemantics)>(DS1_METADATA_ENTRY_CNT);

        constexpr IRenderpassIndependentPipelineMetadata::E_COMMON_SHADER_INPUT types[DS1_METADATA_ENTRY_CNT] = 
        {
            IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_PROJ,
            IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW,
            IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_INVERSE_TRANSPOSE
        };
        constexpr uint32_t sizes[DS1_METADATA_ENTRY_CNT] = 
        {
            sizeof(SBasicViewParameters::MVP),
            sizeof(SBasicViewParameters::MV),
            sizeof(SBasicViewParameters::NormalMat)
        };
        constexpr uint32_t relOffsets[DS1_METADATA_ENTRY_CNT] =
        {
            offsetof(SBasicViewParameters,MVP),
            offsetof(SBasicViewParameters,MV),
            offsetof(SBasicViewParameters,NormalMat)
        };
        for (uint32_t i=0u; i<DS1_METADATA_ENTRY_CNT; ++i)
        {
            auto& semantic = (m_basicViewParamsSemantics->end()-i-1u)[0];
            semantic.type = types[i];
            semantic.descriptorSection.type = IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER;
            semantic.descriptorSection.uniformBufferObject.binding = ds1layout->getDescriptorRedirect(IDescriptor::E_TYPE::ET_UNIFORM_BUFFER).getBindingNumber(0).data;
            semantic.descriptorSection.uniformBufferObject.set = 1u;
            semantic.descriptorSection.uniformBufferObject.relByteoffset = relOffsets[i];
            semantic.descriptorSection.uniformBufferObject.bytesize = sizes[i];
            semantic.descriptorSection.shaderAccessFlags = ICPUShader::ESS_VERTEX;
        }
    }
}
