// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_C_MITSUBA_PIPELINE_METADATA_H_INCLUDED__
#define __NBL_C_MITSUBA_PIPELINE_METADATA_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/core/containers/refctd_dynamic_array.h"
#include "nbl/asset/ICPUDescriptorSet.h"
#include "nbl/asset/IPipelineMetadata.h"

namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{
//TODO make it inherit from IMitsubaMetadata so that it has global mitsuba metadata ptr
class CMitsubaPipelineMetadata final : public asset::IPipelineMetadata
{
public:
    CMitsubaPipelineMetadata(core::smart_refctd_ptr<asset::ICPUDescriptorSet>&& _ds0, core::smart_refctd_dynamic_array<ShaderInputSemantic>&& _inputs) :
        m_ds0(std::move(_ds0)),
        m_shaderInputs(std::move(_inputs))
    {
    }

    core::SRange<const ShaderInputSemantic> getCommonRequiredInputs() const override
    {
        return {m_shaderInputs->begin(), m_shaderInputs->end()};
    }

    asset::ICPUDescriptorSet* getDescriptorSet() const { return m_ds0.get(); }

    _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CMitsubaLoader";
    const char* getLoaderName() const override { return LoaderName; }

private:
    core::smart_refctd_ptr<asset::ICPUDescriptorSet> m_ds0;
    core::smart_refctd_dynamic_array<ShaderInputSemantic> m_shaderInputs;
};

}}}

#endif
