#ifndef __IRR_C_MITSUBA_PIPELINE_METADATA_H_INCLUDED__
#define __IRR_C_MITSUBA_PIPELINE_METADATA_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/core/memory/refctd_dynamic_array.h"
#include "irr/asset/ICPUDescriptorSet.h"
#include "irr/asset/IPipelineMetadata.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

class CMitsubaPipelineMetadata final : public asset::IPipelineMetadata
{
    CMitsubaPipelineMetadata(core::smart_refctd_ptr<asset::ICPUDescriptorSet>&& _ds0, core::smart_refctd_dynamic_array<ShaderInputSemantic>&& _inputs) :
        m_ds0(std::move(_ds0)),
        m_shaderInputs(std::move(_inputs))
    {
    }

private:
    core::smart_refctd_ptr<asset::ICPUDescriptorSet> m_ds0;
    core::smart_refctd_dynamic_array<ShaderInputSemantic> m_shaderInputs;
};

}}}

#endif
