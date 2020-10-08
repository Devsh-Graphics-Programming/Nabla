#ifndef __IRR_C_MITSUBA_PIPELINE_METADATA_H_INCLUDED__
#define __IRR_C_MITSUBA_PIPELINE_METADATA_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/core/containers/refctd_dynamic_array.h"
#include "irr/asset/ICPUDescriptorSet.h"
#include "irr/asset/IPipelineMetadata.h"

namespace irr
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

    _IRR_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CMitsubaLoader";
    const char* getLoaderName() const override { return LoaderName; }

private:
    core::smart_refctd_ptr<asset::ICPUDescriptorSet> m_ds0;
    core::smart_refctd_dynamic_array<ShaderInputSemantic> m_shaderInputs;
};

}}}

#endif
