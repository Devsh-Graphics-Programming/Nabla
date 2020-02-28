#ifndef __IRR_I_CPU_COMPUTE_PIPELINE_H_INCLUDED__
#define __IRR_I_CPU_COMPUTE_PIPELINE_H_INCLUDED__

#include "irr/asset/IComputePipeline.h"
#include "irr/asset/ICPUPipelineLayout.h"
#include "irr/asset/ICPUSpecializedShader.h"
#include "irr/asset/IPipelineMetadata.h"

namespace irr
{
namespace asset
{

class ICPUComputePipeline : public IComputePipeline<ICPUSpecializedShader, ICPUPipelineLayout>, public IAsset
{
    using base_t = IComputePipeline<ICPUSpecializedShader, ICPUPipelineLayout>;

public:
    using base_t::base_t;

    size_t conservativeSizeEstimate() const override { return sizeof(void*)*3u+sizeof(uint8_t); }
    void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
	{
        if (isDummyObjectForCacheAliasing)
            return;
        convertToDummyObject_common(referenceLevelsBelowToConvert);

		if (referenceLevelsBelowToConvert)
		{
            //intentionally parent is not converted
            --referenceLevelsBelowToConvert;
			m_shader->convertToDummyObject(referenceLevelsBelowToConvert);
			m_layout->convertToDummyObject(referenceLevelsBelowToConvert);
		}
	}

    core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
    {
        core::smart_refctd_ptr<ICPUPipelineLayout> layout = (_depth > 0u && m_layout) ? core::smart_refctd_ptr_static_cast<ICPUPipelineLayout>(m_layout->clone(_depth-1u)) : m_layout;
        core::smart_refctd_ptr<ICPUSpecializedShader> shader = (_depth > 0u && m_shader) ? core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(m_shader->clone(_depth-1u)) : m_shader;

        auto cp = core::make_smart_refctd_ptr<ICPUComputePipeline>(std::move(layout), std::move(shader));
        clone_common(cp.get());

        return cp;
    }

    E_TYPE getAssetType() const override { return ET_COMPUTE_PIPELINE; }

    ICPUPipelineLayout* getLayout() { return m_layout.get(); }
    const ICPUPipelineLayout* getLayout() const { return m_layout.get(); }

    ICPUSpecializedShader* getShader() { return m_shader.get(); }
    const ICPUSpecializedShader* getShader() const { return m_shader.get(); }
    void setShader(ICPUSpecializedShader* _cs) { m_shader = core::smart_refctd_ptr<ICPUSpecializedShader>(_cs); }

protected:
    virtual ~ICPUComputePipeline() = default;
};

}
}

#endif