#ifndef __IRR_I_CPU_PIPELINE_LAYOUT_H_INCLUDED__
#define __IRR_I_CPU_PIPELINE_LAYOUT_H_INCLUDED__

#include "irr/asset/IAsset.h"
#include "irr/asset/ICPUDescriptorSetLayout.h"
#include "irr/asset/IPipelineLayout.h"

namespace irr
{
namespace asset
{

class ICPUPipelineLayout : public IAsset, public IPipelineLayout<ICPUDescriptorSetLayout>
{
	public:
		using IPipelineLayout<ICPUDescriptorSetLayout>::IPipelineLayout;

		ICPUDescriptorSetLayout* getDescriptorSetLayout(uint32_t _set) { return m_descSetLayouts[_set].get(); }
		const ICPUDescriptorSetLayout* getDescriptorSetLayout(uint32_t _set) const { return m_descSetLayouts[_set].get(); }

		size_t conservativeSizeEstimate() const override { return m_descSetLayouts.size()*sizeof(void*)+m_pushConstantRanges->size()*sizeof(SPushConstantRange); }
		void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
			if (referenceLevelsBelowToConvert--)
			for (auto it=m_descSetLayouts.begin(); it!=m_descSetLayouts.end(); it++)
			if (it->get())
				it->get()->convertToDummyObject(referenceLevelsBelowToConvert);
			m_pushConstantRanges = nullptr;
		}
		E_TYPE getAssetType() const override { return ET_PIPELINE_LAYOUT; }

	protected:
		virtual ~ICPUPipelineLayout() = default;
};

}
}

#endif