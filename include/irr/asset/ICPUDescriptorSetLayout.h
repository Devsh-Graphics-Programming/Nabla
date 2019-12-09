#ifndef __IRR_I_CPU_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__
#define __IRR_I_CPU_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__

#include "irr/asset/IDescriptorSetLayout.h"
#include "irr/asset/IAsset.h"
#include "irr/asset/ICPUSampler.h"

namespace irr
{
namespace asset
{

class ICPUDescriptorSetLayout : public IDescriptorSetLayout<ICPUSampler>, public IAsset
{
	public:
		using IDescriptorSetLayout<ICPUSampler>::IDescriptorSetLayout;

		size_t conservativeSizeEstimate() const override { return m_bindings->size()*sizeof(SBinding)+m_samplers->size()*sizeof(void*); }
		void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
			m_bindings = nullptr;
			if (referenceLevelsBelowToConvert--)
			{
				for (auto it=m_samplers->begin(); it!=m_samplers->end(); it++)
					it->get()->convertToDummyObject(referenceLevelsBelowToConvert);
			}
			m_samplers = nullptr;
		}
		E_TYPE getAssetType() const override { return ET_DESCRIPTOR_SET_LAYOUT; }

	protected:
		virtual ~ICPUDescriptorSetLayout() = default;
};

}
}

#endif