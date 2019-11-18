#ifndef __IRR_I_CPU_DESCRIPTOR_SET_H_INCLUDED__
#define __IRR_I_CPU_DESCRIPTOR_SET_H_INCLUDED__

#include "irr/asset/IAsset.h"
#include "irr/asset/ICPUBufferView.h"
#include "irr/asset/ICPUImageView.h"
#include "irr/asset/ICPUSampler.h"
#include "irr/asset/ICPUDescriptorSetLayout.h"
#include "irr/asset/IDescriptorSet.h"

namespace irr
{
namespace asset
{


class ICPUDescriptorSet : public IDescriptorSet<ICPUDescriptorSetLayout, ICPUBuffer, ICPUImageView, ICPUBufferView, ICPUSampler>, public IAsset, public impl::IEmulatedDescriptorSet<ICPUDescriptorSetLayout>
{
		using impl_t = impl::IEmulatedDescriptorSet<ICPUDescriptorSetLayout>;
	public:
		using base_t = IDescriptorSet<ICPUDescriptorSetLayout, ICPUBuffer, ICPUImageView, ICPUBufferView, ICPUSampler>;

		//! Contructor preallocating memory for SDescriptorBindings which user can fill later (using non-const getDescriptors()).
		//! @see getDescriptors()
		ICPUDescriptorSet(core::smart_refctd_ptr<ICPUDescriptorSetLayout>&& _layout) : base_t(std::move(_layout)), IAsset(), impl_t(m_layout.get())
		{
		}


		size_t conservativeSizeEstimate() const override
		{
			return m_descriptors->size()*sizeof(SDescriptorInfo)+m_bindingInfo->size()*sizeof(impl::IEmulatedDescriptorSet<ICPUDescriptorSetLayout>::SBindingInfo);
		}
		void convertToDummyObject() override 
		{
			m_descriptors = nullptr;
			m_bindingInfo = nullptr;
		}
		E_TYPE getAssetType() const override { return ET_DESCRIPTOR_SET; }

		ICPUDescriptorSetLayout* getLayout() { return m_layout.get(); }

		//!
		uint32_t getMaxDescriptorBindingIndex() const
		{
			return m_bindingInfo ? m_bindingInfo->size():0u;
		}

		//!
		E_DESCRIPTOR_TYPE getDescriptorsType(uint32_t index) const
		{
			if (m_bindingInfo && index<m_bindingInfo->size())
				return m_bindingInfo->operator[](index).descriptorType;
			return EDT_INVALID;
		}

		//! Can modify the array of descriptors bound to a particular bindings
		core::SRange<SDescriptorInfo> getDescriptors(uint32_t index) 
		{ 
			if (m_bindingInfo && index<m_bindingInfo->size())
			{
				const auto& info = m_bindingInfo->operator[](index);
				auto _begin = m_descriptors->begin()+info.offset;
				if (index+1u!=m_bindingInfo->size())
					return core::SRange<SDescriptorInfo>{_begin, m_descriptors->begin()+m_bindingInfo->operator[](index+1u).offset};
				else
					return core::SRange<SDescriptorInfo>{_begin, m_descriptors->end()};
			}
			else
				core::SRange<SDescriptorInfo>{nullptr, nullptr};
		}

	protected:
		virtual ~ICPUDescriptorSet() = default;
};

}
}

#endif