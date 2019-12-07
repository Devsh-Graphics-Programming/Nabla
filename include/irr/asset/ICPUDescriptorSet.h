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


class ICPUDescriptorSet final : public IDescriptorSet<ICPUDescriptorSetLayout>, public IAsset, public impl::IEmulatedDescriptorSet<ICPUDescriptorSetLayout>
{
		using impl_t = impl::IEmulatedDescriptorSet<ICPUDescriptorSetLayout>;
	public:
		using base_t = IDescriptorSet<ICPUDescriptorSetLayout>;

		//! Contructor preallocating memory for SDescriptorBindings which user can fill later (using non-const getDescriptors()).
		//! @see getDescriptors()
		ICPUDescriptorSet(core::smart_refctd_ptr<ICPUDescriptorSetLayout>&& _layout) : base_t(std::move(_layout)), IAsset(), impl_t(m_layout.get())
		{
		}


		inline size_t conservativeSizeEstimate() const override
		{
			return m_descriptors->size()*sizeof(SDescriptorInfo)+m_bindingInfo->size()*sizeof(impl::IEmulatedDescriptorSet<ICPUDescriptorSetLayout>::SBindingInfo);
		}
		inline void convertToDummyObject() override
		{
			m_descriptors = nullptr;
			m_bindingInfo = nullptr;
		}
		inline E_TYPE getAssetType() const override { return ET_DESCRIPTOR_SET; }

		inline ICPUDescriptorSetLayout* getLayout() { return m_layout.get(); }
		inline const ICPUDescriptorSetLayout* getLayout() const { return m_layout.get(); }

		//!
		inline uint32_t getMaxDescriptorBindingIndex() const
		{
			return m_bindingInfo ? static_cast<uint32_t>(m_bindingInfo->size()):0u;
		}

		//!
		inline E_DESCRIPTOR_TYPE getDescriptorsType(uint32_t index) const
		{
			if (m_bindingInfo && index<m_bindingInfo->size())
				return m_bindingInfo->operator[](index).descriptorType;
			return EDT_INVALID;
		}

		//! Can modify the array of descriptors bound to a particular bindings
		inline core::SRange<SDescriptorInfo> getDescriptors(uint32_t index) 
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
				return core::SRange<SDescriptorInfo>{nullptr, nullptr};
		}
		inline core::SRange<const SDescriptorInfo> getDescriptors(uint32_t index) const
		{
			if (m_bindingInfo && index<m_bindingInfo->size())
			{
				const auto& info = m_bindingInfo->operator[](index);
				auto _begin = m_descriptors->begin()+info.offset;
				if (index+1u!=m_bindingInfo->size())
					return core::SRange<const SDescriptorInfo>{_begin, m_descriptors->begin()+m_bindingInfo->operator[](index+1u).offset};
				else
					return core::SRange<const SDescriptorInfo>{_begin, m_descriptors->end()};
			}
			else
				return core::SRange<const SDescriptorInfo>{nullptr, nullptr};
		}

		inline auto getTotalDescriptorCount() const
		{
			return m_descriptors->size();
		}

	protected:
		virtual ~ICPUDescriptorSet() = default;
};

}
}

#endif