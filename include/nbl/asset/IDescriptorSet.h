// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_DESCRIPTOR_SET_H_INCLUDED__
#define __NBL_ASSET_I_DESCRIPTOR_SET_H_INCLUDED__

#include <algorithm>


#include "nbl/core/declarations.h"

#include "nbl/asset/format/EFormat.h"
#include "nbl/asset/IDescriptor.h"
#include "nbl/asset/IDescriptorSetLayout.h" //for E_DESCRIPTOR_TYPE
#include "nbl/core/SRange.h"
#include "nbl/asset/EImageLayout.h"

namespace nbl
{
namespace asset
{

//! Interface class for various Descriptor Set's resources
/*
	Buffers, Images and Samplers all derive from IDescriptor
	and can be bound under different bindings in a DescriptorSet,
	so that they can be bound to the GPU API together in a single
	API call with efficiency.

	@see ICPUDescriptorSet
	@see IReferenceCounted
*/

template<typename LayoutType>
class IDescriptorSet : public virtual core::IReferenceCounted
{
		using this_type = IDescriptorSet<LayoutType>;

	public:
		struct SDescriptorInfo
		{
                struct SBufferInfo
                {
                    size_t offset;
                    size_t size;//in Vulkan it's called `range` but IMO it's misleading so i changed to `size`
                };
                struct SImageInfo
                {
                    core::smart_refctd_ptr<typename LayoutType::sampler_type> sampler;
                    //! Irrelevant in OpenGL backend
                    E_IMAGE_LAYOUT imageLayout;
                };
                    
				core::smart_refctd_ptr<IDescriptor> desc;
				union
				{
					SBufferInfo buffer;
					SImageInfo image;
				};

				void assign(const SDescriptorInfo& _other, E_DESCRIPTOR_TYPE _type)
				{
					desc = _other.desc;
					if (_type == EDT_COMBINED_IMAGE_SAMPLER || _type == EDT_STORAGE_IMAGE)
						assign_img(_other);
					else
						assign_buf(_other);
				}

				SDescriptorInfo()
				{
					memset(&buffer, 0, core::max<size_t>(sizeof(buffer), sizeof(image)));
				}
				~SDescriptorInfo()
				{
					if (desc && desc->getTypeCategory()==IDescriptor::EC_IMAGE)
						image.sampler.~smart_refctd_ptr();
				}

			private:
				void assign_buf(const SDescriptorInfo& other)
				{
					buffer = other.buffer;
				}
				void assign_img(const SDescriptorInfo& other)
				{
					image = other.image;
				}
		};

		struct SWriteDescriptorSet
		{
			//smart pointer not needed here
			this_type* dstSet;
			uint32_t binding;
			uint32_t arrayElement;
			uint32_t count;
			E_DESCRIPTOR_TYPE descriptorType;
			SDescriptorInfo* info;
		};

		struct SCopyDescriptorSet
		{
			//smart pointer not needed here
			this_type* dstSet;
			const this_type* srcSet;
			uint32_t srcBinding;
			uint32_t srcArrayElement;
			uint32_t dstBinding;
			uint32_t dstArrayElement;
			uint32_t count;
		};

		const LayoutType* getLayout() const { return m_layout.get(); }

	protected:
		IDescriptorSet(core::smart_refctd_ptr<LayoutType>&& _layout) : m_layout(std::move(_layout))
		{
		}
		virtual ~IDescriptorSet() = default;

		core::smart_refctd_ptr<LayoutType> m_layout;
};



namespace impl
{
	
//! Only reason this class exists is because OpenGL back-end implements a similar interface
template<typename LayoutType>
class IEmulatedDescriptorSet
{
	public:
		//! Contructor computes the flattened out array of descriptors
		IEmulatedDescriptorSet(LayoutType* _layout)
		{
			if (!_layout)
				return;

			auto bindings = _layout->getBindings();
            auto lastBnd = (bindings.end()-1);

			m_bindingInfo = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<SBindingInfo> >(lastBnd->binding+1u);
			for (auto it=m_bindingInfo->begin(); it!=m_bindingInfo->end(); it++)
				*it = {~0u,EDT_INVALID};
			
			auto outInfo = m_bindingInfo->begin();
			uint32_t descriptorCount = 0u;
			uint32_t prevBinding = 0;
			// set up the offsets of specified bindings and determine descriptor count
			for (auto it=bindings.begin(); it!=bindings.end(); it++)
			{
				// if bindings are sorted, offsets shall be sorted too
				assert(it==bindings.begin() || it->binding>prevBinding);

				m_bindingInfo->operator[](it->binding) = { descriptorCount,it->type};
				descriptorCount += it->count;
				
				prevBinding = it->binding;
			}
			
			m_descriptors = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<typename IDescriptorSet<LayoutType>::SDescriptorInfo> >(descriptorCount);
			// set up all offsets
			prevBinding = 0u;
			for (auto it=m_bindingInfo->begin(); it!=m_bindingInfo->end(); it++)
			{
				if (it->offset < descriptorCount)
					prevBinding = it->offset;
				else
					it->offset = prevBinding;
			}
		}

	protected:
		virtual ~IEmulatedDescriptorSet() = default;

		struct SBindingInfo
		{
			uint32_t offset;
			E_DESCRIPTOR_TYPE descriptorType = EDT_INVALID;//whatever, default value
		};
		static_assert(sizeof(SBindingInfo)==8ull, "Why is the enum not uint32_t sized!?");
		core::smart_refctd_dynamic_array<SBindingInfo> m_bindingInfo;
		core::smart_refctd_dynamic_array<typename IDescriptorSet<LayoutType>::SDescriptorInfo> m_descriptors;
};

}

}
}

#endif
