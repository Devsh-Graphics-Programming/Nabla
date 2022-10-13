// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_I_DESCRIPTOR_SET_H_INCLUDED_
#define _NBL_ASSET_I_DESCRIPTOR_SET_H_INCLUDED_

#include <algorithm>
#include <compare>


#include "nbl/core/declarations.h"

#include "nbl/asset/format/EFormat.h"
#include "nbl/asset/IDescriptor.h"
#include "nbl/asset/IDescriptorSetLayout.h" //for E_DESCRIPTOR_TYPE
#include "nbl/core/SRange.h"

namespace nbl::asset
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
class NBL_API IDescriptorSet : public virtual core::IReferenceCounted
{
		using this_type = IDescriptorSet<LayoutType>;

	public:
		using layout_t = LayoutType;
		struct NBL_API SDescriptorInfo
		{
                struct SBufferInfo
                {
                    size_t offset;
                    size_t size;//in Vulkan it's called `range` but IMO it's misleading so i changed to `size`

					static constexpr inline size_t WholeBuffer = ~0ull;

					auto operator<=>(const SBufferInfo&) const = default;
                };
                struct SImageInfo
                {
					// This will be ignored if the DS layout already has an immutable sampler specified for the binding.
                    core::smart_refctd_ptr<typename layout_t::sampler_type> sampler;
                    //! Irrelevant in OpenGL backend
                    IImage::E_LAYOUT imageLayout;
                };
                    
				core::smart_refctd_ptr<IDescriptor> desc;
				union
				{
					SBufferInfo buffer;
					SImageInfo image;
				};

				SDescriptorInfo()
				{
					memset(&buffer, 0, core::max<size_t>(sizeof(buffer), sizeof(image)));
				}
				template<typename BufferType>
				SDescriptorInfo(const SBufferBinding<BufferType>& binding) : desc()
				{
					desc = binding.buffer;
					buffer.offset = binding.offset;
					buffer.size = SBufferInfo::WholeBuffer;
				}
				template<typename BufferType>
				SDescriptorInfo(const SBufferRange<BufferType>& range) : desc()
				{
					desc = range.buffer;
					buffer.offset = range.offset;
					buffer.size = range.size;
				}
				SDescriptorInfo(const SDescriptorInfo& other) : SDescriptorInfo()
				{
					operator=(other);
				}
				SDescriptorInfo(SDescriptorInfo&& other): SDescriptorInfo()
				{
					operator=(std::move(other));
				}
				~SDescriptorInfo()
				{
					if (desc && desc->getTypeCategory()==IDescriptor::EC_IMAGE)
						image.sampler = nullptr;
				}

				inline SDescriptorInfo& operator=(const SDescriptorInfo& other)
				{
					if (desc && desc->getTypeCategory()==IDescriptor::EC_IMAGE)
						image.sampler = nullptr;
					desc = other.desc;
					const auto type = desc->getTypeCategory();
					if (type!=IDescriptor::EC_IMAGE)
						buffer = other.buffer;
					else
						image = other.image;
					return *this;
				}
				inline SDescriptorInfo& operator=(SDescriptorInfo&& other)
				{
					if (desc && desc->getTypeCategory()==IDescriptor::EC_IMAGE)
						image = {nullptr,IImage::EL_UNDEFINED};
					desc = std::move(other.desc);
					if (desc)
					{
						const auto type = desc->getTypeCategory();
						if (type!=IDescriptor::EC_IMAGE)
							buffer = other.buffer;
						else
							image = other.image;
					}
					return *this;
				}

				inline bool operator!=(const SDescriptorInfo& other) const
				{
					if (desc != desc)
						return true;
					return buffer != other.buffer;
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

		const layout_t* getLayout() const { return m_layout.get(); }

	protected:
		IDescriptorSet(core::smart_refctd_ptr<layout_t>&& _layout) : m_layout(std::move(_layout))
		{
		}

		virtual ~IDescriptorSet()
		{
			// std::destroy_n(getSamplerRefcountingStorage(), m_layout->getMutableSamplerCount());
			// for (auto type = 0; type < EDT_COUNT; ++type)
			// 	std::destroy_n(getDescriptorRefcountingStorage(type), m_layout->getTotalDescriptorCount(type));
		}

		virtual core::smart_refctd_ptr<IDescriptor>* getDescriptorStorage(const E_DESCRIPTOR_TYPE type) const = 0;
		// virtual core::smart_refctd_ptr<ISampler>* getSamplerRefcountingStorage() = 0;
		// virtual void allocateDescriptors() = 0;

#if 0
		inline void createDescriptors()
		{
			allocateDescriptors();
			std::uninitialized_default_construct_n(getSamplerRefcountingStorage(), m_layout->getMutableSamplerCount());
			for (auto type = 0; type < EDT_COUNT; ++type)
				std::uninitialized_default_construct_n(getDescriptorRefcountingStorage(type), m_layout->getTotalDescriptorCount(type));
		}
#endif

		core::smart_refctd_ptr<layout_t> m_layout;
};



namespace impl
{
	
//! Only reason this class exists is because OpenGL back-end implements a similar interface
template<typename LayoutType>
class NBL_API IEmulatedDescriptorSet
{
	public:
		//! Contructor computes the flattened out array of descriptors
		IEmulatedDescriptorSet(LayoutType* _layout)
		{
			if (!_layout)
				return;

			using bnd_t = typename LayoutType::SBinding;
			auto max_bnd_cmp = [](const bnd_t& a, const bnd_t& b) { return a.binding < b.binding; };

			auto bindings = _layout->getBindings();

			// TODO(achal): We don't want this sparse shit.
            auto lastBnd = std::max_element(bindings.begin(), bindings.end(), max_bnd_cmp);

			m_bindingInfo = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<SBindingInfo> >(lastBnd->binding+1u);
			for (auto it=m_bindingInfo->begin(); it!=m_bindingInfo->end(); it++)
				*it = {~0u,EDT_COUNT};
			
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

			uint32_t offset = descriptorCount;
			
			m_descriptorInfos = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<typename IDescriptorSet<LayoutType>::SDescriptorInfo> >(descriptorCount);

			// set up all offsets, reverse iteration important because "it is for filling gaps with offset of next binding"
			// TODO: rewrite this whole constructor to initialize the `SBindingOffset::offset` to 0 and simply use `std::exclusive_scan` to set it all up
			for (auto it=m_bindingInfo->end()-1; it!=m_bindingInfo->begin()-1; it--)
			{
				if (it->offset < descriptorCount)
					offset = it->offset;
				else
					it->offset = offset;
			}

			// this is vital for getDescriptorCountAtIndex
			uint32_t off = ~0u;
			for (auto it = m_bindingInfo->end() - 1; it != m_bindingInfo->begin() - 1; --it)
			{
				if (it->descriptorType != EDT_COUNT)
					off = it->offset;
				else
					it->offset = off;
			}
		}

	protected:
		virtual ~IEmulatedDescriptorSet() = default;

		struct SBindingInfo
		{
			inline bool operator!=(const SBindingInfo& other) const
			{
				return offset!=other.offset || descriptorType!=other.descriptorType;
			}

			uint32_t offset;
			E_DESCRIPTOR_TYPE descriptorType = EDT_COUNT; //whatever, default value
		};

		static_assert(sizeof(SBindingInfo)==8ull, "Why is the enum not uint32_t sized!?");

		core::smart_refctd_dynamic_array<SBindingInfo> m_bindingInfo;
		core::smart_refctd_dynamic_array<typename IDescriptorSet<LayoutType>::SDescriptorInfo> m_descriptorInfos;
};

}

}

#endif
