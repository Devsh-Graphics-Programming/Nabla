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
#include "nbl/asset/IDescriptorSetLayout.h" //for IDescriptor::E_TYPE
#include "nbl/core/SRange.h"

namespace nbl::asset
{

class IAccelerationStructure;

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
				union SBufferImageInfo
				{
					SBufferImageInfo()
					{
						memset(&buffer, 0, core::max<size_t>(sizeof(buffer), sizeof(image)));
					};

					~SBufferImageInfo() {};

					SBufferInfo buffer;
					SImageInfo image;
				} info;

				SDescriptorInfo() {}

				template<typename BufferType>
				SDescriptorInfo(const SBufferBinding<BufferType>& binding) : desc()
				{
					desc = binding.buffer;
					info.buffer.offset = binding.offset;
					info.buffer.size = SBufferInfo::WholeBuffer;
				}
				template<typename BufferType>
				SDescriptorInfo(const SBufferRange<BufferType>& range) : desc()
				{
					desc = range.buffer;
					info.buffer.offset = range.offset;
					info.buffer.size = range.size;
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
						info.image.sampler = nullptr;
				}

				inline SDescriptorInfo& operator=(const SDescriptorInfo& other)
				{
					if (desc && desc->getTypeCategory()==IDescriptor::EC_IMAGE)
						info.image.sampler = nullptr;
					desc = other.desc;
					const auto type = desc->getTypeCategory();
					if (type!=IDescriptor::EC_IMAGE)
						info.buffer = other.info.buffer;
					else
						info.image = other.info.image;
					return *this;
				}
				inline SDescriptorInfo& operator=(SDescriptorInfo&& other)
				{
					if (desc && desc->getTypeCategory()==IDescriptor::EC_IMAGE)
						info.image = {nullptr,IImage::EL_UNDEFINED};
					desc = std::move(other.desc);
					if (desc)
					{
						const auto type = desc->getTypeCategory();
						if (type!=IDescriptor::EC_IMAGE)
							info.buffer = other.info.buffer;
						else
							info.image = other.info.image;
					}
					return *this;
				}

				inline bool operator!=(const SDescriptorInfo& other) const
				{
					if (desc != desc)
						return true;
					return info.buffer != other.info.buffer;
				}
		};

		struct SWriteDescriptorSet
		{
			//smart pointer not needed here
			this_type* dstSet;
			uint32_t binding;
			uint32_t arrayElement;
			uint32_t count;
			IDescriptor::E_TYPE descriptorType;
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
		IDescriptorSet(core::smart_refctd_ptr<layout_t>&& _layout) : m_layout(std::move(_layout)) {}
		virtual ~IDescriptorSet() {}

		virtual void allocateDescriptors() = 0;

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

			for (uint32_t t = 0u; t < static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); ++t)
			{
				const auto type = static_cast<IDescriptor::E_TYPE>(t);
				const uint32_t count = _layout->getTotalDescriptorCount(type);
				if (count == 0u)
					continue;

				m_descriptors[t] = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<core::smart_refctd_ptr<IDescriptor>>>(count);
			}

			const uint32_t mutableSamplerCount = _layout->getTotalMutableSamplerCount();
			if (mutableSamplerCount > 0u)
				m_mutableSamplers = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<core::smart_refctd_ptr<ICPUSampler>>>(mutableSamplerCount);
		}

	protected:
		virtual ~IEmulatedDescriptorSet() = default;

	protected:
		core::smart_refctd_dynamic_array<core::smart_refctd_ptr<IDescriptor>> m_descriptors[static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT)];
		core::smart_refctd_dynamic_array<core::smart_refctd_ptr<ICPUSampler>> m_mutableSamplers = nullptr;
};

}

}

#endif
