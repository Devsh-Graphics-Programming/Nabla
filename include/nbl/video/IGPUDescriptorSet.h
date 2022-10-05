// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_DESCRIPTOR_SET_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_DESCRIPTOR_SET_H_INCLUDED__


#include "nbl/asset/IDescriptorSet.h"

#include "nbl/video/IGPUBuffer.h"
#include "nbl/video/IGPUBufferView.h"
#include "nbl/video/IGPUImageView.h"
#include "nbl/video/IGPUSampler.h"
#include "nbl/video/IGPUDescriptorSetLayout.h"


namespace nbl::video
{

//! GPU Version of Descriptor Set
/*
	@see IDescriptorSet
*/

class NBL_API IGPUDescriptorSet : public asset::IDescriptorSet<const IGPUDescriptorSetLayout>, public IBackendObject
{
		using base_t = asset::IDescriptorSet<const IGPUDescriptorSetLayout>;

	public:
		IGPUDescriptorSet(core::smart_refctd_ptr<const ILogicalDevice>&& dev, core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& _layout, core::smart_refctd_ptr<IDescriptorPool>&& pool, IDescriptorPool::SDescriptorOffsets&& descriptorStorageOffsets)
			: base_t(std::move(_layout)), IBackendObject(std::move(dev)), m_pool(std::move(pool)), m_descriptorStorageOffsets(std::move(descriptorStorageOffsets))
		{}

		uint8_t* getDescriptorMemory(const asset::E_DESCRIPTOR_TYPE type, const uint32_t binding) const;

	protected:
		virtual ~IGPUDescriptorSet()
		{
			for (const auto& b : getLayout()->getBindings())
			{
				assert(m_descriptorStorageOffsets.data[b.type] != ~0u && "Descriptor of this type doesn't exist in the set!");

				auto* descriptorMemory = getDescriptorMemory(b.type, b.binding);
				assert(descriptorMemory);

				auto* descriptors = reinterpret_cast<core::smart_refctd_ptr<const asset::IDescriptor>*>(descriptorMemory);

				for (auto i = 0; i < b.count; ++i)
					descriptors[i].~smart_refctd_ptr();
			}
		}

	private:
#if 0
		void allocateDescriptors() override
		{
			// No-OP because allocation is already been done in ILogicalDevice::createDescriptorSet
		}
#endif

		core::smart_refctd_ptr<IDescriptorPool> m_pool;
		IDescriptorPool::SDescriptorOffsets m_descriptorStorageOffsets;
};

}

#endif