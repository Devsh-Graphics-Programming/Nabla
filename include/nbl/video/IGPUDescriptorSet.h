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
		IGPUDescriptorSet(core::smart_refctd_ptr<const ILogicalDevice>&& dev, core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& _layout, core::smart_refctd_ptr<IDescriptorPool>&& pool, const uint32_t* descriptorStorageOffsets)
			: base_t(std::move(_layout)), IBackendObject(std::move(dev)), m_pool(std::move(pool))
		{
			memcpy(m_descriptorStorageOffsets, descriptorStorageOffsets, asset::EDT_COUNT * sizeof(uint32_t));

			// VkWriteDescriptorSet::dstArrayElement -> Specifies from where to START updating this descriptor set's binding in case it expects an array of descriptors.
			// VkWriteDescriptorSet::descriptorCount -> Specifies how many descriptors to update this binding with --this is the number of pBufferInfo or pImageInfo structs.
			// VkWriteDescriptorSet::p{Image/Buffer}Info -> The array of descriptorCount elements used to update this binding with.
			// In conclusion, 1 descriptor set binding will require 1 VkWriteDescriptorSet structure but that VkWriteDescriptorSetStructure can update the binding with more than
			// 1 descriptors.
		}

	protected:
		virtual ~IGPUDescriptorSet() = default;

	private:
		core::smart_refctd_ptr<IDescriptorPool> m_pool;
		uint32_t m_descriptorStorageOffsets[asset::EDT_COUNT];
};

}

#endif