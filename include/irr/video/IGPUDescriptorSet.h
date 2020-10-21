// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_DESCRIPTOR_SET_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_DESCRIPTOR_SET_H_INCLUDED__

#include "irr/asset/IDescriptorSet.h"

#include "IGPUBuffer.h"
#include "irr/video/IGPUBufferView.h"
#include "irr/video/IGPUImageView.h"
#include "irr/video/IGPUSampler.h"
#include "irr/video/IGPUDescriptorSetLayout.h"

namespace irr
{
namespace video
{

class IGPUDescriptorSet : public asset::IDescriptorSet<const IGPUDescriptorSetLayout>
{
	public:
		using asset::IDescriptorSet<const IGPUDescriptorSetLayout>::IDescriptorSet;

	protected:
		virtual ~IGPUDescriptorSet() = default;
};

}
}

#endif