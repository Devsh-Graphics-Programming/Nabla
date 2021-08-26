// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VIDEO_I_GPU_ACCELERATION_STRUCTURE_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_ACCELERATION_STRUCTURE_H_INCLUDED_


#include "nbl/asset/IAccelerationStructure.h"
#include "nbl/video/IGPUBuffer.h"


namespace nbl::video
{
class IGPUAccelerationStructure : public asset::IAccelerationStructure<IGPUBuffer>, public IBackendObject
{
	using Base = asset::IAccelerationStructure<IGPUBuffer>;

	public:
        const SCreationParams& getCreationParameters() const { return params; }

	protected:
		IGPUAccelerationStructure(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& _params) : Base(std::move(_params)), IBackendObject(std::move(dev)) {}
		virtual ~IGPUAccelerationStructure() = default;
};
}

#endif