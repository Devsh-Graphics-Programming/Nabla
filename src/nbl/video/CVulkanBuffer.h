// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_VULKAN_BUFFER_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_BUFFER_H_INCLUDED_


#include "nbl/video/CVulkanDeviceMemoryBacked.h"


namespace nbl::video
{

class ILogicalDevice;

class CVulkanBuffer : public CVulkanDeviceMemoryBacked<IGPUBuffer>
{
       using base_t = CVulkanDeviceMemoryBacked<IGPUBuffer>;

    public:
        inline CVulkanBuffer(const ILogicalDevice* dev, IGPUBuffer::SCreationParams&& creationParams, const VkBuffer buffer) : base_t(dev,std::move(creationParams),buffer) {}
    
        void setObjectDebugName(const char* label) const override;

    private:
        ~CVulkanBuffer();
};

}

#endif
