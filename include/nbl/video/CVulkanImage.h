// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_VULKAN_IMAGE_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_IMAGE_H_INCLUDED_


#include "nbl/video/CVulkanDeviceMemoryBacked.h"


namespace nbl::video
{

class ILogicalDevice;

class CVulkanImage : public CVulkanDeviceMemoryBacked<IGPUImage>
{
	public:
		inline CVulkanImage(const ILogicalDevice* dev, IGPUImage::SCreationParams&& _params, const VkImage _vkimg) : CVulkanDeviceMemoryBacked<IGPUImage>(dev,std::move(_params),_vkimg) {}

		void setObjectDebugName(const char* label) const override;

	protected:
		virtual ~CVulkanImage();
};

} // end namespace nbl::video

#endif
