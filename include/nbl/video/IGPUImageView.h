// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_TEXTURE_VIEW_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_TEXTURE_VIEW_H_INCLUDED__


#include "nbl/asset/IImageView.h"

#include "nbl/video/IGPUImage.h"


namespace nbl::video
{

class NBL_API IGPUImageView : public asset::IImageView<IGPUImage>, public IBackendObject
{
	public:
        const SCreationParams& getCreationParameters() const { return params; }

		// OpenGL: const GLuint* handle of GL_TEXTURE_VIEW target
		// Vulkan: const VkImageView*
		virtual const void* getNativeHandle() const = 0;

	protected:
		IGPUImageView(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& _params) : IImageView<IGPUImage>(std::move(_params)), IBackendObject(std::move(dev)) {}
		virtual ~IGPUImageView() = default;
};

}

#endif