// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_BUFFER_VIEW_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_BUFFER_VIEW_H_INCLUDED__


#include "nbl/asset/IBufferView.h"

#include <utility>

#include "nbl/video/decl/IBackendObject.h"
#include "nbl/video/IGPUBuffer.h"


namespace nbl::video
{

class IGPUBufferView : public asset::IBufferView<const IGPUBuffer>, public IBackendObject
{
    public:
        IGPUBufferView(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const asset::SBufferRange<const IGPUBuffer>& underlying, const asset::E_FORMAT _format) :
            asset::IBufferView<const IGPUBuffer>(underlying,_format), IBackendObject(std::move(dev)) {}

        // OpenGL: const GLuint* handle of GL_TEXTURE_BUFFER
        // Vulkan: const VkBufferView*
        virtual const void* getNativeHandle() const = 0;

    protected:
        virtual ~IGPUBufferView() = default;
};

}

#endif