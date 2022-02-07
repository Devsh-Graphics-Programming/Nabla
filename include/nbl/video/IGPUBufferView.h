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
class IGPUBufferView : public asset::IBufferView<IGPUBuffer>, public IBackendObject
{
public:
    IGPUBufferView(core::smart_refctd_ptr<const ILogicalDevice>&& dev, core::smart_refctd_ptr<IGPUBuffer> _buffer, asset::E_FORMAT _format, size_t _offset = 0ull, size_t _size = IGPUBufferView::whole_buffer)
        : asset::IBufferView<IGPUBuffer>(std::move(_buffer), _format, _offset, _size), IBackendObject(std::move(dev))
    {}

protected:
    virtual ~IGPUBufferView() = default;
};

}

#endif