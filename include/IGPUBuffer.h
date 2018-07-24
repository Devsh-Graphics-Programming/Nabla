// Copyright (C) 2016 Mateusz "DeVsh" Kielan
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __I_GPU_BUFFER_H_INCLUDED__
#define __I_GPU_BUFFER_H_INCLUDED__

#include "IBuffer.h"
#include "IDriverMemoryBacked.h"

namespace irr
{
namespace video
{

//! GPU Buffer class, where the memory is provided by the driver, does not support resizing.
/** For additional OpenGL DSA state-free operations such as flushing mapped ranges or
buffer to buffer copies, one needs a command buffer in Vulkan as these operations are
performed by the GPU and not wholly by the driver, so look for them in IDriver and IVideoDriver. */
class IGPUBuffer : public core::IBuffer, public IDriverMemoryBacked
{
    public:
        //deprecated, delegate this to command buffer
        virtual void updateSubRange(const size_t& offset, const size_t& size, const void* data) = 0;

        virtual bool canUpdateSubRange() const = 0;
};

} // end namespace scene
} // end namespace irr

#endif

