// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_I_GPU_BUFFER_H_INCLUDED__
#define __NBL_I_GPU_BUFFER_H_INCLUDED__

#include "nbl/asset/IBuffer.h"
#include "IDriverMemoryBacked.h"
#include "nbl/asset/IDescriptor.h"
#include "nbl/video/IBackendObject.h"

namespace nbl
{
namespace video
{

//! GPU Buffer class, where the memory is provided by the driver, does not support resizing.
/** For additional OpenGL DSA state-free operations such as flushing mapped ranges or
buffer to buffer copies, one needs a command buffer in Vulkan as these operations are
performed by the GPU and not wholly by the driver, so look for them in IDriver and IVideoDriver. */
class IGPUBuffer : public asset::IBuffer, public IDriverMemoryBacked, public IBackendObject
{
    protected:
        IGPUBuffer(ILogicalDevice* dev, const IDriverMemoryBacked::SDriverMemoryRequirements& reqs) : IDriverMemoryBacked(reqs), IBackendObject(dev) {}

    public:
        //! Get usable buffer byte size.
        inline const uint64_t& getSize() const {return cachedMemoryReqs.vulkanReqs.size;}

        //! Whether calling updateSubRange will produce any effects.
        virtual bool canUpdateSubRange() const = 0;
};

} // end namespace scene
} // end namespace nbl

#endif

