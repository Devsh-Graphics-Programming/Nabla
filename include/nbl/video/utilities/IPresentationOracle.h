// Copyright (C) 2018-2021 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_PRESENTATION_ORACLE_H_INCLUDED_
#define _NBL_VIDEO_I_PRESENTATION_ORACLE_H_INCLUDED_

#include "nbl/video/ISwapchain.h"

namespace nbl::video
{
#if 0 // TODO: port
class IPresentationOracle
{
    public:
        //! demark CPU work on a frame start and end
        virtual void reportBeginFrameRecord() = 0;
        virtual void reportEndFrameRecord() = 0;

        //! wraps image acquire, could inject timings and queries before and after to obtain frame presentation data
        virtual std::chrono::microseconds acquireNextImage(ISwapchain* swapchain, IGPUSemaphore* acquireSemaphore, IGPUFence* fence, uint32_t* imageNumber) = 0;

        // TODO: Actually start using this in CommonAPI of the examples
        //! wraps image acquire, could inject timings and queries before and after to obtain frame presentation data
        virtual void present(nbl::video::ILogicalDevice* device, nbl::video::ISwapchain* swapchain, nbl::video::IQueue* queue, nbl::video::IGPUSemaphore* renderFinishedSemaphore, const uint32_t imageNumer) = 0;
};
#endif
}

#endif // __NBL_VIDEO_I_PRESENTATION_ORACLE__H_INCLUDED__