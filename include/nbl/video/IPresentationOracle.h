// Copyright (C) 2018-2021 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_PRESENTATION_ORACLE__H_INCLUDED__
#define __NBL_VIDEO_I_PRESENTATION_ORACLE__H_INCLUDED__

#include <nabla.h>

namespace nbl
{
	namespace video
	{
        class IPresentationOracle
        {
            public:
                //! demark CPU work on a frame start and end
                virtual void reportBeginFrameRecord() = 0;
                virtual void reportEndFrameRecord() = 0;

                //! wraps image acquire, could inject timings and queries before and after to obtain frame presentation data
                virtual void acquireNextImage(ISwapchain* swapchain, uint64_t timeout, IGPUSemaphore* acquireSemaphore, IGPUFence* fence, uint32_t& imageNumber) = 0;

                //! wraps image acquire, could inject timings and queries before and after to obtain frame presentation data
                virtual void present(nbl::video::ILogicalDevice* device, nbl::video::ISwapchain* swapchain, nbl::video::IGPUQueue* queue, nbl::video::IGPUSemaphore* renderFinishedSemaphore, uint32_t imageNumer) = 0;
        };
	}
}

#endif // __NBL_VIDEO_I_PRESENTATION_ORACLE__H_INCLUDED__