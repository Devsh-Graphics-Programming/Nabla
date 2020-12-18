#ifndef __NBL_I_SWAPCHAIN_H_INCLUDED__
#define __NBL_I_SWAPCHAIN_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/video/surface/ISurface.h"
#include "nbl/video/IGPUImage.h"
#include "nbl/video/IGPUSemaphore.h"
#include "nbl/video/IGPUFence.h"

namespace nbl {
namespace video
{

class ISwapchain : public core::IReferenceCounted
{
public:
    struct SCreationParams
    {
        core::smart_refctd_ptr<ISurface> surface;
        uint32_t minImageCount;
        ISurface::SFormat surfaceFormat;
        ISurface::E_PRESENT_MODE presentMode;
        uint32_t width;
        uint32_t height;
        uint32_t arrayLayers = 1u;
        //VkImageUsageFlags imageUsage;
        //VkSharingMode imageSharingMode;
        //uint32_t queueFamilyIndexCount;
        //const uint32_t* pQueueFamilyIndices;
        //VkSurfaceTransformFlagBitsKHR preTransform;
        //VkCompositeAlphaFlagBitsKHR compositeAlpha;
        //VkBool32 clipped;
        //VkSwapchainKHR oldSwapchain;
    };

    enum E_ACQUIRE_IMAGE_RESULT
    {
        EAIR_SUCCESS,
        EAIR_TIMEOUT,
        EAIR_NOT_READY,
        EAIR_SUBOPTIMAL
    };

    virtual uint32_t getImageCount() const = 0;
    virtual bool getImages(core::smart_refctd_ptr<IGPUImage>* out_images) const = 0;

    virtual E_ACQUIRE_IMAGE_RESULT acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx) = 0;
};

}
}

#endif