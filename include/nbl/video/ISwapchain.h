#ifndef __NBL_I_SWAPCHAIN_H_INCLUDED__
#define __NBL_I_SWAPCHAIN_H_INCLUDED__


#include "nbl/video/surface/ISurface.h"
#include "nbl/video/IGPUImage.h"
#include "nbl/video/IGPUSemaphore.h"
#include "nbl/video/IGPUFence.h"


namespace nbl::video
{

// TODO: decouple swapchain from queue some more (make presentation a method of swapchain), then we can have fake UE5, Unity, Qt6 swapchains
class ISwapchain : public core::IReferenceCounted, public IBackendObject
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
            core::smart_refctd_dynamic_array<uint32_t> queueFamilyIndices;

        asset::IImage::E_USAGE_FLAGS imageUsage;
        asset::E_SHARING_MODE imageSharingMode;
        // ISurface::E_SURFACE_TRANSFORM_FLAGS preTransform;
        //VkCompositeAlphaFlagBitsKHR compositeAlpha;
        //VkBool32 clipped;
        //VkSwapchainKHR oldSwapchain;
    };

        enum E_ACQUIRE_IMAGE_RESULT
        {
            EAIR_SUCCESS,
            EAIR_TIMEOUT,
            EAIR_NOT_READY,
            EAIR_SUBOPTIMAL,
            EAIR_ERROR
        };

        uint32_t getImageCount() const { return m_images->size(); }
        core::SRange<core::smart_refctd_ptr<IGPUImage>> getImages()
        {
            return { m_images->begin(), m_images->end() };
        }

        virtual E_ACQUIRE_IMAGE_RESULT acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx) = 0;

        ISwapchain(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params) : IBackendObject(std::move(dev)), m_params(std::move(params)) {}

    protected:
        virtual ~ISwapchain() = default;

        SCreationParams m_params;
        using images_array_t = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<IGPUImage>>;
        images_array_t m_images;
};

}

#endif