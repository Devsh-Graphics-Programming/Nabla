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
        using images_array_t = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<IGPUImage>>;

        struct SCreationParams
        {
            core::smart_refctd_ptr<ISurface> surface;
            uint32_t minImageCount;
            ISurface::SFormat surfaceFormat;
            ISurface::E_PRESENT_MODE presentMode;
            uint32_t width;
            uint32_t height;
            uint32_t arrayLayers = 1u;
            uint32_t queueFamilyIndexCount;
            const uint32_t* queueFamilyIndices;
            asset::IImage::E_USAGE_FLAGS imageUsage;
            asset::E_SHARING_MODE imageSharingMode;
            ISurface::E_SURFACE_TRANSFORM_FLAGS preTransform;
            ISurface::E_COMPOSITE_ALPHA compositeAlpha;
            core::smart_refctd_ptr<ISwapchain> oldSwapchain = nullptr;
        };

        enum E_ACQUIRE_IMAGE_RESULT
        {
            EAIR_SUCCESS,
            EAIR_TIMEOUT,
            EAIR_NOT_READY,
            EAIR_SUBOPTIMAL,
            EAIR_ERROR
        };

        enum E_PRESENT_RESULT
        {
            EPR_SUCCESS = 0,
            EPR_SUBOPTIMAL,
            EPR_ERROR // There are other types of errors as well for if they are ever required in the future
        };

        uint32_t getImageCount() const { return m_images->size(); }
        core::SRange<core::smart_refctd_ptr<IGPUImage>> getImages()
        {
            return { m_images->begin(), m_images->end() };
        }

        virtual E_ACQUIRE_IMAGE_RESULT acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx) = 0;
        // 100% blocking version, guaranteed to **not** return TIMEOUT or NOT_READY
        virtual E_ACQUIRE_IMAGE_RESULT acquireNextImage(IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx)
        {
            E_ACQUIRE_IMAGE_RESULT result=EAIR_NOT_READY;
            while (result==EAIR_NOT_READY||result==EAIR_TIMEOUT)
            {
                result = acquireNextImage(999999999ull,semaphore,fence,out_imgIx);
                if (result==EAIR_ERROR)
                {
                    assert(false);
                    break;
                }
            }
            return result;
        }

        ISwapchain(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params,
            images_array_t&& images)
            : IBackendObject(std::move(dev)), m_params(std::move(params)), m_images(std::move(images))
        {}
        
        inline const auto& getCreationParameters() const
        {
            return m_params;
        }

    protected:
        virtual ~ISwapchain() = default;

        SCreationParams m_params;
        images_array_t m_images;
};

}

#endif