#ifndef __NBL_I_SWAPCHAIN_H_INCLUDED__
#define __NBL_I_SWAPCHAIN_H_INCLUDED__


#include "nbl/video/surface/ISurface.h"
#include "nbl/video/IGPUImage.h"
#include "nbl/video/IGPUSemaphore.h"
#include "nbl/video/IGPUFence.h"
#include "nbl/core/util/bitflag.h"

namespace nbl::video
{

class NBL_API ISwapchain : public core::IReferenceCounted, public IBackendObject
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
            core::bitflag<asset::IImage::E_USAGE_FLAGS> imageUsage;
            asset::E_SHARING_MODE imageSharingMode;
            ISurface::E_SURFACE_TRANSFORM_FLAGS preTransform;
            ISurface::E_COMPOSITE_ALPHA compositeAlpha;
            core::smart_refctd_ptr<ISwapchain> oldSwapchain = nullptr;
        };

        struct SPresentInfo
        {
            uint32_t waitSemaphoreCount;
            IGPUSemaphore* const* waitSemaphores;
            uint32_t imgIndex;
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

        inline uint32_t getImageCount() const { return m_images->size(); }
        inline core::SRange<core::smart_refctd_ptr<IGPUImage>> getImages()
        {
            return { m_images->begin(), m_images->end() };
        }

        // The value passed to `preTransform` when creating the swapchain. "pre" refers to the transform happening
        // as an operation before the presentation engine presents the image.
        inline ISurface::E_SURFACE_TRANSFORM_FLAGS getPreTransform() const { return m_params.preTransform; }

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

        virtual E_PRESENT_RESULT present(IGPUQueue* queue, const SPresentInfo& info) = 0;

        inline ISwapchain(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params, images_array_t&& images)
            : IBackendObject(std::move(dev)), m_params(std::move(params)), m_images(std::move(images))
        {
            m_imageExists = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<std::atomic<bool>>>(m_images->size());
        }
        
        inline const auto& getCreationParameters() const
        {
            return m_params;
        }

        // OpenGL: const egl::CEGL::Context*
        // Vulkan: const VkSwapchainKHR*
        virtual const void* getNativeHandle() const = 0;

        void freeImageExists(uint32_t ix) { m_imageExists->begin()[ix].store(false); }

    protected:
        virtual ~ISwapchain() = default;

        SCreationParams m_params;
        images_array_t m_images;
        core::smart_refctd_dynamic_array<std::atomic<bool>> m_imageExists;
        
        // Returns false if the image already existed
        bool setImageExists(uint32_t ix) { return !m_imageExists->begin()[ix].exchange(true); }

};

}

#endif