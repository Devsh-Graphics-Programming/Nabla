#ifndef __NBL_I_SWAPCHAIN_H_INCLUDED__
#define __NBL_I_SWAPCHAIN_H_INCLUDED__


#include "nbl/video/surface/ISurface.h"
#include "nbl/video/IGPUSemaphore.h"
#include "nbl/video/IGPUFence.h"
#include "nbl/video/IGPUImage.h"
#include "nbl/core/util/bitflag.h"

namespace nbl::video
{
class IGPUQueue;

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
            core::bitflag<asset::IImage::E_USAGE_FLAGS> imageUsage;
            ISurface::E_SURFACE_TRANSFORM_FLAGS preTransform;
            ISurface::E_COMPOSITE_ALPHA compositeAlpha;
            core::smart_refctd_ptr<ISwapchain> oldSwapchain = nullptr;

            inline bool isConcurrentSharing() const {return queueFamilyIndexCount!=0u;}
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

        inline uint8_t getImageCount() const { return m_imageCount; }

        // The value passed to `preTransform` when creating the swapchain. "pre" refers to the transform happening
        // as an operation before the presentation engine presents the image.
        inline ISurface::E_SURFACE_TRANSFORM_FLAGS getPreTransform() const { return m_params.preTransform; }

        virtual E_ACQUIRE_IMAGE_RESULT acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx) = 0;
        // 100% blocking version, guaranteed to **not** return TIMEOUT or NOT_READY
        virtual E_ACQUIRE_IMAGE_RESULT acquireNextImage(IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx)
        {
            return acquireNextImage(std::numeric_limits<uint64_t>::max(),semaphore,fence,out_imgIx);
        }

        virtual E_PRESENT_RESULT present(IGPUQueue* queue, const SPresentInfo& info) = 0;

        virtual core::smart_refctd_ptr<IGPUImage> createImage(const uint32_t imageIndex) = 0;

        inline ISwapchain(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params, IGPUImage::SCreationParams&& imgCreationParams, uint8_t imageCount)
            : IBackendObject(std::move(dev)), m_params(std::move(params)), m_imageCount(imageCount), m_imgCreationParams(std::move(imgCreationParams))
        {
            // don't need to keep a reference to the old swapchain anymore
            m_params.oldSwapchain = nullptr;
        }
        
        inline const auto& getCreationParameters() const
        {
            return m_params;
        }

        // OpenGL: const egl::CEGL::Context*
        // Vulkan: const VkSwapchainKHR*
        virtual const void* getNativeHandle() const = 0;

        struct CCleanupSwapchainReference : public ICleanup
        {
            core::smart_refctd_ptr<ISwapchain> m_swapchain;
            uint32_t m_imageIndex;

            ~CCleanupSwapchainReference()
            {
                m_swapchain->freeImageExists(m_imageIndex);
            }
        };

    protected:
        virtual ~ISwapchain() = default;

        SCreationParams m_params;
        uint8_t m_imageCount;
        std::atomic_uint32_t m_imageExists = 0;
        IGPUImage::SCreationParams m_imgCreationParams;

        friend class CCleanupSwapchainReference;
        void freeImageExists(uint32_t ix) { m_imageExists.fetch_and(~(1U << ix)); }

        // Returns false if the image already existed
        bool setImageExists(uint32_t ix) { return (m_imageExists.fetch_or(1U << ix) & (1U << ix)) == 0; }

    public:
        static inline constexpr uint32_t MaxImages = sizeof(m_imageExists) * 8u;

};

}

#endif