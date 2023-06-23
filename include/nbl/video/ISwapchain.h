#ifndef _NBL_VIDEO_I_SWAPCHAIN_H_INCLUDED_
#define _NBL_VIDEO_I_SWAPCHAIN_H_INCLUDED_

#include "nbl/core/util/bitflag.h"

#include "nbl/video/IGPUImage.h"
#include "nbl/video/CThreadSafeQueueAdapter.h"
#include "nbl/video/surface/ISurface.h"


namespace nbl::video
{
class IQueue;

class ISwapchain : public core::IReferenceCounted, public IBackendObject
{
    public:
        struct SCreationParams
        {
            core::smart_refctd_ptr<ISurface> surface = {};
            uint32_t minImageCount = 3u;
            ISurface::SFormat surfaceFormat = {};
            ISurface::E_PRESENT_MODE presentMode = ISurface::EPM_FIFO;
            uint32_t width = ~0u;
            uint32_t height = ~0u;
            uint32_t arrayLayers = 1u;
            uint32_t queueFamilyIndexCount = 0u;
            const uint32_t* queueFamilyIndices = nullptr;
            core::bitflag<IGPUImage::E_USAGE_FLAGS> imageUsage = IGPUImage::EUF_NONE;
            ISurface::E_SURFACE_TRANSFORM_FLAGS preTransform = ISurface::E_SURFACE_TRANSFORM_FLAGS::EST_IDENTITY_BIT;
            ISurface::E_COMPOSITE_ALPHA compositeAlpha = ISurface::E_COMPOSITE_ALPHA::ECA_OPAQUE_BIT;
            core::smart_refctd_ptr<ISwapchain> oldSwapchain = nullptr; // TODO: investigate if we can abuse this to retire old swapchain tied resources
            // If you set it to something else then your Swapchain will be created with Mutable Format capability
            // NOTE: If you do that, then the bitset needs to contain `viewFormats[surfaceFormat.format] = true`
            std::bitset<asset::E_FORMAT::EF_COUNT> viewFormats = {};

            inline bool isConcurrentSharing() const {return queueFamilyIndexCount!=0u;}
        };
        inline const auto& getCreationParameters() const
        {
            return m_params;
        }

        inline uint8_t getImageCount() const { return m_imageCount; }

        // The value passed to `preTransform` when creating the swapchain. "pre" refers to the transform happening
        // as an operation before the presentation engine presents the image.
        inline ISurface::E_SURFACE_TRANSFORM_FLAGS getPreTransform() const { return m_params.preTransform; }

        // acquire
        struct SAcquireInfo
        {
            IQueue* queue;
            // If you don't change the default it will 100% block and acquire will not return TIMEOUT or NOT_READY
            uint64_t timeout = std::numeric_limits<uint64_t>::max();
            uint32_t signalSemaphoreCount = 0u;
            const IQueue::SSubmitInfo::SSemaphoreInfo* signalSemaphores = nullptr;
        };
        enum class ACQUIRE_IMAGE_RESULT : uint8_t
        {
            SUCCESS,
            TIMEOUT,
            NOT_READY,
            SUBOPTIMAL,
            _ERROR // GDI macros getting in the way
        };
        // Even though in Vulkan image acquisition is not a queue operation, we perform a micro-submit to adapt a Timeline Semaphore to work with it 
        inline ACQUIRE_IMAGE_RESULT acquireNextImage(SAcquireInfo info, uint32_t* const out_imgIx)
        {
            if (!out_imgIx || !info.queue)
                return ACQUIRE_IMAGE_RESULT::_ERROR;

            auto* threadsafeQ = dynamic_cast<CThreadSafeQueueAdapter*>(info.queue);
            if (threadsafeQ)
            {
                threadsafeQ->m.lock();
                info.queue = threadsafeQ->getUnderlyingQueue();
            }
            const auto retval = acquireNextImage_impl(info,out_imgIx);
            if (threadsafeQ)
                threadsafeQ->m.unlock();
            return retval;
        }

        // present
        struct SPresentInfo
        {
            IQueue* queue;
            uint32_t imgIndex;
            uint32_t waitSemaphoreCount = 0u;
            const IQueue::SSubmitInfo::SSemaphoreInfo* waitSemaphores = nullptr;
        };
        enum class PRESENT_RESULT : uint8_t
        {
            SUCCESS = 0,
            SUBOPTIMAL,
            _ERROR // There are other types of errors as well for if they are ever required in the future
        };
        inline PRESENT_RESULT present(SPresentInfo info)
        {
            if (!info.queue || info.imgIndex>=m_imageCount)
                return PRESENT_RESULT::_ERROR;

            auto* threadsafeQ = dynamic_cast<CThreadSafeQueueAdapter*>(info.queue);
            if (threadsafeQ)
            {
                threadsafeQ->m.lock();
                info.queue = threadsafeQ->getUnderlyingQueue();
            }
            const auto retval = present_impl(info);
            if (threadsafeQ)
                threadsafeQ->m.unlock();
            return retval;
        }

        //
        virtual core::smart_refctd_ptr<IGPUImage> createImage(const uint32_t imageIndex) = 0;
        // for cleaning up external images from `createImage`
        struct CCleanupSwapchainReference : public ICleanup
        {
            core::smart_refctd_ptr<ISwapchain> m_swapchain;
            uint32_t m_imageIndex;

            ~CCleanupSwapchainReference()
            {
                m_swapchain->freeImageExists(m_imageIndex);
            }
        };

        // Vulkan: const VkSwapchainKHR*
        virtual const void* getNativeHandle() const = 0;

    protected: // TODO: move all definitions to a .cpp
        inline ISwapchain(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params, const uint8_t imageCount)
            : IBackendObject(std::move(dev)), m_params(std::move(params)), m_imageCount(imageCount), m_imgCreationParams{}
        {
            assert(imageCount <= ISwapchain::MaxImages);

            m_imgCreationParams.type = IGPUImage::ET_2D;
            m_imgCreationParams.samples = IGPUImage::ESCF_1_BIT;
            m_imgCreationParams.format = m_params.surfaceFormat.format;
            m_imgCreationParams.extent = { m_params.width, m_params.height, 1u };
            m_imgCreationParams.mipLevels = 1u;
            m_imgCreationParams.arrayLayers = m_params.arrayLayers;
            m_imgCreationParams.flags = m_params.viewFormats.count()>1u ? IGPUImage::ECF_MUTABLE_FORMAT_BIT:IGPUImage::ECF_NONE;
            m_imgCreationParams.usage = m_params.imageUsage;
            if (!(getOriginDevice()->getPhysicalDevice()->getImageFormatUsagesOptimalTiling()[m_imgCreationParams.format]<m_imgCreationParams.usage))
                m_imgCreationParams.flags |= IGPUImage::ECF_EXTENDED_USAGE_BIT;
            m_imgCreationParams.viewFormats = m_params.viewFormats;

            // don't need to keep a reference to the old swapchain anymore
            m_params.oldSwapchain = nullptr;
        }
        virtual inline ~ISwapchain()
        {
            assert(m_imageExists.load()==0u);
        }

        virtual ACQUIRE_IMAGE_RESULT acquireNextImage_impl(const SAcquireInfo& info, uint32_t* const out_imgIx) = 0;
        virtual PRESENT_RESULT present_impl(const SPresentInfo& info) = 0;

        // Returns false if the image already existed
        bool setImageExists(uint32_t ix) { return (m_imageExists.fetch_or(1U << ix) & (1U << ix)) == 0; }

        SCreationParams m_params;
        uint8_t m_imageCount;
        std::atomic_uint32_t m_imageExists = 0;
        IGPUImage::SCreationParams m_imgCreationParams;

    private:
        friend class CCleanupSwapchainReference;
        //
        void freeImageExists(uint32_t ix) { m_imageExists.fetch_and(~(1U << ix)); }

    public:
        static inline constexpr uint32_t MaxImages = sizeof(m_imageExists)*8u;

};

}

#endif