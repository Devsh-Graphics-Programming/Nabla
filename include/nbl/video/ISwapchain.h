#ifndef _NBL_VIDEO_I_SWAPCHAIN_H_INCLUDED_
#define _NBL_VIDEO_I_SWAPCHAIN_H_INCLUDED_


#include "nbl/core/util/bitflag.h"

#include "nbl/video/surface/ISurface.h"
#include "nbl/video/ILogicalDevice.h"


namespace nbl::video
{
class IQueue;

class ISwapchain : public IBackendObject
{
        std::atomic_uint32_t m_imageExists = 0;

    public:
        static inline constexpr uint32_t MaxImages = sizeof(m_imageExists)*8u;

        struct SSharedCreationParams
        {
            inline bool valid(const IPhysicalDevice* physDev, const ISurface* surface) const
            {
                ISurface::SCapabilities caps;
                if (!physDev || !surface || !surface->getSurfaceCapabilitiesForPhysicalDevice(physDev,caps))
                    return false;

                if (!caps.supportedUsageFlags.hasFlags(imageUsage))
                    return false;
                if (minImageCount<caps.minImageCount)
                    return false;
                if (hlsl::bitCount(presentMode.value)!=1 || !surface->getAvailablePresentModesForPhysicalDevice(physDev).hasFlags(presentMode))
                    return false;
                if (width<caps.minImageExtent.width || width>caps.maxImageExtent.width)
                    return false;
                if (height<caps.minImageExtent.height || height>caps.maxImageExtent.height)
                    return false;
                if (hlsl::bitCount(compositeAlpha.value)!=1 || !caps.supportedCompositeAlpha.hasFlags(compositeAlpha))
                    return false;
                if (arrayLayers==0 || arrayLayers>caps.maxImageArrayLayers)
                    return false;
                if (hlsl::bitCount(preTransform.value)!=1 || !caps.supportedTransforms.hasFlags(preTransform))
                    return false;
                return true;
            }

            constexpr static inline ISurface::E_PRESENT_MODE DefaultPreferredPresentModes[] = {
                ISurface::EPM_FIFO_RELAXED,
                ISurface::EPM_FIFO,
                ISurface::EPM_MAILBOX,
                ISurface::EPM_IMMEDIATE
            };
            constexpr static inline ISurface::E_COMPOSITE_ALPHA DefaultPreferredCompositeAlphas[] = {
                ISurface::ECA_OPAQUE_BIT,
                ISurface::ECA_PRE_MULTIPLIED_BIT,
                ISurface::ECA_POST_MULTIPLIED_BIT,
                ISurface::ECA_INHERIT_BIT
            };
            constexpr static inline ISurface::E_SURFACE_TRANSFORM_FLAGS DefaultPreferredTransforms[] = {
                ISurface::EST_IDENTITY_BIT
                // nothing else will work out the box without us explicitly having to handle it
            };
            inline bool deduce(
                const IPhysicalDevice* physDev, const ISurface* surface,
                std::span<const ISurface::E_PRESENT_MODE> preferredPresentModes=DefaultPreferredPresentModes,
                std::span<const ISurface::E_COMPOSITE_ALPHA> preferredCompositeAlphas=DefaultPreferredCompositeAlphas,
                std::span<const ISurface::E_SURFACE_TRANSFORM_FLAGS> preferredTransforms=DefaultPreferredTransforms
            )
            {
                ISurface::SCapabilities caps;
                if (!physDev || !surface || !surface->getSurfaceCapabilitiesForPhysicalDevice(physDev,caps))
                    return false;

                if (!imageUsage)
                    imageUsage = caps.supportedUsageFlags;
                else if (!caps.supportedUsageFlags.hasFlags(imageUsage))
                    return false;

                caps.maxImageCount = core::min<uint32_t>(caps.maxImageCount,MaxImages);
                minImageCount = core::clamp(minImageCount,caps.minImageCount,caps.maxImageCount);

                auto cantPickPreferred = []<typename T>(core::bitflag<T>& options, std::span<const T> mostToLeastPreferred) -> bool
                {
                    switch (hlsl::bitCount(options.value))
                    {
                        case 0:
                            return true;
                        case 1:
                            break;
                        default:
                            for (const auto nextMostPreferred : mostToLeastPreferred)
                            {
                                assert(hlsl::bitCount(nextMostPreferred)==1);
                                if (options.hasFlags(nextMostPreferred))
                                {
                                    options = nextMostPreferred;
                                    break;
                                }
                            }
                            break;
                    }
                    return false;
                };

                presentMode &= surface->getAvailablePresentModesForPhysicalDevice(physDev);
                if (cantPickPreferred(presentMode,preferredPresentModes))
                    return false;
                // in case of no preferred list
                presentMode = static_cast<ISurface::E_PRESENT_MODE>(0x1u<<hlsl::findMSB(presentMode));
                
                if (!width)
                    width = caps.currentExtent.width;
                else
                    width = core::clamp(width,caps.minImageExtent.width,caps.maxImageExtent.width);

                if (!height)
                    height = caps.currentExtent.height;
                else
                    height = core::clamp(height,caps.minImageExtent.height,caps.maxImageExtent.height);

                compositeAlpha &= caps.supportedCompositeAlpha;
                if (cantPickPreferred(compositeAlpha,preferredCompositeAlphas))
                    return false;
                // in case of no preferred list
                compositeAlpha = static_cast<ISurface::E_COMPOSITE_ALPHA>(0x1u<<hlsl::findLSB(compositeAlpha));

                arrayLayers = core::min(arrayLayers,caps.maxImageArrayLayers);

                preTransform &= caps.supportedTransforms;
                if (cantPickPreferred(preTransform,preferredTransforms))
                    return false;
                assert(caps.supportedTransforms.hasFlags(caps.currentTransform));
                // in case of no preferred list
                if (preTransform.hasFlags(caps.currentTransform))
                    preTransform = caps.currentTransform;

                assert(valid(physDev,surface));
                return true;
            }

            // default means "all supported"
            core::bitflag<IGPUImage::E_USAGE_FLAGS> imageUsage = IGPUImage::E_USAGE_FLAGS::EUF_NONE;
            // can also treat as them as bitflags of valid transforms
            uint8_t minImageCount = 0u;
            core::bitflag<ISurface::E_PRESENT_MODE> presentMode = ISurface::EPM_ALL_BITS;
            uint16_t width = 0u;
            uint16_t height = 0u;
            core::bitflag<ISurface::E_COMPOSITE_ALPHA> compositeAlpha = ISurface::ECA_ALL_BITS;
            uint8_t arrayLayers = 1u;
            core::bitflag<ISurface::E_SURFACE_TRANSFORM_FLAGS> preTransform = ISurface::EST_ALL_BITS;
        };
        struct SCreationParams
        {
            inline bool isConcurrentSharing() const {return !queueFamilyIndices.empty();}

            inline bool valid(const IPhysicalDevice* physDev) const
            {
                if (!sharedParams.valid(physDev,surface.get()))
                    return false;

                core::vector<ISurface::SFormat> availableFormats;
                {
                    uint32_t availableFormatCount = 0;
                    surface->getAvailableFormatsForPhysicalDevice(physDev,availableFormatCount,nullptr);
                    availableFormats.resize(availableFormatCount);
                    surface->getAvailableFormatsForPhysicalDevice(physDev,availableFormatCount,availableFormats.data());
                }
                return std::find(availableFormats.begin(),availableFormats.end(),surfaceFormat)!=availableFormats.end();
            }

            constexpr static inline asset::E_FORMAT DefaultPreferredFormats[] = {
                asset::EF_R8G8B8A8_SRGB,
                asset::EF_B8G8R8A8_SRGB,
                asset::EF_R16G16B16A16_SFLOAT,
                asset::EF_A2R10G10B10_UNORM_PACK32,
                asset::EF_A2B10G10R10_UNORM_PACK32,
                asset::EF_R16G16B16A16_UNORM,
                asset::EF_R8G8B8_SRGB,
                asset::EF_B8G8R8_SRGB,
                asset::EF_E5B9G9R9_UFLOAT_PACK32,
                asset::EF_R16G16B16_SFLOAT,
                asset::EF_R16G16B16_UNORM
            };
            constexpr static inline asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION DefaultEOTFs[] = {
                asset::EOTF_sRGB
                // nothing else will work out the box without adjusting everything while rendering/composting
            };
            constexpr static inline asset::E_COLOR_PRIMARIES DefaultColorPrimaries[] = {
                asset::ECP_SRGB
                // nothing else will work out the box without adjusting everything while rendering/composting
            };
            // The preferences get ignored if you set the fields to a known value.
            inline bool deduceFormat(const IPhysicalDevice* physDev,
                std::span<const asset::E_FORMAT> preferredFormats=DefaultPreferredFormats,
                std::span<const asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION> preferredEOTFs=DefaultEOTFs,
                std::span<const asset::E_COLOR_PRIMARIES> preferredColorPrimaries=DefaultColorPrimaries
            )
            {
                core::vector<ISurface::SFormat> availableFormats;
                {
                    uint32_t availableFormatCount = 0;
                    surface->getAvailableFormatsForPhysicalDevice(physDev,availableFormatCount,nullptr);
                    availableFormats.resize(availableFormatCount);
                    surface->getAvailableFormatsForPhysicalDevice(physDev,availableFormatCount,availableFormats.data());
                }

                // override preferred if set to known value already
                if (surfaceFormat.format!=asset::EF_UNKNOWN)
                    preferredFormats = {&surfaceFormat.format,1};
                if (surfaceFormat.colorSpace.eotf!=asset::EOTF_UNKNOWN)
                    preferredEOTFs = {&surfaceFormat.colorSpace.eotf,1};
                if (surfaceFormat.colorSpace.primary!=asset::ECP_COUNT)
                    preferredColorPrimaries = {&surfaceFormat.colorSpace.primary,1};

                // the color space if the most important thing, we do everything else according to that
                auto fullComparator = [](const ISurface::SFormat& lhs, const ISurface::SFormat& rhs)->bool
                {
                    if (lhs.colorSpace.primary==rhs.colorSpace.primary)
                    {
                        if (lhs.colorSpace.eotf==rhs.colorSpace.eotf)
                            return lhs.format<rhs.format;
                        return lhs.colorSpace.eotf<rhs.colorSpace.eotf;
                    }
                    return lhs.colorSpace.primary<rhs.colorSpace.primary;
                };
                std::sort(availableFormats.begin(),availableFormats.end(),fullComparator);
        
                auto primaryComparator = [](const ISurface::SFormat& lhs, const ISurface::SFormat& rhs)->bool
                {
                    return lhs.colorSpace.primary<rhs.colorSpace.primary;
                };
                auto eotfComparator = [](const ISurface::SFormat& lhs, const ISurface::SFormat& rhs)->bool
                {
                    return lhs.colorSpace.eotf<rhs.colorSpace.eotf;
                };
                ISurface::SFormat val = {};
                for (const auto primary : preferredColorPrimaries)
                {
                    val.colorSpace.primary = primary;
                    const auto eotfBegin = std::lower_bound(availableFormats.begin(),availableFormats.end(),val,primaryComparator);
                    const auto eotfEnd = std::upper_bound(eotfBegin,availableFormats.end(),val,primaryComparator);
                    if (eotfBegin<eotfEnd)
                    for (const auto eotf : preferredEOTFs)
                    {
                        val.colorSpace.eotf = eotf;
                        const auto formatBegin = std::lower_bound(eotfBegin,eotfEnd,val,eotfComparator);
                        const auto formatEnd = std::upper_bound(formatBegin,eotfEnd,val,eotfComparator);
                        for (const auto format : preferredFormats)
                        {
                            val.format = format;
                            if (std::binary_search(formatBegin,formatEnd,val,fullComparator))
                            {
                                surfaceFormat.format = format;
                                surfaceFormat.colorSpace = {primary,eotf};
                                return true;
                            }
                        }
                    }
                }
                return false;
            }

            inline core::bitflag<IGPUImage::E_CREATE_FLAGS> computeImageCreationFlags(const IPhysicalDevice* physDev) const
            {
                core::bitflag<IGPUImage::E_CREATE_FLAGS> retval = IGPUImage::ECF_NONE;
                if (viewFormats.count()>1)
                    retval |= IGPUImage::ECF_MUTABLE_FORMAT_BIT;
                if (!(physDev->getImageFormatUsagesOptimalTiling()[surfaceFormat.format]<sharedParams.imageUsage))
                    retval |= IGPUImage::ECF_EXTENDED_USAGE_BIT;
                return retval;
            }

            core::smart_refctd_ptr<ISurface> surface = {};
            // these we can deduce
            ISurface::SFormat surfaceFormat = {};
            SSharedCreationParams sharedParams = {};
            // If you set it to something else then your Swapchain will be created with Mutable Format capability
            // NOTE: If you do that, then the bitset needs to contain `viewFormats[surfaceFormat.format] = true`
            std::bitset<asset::E_FORMAT::EF_COUNT> viewFormats = {};
            std::span<const uint8_t> queueFamilyIndices = {};
        };

        inline const auto& getCreationParameters() const {return m_params;}

        inline uint8_t getImageCount() const { return m_imageCount; }

        // The value passed to `preTransform` when creating the swapchain. "pre" refers to the transform happening
        // as an operation before the presentation engine presents the image.
        inline ISurface::E_SURFACE_TRANSFORM_FLAGS getPreTransform() const { return m_params.sharedParams.preTransform.value; }

        // acquire
        struct SAcquireInfo
        {
            IQueue* queue;
            // If you don't change the default it will 100% block and acquire will not return TIMEOUT or NOT_READY
            uint64_t timeout = std::numeric_limits<uint64_t>::max();
            std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> signalSemaphores = {};
        };
        enum class ACQUIRE_IMAGE_RESULT : uint8_t
        {
            SUCCESS,
            TIMEOUT,
            NOT_READY,
            SUBOPTIMAL,
            _ERROR // GDI macros getting in the way of just ERROR
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

            switch (retval)
            {
                case ACQUIRE_IMAGE_RESULT::SUCCESS: [[fallthrough]];
                case ACQUIRE_IMAGE_RESULT::SUBOPTIMAL:
                    // TODO: pop deferred event slot
                    if (m_oldSwapchain)
                    {
                        bool canDrop = false;//count>m_imageCount && count>m_oldSwapchain->getImageCount();
                        for (;false;)
                        if (false) // TODO: just poll the event slots
                        {
                            canDrop = false;
                        }
                        if (canDrop)
                            m_oldSwapchain = nullptr;
                    }
                    break;
                default:
                    break;
            }
            return retval;
        }

        // present
        struct SPresentInfo
        {
            IQueue* queue;
            uint32_t imgIndex;
            std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> waitSemaphores = {};
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

            // If not error (so you can re-push)
            if (retval!=PRESENT_RESULT::_ERROR)
            {
                // TODO: push onto deferred event slot even 
                // latch the frees on the wait semaphores
            }
            return retval;
        }

        //
        virtual core::smart_refctd_ptr<IGPUImage> createImage(const uint32_t imageIndex) = 0;
        // for cleaning up external images from `createImage`
        struct CCleanupSwapchainReference : public ICleanup
        {
            CCleanupSwapchainReference(core::smart_refctd_ptr<ISwapchain>&& _swapchain, const uint8_t _imageIndex) :
                m_swapchain(std::move(_swapchain)), m_imageIndex(_imageIndex) {}
            inline virtual ~CCleanupSwapchainReference()
            {
                m_swapchain->freeImageExists(m_imageIndex);
            }

            const core::smart_refctd_ptr<ISwapchain> m_swapchain;
            const uint8_t m_imageIndex;
        };

        //
        inline core::smart_refctd_ptr<ISwapchain> recreate(SSharedCreationParams&& params={}) const
        {
            if (!params.deduce(getOriginDevice()->getPhysicalDevice(),m_params.surface.get(),{&m_params.sharedParams.presentMode.value,1},{&m_params.sharedParams.compositeAlpha.value,1},{&m_params.sharedParams.preTransform.value,1}))
                return nullptr;
            return recreate_impl(std::move(params));
        }

        // Vulkan: const VkSwapchainKHR*
        virtual const void* getNativeHandle() const = 0;

    protected:
        ISwapchain(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params, const uint8_t imageCount, core::smart_refctd_ptr<const ISwapchain>&& oldSwapchain);
        virtual inline ~ISwapchain()
        {
            assert(m_imageExists.load()==0u);

            // TODO: block until all deferred event slots are finished
        }

        inline const auto& getImageCreationParams() const {return m_imgCreationParams;}

        // Returns false if the image already existed
        inline bool setImageExists(const uint8_t ix)
        {
            const uint32_t ixMask = 0x1u<<ix;
            return (m_imageExists.fetch_or(ixMask)&ixMask)==0;
        }

        virtual ACQUIRE_IMAGE_RESULT acquireNextImage_impl(const SAcquireInfo& info, uint32_t* const out_imgIx) = 0;
        virtual PRESENT_RESULT present_impl(const SPresentInfo& info) = 0;

        virtual core::smart_refctd_ptr<ISwapchain> recreate_impl(SSharedCreationParams&& params) const = 0;

    private:
        friend class CCleanupSwapchainReference;
        //
        inline void freeImageExists(const uint8_t ix)
        {
            m_imageExists.fetch_and(~(0x1u<<ix));
        }

        SCreationParams m_params;
        const asset::IImage::SCreationParams m_imgCreationParams;
        // The user needs to hold onto the old swapchain by themselves as well, because without the `KHR_swapchain_maintenance1` extension:
        // - Images cannot be "unacquired", so all already acquired images of a swapchain (even if its retired) must be presented
        // - No way to query that the presentation is finished, so we can't destroy a swapchain just because we issued a present on all previously acquired images
        // This means new swapchain should hold onto the old swapchain for `getImageCount()` acquires or until an acquisition error
        core::smart_refctd_ptr<const ISwapchain> m_oldSwapchain;
        std::array<uint8_t,ILogicalDevice::SCreationParams::MaxQueueFamilies> m_queueFamilies;
        const uint8_t m_imageCount;
};

}

#endif