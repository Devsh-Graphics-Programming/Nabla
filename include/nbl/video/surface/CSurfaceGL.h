#ifndef __NBL_I_SURFACE_GL_H_INCLUDED__
#define __NBL_I_SURFACE_GL_H_INCLUDED__

#include "nbl/ui/IWindowWin32.h"
#include "nbl/ui/IWindowAndroid.h"
#include "nbl/ui/IWindowX11.h"
#include "nbl/ui/IWindowWayland.h"

#include "nbl/video/surface/ISurface.h"
#include "nbl/video/COpenGL_Connection.h"

namespace nbl::video
{

template<class Window, template<typename,typename> typename Base, class CRTP = void>
class CSurfaceGLImpl : public Base<Window,ISurface>
{
    public:
        using this_t = std::conditional_t<std::is_void_v<CRTP>,CSurfaceGLImpl<Window,Base>,CRTP>;
        using base_t = Base<Window,ISurface>;

        template<video::E_API_TYPE API_TYPE>
        static inline core::smart_refctd_ptr<this_t> create(core::smart_refctd_ptr<video::COpenGL_Connection<API_TYPE>>&& api, core::smart_refctd_ptr<Window>&& window)
        {
            if (!api || !window)
                return nullptr;
            auto retval = new this_t(std::move(window), std::move(api));
            return core::smart_refctd_ptr<this_t>(retval,core::dont_grab);
        }
        template<video::E_API_TYPE API_TYPE>
        static inline core::smart_refctd_ptr<this_t> create(core::smart_refctd_ptr<video::COpenGL_Connection<API_TYPE>>&& api, typename Window::native_handle_t ntv_window)
        {
            if (!api || !ntv_window)
                return nullptr;
            auto retval = new this_t(std::move(api), ntv_window);
            return core::smart_refctd_ptr<this_t>(retval, core::dont_grab);
        }

        inline bool isSupportedForPhysicalDevice(const IPhysicalDevice* dev, uint32_t _queueFamIx) const override
        {
            const E_API_TYPE pdev_api = dev->getAPIType();
            // GL/GLES backends have just 1 queue family and device
            assert(dev->getQueueFamilyProperties().size()==1u);
            assert(base_t::m_api->getPhysicalDevices().size()==1u);
            return _queueFamIx==0u && dev==*base_t::m_api->getPhysicalDevices().begin();
        }
        
        inline void getAvailableFormatsForPhysicalDevice(const IPhysicalDevice* physicalDevice, uint32_t& formatCount, ISurface::SFormat* formats) const override
        {
            // Todo(achal): Need to properly map asset::E_FORMAT which would also dictate
            // formatCount
            formatCount = 1u;

            if (!formats)
                return;

            formats[0].format = asset::EF_R8G8B8A8_SRGB;

            // formats[0].colorSpace.eotf = ;
            // formats[0].colorSpace.primary = ;
        }

        inline ISurface::E_PRESENT_MODE getAvailablePresentModesForPhysicalDevice(const IPhysicalDevice* physicalDevice) const override
        {
            return static_cast<ISurface::E_PRESENT_MODE>(ISurface::EPM_IMMEDIATE | ISurface::EPM_FIFO | ISurface::EPM_FIFO_RELAXED);
        }

        inline bool getSurfaceCapabilitiesForPhysicalDevice(const IPhysicalDevice* physicalDevice, ISurface::SCapabilities& capabilities) const override
        {
            capabilities.minImageCount = 2u;
            capabilities.maxImageCount = ~0u;
            capabilities.currentExtent = { this->getWidth(), this->getHeight() };
            capabilities.minImageExtent = { 1u, 1u };
            capabilities.maxImageExtent = { static_cast<uint32_t>(physicalDevice->getLimits().maxImageDimension2D), static_cast<uint32_t>(physicalDevice->getLimits().maxImageDimension2D) };
            capabilities.maxImageArrayLayers = physicalDevice->getLimits().maxImageArrayLayers;
            // Only supported transform exposed for OpenGL is mirror rotate 180 (effectively flip Y) to match Vulkan
            capabilities.supportedTransforms = ISurface::EST_HORIZONTAL_MIRROR_ROTATE_180_BIT;
            capabilities.currentTransform = ISurface::EST_HORIZONTAL_MIRROR_ROTATE_180_BIT;
            capabilities.supportedCompositeAlpha = static_cast<ISurface::E_COMPOSITE_ALPHA>(ISurface::ECA_OPAQUE_BIT | ISurface::ECA_PRE_MULTIPLIED_BIT | ISurface::ECA_POST_MULTIPLIED_BIT);
            capabilities.supportedUsageFlags = static_cast<asset::IImage::E_USAGE_FLAGS>(
                asset::IImage::EUF_TRANSFER_SRC_BIT | asset::IImage::EUF_TRANSFER_DST_BIT |
                asset::IImage::EUF_SAMPLED_BIT | asset::IImage::EUF_STORAGE_BIT |
                asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_DEPTH_STENCIL_ATTACHMENT_BIT);

            return true;
        }

    protected:
        using base_t::base_t;
};

template <typename Window>
using CSurfaceGL = CSurfaceGLImpl<Window, CSurface>;
template <typename Window, typename CRTP>
using CSurfaceNativeGL = CSurfaceGLImpl<Window, CSurfaceNative, CRTP>;

// TODO: conditional defines
#ifdef _NBL_PLATFORM_WINDOWS_
using CSurfaceGLWin32 = CSurfaceGL<ui::IWindowWin32>;
class CSurfaceNativeGLWin32 : public CSurfaceNativeGL<ui::IWindowWin32,CSurfaceNativeGLWin32>
{
    protected:
        using base_t = CSurfaceNativeGL<ui::IWindowWin32,CSurfaceNativeGLWin32>;
        using base_t::base_t;

        inline uint32_t getWidth() const override 
        { 
            RECT wr;
            GetWindowRect(m_handle, &wr);
            return wr.right - wr.left;
        }
        inline uint32_t getHeight() const override 
        { 
            RECT wr;
            GetWindowRect(m_handle, &wr);
            return wr.top - wr.bottom;
        }
};
#elif defined(_NBL_PLATFORM_LINUX_)
using CSurfaceGLX11 = CSurfaceGL<ui::IWindowX11>;
#elif defined(_NBL_PLATFORM_ANDROID_)
using CSurfaceGLAndroid = CSurfaceGL<ui::IWindowAndroid>;
#endif

}

#endif