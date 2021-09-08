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

template<class Window>
class CSurfaceGL final : public CSurface<Window>
{
    public:
        using this_t = CSurfaceGL<Window>;
        using base_t = CSurface<Window>;

        template<video::E_API_TYPE API_TYPE>
        static inline core::smart_refctd_ptr<this_t> create(core::smart_refctd_ptr<video::COpenGL_Connection<API_TYPE>>&& api, core::smart_refctd_ptr<Window>&& window)
        {
            if (!api || !window)
                return nullptr;
            auto retval = new this_t(std::move(api),std::move(window));
            return core::smart_refctd_ptr<this_t>(retval,core::dont_grab);
        }

        bool isSupportedForPhysicalDevice(const IPhysicalDevice* dev, uint32_t _queueFamIx) const override
        {
            const E_API_TYPE pdev_api = dev->getAPIType();
            // GL/GLES backends have just 1 queue family and device
            assert(dev->getQueueFamilyProperties().size()==1u);
            assert(base_t::m_api->getPhysicalDevices().size()==1u);
            return _queueFamIx==0u && dev==*base_t::m_api->getPhysicalDevices().begin();
        }
        
        void getAvailableFormatsForPhysicalDevice(const IPhysicalDevice* physicalDevice, uint32_t& formatCount, ISurface::SFormat* formats) const override
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

        ISurface::E_PRESENT_MODE getAvailablePresentModesForPhysicalDevice(const IPhysicalDevice* physicalDevice) const override
        {
            return static_cast<ISurface::E_PRESENT_MODE>(ISurface::EPM_IMMEDIATE | ISurface::EPM_FIFO | ISurface::EPM_FIFO_RELAXED);
        }

        bool getSurfaceCapabilitiesForPhysicalDevice(const IPhysicalDevice* physicalDevice, ISurface::SCapabilities& capabilities) const override
        {
            // Todo(achal): Fill it properly
            capabilities.minImageCount = 2u;
            capabilities.maxImageCount = 8u;
            capabilities.currentExtent = { base_t::m_window->getWidth(), base_t::m_window->getHeight() };
            capabilities.minImageExtent = { 1u, 1u };
            capabilities.maxImageExtent = { base_t::m_window->getWidth(), base_t::m_window->getHeight() };
            capabilities.maxImageArrayLayers = 1u;
            capabilities.supportedUsageFlags = static_cast<asset::IImage::E_USAGE_FLAGS>(0u);

            return true;
        }

        inline const void* getNativeWindowHandle() const override
        {
            return &base_t::m_window->getNativeHandle();
        }

    protected:
        template<video::E_API_TYPE API_TYPE>
        explicit CSurfaceGL(core::smart_refctd_ptr<video::COpenGL_Connection<API_TYPE>>&& api, core::smart_refctd_ptr<Window>&& window) : base_t(std::move(api),std::move(window))
        {
        }
};


// TODO: conditional defines
using CSurfaceGLWin32 = CSurfaceGL<ui::IWindowWin32>;
//using CSurfaceGLAndroid = CSurfaceGL<ui::IWindowAndroid>;
//using CSurfaceGLX11 = CSurfaceGL<ui::IWindowX11>;
//using CSurfaceGLWayland = CSurfaceGL<ui::IWindowWayland>;

}

#endif