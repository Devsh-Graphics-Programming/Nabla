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

        bool isSupported(const IPhysicalDevice* dev, uint32_t _queueFamIx) const override
        {
            const E_API_TYPE pdev_api = dev->getAPIType();
            // GL/GLES backends have just 1 queue family and device
            assert(dev->getQueueFamilyProperties().size()==1u);
            assert(base_t::m_api->getPhysicalDevices().size()==1u);
            return _queueFamIx==0u && dev==base_t::m_api->getPhysicalDevices().begin()->get();
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


#ifdef _NBL_PLATFORM_WINDOWS_
    using CSurfaceGLWin32 = CSurfaceGL<ui::IWindowWin32>;
#endif

#ifdef _NBL_PLATFORM_ANDROID_
    using CSurfaceGLAndroid = CSurfaceGL<ui::IWindowAndroid>;
#endif

#ifdef _NBL_PLATFORM_LINUX_
    using CSurfaceGLX11 = CSurfaceGL<ui::IWindowX11>;
#endif

#ifdef _NBL_BUILD_WITH_WAYLAND // TODO: Probably in the future - _NBL_BUILD_PLATFORM_WAYLAND_
    using CSurfaceGLWayland = CSurfaceGL<ui::IWindowWayland>;
#endif

}

#endif