#ifndef __NBL_C_OPENGL__CONNECTION_H_INCLUDED__
#define __NBL_C_OPENGL__CONNECTION_H_INCLUDED__

#include "nbl/video/IAPIConnection.h"
#include "nbl/video/CEGL.h"
#if defined(_NBL_PLATFORM_WINDOWS_)
#   include "nbl/ui/IWindowWin32.h"
#   include "nbl/video/surface/CSurfaceGLWin32.h"
#elif defined(_NBL_BUILD_WITH_WAYLAND) && defined(_NBL_TEST_WAYLAND)
#   include "nbl/ui/IWindowWayland.h"
#   include "nbl/video/surface/CSurfaceGLWayland.h"
#elif defined(_NBL_PLATFORM_LINUX_)
#   include "nbl/ui/IWindowX11.h"
#   include "nbl/video/surface/CSurfaceGLX11.h"
#elif defined(_NBL_PLATFORM_ANDROID_)
#   include "nbl/ui/IWindowAndroid.h"
#   include "nbl/video/surface/CSurfaceGLAndroid.h"
#endif // TODO more platforms

namespace nbl::video
{

template <typename PhysicalDeviceType, E_API_TYPE API_TYPE>
class COpenGL_Connection final : public IAPIConnection
{
public:
    COpenGL_Connection(SDebugCallback& dbgCb) : IAPIConnection(dbgCb)
    {
        // would be nice to initialize this in create() and return nullptr on failure
        // but DynamicFunctionCallerBase is unmovable!! why?? So i cannot move into m_egl
        if (m_egl.initialize())
            m_pdevice = PhysicalDeviceType::create(core::smart_refctd_ptr(m_system), core::smart_refctd_ptr(m_GLSLCompiler), &m_egl, const_cast<SDebugCallback*>(&dbgCb));
    }

    E_API_TYPE getAPIType() const override
    {
        return API_TYPE;
    }

    core::SRange<const core::smart_refctd_ptr<IPhysicalDevice>> getPhysicalDevices() const override
    {          
        if (!m_pdevice)
            return core::SRange<const core::smart_refctd_ptr<IPhysicalDevice>>{ nullptr, nullptr };

        return core::SRange<const core::smart_refctd_ptr<IPhysicalDevice>>{ &m_pdevice, &m_pdevice + 1 };
    }

    core::smart_refctd_ptr<ISurface> createSurface(ui::IWindow* window) const override
    {
        // TODO surface creation needs to be reorganized
        // on linux both X11 and Wayland windows are possible, and theres no way to distinguish between them
        // (so now in case of Wayland installed, createSurface always expects Wayland window)
#if defined(_NBL_PLATFORM_WINDOWS_)
        {
            ui::IWindowWin32* w32 = static_cast<ui::IWindowWin32*>(window);

            CSurfaceGLWin32::SCreationParams params;
            params.hinstance = GetModuleHandle(NULL);
            params.hwnd = w32->getNativeHandle();

            return core::make_smart_refctd_ptr<CSurfaceGLWin32>(std::move(params));
        }
#elif defined(_NBL_BUILD_WITH_WAYLAND) && defined(_NBL_TEST_WAYLAND)
        {
            ui::IWindowWayland* win = static_cast<ui::IWindowWayland*>(window);

            CSurfaceGLWayland::SCreationParams params;
            params.dpy = win->getDisplay();
            params.window = win->getNativeHandle();

            return core::make_smart_refctd_ptr<CSurfaceGLWayland>(std::move(params));
        }
#elif defined(_NBL_PLATFORM_LINUX_)
        {
            ui::IWindowX11* win = static_cast<ui::IWindowX11*>(window);

            CSurfaceGLX11::SCreationParams params;
            params.dpy = win->getDisplay();
            params.window = win->getNativeHandle();

            return core::make_smart_refctd_ptr<CSurfaceGLX11>(std::move(params));
        }
#elif defined(_NBL_PLATFORM_ANDROID_)
        {
            ui::IWindowAndroid* win = static_cast<ui::IWindowAndroid*>(window);

            CSurfaceGLAndroid::SCreationParams params;
            params.anw = win->getNativeHandle();

            return core::make_smart_refctd_ptr<CSurfaceGLAndroid>(std::move(params));
        }
#else // TODO more platforms
        return nullptr;
#endif
    }

protected:
    ~COpenGL_Connection()
    {
        m_egl.deinitialize();
    }

private:
    egl::CEGL m_egl;
    core::smart_refctd_ptr<IPhysicalDevice> m_pdevice;
};

}

#endif