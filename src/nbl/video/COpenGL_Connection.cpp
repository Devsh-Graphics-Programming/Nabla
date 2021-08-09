#include "nbl/video/COpenGL_Connection.h"
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
#endif // TODO more platforms
#include "nbl/video/COpenGLPhysicalDevice.h"
#include "nbl/video/COpenGLESPhysicalDevice.h"


namespace nbl::video
{

template<E_API_TYPE API_TYPE>
static core::smart_refctd_ptr<COpenGL_Connection<API_TYPE>> COpenGL_Connection<API_TYPE>::create(core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t appVer, const char* appName, COpenGLDebugCallback&& dbgCb)
{
    egl::CEGL egl;
    if (!egl.initialize())
        return nullptr;

    core::smart_refctd_ptr<IPhysicalDevice> pdevice;
    if constexpr (API_TYPE==EAT_OPENGL)
        pdevice = COpenGLPhysicalDevice::create(std::move(m_system),std::move(m_GLSLCompiler),std::move(egl),std::move(dbgCb));
    else
        pdevice = COpenGLESPhysicalDevice::create(std::move(m_system),std::move(m_GLSLCompiler),std::move(egl),std::move(dbgCb));

    if (!pdevice)
        return nullptr;

    auto retval = core::make_smart_refctd_ptr<COpenGL_Connection<API_TYPE>>(std::move(pdevice));
    return retval;
}

template<E_API_TYPE API_TYPE>
core::smart_refctd_ptr<ISurface> COpenGL_Connection<API_TYPE>::createSurface(ui::IWindow* window)
{
    // TODO surface creation needs to be reorganized
    // ALSO CREATE AND VALIDATE BEFORE CALLING CTORS!
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
#else // TODO more platforms
    return nullptr;
#endif
}

template<E_API_TYPE API_TYPE>
IDebugCallback* COpenGL_Connection<API_TYPE>::getDebugCallback()
{
    if constexpr (API_TYPE == EAT_OPENGL)
        return static_cast<const IOpenGL_PhysicalDeviceBase<COpenGLLogicalDevice>*>(m_pdevice)->getDebugCallback();
    else
        return static_cast<const IOpenGL_PhysicalDeviceBase<COpenGLESLogicalDevice>*>(m_pdevice)->getDebugCallback();
}


}