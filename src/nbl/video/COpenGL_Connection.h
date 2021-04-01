#ifndef __NBL_C_OPENGL__CONNECTION_H_INCLUDED__
#define __NBL_C_OPENGL__CONNECTION_H_INCLUDED__

#include "nbl/video/IAPIConnection.h"
#include "nbl/video/CEGL.h"
#if defined(_NBL_PLATFORM_WINDOWS_)
#   include "nbl/system/IWindowWin32.h"
#   include "nbl/video/surface/CSurfaceGLWin32.h"
#elif defined(_NBL_PLATFORM_LINUX_)
#   include "nbl/system/IWindowLinux.h"
#   include "nbl/video/surface/CSurfaceGLLinux.h"
#endif // TODO more platforms

namespace nbl {
namespace video
{

template <typename PhysicalDeviceType, E_API_TYPE API_TYPE>
class COpenGL_Connection final : public IAPIConnection
{
public:
    COpenGL_Connection(SDebugCallback* dbgCb)
    {
        // would be nice to initialize this in create() and return nullptr on failure
        // but DynamicFunctionCallerBase is unmovable!! why?? So i cannot move into m_egl
        if (m_egl.initialize())
            m_pdevice = PhysicalDeviceType::create(core::smart_refctd_ptr(m_fs), core::smart_refctd_ptr(m_GLSLCompiler), &m_egl, dbgCb);
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

    core::smart_refctd_ptr<ISurface> createSurface(system::IWindow* window) const override
    {
#if defined(_NBL_PLATFORM_WINDOWS_)
        {
            system::IWindowWin32* w32 = static_cast<system::IWindowWin32*>(window);

            CSurfaceGLWin32::SCreationParams params;
            params.hinstance = GetModuleHandle(NULL);
            params.hwnd = w32->getNativeHandle();

            return core::make_smart_refctd_ptr<CSurfaceGLWin32>(std::move(params));
        }
#elif defined(_NBL_PLATFORM_LINUX_)
        {
            system::IWindowLinux* win = static_cast<system::IWindowLinux*>(window);

            CSurfaceGLLinux::SCreationParams params;
            params.dpy = win->getDisplay();
            params.window = win->getNativeHandle();

            return core::make_smart_refctd_ptr<CSurfaceGLLinux>(std::move(params));
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
}

#endif