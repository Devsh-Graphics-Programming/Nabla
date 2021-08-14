#include "nbl/video/COpenGL_Connection.h"

#include "nbl/video/CEGL.h"
#include "nbl/video/COpenGLPhysicalDevice.h"
#include "nbl/video/COpenGLESPhysicalDevice.h"


namespace nbl::video
{

template<E_API_TYPE API_TYPE>
core::smart_refctd_ptr<COpenGL_Connection<API_TYPE>> COpenGL_Connection<API_TYPE>::create(
    core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t appVer, const char* appName, COpenGLDebugCallback&& dbgCb
)
{
    egl::CEGL egl;
    if (!egl.initialize())
        return nullptr;

    core::smart_refctd_ptr<IPhysicalDevice> pdevice;
    if constexpr (API_TYPE==EAT_OPENGL)
        pdevice = COpenGLPhysicalDevice::create(std::move(sys),std::move(egl),std::move(dbgCb));
    else
        pdevice = COpenGLESPhysicalDevice::create(std::move(sys),std::move(egl),std::move(dbgCb));

    if (!pdevice)
        return nullptr;

    auto retval = new COpenGL_Connection<API_TYPE>(std::move(pdevice));
    return core::smart_refctd_ptr<COpenGL_Connection<API_TYPE>>(retval,core::dont_grab);
}

template<E_API_TYPE API_TYPE>
IDebugCallback* COpenGL_Connection<API_TYPE>::getDebugCallback() const
{
    if constexpr (API_TYPE == EAT_OPENGL)
        return static_cast<IOpenGL_PhysicalDeviceBase<COpenGLLogicalDevice>*>(m_pdevice.get())->getDebugCallback();
    else
        return static_cast<IOpenGL_PhysicalDeviceBase<COpenGLESLogicalDevice>*>(m_pdevice.get())->getDebugCallback();
}


template class COpenGL_Connection<EAT_OPENGL>;
template class COpenGL_Connection<EAT_OPENGL_ES>;

}