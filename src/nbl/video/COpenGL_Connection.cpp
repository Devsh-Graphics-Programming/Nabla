#include "nbl/video/COpenGL_Connection.h"

#include "nbl/video/COpenGLPhysicalDevice.h"
#include "nbl/video/COpenGLESPhysicalDevice.h"


namespace nbl::video
{

template<E_API_TYPE API_TYPE>
core::smart_refctd_ptr<COpenGL_Connection<API_TYPE>> COpenGL_Connection<API_TYPE>::create(
    core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t appVer, const char* appName, COpenGLDebugCallback&& dbgCb
)
{
    auto cwdBackup = std::filesystem::current_path();
    std::string eglLibraryName;
    if (std::getenv("NBL_EGL_PATH"))
    {
        system::path eglPath = std::getenv("NBL_EGL_PATH");
        std::filesystem::current_path(eglPath.parent_path());
        eglLibraryName = eglPath.filename().string();
    }
    egl::CEGL egl(eglLibraryName.c_str());
    if (!egl.initialize())
        return nullptr;
    std::filesystem::current_path(cwdBackup);
    
    SFeatures enabledFeatures = {};
    enabledFeatures.swapchainMode = E_SWAPCHAIN_MODE::ESM_SURFACE;
    auto retval = new COpenGL_Connection<API_TYPE>(enabledFeatures, core::make_smart_refctd_ptr<asset::IGLSLCompiler>(sys.get()));
    std::unique_ptr<IPhysicalDevice> physicalDevice;
    if constexpr (API_TYPE==EAT_OPENGL)
        physicalDevice.reset(COpenGLPhysicalDevice::create(retval,retval->m_rdoc_api,std::move(sys),std::move(egl),std::move(dbgCb)));
    else
        physicalDevice.reset(COpenGLESPhysicalDevice::create(retval,retval->m_rdoc_api,std::move(sys),std::move(egl),std::move(dbgCb)));

    if (!physicalDevice)
    {
        retval->drop(); // maual drop needed, haven't made a smart pointer yet
        return nullptr;
    }
    retval->m_physicalDevices.push_back(std::move(physicalDevice));
    return core::smart_refctd_ptr<COpenGL_Connection<API_TYPE>>(retval,core::dont_grab);
}

template<E_API_TYPE API_TYPE>
IDebugCallback* COpenGL_Connection<API_TYPE>::getDebugCallback() const
{
    if constexpr (API_TYPE == EAT_OPENGL)
        return static_cast<IOpenGL_PhysicalDeviceBase<COpenGLLogicalDevice>*>(*getPhysicalDevices().begin())->getDebugCallback();
    else
        return static_cast<IOpenGL_PhysicalDeviceBase<COpenGLESLogicalDevice>*>(*getPhysicalDevices().begin())->getDebugCallback();
}

template<E_API_TYPE API_TYPE>
const egl::CEGL& COpenGL_Connection<API_TYPE>::getInternalObject() const
{
    return static_cast<const IOpenGLPhysicalDeviceBase*>(m_physicalDevices.front().get())->getInternalObject();
}


template class COpenGL_Connection<EAT_OPENGL>;
template class COpenGL_Connection<EAT_OPENGL_ES>;

}