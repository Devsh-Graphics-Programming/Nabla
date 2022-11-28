#include "nbl/video/IAPIConnection.h"

#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/utilities/renderdoc.h"

namespace nbl::video
{

core::SRange<IPhysicalDevice* const> IAPIConnection::getPhysicalDevices() const
{
    static_assert(sizeof(std::unique_ptr<IPhysicalDevice>) == sizeof(void*));

    return core::SRange<IPhysicalDevice* const>(
        reinterpret_cast<IPhysicalDevice* const*>(m_physicalDevices.data()),
        reinterpret_cast<IPhysicalDevice* const*>(m_physicalDevices.data()) + m_physicalDevices.size());
}

IAPIConnection::IAPIConnection(const SFeatures& enabledFeatures, core::smart_refctd_ptr<asset::CGLSLCompiler>&& glslc) 
    : m_physicalDevices()
    , m_rdoc_api(nullptr)
    , m_enabledFeatures(enabledFeatures)
    , m_glslCompiler(std::move(glslc))
{
#ifdef _NBL_PLATFORM_WINDOWS_
    if (HMODULE mod = GetModuleHandleA("renderdoc.dll"))
#elif defined(_NBL_PLATFORM_ANDROID_)
    if (void* mod = dlopen("libVkLayer_GLES_RenderDoc.so", RTLD_NOW | RTLD_NOLOAD))
#elif defined(_NBL_PLATFORM_LINUX_)
    if (void* mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD))
#else
#error "Nabla Unsupported Platform!"
#endif
    {
    #if defined(_NBL_PLATFORM_WINDOWS_)
        pRENDERDOC_GetAPI RENDERDOC_GetAPI =
            (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
        int ret = RENDERDOC_GetAPI(MinRenderdocVersion, (void**)&m_rdoc_api);
        assert(ret == 1);
    #elif defined(_NBL_PLATFORM_ANDROID_) || defined(_NBL_PLATFORM_LINUX_)
        pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
        int ret = RENDERDOC_GetAPI(MinRenderdocVersion, (void**)&m_rdoc_api);
        assert(ret == 1);
    #endif
    }
}

}