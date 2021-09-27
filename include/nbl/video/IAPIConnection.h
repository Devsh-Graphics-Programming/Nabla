#ifndef __NBL_I_API_CONNECTION_H_INCLUDED__
#define __NBL_I_API_CONNECTION_H_INCLUDED__

#include "nbl/core/declarations.h"

#include "nbl/video/EApiType.h"
#include "nbl/video/debug/IDebugCallback.h"
#include "nbl/video/utilities/renderdoc.h"

namespace nbl::video
{

class IPhysicalDevice;

class IAPIConnection : public core::IReferenceCounted
{
public:
    enum E_FEATURE
    {
        EF_SURFACE = 0,
        EF_GET_DISPLAY_PROPERTIES2,
        EF_COUNT
    };

    virtual E_API_TYPE getAPIType() const = 0;

    virtual IDebugCallback* getDebugCallback() const = 0;

    core::SRange<IPhysicalDevice* const> getPhysicalDevices() const;

    static core::SRange<E_FEATURE> getDependentFeatures(const E_FEATURE feature);

protected:
    inline IAPIConnection() : m_physicalDevices(), m_rdoc_api(nullptr)
    {
#ifdef _NBL_PLATFORM_WINDOWS_
        if (HMODULE mod = GetModuleHandleA("renderdoc.dll"))
#elif defined(_NBL_PLATFORM_ANDROID_)
        if (void* mod = dlopen("libVkLayer_GLES_RenderDoc.so", RTLD_NOW | RTLD_NOLOAD))
#elif defined(_NBL_PLATFORM_LINUX_)
        if (void* mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD))
#else
        if (false)
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

    std::vector<std::unique_ptr<IPhysicalDevice>> m_physicalDevices;
    renderdoc_api_t* m_rdoc_api;
};

}


#endif