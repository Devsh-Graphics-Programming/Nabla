#include "nbl/video/IAPIConnection.h"

#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/utilities/renderdoc.h"

#include "nbl/system/CSystemWin32.h"

#ifdef NBL_BUILD_WITH_NGFX
#include "NGFX_Injection.h"
#endif

#if defined(_NBL_POSIX_API_)
    #include <dlfcn.h>
#endif

namespace nbl::video
{

void IAPIConnection::loadDebuggers()
{
    // there's a debugger already attached to the process
    if (m_debugger!=EDebuggerType::None)
        return;

    // NGFX takes precedence, but don't want to use NGFX SDK unless NSight launched our application.
    // Otherwise we end up starting random NSight GUI instances if only the NGFX DLL is found (interferes with Renderdoccing) 
    if (std::getenv("NVTX_INJECTION64_PATH"))
    {
        if (loadNGFX())
            m_debugger = EDebuggerType::NSight;
        return;
    }

    // now load renderdoc
    if (m_rdoc_api=loadRenderdoc(); m_rdoc_api)
        m_debugger = EDebuggerType::Renderdoc;
}

bool IAPIConnection::loadNGFX()
{
#ifdef NBL_BUILD_WITH_NGFX
    //! absolute path to official install NGFX SDK runtime directory
    auto getOfficialRuntimeDirectory = []()
    {
        const char* sdk = std::getenv("NGFX_SDK");
        const char* version = std::getenv("NGFX_VERSION");
        const bool composed = sdk && version;

        if (composed)
        {
            const auto directory = system::path(sdk) / system::path(version) / "lib" / "x64";

            if (std::filesystem::exists(directory))
                return directory;
        }

        return system::path("");
    };

    //! batch request with priority order & custom Nabla runtime search, I'm assuming we are loading the runtime from official SDK not custom location
    //! one question is if we should have any constraints for min/max version, maybe force the "version" 
    //! to match the "NGFX_VERSION" define so to "what we built with", or don't have any - just like now

    bool isAlreadyLoaded = false;
#if defined(_NBL_PLATFORM_WINDOWS_)
    static constexpr std::string_view NGFXMODULE = "NGFX_Injection.dll";
    isAlreadyLoaded = GetModuleHandleA(NGFXMODULE.data());
    if (!isAlreadyLoaded)
    {
        const auto dll = getOfficialRuntimeDirectory() / NGFXMODULE.data();
        const HRESULT hook = system::CSystemWin32::delayLoadDLL(NGFXMODULE.data(), { NGFX_INJECTION_DLL_DIR, dll.parent_path() });

        //! don't be scared if you see "No symbols loaded" - you will not hit "false" in this case, the DLL will get loaded if found,
        //! proc addresses will be resolved correctly but status will scream "FAILED" because we don't have any PDB to load
        if (FAILED(hook))
            return false;
    }
#else
    #error "TODO!"
#endif

    if (isAlreadyLoaded)
        return true;

    // now call the APIs for the first time
    uint32_t numInstallations = 0;
    auto result = NGFX_Injection_EnumerateInstallations(&numInstallations, nullptr);
    if (numInstallations==0 || NGFX_INJECTION_RESULT_OK!=result)
        return false;

    std::vector<NGFX_Injection_InstallationInfo> installations(numInstallations);
    result = NGFX_Injection_EnumerateInstallations(&numInstallations,installations.data());
    if (NGFX_INJECTION_RESULT_OK!=result)
        return false;
    assert(installations.size()==numInstallations);

    // get latest installation
    const NGFX_Injection_InstallationInfo& versionInfo = installations.back();

    uint32_t numActivities = 0;
    result = NGFX_Injection_EnumerateActivities(&versionInfo, &numActivities, nullptr);
    if (numActivities==0 || NGFX_INJECTION_RESULT_OK!=result)
        return false;

    std::vector<NGFX_Injection_Activity> activities(numActivities);
    result = NGFX_Injection_EnumerateActivities(&versionInfo, &numActivities, activities.data());
    if (NGFX_INJECTION_RESULT_OK != result)
        return false;
    assert(activities.size()==numActivities);

    for (const auto& activity : activities)
    if (activity.type==NGFX_INJECTION_ACTIVITY_FRAME_DEBUGGER)
    {
        result = NGFX_Injection_InjectToProcess(&versionInfo,&activity);
        if (result==NGFX_INJECTION_RESULT_DRIVER_STILL_LOADED)
            return true;
    }
#endif
    // no NGFX build -> no API to load
    return false;
}

renderdoc_api_t* IAPIConnection::loadRenderdoc()
{
    pRENDERDOC_GetAPI RENDERDOC_GetAPI = nullptr;

#ifdef _NBL_PLATFORM_WINDOWS_
    if (HMODULE mod = GetModuleHandleA("renderdoc.dll"); mod)
       RENDERDOC_GetAPI = reinterpret_cast<pRENDERDOC_GetAPI>(reinterpret_cast<void*>(GetProcAddress(mod, "RENDERDOC_GetAPI")));
#elif defined(_NBL_PLATFORM_ANDROID_)
    if (void* mod=dlopen("libVkLayer_GLES_RenderDoc.so",RTLD_NOW|RTLD_NOLOAD); mod)
        RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)dlsym(mod,"RENDERDOC_GetAPI");
#elif defined(_NBL_PLATFORM_LINUX_)
    if (void* mod=dlopen("librenderdoc.so",RTLD_NOW|RTLD_NOLOAD); mod)
        RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)dlsym(mod,"RENDERDOC_GetAPI");
#else
#error "Nabla Unsupported Platform!"
#endif

    if (RENDERDOC_GetAPI)
    {
        renderdoc_api_t* retval = nullptr;
        if (RENDERDOC_GetAPI(MinRenderdocVersion,(void**)&retval) == 1)
            return retval;
    }

    return nullptr;
}

IAPIConnection::IAPIConnection(const SFeatures& enabledFeatures) : m_physicalDevices(), m_enabledFeatures(enabledFeatures)
{
    loadDebuggers();
}

std::span<IPhysicalDevice* const> IAPIConnection::getPhysicalDevices() const
{
    static_assert(sizeof(std::unique_ptr<IPhysicalDevice>) == sizeof(void*));

    IPhysicalDevice* const* const begin = reinterpret_cast<IPhysicalDevice* const*>(m_physicalDevices.data());
    return {begin,m_physicalDevices.size()};
}

// Current NGFX SDK API is extremely dumb, this will literally pop up a new NSight window every frame.
// It also still fails to capture anything happening off the Queue that's acquiring and presenting to the swapchain or before the very first acquire.
void IAPIConnection::executeNGFXCommand()
{
    assert(m_debugger==EDebuggerType::NSight);
#ifdef NBL_BUILD_WITH_NGFX
    const auto result = NGFX_Injection_ExecuteActivityCommand();
    if (result!=NGFX_INJECTION_RESULT_OK)
    {
        // TODO: add logger to IAPIConnection and log failures!
    }
#endif // NBL_BUILD_WITH_NGFX
}

}