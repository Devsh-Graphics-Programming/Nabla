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

std::span<IPhysicalDevice* const> IAPIConnection::getPhysicalDevices() const
{
    static_assert(sizeof(std::unique_ptr<IPhysicalDevice>) == sizeof(void*));

    IPhysicalDevice* const* const begin = reinterpret_cast<IPhysicalDevice* const*>(m_physicalDevices.data());
    return {begin,m_physicalDevices.size()};
}

IAPIConnection::IAPIConnection(const SFeatures& enabledFeatures) 
    : m_physicalDevices()
    , m_rdoc_api(nullptr)
    , m_enabledFeatures(enabledFeatures)
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

        // probably is platform agnostic, for now
        m_ngfx_api.injectNGFXToProcess();

        m_debuggerType = m_ngfx_api.useNGFX ? EDT_NGFX :      // ngfx takes priority?
            m_rdoc_api ? EDT_RENDERDOC : EDT_NONE;
    }
}

bool IAPIConnection::SNGFXIntegration::injectNGFXToProcess()
{
    #ifdef NBL_BUILD_WITH_NGFX
    if (m_loaded) //! this check is mandatory!
    {
        uint32_t numInstallations = 0;
        auto result = NGFX_Injection_EnumerateInstallations(&numInstallations, nullptr);
        if (numInstallations == 0 || NGFX_INJECTION_RESULT_OK != result)
        {
            useNGFX = false;
            return false;
        }

        std::vector<NGFX_Injection_InstallationInfo> installations(numInstallations);
        result = NGFX_Injection_EnumerateInstallations(&numInstallations, installations.data());
        if (numInstallations == 0 || NGFX_INJECTION_RESULT_OK != result)
        {
            useNGFX = false;
            return false;
        }

        // get latest installation
        NGFX_Injection_InstallationInfo versionInfo = installations.back();

        uint32_t numActivities = 0;
        result = NGFX_Injection_EnumerateActivities(&versionInfo, &numActivities, nullptr);
        if (numActivities == 0 || NGFX_INJECTION_RESULT_OK != result)
        {
            useNGFX = false;
            return false;
        }

        std::vector<NGFX_Injection_Activity> activities(numActivities);
        result = NGFX_Injection_EnumerateActivities(&versionInfo, &numActivities, activities.data());
        if (NGFX_INJECTION_RESULT_OK != result)
        {
            useNGFX = false;
            return false;
        }

        const NGFX_Injection_Activity* pActivityToInject = nullptr;
        for (const NGFX_Injection_Activity& activity : activities)
        {
            if (activity.type == NGFX_INJECTION_ACTIVITY_FRAME_DEBUGGER)    // only want frame debugger
            {
                pActivityToInject = &activity;
                break;
            }
        }

        if (!pActivityToInject) {
            useNGFX = false;
            return false;
        }

        result = NGFX_Injection_InjectToProcess(&versionInfo, pActivityToInject);
        if (NGFX_INJECTION_RESULT_OK != result)
        {
            useNGFX = false;
            return false;
        }

        useNGFX = true;

        return true;
    } // optional TOOD: could log on "else"
    #endif // NBL_BUILD_WITH_NGFX

    return false;
}

bool IAPIConnection::SNGFXIntegration::executeNGFXCommand()
{
    #ifdef NBL_BUILD_WITH_NGFX
    if(m_loaded) //! this check is mandatory!
        return NGFX_Injection_ExecuteActivityCommand() == NGFX_INJECTION_RESULT_OK; // optional TOOD: could log on "else"
    #endif // NBL_BUILD_WITH_NGFX

    return false;
}

IAPIConnection::SNGFXIntegration::SNGFXIntegration()
    : useNGFX(false /*??*/), m_loaded([]() -> bool
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

        #if defined(_NBL_PLATFORM_WINDOWS_)
        static constexpr std::string_view NGFXMODULE = "NGFX_Injection.dll";
        HMODULE isAlreadyLoaded = GetModuleHandleA(NGFXMODULE.data());

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

        return true;
#else
        return false; // no NGFX build -> no API to load
#endif
    }())
{}

}