#ifndef _NBL_SYSTEM_C_SYSTEM_WIN32_H_INCLUDED_
#define _NBL_SYSTEM_C_SYSTEM_WIN32_H_INCLUDED_

#include "nbl/system/ISystem.h"

#ifdef _NBL_PLATFORM_WINDOWS_
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <delayimp.h>

namespace nbl::system
{

class NBL_API2 CSystemWin32 : public ISystem
{
    protected:
        class CCaller final : public ICaller
        {
            public:
                CCaller(ISystem* _system) : ICaller(_system) {}

                core::smart_refctd_ptr<ISystemFile> createFile(const std::filesystem::path& filename, const core::bitflag<IFile::E_CREATE_FLAGS> flags) override final;
        };
        
    public:
        inline CSystemWin32() : ISystem(core::make_smart_refctd_ptr<CCaller>(this)) {}

        SystemInfo getSystemInfo() const override;

        template<typename PathContainer=core::vector<system::path>>
        static inline HRESULT delayLoadDLL(const char* dllName, const PathContainer& paths)
        {
            #ifdef NBL_EXPLICIT_MODULE_LOAD_LOG
            auto getModulePath = [](HMODULE hModule) -> std::string
            {
                char path[MAX_PATH];
                if (GetModuleFileNameA(hModule, path, MAX_PATH) == 0)
                    return {};

                return std::string(path);
            };
            #endif // NBL_EXPLICIT_MODULE_LOAD_LOG

            const auto executableDirectory = []() -> std::filesystem::path
            {
                wchar_t path[MAX_PATH] = { 0 };
                GetModuleFileNameW(NULL, path, MAX_PATH);

                return std::filesystem::path(path).parent_path();
            }();

            // load from right next to the executable (always be able to override like this)
            HMODULE res = LoadLibraryExA(dllName, NULL, LOAD_LIBRARY_SEARCH_APPLICATION_DIR);

            // now lets try our custom dirs, always attempt to resolve relative paths
            for (system::path dir : paths)
            {
                auto requestModulePath = dir.make_preferred() / dllName;

                // first try relative to CWD
                {
                    const auto path = std::filesystem::absolute(requestModulePath).string(); 
                    if (res = LoadLibraryExA(path.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH))
                        break;
                }

                // then relative to the executable's directory
                {
                    const auto path = std::filesystem::absolute(executableDirectory / requestModulePath).string();
                    if (res = LoadLibraryExA(path.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH))
                        break;
                }
            }

            // if still can't find, try looking for a system wide install
            if (!res)
                res = LoadLibraryExA(dllName, NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
                  
            #ifdef NBL_EXPLICIT_MODULE_LOAD_LOG
            if (res)
            {
                const auto modulePath = getModulePath(res);
                printf("[INFO]: Loaded \"%s\" module\n", modulePath.c_str());
            }
            #endif // NBL_EXPLICIT_MODULE_LOAD_LOG

            if (!res)
            {
                #ifdef NBL_EXPLICIT_MODULE_LOAD_LOG
                printf("[ERROR]: Could not load \"%s\" module\n", dllName);
                #endif // NBL_EXPLICIT_MODULE_LOAD_LOG

                return E_FAIL;
            }

            // VS 17.9.6 bug
            __HrLoadAllImportsForDll(dllName);
            return true;
        }
};

}
#endif


#endif