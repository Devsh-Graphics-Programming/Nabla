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
            // load from right next to the executable (always be able to override like this)
            HMODULE res = LoadLibraryExA(dllName, NULL, LOAD_LIBRARY_SEARCH_APPLICATION_DIR);

            #ifdef NBL_EXPLICIT_MODULE_LOAD_LOG
            printf("[LOG]: Loaded \"%s\" module next to the executable\n", dllName);
            #endif // NBL_EXPLICIT_MODULE_LOAD_LOG

            // now lets try our custom dirs
            for (system::path dir : paths)
            {
                const auto pathStr = std::filesystem::absolute(dir.make_preferred()/dllName).string(); // always attempt to resolve relative paths
                if (res = LoadLibraryExA(pathStr.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH))
                {
                    #ifdef NBL_EXPLICIT_MODULE_LOAD_LOG
                    printf("LOG]: Loaded \"%s\" module, overriding module next to the executable\n", pathStr.c_str());
                    #endif // NBL_EXPLICIT_MODULE_LOAD_LOG
                    break;
                }
            }
            // if still can't find, try looking for a system wide install
            if (!res)
                if (res = LoadLibraryExA(dllName, NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS))
                {
                    #ifdef NBL_EXPLICIT_MODULE_LOAD_LOG
                    printf("LOG]: Loaded \"%s\" sytem-wide module\n", dllName);
                    #endif // NBL_EXPLICIT_MODULE_LOAD_LOG
                }
                    
            if(!res)
                return E_FAIL;

            return __HrLoadAllImportsForDll(dllName);
        }
};

}
#endif


#endif