#ifndef _NBL_SYSTEM_C_SYSTEM_WIN32_H_INCLUDED_
#define _NBL_SYSTEM_C_SYSTEM_WIN32_H_INCLUDED_

#include "ISystem.h"

namespace nbl::system
{

#ifdef _NBL_PLATFORM_WINDOWS_
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
};
#endif

}

#endif