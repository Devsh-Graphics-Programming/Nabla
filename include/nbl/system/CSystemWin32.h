#ifndef _NBL_SYSTEM_CSYSTEMWIN32_H_INCLUDED_
#define _NBL_SYSTEM_CSYSTEMWIN32_H_INCLUDED_
#ifdef _NBL_PLATFORM_WINDOWS_
#include "ISystem.h"
#include "CFileWin32.h"

namespace nbl::system
{
	
class CSystemCallerWin32 final : public ISystemCaller        
{
    protected:
        ~CSystemCallerWin32() override = default;

    public:
        core::smart_refctd_ptr<IFile> createFile_impl(core::smart_refctd_ptr<ISystem>&& sys, const std::filesystem::path& filename, core::bitflag<IFile::E_CREATE_FLAGS> flags) override final
        {
            auto f = core::make_smart_refctd_ptr<CFileWin32>(std::move(sys), filename, flags);
            return f->isOpenedProperly() ? f : nullptr;
        }
};

class CSystemWin32 : public ISystem
{

};

}

#endif
#endif