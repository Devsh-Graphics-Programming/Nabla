#ifndef _NBL_SYSTEM_C_SYSTEM_LINUX_H_INCLUDED_
#define _NBL_SYSTEM_C_SYSTEM_LINUX_H_INCLUDED_
#ifdef _NBL_PLATFORM_LINUX_
#include "nbl/system/ISystem.h"
namespace nbl::system
{
    class CSystemCallerPOSIX final : public ISystemCaller
    {
    protected:
        ~CSystemCallerPOSIX() override = default;

    public:
        core::smart_refctd_ptr<IFile> createFile_impl(core::smart_refctd_ptr<ISystem>&& sys, const std::filesystem::path& filename, core::bitflag<IFile::E_CREATE_FLAGS> flags) override final
        {
            assert(false); // TODO: posix files (hopefully won't need those on android yet)
        }
    };
	class CSystemLinux : public ISystem
	{

	};
}
#endif
#endif