#ifndef _NBL_SYSTEM_C_SYSTEM_CALLER_POSIX_INCLUDED_
#define _NBL_SYSTEM_C_SYSTEM_CALLER_POSIX_INCLUDED_

#include "nbl/system/ISystem.h"

namespace nbl::system
{

#if defined(__unix__)
class ISystemPOSIX : public ISystem
{
    protected:
        class CCaller final : public ISystem::ICaller
        {
            public:
                core::smart_refctd_ptr<IFile> createFile_impl(const std::filesystem::path& filename, core::bitflag<IFile::E_CREATE_FLAGS> flags) override final;
        };

        ISystemPOSIX() : ISystem(core::make_smart_refctd_ptr<CCaller>(this)) {}
};
#endif

}
#endif // ! _NBL_SYSTEM_C_SYSTEM_CALLER_POSIX_INCLUDED_
