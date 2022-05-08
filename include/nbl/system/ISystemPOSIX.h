#ifndef _NBL_SYSTEM_C_SYSTEM_CALLER_POSIX_INCLUDED_
#define _NBL_SYSTEM_C_SYSTEM_CALLER_POSIX_INCLUDED_

#include "nbl/system/ISystem.h"

namespace nbl::system
{

#if defined(__unix__)
class NBL_API ISystemPOSIX : public ISystem
{
    protected:
        class CCaller final : public ISystem::ICaller
        {
            public:
                CCaller(ISystemPOSIX* _system) : ICaller(_system) {}

                core::smart_refctd_ptr<ISystemFile> createFile(const std::filesystem::path& filename, const core::bitflag<IFile::E_CREATE_FLAGS> flags) override;
        };

        ISystemPOSIX() : ISystem(core::make_smart_refctd_ptr<CCaller>(this)) {}
};
#endif

}
#endif // ! _NBL_SYSTEM_C_SYSTEM_CALLER_POSIX_INCLUDED_
