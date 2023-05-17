#ifndef _NBL_SYSTEM_C_SYSTEM_CALLER_POSIX_INCLUDED_
#define _NBL_SYSTEM_C_SYSTEM_CALLER_POSIX_INCLUDED_

#include "nbl/core/util/bitflag.h"

#include "nbl/system/ISystem.h"
#include "nbl/system/IFile.h"

namespace nbl::system
{

#if defined(_NBL_PLATFORM_LINUX_) || defined (_NBL_PLATFORM_ANDROID_) || defined (_NBL_PLATFORM_MACOS_)
class ISystemPOSIX : public ISystem
{
    protected:
        class CCaller final : public ISystem::ICaller
        {
            public:
                inline CCaller(ISystemPOSIX* _system) : ICaller(_system) {}

                NBL_API2 core::smart_refctd_ptr<ISystemFile> createFile(const std::filesystem::path& filename, const core::bitflag<IFile::E_CREATE_FLAGS> flags) override;
        };

        inline ISystemPOSIX() : ISystem(core::make_smart_refctd_ptr<CCaller>(this)) {}
};
#endif

}
#endif // ! _NBL_SYSTEM_C_SYSTEM_CALLER_POSIX_INCLUDED_
