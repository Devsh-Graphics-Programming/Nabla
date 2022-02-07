#ifndef _NBL_SYSTEM_C_SYSTEM_CALLER_POSIX_INCLUDED_
#define _NBL_SYSTEM_C_SYSTEM_CALLER_POSIX_INCLUDED_

#if defined(__unix__)
#include "nbl/system/CFilePOSIX.h"
namespace nbl::system
{
class CSystemCallerPOSIX final : public ISystemCaller
{
protected:
    ~CSystemCallerPOSIX() override = default;

public:
    core::smart_refctd_ptr<IFile> createFile_impl(core::smart_refctd_ptr<ISystem>&& sys, const std::filesystem::path& filename, core::bitflag<IFile::E_CREATE_FLAGS> flags) override final
    {
        auto f = core::make_smart_refctd_ptr<CFilePOSIX>(std::move(sys), filename, flags);
        return f->isOpenedProperly() ? f : nullptr;
    }
};
}
#endif
#endif  // ! _NBL_SYSTEM_C_SYSTEM_CALLER_POSIX_INCLUDED_
