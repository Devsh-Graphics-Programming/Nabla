#ifndef _NBL_SYSTEM_C_SYSTEM_CALLER_POSIX_INCLUDED_
#define _NBL_SYSTEM_C_SYSTEM_CALLER_POSIX_INCLUDED_

#if defined(__unix__)

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
}
#endif
#endif // ! _NBL_SYSTEM_C_SYSTEM_CALLER_POSIX_INCLUDED_
