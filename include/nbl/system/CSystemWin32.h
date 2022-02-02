#ifndef _NBL_SYSTEM_CSYSTEMWIN32_H_INCLUDED_
#define _NBL_SYSTEM_CSYSTEMWIN32_H_INCLUDED_
#ifdef _NBL_PLATFORM_WINDOWS_
#include <windows.h>
#include <powerbase.h>
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
    //LOL the struct definition wasn't added to winapi headers do they ask to declare them yourself
    typedef struct _PROCESSOR_POWER_INFORMATION {
        ULONG Number;
        ULONG MaxMhz;
        ULONG CurrentMhz;
        ULONG MhzLimit;
        ULONG MaxIdleState;
        ULONG CurrentIdleState;
    } PROCESSOR_POWER_INFORMATION, * PPROCESSOR_POWER_INFORMATION;

    SystemInfo getSystemInfo() const override
    {
        SystemInfo info;
        PROCESSOR_POWER_INFORMATION cpuInfo;
        CallNtPowerInformation(ProcessorInformation, nullptr, 0, &cpuInfo, sizeof(cpuInfo));
        info.cpuFrequency = cpuInfo.MaxMhz;

        info.desktopResX = GetSystemMetrics(SM_CXSCREEN);
        info.desktopResY = GetSystemMetrics(SM_CYSCREEN);

        return info;
    }
};

}

#endif
#endif