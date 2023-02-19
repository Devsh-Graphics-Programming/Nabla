#include "nbl/system/CSystemLinux.h"

using namespace nbl;
using namespace nbl::system;

#ifdef _NBL_PLATFORM_LINUX_

#include <sys/sysinfo.h>
ISystem::SystemInfo CSystemLinux::getSystemInfo() const
{
    SystemInfo info;

    // TODO
    // info.cpuFrequencyHz = 3000000000u;

    struct sysinfo linuxSystemInfo;
    sysinfo(&linuxSystemInfo);
    info.totalMemory = linuxSystemInfo.totalram;
    info.availableMemory = linuxSystemInfo.freeram;
    info.totalMemory *= linuxSystemInfo.mem_unit;
    info.availableMemory *= linuxSystemInfo.mem_unit;

    // TODO
    info.desktopResX = 0xdeadbeefu;
    info.desktopResY = 0xdeadbeefu;

    return info;
}
#endif