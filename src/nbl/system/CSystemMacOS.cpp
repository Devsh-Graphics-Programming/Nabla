#include "nbl/system/CSystemMacOS.h"

using namespace nbl;
using namespace nbl::system;

#ifdef _NBL_PLATFORM_MACOS_

#include <sys/types.h>
#include <sys/sysctl.h>
#include <stdio.h>
#include <string>

ISystem::SystemInfo CSystemMacOS::getSystemInfo() const
{
    #define BUFFERLEN 128
    
    auto getProperty = [](const std::string_view property) -> int32_t
    {
        std::string buffer;
        buffer.resize(BUFFERLEN);
        size_t bufferCopyLen = buffer.size();
        sysctlbyname(property.data(), buffer.data(), &bufferCopyLen, nullptr, 0);
        return std::stoi(buffer.substr(0, bufferCopyLen));
    };
    
    SystemInfo info;
    
    // info.OSFullName = ;
    info.cpuFrequencyHz = 3000000000u; // TODO
    info.desktopResX = 0xdeadbeefu; // TODO
    info.desktopResY = 0xdeadbeefu; // TODO
    info.totalMemory = getProperty("hw.memsize"); //hw.memsize
    // info.availableMemory = ; // TODO

    return info;
}
#endif // _NBL_PLATFORM_MACOS_
