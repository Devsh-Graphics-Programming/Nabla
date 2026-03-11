#include "nbl/system/CSystemLinux.h"

using namespace nbl;
using namespace nbl::system;

#ifdef _NBL_PLATFORM_LINUX_

#include <algorithm>
#include <cctype>
#include <fstream>
#include <string>
#include <unordered_set>
#include <sys/sysinfo.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace
{

std::string trimCopy(std::string value)
{
    auto notSpace = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), notSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), notSpace).base(), value.end());
    return value;
}

}

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

    std::ifstream cpuInfo("/proc/cpuinfo");
    std::unordered_set<std::string> uniquePhysicalCores;
    std::string currentPhysicalId;
    std::string currentCoreId;
    auto flushCurrentCore = [&]()
    {
        if (!currentPhysicalId.empty() || !currentCoreId.empty())
            uniquePhysicalCores.insert(currentPhysicalId + ":" + currentCoreId);
        currentPhysicalId.clear();
        currentCoreId.clear();
    };

    for (std::string line; std::getline(cpuInfo, line);)
    {
        if (line.empty())
        {
            flushCurrentCore();
            continue;
        }

        if (line.starts_with("model name"))
        {
            const auto separator = line.find(':');
            if (separator != std::string::npos && info.cpuName == "Unknown")
                info.cpuName = trimCopy(line.substr(separator + 1u));
            continue;
        }

        if (line.starts_with("physical id"))
        {
            const auto separator = line.find(':');
            if (separator != std::string::npos)
                currentPhysicalId = trimCopy(line.substr(separator + 1u));
            continue;
        }

        if (line.starts_with("core id"))
        {
            const auto separator = line.find(':');
            if (separator != std::string::npos)
                currentCoreId = trimCopy(line.substr(separator + 1u));
            continue;
        }
    }
    flushCurrentCore();
    info.physicalCoreCount = static_cast<uint32_t>(uniquePhysicalCores.size());

    return info;
}

bool isDebuggerAttached()
{
   constexpr char tracerPidStr[] = "TracerPid:";
   char buf[4096];

   const int status = open("/proc/self/status");
   if (status == -1)
      return false;

   const size_t numRead = read(status, static_cast<void*>(buf), sizeof(buf) - 1);
   close(status);

   buf[numRead] = '\0';
   const auto offset = strstr(buf, tracerPidStr);
   if (not offset)
      return false;

   // few helper lambdas
   auto isSpace = [](char c) { return c == ' '; };
   auto isDigit = [](char c) { return c >= '0' && c <= '9'; };

   for (const char* cPtr = offset + sizeof(tracerPidStr) - 1; cPtr <= buf + numRead; cPtr++)
   {
      if (isSpace(*cPtr))
         continue;
      else
         return isDigit(*cPtr) && *cPtr != '0';
   }

   return false;
}

#endif
