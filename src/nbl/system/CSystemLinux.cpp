#include "nbl/system/CSystemLinux.h"

using namespace nbl;
using namespace nbl::system;

#ifdef _NBL_PLATFORM_LINUX_

#include <sys/sysinfo.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
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