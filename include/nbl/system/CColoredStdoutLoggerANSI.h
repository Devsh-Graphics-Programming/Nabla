#ifndef _NBL_SYSTEM_C_COLORED_STDOUT_LOGGER_ANSI_INCLUDED_
#define _NBL_SYSTEM_C_COLORED_STDOUT_LOGGER_ANSI_INCLUDED_

#include "nbl/system/IThreadsafeLogger.h"

#include <string_view>

namespace nbl::system
{

// logging using ANSI escape codes
class NBL_API2 CColoredStdoutLoggerANSI : public IThreadsafeLogger
{
  public:
    CColoredStdoutLoggerANSI(core::bitflag<E_LOG_LEVEL> logLevelMask = ILogger::defaultLogMask()) : IThreadsafeLogger(logLevelMask) {}

  private:
    // more info about how this works: https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
    virtual void threadsafeLog_impl(const std::string_view &fmt, E_LOG_LEVEL logLevel, va_list args) override
    {
      auto str = constructLogString(fmt, logLevel, args);
      switch (logLevel)
      {
        case ELL_DEBUG:
          printf("\x1b[37m%s", str.data()); // White
          break;
        case ELL_INFO:
          printf("\x1b[37m%s", str.data()); // White
          break;
        case ELL_WARNING:
          printf("\x1b[33m%s", str.data()); // yellow
          break;
        case ELL_ERROR:
          printf("\x1b[31m%s", str.data()); // red
          break;
        case ELL_PERFORMANCE:
          printf("\x1b[34m%s", str.data()); // blue
          break;
        case ELL_NONE:
          assert(false);
          break;
      }
      fflush(stdout);
    }
};

} // namespace nbl::system

#endif
