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
    virtual void threadsafeLog_impl(const std::string_view &fmt, E_LOG_LEVEL logLevel, va_list args) override;
};

} // namespace nbl::system

#endif
