#ifndef _NBL_SYSTEM_C_COLORFUL_STDOUT_LOGGER_ANSI_INCLUDED_
#define _NBL_SYSTEM_C_COLORFUL_STDOUT_LOGGER_ANSI_INCLUDED_

#include "nbl/system/IThreadsafeLogger.h"

#include <string_view>

namespace nbl::system {

#include "nbl/system/DefaultFuncPtrLoader.h"

// logging using ANSI escape codes
class NBL_API CColoredStdoutLoggerANSI : public IThreadsafeLogger {

public:
  CColoredStdoutLoggerANSI(
      core::bitflag<E_LOG_LEVEL> logLevelMask = ILogger::defaultLogMask())
      : IThreadsafeLogger(logLevelMask) {}

private:
  virtual void threadsafeLog_impl(const std::string_view &fmt,
                                  E_LOG_LEVEL logLevel, va_list args) override;
};

} // namespace nbl::system

#endif