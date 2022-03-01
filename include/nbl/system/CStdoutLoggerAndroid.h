#ifndef _NBL_SYSTEM_C_STDOUT_LOGGER_ANDROID_H_INCLUDED_
#define _NBL_SYSTEM_C_STDOUT_LOGGER_ANDROID_H_INCLUDED_

#include "nbl/system/IThreadsafeLogger.h"

#ifdef _NBL_PLATFORM_ANDROID_
namespace nbl::system
{

class CStdoutLoggerAndroid : public IThreadsafeLogger
{
	public:
		CStdoutLoggerAndroid(core::bitflag<E_LOG_LEVEL> logLevelMask = ILogger::defaultLogMask()) : IThreadsafeLogger(logLevelMask) {}

	private:
		void threadsafeLog_impl(const std::string_view& fmt, E_LOG_LEVEL logLevel, va_list args) override;
};

}

#endif
#endif