#ifndef _NBL_SYSTEM_C_STDOUT_LOGGER_INCLUDED_
#define _NBL_SYSTEM_C_STDOUT_LOGGER_INCLUDED_

#include "IThreadsafeLogger.h"
#include <iostream>

namespace nbl::system
{
	class CStdoutLogger : public IThreadsafeLogger
	{
		CStdoutLogger(core::bitflag<E_LOG_LEVEL> logLevelMask = ILogger::defaultLogMask()) : IThreadsafeLogger(logLevelMask) {}
		virtual void threadsafeLog_impl(const std::string_view& fmt, E_LOG_LEVEL logLevel, va_list args) override
		{
			printf(constructLogString(fmt, logLevel, args).data());
			fflush(stdout);
		}

	};
}

#endif