#ifndef _NBL_SYSTEM_C_COLORFUL_STDOUT_LOGGER_WIN32_INCLUDED_
#define _NBL_SYSTEM_C_COLORFUL_STDOUT_LOGGER_WIN32_INCLUDED_

#include "nbl/system/IThreadsafeLogger.h"

namespace nbl::system
{
#ifdef _NBL_PLATFORM_WINDOWS_
class NBL_API2 CColoredStdoutLoggerWin32 : public IThreadsafeLogger
{
		void* m_native_console;

		void threadsafeLog_impl(const std::string_view& fmt, E_LOG_LEVEL logLevel, va_list args) override;

		inline int getConsoleColor(E_LOG_LEVEL level)
		{
			switch (level)
			{
				case ELL_DEBUG: // Gray
				{
					return 8;
				}
				case ELL_INFO: // White
				{
					return 7;
				}
				case ELL_WARNING: // Yellow
				{
					return 14;
				}
				case ELL_ERROR: // Red
				{
					return 12;
				}
				case ELL_PERFORMANCE: // Blue
				{
					return 11;
				}
				case ELL_NONE: 
				{
					assert(false); // how did this happen?? Btw, do we even need this log level?
					break;
				}
			}
			return 0;
		}

	public:
		CColoredStdoutLoggerWin32(core::bitflag<E_LOG_LEVEL> logLevelMask = ILogger::defaultLogMask());
};

#endif
}

#endif