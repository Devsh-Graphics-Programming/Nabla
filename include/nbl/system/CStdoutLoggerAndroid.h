#ifndef _NBL_SYSTEM_C_STDOUT_LOGGER_ANDROID_H_INCLUDED_
#define _NBL_SYSTEM_C_STDOUT_LOGGER_ANDROID_H_INCLUDED_
#ifdef _NBL_PLATFORM_ANDROID_
#include "nbl/system/IThreadsafeLogger.h"

namespace nbl::system
{
	class CStdoutLoggerAndroid : public IThreadsafeLogger
	{
	public:
		CStdoutLoggerAndroid(std::underlying_type_t<E_LOG_LEVEL> logLevelMask = ILogger::defaultLogMask()) : IThreadsafeLogger(logLevelMask) {}

	private:
		auto getNativeLogLevel(E_LOG_LEVEL logLevel)
		{
			switch (logLevel)
			{
			case ELL_DEBUG:
			{
				return ANDROID_LOG_DEBUG;
			}
			case ELL_INFO:
			{
				return ANDROID_LOG_INFO;
			}
			case ELL_WARNING:
			{
				return ANDROID_LOG_WARN;
			}
			case ELL_ERROR:
			{
				return ANDROID_LOG_ERROR;
			}
			case ELL_PERFORMANCE:
			{
				return ANDROID_LOG_INFO;
			}
			default:
			{
				assert(false);
				return ANDROID_LOG_UNKNOWN;
			}
			}
		}
		void threadsafeLog_impl(const std::string_view& fmt, E_LOG_LEVEL logLevel, va_list args) override
		{
			(void)__android_log_print(getNativeLogLevel(logLevel), "native-activity", fmt.data(), args);
		}
	};
}

#endif
#endif