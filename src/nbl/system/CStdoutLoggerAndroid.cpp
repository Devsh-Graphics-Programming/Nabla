#include "nbl/system/CStdoutLoggerAndroid.h"

#ifdef _NBL_PLATFORM_ANDROID_
#include <android/log.h>

using namespace nbl::system;

void CStdoutLoggerAndroid::threadsafeLog_impl(const std::string_view& fmt, E_LOG_LEVEL logLevel, va_list args) override
{
	auto nativeLogLevel = ANDROID_LOG_UNKNOWN;
	switch (logLevel)
	{
		case ELL_DEBUG:
			nativeLogLevel = ANDROID_LOG_DEBUG;
			break;
		case ELL_INFO:
			nativeLogLevel = ANDROID_LOG_INFO;
			break;
		case ELL_WARNING:
			nativeLogLevel = ANDROID_LOG_WARN;
			break;
		case ELL_ERROR:
			nativeLogLevel = ANDROID_LOG_ERROR;
			break;
		case ELL_PERFORMANCE:
			nativeLogLevel = ANDROID_LOG_INFO;
			break;
		default:
			assert(false);
			break;
	}
	(void)__android_log_print(nativeLogLevel, "Nabla Engine: ", "%s", constructLogString(fmt,logLevel,args).c_str());
}
#endif