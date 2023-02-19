#include "nbl/system/CColoredStdoutLoggerANSI.h"

using namespace nbl::system;

void CColoredStdoutLoggerANSI::threadsafeLog_impl(const std::string_view &fmt, E_LOG_LEVEL logLevel, va_list args) {
	auto str = constructLogString(fmt, logLevel, args);
	switch (logLevel) {
		case ELL_DEBUG: {
			printf("\x1b[37m%s", str.data()); // White
			break;
		}
		case ELL_INFO: {
			printf("\x1b[37m%s", str.data()); // White
			break;
		}
		case ELL_WARNING: {
			printf("\x1b[33m%s", str.data()); // yellow
			break;
		}
		case ELL_ERROR: {
			printf("\x1b[31m%s", str.data()); // red
			break;
		}
		case ELL_PERFORMANCE: {
			printf("\x1b[34m%s", str.data()); // blue
			break;
		}
		case ELL_NONE: {
			assert(false);
			break;
		}
	}
	fflush(stdout);
}

