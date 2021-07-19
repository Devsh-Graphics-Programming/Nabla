#include "IThreadsafeLogger.h"
#include <iostream>

namespace nbl::system
{
	class CStdoutLogger : public IThreadsafeLogger
	{
		virtual void log_impl(const std::string_view& fmt, E_LOG_LEVEL logLevel, va_list args) override
		{
			std::cout << constructLogString(fmt, logLevel, args);
		}
		
		virtual void log_impl(const std::wstring_view& fmt, E_LOG_LEVEL logLevel, va_list args) override
		{
			std::wcout << constructLogWstring(fmt, logLevel, args);
		}
	};
}