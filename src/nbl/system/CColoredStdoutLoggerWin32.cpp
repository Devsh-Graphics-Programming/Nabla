#include "nbl/system/CColoredStdoutLoggerWin32.h"

using namespace nbl;
using namespace nbl::system;

#ifdef _NBL_PLATFORM_WINDOWS_
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

CColoredStdoutLoggerWin32::CColoredStdoutLoggerWin32(core::bitflag<E_LOG_LEVEL> logLevelMask) : IThreadsafeLogger(logLevelMask)
{
	m_native_console = GetStdHandle(STD_OUTPUT_HANDLE);
}

void CColoredStdoutLoggerWin32::threadsafeLog_impl(const std::string_view& fmt, E_LOG_LEVEL logLevel, va_list args)
{
	SetConsoleTextAttribute(m_native_console, getConsoleColor(logLevel));
	printf(constructLogString(fmt, logLevel, args).data());
	fflush(stdout);
	SetConsoleTextAttribute(m_native_console, 15); // restore to white
}
#endif