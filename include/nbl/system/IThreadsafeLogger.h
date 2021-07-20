#ifndef _NBL_SYSTEM_I_THREADSAFE_LOGGER_INCLUDED_
#define _NBL_SYSTEM_I_THREADSAFE_LOGGER_INCLUDED_

#include "ILogger.h"
#include <mutex>
#include <cstdarg>

namespace nbl::system
{
class IThreadsafeLogger : public ILogger
{
	mutable std::mutex m_mutex;
public:
	// Inherited via ILogger
	void log(const std::string_view& fmtString, E_LOG_LEVEL logLevel, ...) override final
	{
		va_list args;
		va_start(args, logLevel);
		auto l = lock();
		log_impl(fmtString, logLevel, args);
		va_end(args);
	}
	void log(const std::wstring_view& fmtString, E_LOG_LEVEL logLevel, ...) override final
	{
		va_list args;
		va_start(args, logLevel);
		auto l = lock();
		log_impl(fmtString, logLevel, args);
		va_end(args);
	}
private:
	virtual void log_impl(const std::string_view&, E_LOG_LEVEL logLevel, va_list args) = 0;
	virtual void log_impl(const std::wstring_view&, E_LOG_LEVEL logLevel, va_list args) = 0;
	
	std::unique_lock<std::mutex> lock() const
	{
		return std::unique_lock<std::mutex>(m_mutex);
	}

};
}
#endif