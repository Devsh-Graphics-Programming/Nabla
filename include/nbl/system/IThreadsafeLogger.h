#ifndef _NBL_SYSTEM_I_THREADSAFE_LOGGER_INCLUDED_
#define _NBL_SYSTEM_I_THREADSAFE_LOGGER_INCLUDED_

#include "ILogger.h"
#include <mutex>
#include <cstdarg>

namespace nbl::system
{
class NBL_API IThreadsafeLogger : public ILogger
{
	mutable std::mutex m_mutex;
public:
	IThreadsafeLogger(core::bitflag<E_LOG_LEVEL> logLevelMask) : ILogger(logLevelMask) {}
	// Inherited via ILogger
	void log_impl(const std::string_view& fmtString, E_LOG_LEVEL logLevel, va_list args) override final
	{
		auto l = lock();
		threadsafeLog_impl(fmtString, logLevel, args);
	}
private:
	virtual void threadsafeLog_impl(const std::string_view&, E_LOG_LEVEL logLevel, va_list args) = 0;
	
	std::unique_lock<std::mutex> lock() const
	{
		return std::unique_lock<std::mutex>(m_mutex);
	}

};
}
#endif