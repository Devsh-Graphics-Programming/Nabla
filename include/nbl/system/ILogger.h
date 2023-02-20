#ifndef _NBL_SYSTEM_I_LOGGER_INCLUDED_
#define _NBL_SYSTEM_I_LOGGER_INCLUDED_

#include "nbl/core/IReferenceCounted.h"
#include "nbl/core/decl/smart_refctd_ptr.h"
#include "nbl/core/util/bitflag.h"

#include <string>
#include <cstdint>
#include <chrono>
#include <cassert>
#include <sstream>
#include <iomanip>
#include <regex>
#include <cstdarg>


namespace nbl::system
{

class NBL_API ILogger : public core::IReferenceCounted
{
	public:
		enum E_LOG_LEVEL : uint8_t
		{
			ELL_NONE = 0,
			ELL_DEBUG = 1,
			ELL_INFO = 2,
			ELL_WARNING = 4,
			ELL_PERFORMANCE = 8,
			ELL_ERROR = 16
		};

		void log(const std::string_view& fmtString, E_LOG_LEVEL logLevel = ELL_DEBUG, ...)
		{
			if (logLevel & m_logLevelMask.value)
			{
				va_list args;
				va_start(args, logLevel);
				log_impl(fmtString, logLevel, args);
				va_end(args);
			}
		}

		inline core::bitflag<E_LOG_LEVEL> getLogLevelMask() const
		{
			return m_logLevelMask;
		}

	protected:
		static core::bitflag<E_LOG_LEVEL> defaultLogMask();

		ILogger(core::bitflag<E_LOG_LEVEL> logLevelMask) : m_logLevelMask(logLevelMask) {}

		virtual void log_impl(const std::string_view& fmtString, E_LOG_LEVEL logLevel, va_list args) = 0;
		virtual std::string constructLogString(const std::string_view& fmtString, E_LOG_LEVEL logLevel, va_list l)
		{
			using namespace std::literals;
			using namespace std::chrono;
			auto currentTime = std::chrono::system_clock::now();
			const std::time_t t = std::chrono::system_clock::to_time_t(currentTime);
			
			// Since there is no real way in c++ to get current time with microseconds, this is my weird approach
			auto time_since_epoch = duration_cast<microseconds>(system_clock::now().time_since_epoch());
			auto time_since_epoch_s = duration_cast<seconds>(system_clock::now().time_since_epoch());
			time_since_epoch -= duration_cast<microseconds>(time_since_epoch_s);

			// This while is for the microseconds which are less that 6 digits long to be aligned with the others
			while (time_since_epoch.count() / 100000 == 0) time_since_epoch *= 10;

			auto time = std::localtime(&t);

			constexpr size_t DATE_STR_LENGTH = 28;
			std::string timeStr(DATE_STR_LENGTH, '\0');
			sprintf(timeStr.data(), "[%02d.%02d.%d %02d:%02d:%02d:%d]", time->tm_mday, time->tm_mon + 1, 1900 + time->tm_year, time->tm_hour, time->tm_min, time->tm_sec, (int)time_since_epoch.count());
			
			std::string messageTypeStr;
			switch (logLevel)
			{
			case ELL_DEBUG:
				messageTypeStr = "[DEBUG]";
				break;
			case ELL_INFO:
				messageTypeStr = "[INFO]";
				break;
			case ELL_WARNING:
				messageTypeStr = "[WARNING]";
				break;
			case ELL_PERFORMANCE:
				messageTypeStr = "[PERFORMANCE]";
				break;
			case ELL_ERROR:
				messageTypeStr = "[ERROR]";
				break;
			case ELL_NONE:
				return "";
			}

			va_list testArgs; // copy of va_list since it is not safe to use it twice
			va_copy(testArgs, l);
			int formatSize = vsnprintf(nullptr, 0, fmtString.data(), testArgs) + 1;
			std::string message(formatSize, '\0'); 
			vsnprintf(message.data(), formatSize, fmtString.data(), l);

			std::string out_str(timeStr.length() + messageTypeStr.length() + formatSize + 3, '\0');
			sprintf(out_str.data(), "%s%s: %s\n", timeStr.data(), messageTypeStr.data(), message.data());
 			return out_str;
			return "";
		}

	private:
		core::bitflag<E_LOG_LEVEL> m_logLevelMask;
};

class NBL_API logger_opt_ptr final
{
	public:
		logger_opt_ptr(ILogger* const _logger) : logger(_logger) {}
		~logger_opt_ptr() = default;

		template<typename... Args>
		void log(const std::string_view& fmtString, ILogger::E_LOG_LEVEL logLevel = ILogger::ELL_DEBUG, Args&&... args) const
		{
			if (logger != nullptr)
				return logger->log(fmtString, logLevel, std::forward<Args>(args)...);
		}

		ILogger* get() const { return logger; }
	private:
		mutable ILogger* logger;
};

class NBL_API logger_opt_smart_ptr final
{
	public:
		logger_opt_smart_ptr(core::smart_refctd_ptr<ILogger>&& _logger) : logger(std::move(_logger)) {}
		logger_opt_smart_ptr(std::nullptr_t t) : logger(nullptr) {}
		~logger_opt_smart_ptr() = default;

		template<typename... Args>
		void log(const std::string_view& fmtString, ILogger::E_LOG_LEVEL logLevel = ILogger::ELL_DEBUG, Args&&... args) const
		{
			if (logger.get() != nullptr)
				return logger->log(fmtString, logLevel, std::forward<Args>(args)...);
		}

		ILogger* getRaw() const { return logger.get(); }
		logger_opt_ptr getOptRawPtr() const { return logger_opt_ptr(logger.get()); }
		const core::smart_refctd_ptr<ILogger>& get() const { return logger; }

	private:
		mutable core::smart_refctd_ptr<ILogger> logger;
};



}

#endif