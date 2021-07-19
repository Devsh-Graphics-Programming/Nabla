#include <string>
#include <cstdint>
#include <chrono>
#include <cassert>
#include <sstream>
#include <iomanip>
namespace nbl::system
{
	class ILogger
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

		virtual void log(const std::string_view& fmtString, E_LOG_LEVEL logLevel, ...) = 0;
		virtual void log(const std::wstring_view& fmtString, E_LOG_LEVEL logLevel, ...) = 0;
	protected:
		virtual std::string constructLogString(const std::string_view& fmtString, E_LOG_LEVEL logLevel, va_list l)
		{
			using namespace std::literals;
			auto currentTime = std::chrono::system_clock::now();
			const std::time_t t = std::chrono::system_clock::to_time_t(currentTime);
			std::stringstream ss;
			ss << '[' << std::put_time(std::localtime(&t), "%F %T.\n") << ']';
			switch (logLevel)
			{
			case ELL_DEBUG:
				ss << "[DEBUG]: ";
				break;
			case ELL_INFO:
				ss << "[INFO]: ";
				break;
			case ELL_WARNING:
				ss << "[WARNING]: ";
				break;
			case ELL_PERFORMANCE:
				ss << "[PERFORMANCE]: ";
				break;
			case ELL_ERROR:
				ss << "[ERROR]: ";
				break;
			case ELL_NONE:
				return "";
			}

			std::string message(fmtString); 
			assert(false); //TODO: calculate the length of the output message
			vsprintf(message.data(), fmtString.data(), l);
			ss << message << '\n';
			return ss.str();
		}
		virtual std::wstring constructLogWstring(const std::wstring_view& fmtString, E_LOG_LEVEL logLevel, va_list l)
		{
			assert(false);
		}
	};
}