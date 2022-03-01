#ifndef _NBL_SYSTEM_C_FILE_LOGGER_INCLUDED_
#define _NBL_SYSTEM_C_FILE_LOGGER_INCLUDED_

#include <filesystem>
#include <fstream>

#include "IThreadsafeLogger.h"

namespace nbl::system
{

class CFileLogger : public IThreadsafeLogger
{
	public:
		static core::smart_refctd_ptr<CFileLogger> create(const std::filesystem::path& outputFileName)
		{
			auto ret = core::smart_refctd_ptr<CFileLogger>(new CFileLogger(outputFileName));
			if (!ret->m_ofs.is_open()) return nullptr;
			return ret;
		}
		~CFileLogger()
		{
		}
	private:
		std::ofstream m_ofs;
		CFileLogger(const std::filesystem::path& outputFileName, core::bitflag<E_LOG_LEVEL> logLevelMask = ILogger::defaultLogMask()) : IThreadsafeLogger(logLevelMask), m_ofs(outputFileName, std::ios_base::app){}

		virtual void threadsafeLog_impl(const std::string_view& fmt, E_LOG_LEVEL logLevel, va_list args) override
		{
			m_ofs << constructLogString(fmt, logLevel, args).data() << std::flush;
		}
};

}

#endif