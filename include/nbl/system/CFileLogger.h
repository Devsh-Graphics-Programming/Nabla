#ifndef _NBL_SYSTEM_C_FILE_LOGGER_INCLUDED_
#define _NBL_SYSTEM_C_FILE_LOGGER_INCLUDED_

#include "nbl/system/IThreadsafeLogger.h"
#include "nbl/system/IFile.h"

namespace nbl::system
{

class CFileLogger : public IThreadsafeLogger
{
	public:
		CFileLogger(core::smart_refctd_ptr<IFile>&& _file, const core::bitflag<E_LOG_LEVEL> logLevelMask=ILogger::defaultLogMask())
			: IThreadsafeLogger(logLevelMask), m_file(std::move(_file))
		{
		}

	protected:
		~CFileLogger() = default;

		virtual void threadsafeLog_impl(const std::string_view& fmt, E_LOG_LEVEL logLevel, va_list args) override
		{
			const auto str = constructLogString(fmt, logLevel, args);
			ISystem::future_t<size_t> future;
			m_file->write(future,str.data(),m_file->getSize(),str.length());
		}

		core::smart_refctd_ptr<IFile> m_file;
};

}

#endif