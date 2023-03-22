#ifndef _NBL_SYSTEM_C_FILE_LOGGER_INCLUDED_
#define _NBL_SYSTEM_C_FILE_LOGGER_INCLUDED_

#include "nbl/system/IThreadsafeLogger.h"
#include "nbl/system/IFile.h"

namespace nbl::system
{

class CFileLogger : public IThreadsafeLogger
{
	public:
		CFileLogger(core::smart_refctd_ptr<IFile>&& _file, const bool append, const core::bitflag<E_LOG_LEVEL> logLevelMask=ILogger::defaultLogMask())
			: IThreadsafeLogger(logLevelMask), m_file(std::move(_file)), m_pos(append ? m_file->getSize():0ull)
		{
		}

	protected:
		~CFileLogger() = default;

		virtual void threadsafeLog_impl(const std::string_view& fmt, E_LOG_LEVEL logLevel, va_list args) override
		{
			const auto str = constructLogString(fmt, logLevel, args);
			IFile::success_t succ;
			m_file->write(succ,str.data(),m_pos,str.length());
			m_pos += succ.getBytesProcessed();
		}

		core::smart_refctd_ptr<IFile> m_file;
		size_t m_pos;
};

}

#endif