// Copyright (C) 2023-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_APPLICATION_TEMPLATES_MONO_SYSTEM_MONO_LOGGER_APPLICATION_HPP_INCLUDED_
#define _NBL_EXAMPLES_APPLICATION_TEMPLATES_MONO_SYSTEM_MONO_LOGGER_APPLICATION_HPP_INCLUDED_

// always include nabla first
#include "nabla.h"

// TODO: get these all included by the appropriate namespace headers!
#include "nbl/system/CColoredStdoutLoggerANSI.h"
#include "nbl/system/IApplicationFramework.h"

namespace nbl::application_templates
{

// Virtual Inheritance because apps might end up doing diamond inheritance
class MonoSystemMonoLoggerApplication : public virtual system::IApplicationFramework
{
		using base_t = system::IApplicationFramework;

	public:
		using base_t::base_t;

		virtual bool onAppTerminated() override
		{
			m_logger->log("Example Terminated Successfully!",system::ILogger::ELL_INFO);
			m_logger = nullptr;
			m_system = nullptr;
			return true;
		}

	protected:
		// need this one for skipping passing all args into ApplicationFramework
		MonoSystemMonoLoggerApplication() = default;

		virtual bool onAppInitialized(core::smart_refctd_ptr<system::ISystem>&& system) override
		{
			// protect against double initialization call (diamond inheritance)
			if (!m_system)
			{
				// This is a weird pattern, basically on some platforms all file & system operations need to go through a "God Object" only handed to you in some plaform specific way
				// On "normal" platforms like win32 and Linux we can just create system objects at will and there's no special state we need to find.
				if (system)
					m_system = std::move(system);
				else
					m_system = system::IApplicationFramework::createSystem();
			}

			// create a logger with default logging level masks
			if (!m_logger)
			{
				m_logger = core::make_smart_refctd_ptr<system::CColoredStdoutLoggerANSI>(getLogLevelMask());
				m_logger->log("Logger Created!",system::ILogger::ELL_INFO);
			}
			return true;
		}

		// some examples may need to override this because they're Headless (no window output)
		virtual core::bitflag<system::ILogger::E_LOG_LEVEL> getLogLevelMask() //TODO: const
		{
			// @Hazardu probably need a commandline option to override
			return system::ILogger::DefaultLogMask();
		}

		// made it return false so we can save some lines writing `if (failCond) {logFail(); return false;}`
		template<typename... Args>
		inline bool logFail(const char* msg, Args&&... args)
		{
			m_logger->log(msg,system::ILogger::ELL_ERROR,std::forward<Args>(args)...);
			return false;
		}

		core::smart_refctd_ptr<system::ISystem> m_system;
		core::smart_refctd_ptr<system::ILogger> m_logger;
};

}

#endif // _CAMERA_IMPL_