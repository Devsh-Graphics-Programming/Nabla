// Internal src-only header.
// Do not include from public headers.
#ifndef _NBL_ASSET_IMPL_S_IO_DIAGNOSTICS_H_INCLUDED_
#define _NBL_ASSET_IMPL_S_IO_DIAGNOSTICS_H_INCLUDED_

#include "nbl/asset/interchange/SInterchangeIO.h"
#include "nbl/system/ILogger.h"


namespace nbl::asset::impl
{

class SIODiagnostics
{
	public:
		template<typename Logger>
		static inline bool logInvalidPlan(Logger& logger, const char* const owner, const char* const fileName, const SResolvedFileIOPolicy& ioPlan)
		{
			if (ioPlan.isValid())
				return false;
			logger.log("%s: invalid io policy for %s reason=%s", system::ILogger::ELL_ERROR, owner, fileName, ioPlan.reason);
			return true;
		}

		template<typename Logger>
		static inline void logTinyIO(Logger& logger, const char* const owner, const char* const fileName, const SInterchangeIO::STelemetry& telemetry, const uint64_t payloadBytes, const SFileIOPolicy& ioPolicy, const char* const opName)
		{
			if (!SInterchangeIO::isTinyIOTelemetryLikely(telemetry, payloadBytes, ioPolicy))
				return;
			logger.log("%s tiny-io guard: file=%s %s=%llu min=%llu avg=%llu",
				system::ILogger::ELL_WARNING, owner, fileName, opName,
				static_cast<unsigned long long>(telemetry.callCount),
				static_cast<unsigned long long>(telemetry.getMinOrZero()),
				static_cast<unsigned long long>(telemetry.getAvgOrZero()));
		}
};

}

#endif
