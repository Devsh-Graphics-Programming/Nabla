// Internal src-only header.
// Do not include from public headers.
#ifndef _NBL_ASSET_IMPL_S_LOAD_SESSION_H_INCLUDED_
#define _NBL_ASSET_IMPL_S_LOAD_SESSION_H_INCLUDED_

#include "SFileAccess.h"
#include "SIODiagnostics.h"

#include <string>


namespace nbl::asset::impl
{

class SLoadSession
{
	public:
		system::IFile* file = nullptr;
		const SFileIOPolicy* requestedPolicy = nullptr;
		SResolvedFileIOPolicy ioPlan = {};
		uint64_t payloadBytes = 0ull;
		const char* owner = nullptr;
		std::string fileName = {};

		template<typename Logger>
		static inline bool begin(Logger& logger, const char* const owner, system::IFile* file, const SFileIOPolicy& ioPolicy, const uint64_t payloadBytes, const bool sizeKnown, SLoadSession& out)
		{
			out = {};
			if (!file)
				return false;

			out.file = file;
			out.requestedPolicy = &ioPolicy;
			out.ioPlan = SFileAccess::resolvePlan(ioPolicy, payloadBytes, sizeKnown, file);
			out.payloadBytes = payloadBytes;
			out.owner = owner;
			out.fileName = file->getFileName().string();
			return !SIODiagnostics::logInvalidPlan(logger, owner, out.fileName.c_str(), out.ioPlan);
		}

		inline bool isWholeFile() const
		{
			return ioPlan.strategy == SResolvedFileIOPolicy::Strategy::WholeFile;
		}

		inline const uint8_t* mappedPointer() const
		{
			if (!file || !isWholeFile())
				return nullptr;
			return reinterpret_cast<const uint8_t*>(static_cast<const system::IFile*>(file)->getMappedPointer());
		}

		inline const uint8_t* readRange(const size_t offset, const size_t bytes, core::vector<uint8_t>& storage, SFileReadTelemetry* const ioTelemetry = nullptr, const bool zeroTerminate = false) const
		{
			return SFileAccess::readRange(file, offset, bytes, storage, ioPlan, ioTelemetry, zeroTerminate);
		}

		inline const uint8_t* mapOrReadWholeFile(core::vector<uint8_t>& storage, SFileReadTelemetry* const ioTelemetry = nullptr, bool* const wasMapped = nullptr, const bool zeroTerminate = false) const
		{
			return SFileAccess::mapOrReadWholeFile(file, static_cast<size_t>(payloadBytes), storage, ioPlan, ioTelemetry, wasMapped, zeroTerminate);
		}

		template<typename Logger, typename Telemetry>
		inline void logTinyIO(Logger& logger, const Telemetry& telemetry, const char* const opName = "reads") const
		{
			if (!requestedPolicy)
				return;
			SIODiagnostics::logTinyIO(logger, owner, fileName.c_str(), telemetry, payloadBytes, *requestedPolicy, opName);
		}
};

}

#endif
