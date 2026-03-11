// Internal src-only header. Do not include from public headers.
#ifndef _NBL_ASSET_IMPL_S_FILE_ACCESS_H_INCLUDED_
#define _NBL_ASSET_IMPL_S_FILE_ACCESS_H_INCLUDED_
#include "nbl/core/declarations.h"
#include "nbl/asset/interchange/SInterchangeIO.h"
#include "nbl/system/ILogger.h"
#include <string>
namespace nbl::asset::impl
{
//! Small file access helper shared by interchange loaders.
class SFileAccess
{
	public:
		//! Returns true when the file exposes a mapped pointer.
		static inline bool isMappable(const system::IFile* file) { return file && core::bitflag<system::IFile::E_CREATE_FLAGS>(file->getFlags()).hasAnyFlag(system::IFile::ECF_MAPPABLE); }
		//! Resolves the final IO plan after considering payload size and mapping support.
		static inline SResolvedFileIOPolicy resolvePlan(const SFileIOPolicy& ioPolicy, const uint64_t payloadBytes, const bool sizeKnown, const system::IFile* file) { return SResolvedFileIOPolicy(ioPolicy, payloadBytes, sizeKnown, isMappable(file)); }

		//! Logs an invalid plan and returns true when the caller should abort.
		template<typename Logger>
		static inline bool logInvalidPlan(Logger& logger, const char* const owner, const char* const fileName, const SResolvedFileIOPolicy& ioPlan)
		{
			if (ioPlan.isValid())
				return false;
			logger.log("%s: invalid io policy for %s reason=%s", system::ILogger::ELL_ERROR, owner, fileName, ioPlan.reason);
			return true;
		}

		//! Emits the shared tiny-IO warning when telemetry indicates suspiciously small reads.
		template<typename Logger>
		static inline void logTinyIO(Logger& logger, const char* const owner, const char* const fileName, const SInterchangeIO::STelemetry& telemetry, const uint64_t payloadBytes, const SFileIOPolicy& ioPolicy, const char* const opName)
		{
			if (!SInterchangeIO::isTinyIOTelemetryLikely(telemetry, payloadBytes, ioPolicy))
				return;
			logger.log("%s tiny-io guard: file=%s %s=%llu min=%llu avg=%llu", system::ILogger::ELL_WARNING, owner, fileName, opName, static_cast<unsigned long long>(telemetry.callCount), static_cast<unsigned long long>(telemetry.getMinOrZero()), static_cast<unsigned long long>(telemetry.getAvgOrZero()));
		}
		//! Reads one byte range using the already resolved IO plan.
		static inline const uint8_t* readRange(system::IFile* file, const size_t offset, const size_t bytes, core::vector<uint8_t>& storage, const SResolvedFileIOPolicy& ioPlan, SFileReadTelemetry* ioTelemetry = nullptr, const bool zeroTerminate = false)
		{
			storage.resize(bytes + (zeroTerminate ? 1ull : 0ull), 0u);
			if (!SInterchangeIO::readFileWithPolicy(file, storage.data(), offset, bytes, ioPlan, ioTelemetry))
				return nullptr;
			if (zeroTerminate)
				storage[bytes] = 0u;
			return storage.data();
		}
		//! Uses the mapped pointer for whole-file mode when available, otherwise falls back to `readRange`.
		static inline const uint8_t* mapOrReadWholeFile(system::IFile* file, const size_t bytes, core::vector<uint8_t>& storage, const SResolvedFileIOPolicy& ioPlan, SFileReadTelemetry* ioTelemetry = nullptr, bool* wasMapped = nullptr, const bool zeroTerminate = false)
		{
			if (wasMapped)
				*wasMapped = false;
			if (ioPlan.strategy == SResolvedFileIOPolicy::Strategy::WholeFile)
			{
				const auto* mapped = reinterpret_cast<const uint8_t*>(static_cast<const system::IFile*>(file)->getMappedPointer());
				if (mapped)
				{
					if (ioTelemetry) ioTelemetry->account(bytes);
					if (wasMapped) *wasMapped = true;
					return mapped;
				}
			}
			return readRange(file, 0ull, bytes, storage, ioPlan, ioTelemetry, zeroTerminate);
		}
};
//! Per-load session state shared across the loader entry points.
class SLoadSession
{
	public:
		system::IFile* file = nullptr; //!< File being processed by the loader.
		const SFileIOPolicy* requestedPolicy = nullptr; //!< Original policy requested by the caller.
		SResolvedFileIOPolicy ioPlan = {}; //!< Final plan chosen for this payload.
		uint64_t payloadBytes = 0ull; //!< Logical payload size covered by `ioPlan`.
		const char* owner = nullptr; //!< Human-readable loader name used in logs.
		std::string fileName = {}; //!< Cached file name used in diagnostics.

		//! Initializes the session and resolves the IO plan.
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
			return !SFileAccess::logInvalidPlan(logger, owner, out.fileName.c_str(), out.ioPlan);
		}
		//! Returns true when the resolved plan prefers whole-file access.
		inline bool isWholeFile() const { return ioPlan.strategy == SResolvedFileIOPolicy::Strategy::WholeFile; }
		//! Returns the mapped pointer for whole-file mode or `nullptr` when unavailable.
		inline const uint8_t* mappedPointer() const { return file && isWholeFile() ? reinterpret_cast<const uint8_t*>(static_cast<const system::IFile*>(file)->getMappedPointer()) : nullptr; }
		//! Convenience wrapper over `SFileAccess::readRange` bound to this session.
		inline const uint8_t* readRange(const size_t offset, const size_t bytes, core::vector<uint8_t>& storage, SFileReadTelemetry* const ioTelemetry = nullptr, const bool zeroTerminate = false) const { return SFileAccess::readRange(file, offset, bytes, storage, ioPlan, ioTelemetry, zeroTerminate); }
		//! Convenience wrapper over `SFileAccess::mapOrReadWholeFile` bound to this session.
		inline const uint8_t* mapOrReadWholeFile(core::vector<uint8_t>& storage, SFileReadTelemetry* const ioTelemetry = nullptr, bool* const wasMapped = nullptr, const bool zeroTerminate = false) const { return SFileAccess::mapOrReadWholeFile(file, static_cast<size_t>(payloadBytes), storage, ioPlan, ioTelemetry, wasMapped, zeroTerminate); }

		//! Emits the shared tiny-IO diagnostic for this session.
		template<typename Logger, typename Telemetry>
		inline void logTinyIO(Logger& logger, const Telemetry& telemetry, const char* const opName = "reads") const { if (requestedPolicy) SFileAccess::logTinyIO(logger, owner, fileName.c_str(), telemetry, payloadBytes, *requestedPolicy, opName); }
};
}
#endif
