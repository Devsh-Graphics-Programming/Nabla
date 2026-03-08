// Internal src-only header.
// Do not include from public headers.
#ifndef _NBL_ASSET_IMPL_S_FILE_ACCESS_H_INCLUDED_
#define _NBL_ASSET_IMPL_S_FILE_ACCESS_H_INCLUDED_

#include "nbl/core/declarations.h"
#include "nbl/asset/interchange/SInterchangeIO.h"


namespace nbl::asset::impl
{

class SFileAccess
{
	public:
		static inline bool isMappable(const system::IFile* file)
		{
			return file && core::bitflag<system::IFile::E_CREATE_FLAGS>(file->getFlags()).hasAnyFlag(system::IFile::ECF_MAPPABLE);
		}

		static inline SResolvedFileIOPolicy resolvePlan(const SFileIOPolicy& ioPolicy, const uint64_t payloadBytes, const bool sizeKnown, const system::IFile* file)
		{
			return SResolvedFileIOPolicy(ioPolicy, payloadBytes, sizeKnown, isMappable(file));
		}

		static inline const uint8_t* readRange(system::IFile* file, const size_t offset, const size_t bytes, core::vector<uint8_t>& storage, const SResolvedFileIOPolicy& ioPlan, SFileReadTelemetry* ioTelemetry = nullptr, const bool zeroTerminate = false)
		{
			storage.resize(bytes + (zeroTerminate ? 1ull : 0ull), 0u);
			if (!SInterchangeIO::readFileWithPolicy(file, storage.data(), offset, bytes, ioPlan, ioTelemetry))
				return nullptr;
			if (zeroTerminate)
				storage[bytes] = 0u;
			return storage.data();
		}

		static inline const uint8_t* mapOrReadWholeFile(system::IFile* file, const size_t bytes, core::vector<uint8_t>& storage, const SResolvedFileIOPolicy& ioPlan, SFileReadTelemetry* ioTelemetry = nullptr, bool* wasMapped = nullptr, const bool zeroTerminate = false)
		{
			if (wasMapped)
				*wasMapped = false;

			if (ioPlan.strategy == SResolvedFileIOPolicy::Strategy::WholeFile)
			{
				const auto* mapped = reinterpret_cast<const uint8_t*>(static_cast<const system::IFile*>(file)->getMappedPointer());
				if (mapped)
				{
					if (ioTelemetry)
						ioTelemetry->account(bytes);
					if (wasMapped)
						*wasMapped = true;
					return mapped;
				}
			}

			return readRange(file, 0ull, bytes, storage, ioPlan, ioTelemetry, zeroTerminate);
		}
};

}

#endif
