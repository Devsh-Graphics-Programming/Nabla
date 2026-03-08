// Internal src-only header.
// Do not include from public headers.
#ifndef _NBL_ASSET_IMPL_S_CONTENT_HASH_BUILD_H_INCLUDED_
#define _NBL_ASSET_IMPL_S_CONTENT_HASH_BUILD_H_INCLUDED_

#include "nbl/core/declarations.h"
#include "nbl/asset/ICPUBuffer.h"

#include <thread>


namespace nbl::asset::impl
{

	class SContentHashBuild
{
	public:
		bool enabled = false;
		bool inlineHash = false;
		core::vector<core::smart_refctd_ptr<ICPUBuffer>> hashedBuffers = {};
		std::jthread deferredThread = {};

		static inline SContentHashBuild create(const bool enabled, const bool inlineHash)
		{
			return {.enabled = enabled, .inlineHash = inlineHash};
		}

		inline bool hashesInline() const
		{
			return enabled && inlineHash;
		}

		inline bool hashesDeferred() const
		{
			return enabled && !inlineHash;
		}

		inline void hashNow(ICPUBuffer* const buffer)
		{
			if (!hashesInline() || !buffer)
				return;
			if (buffer->getContentHash() != IPreHashed::INVALID_HASH)
				return;
			for (const auto& hashed : hashedBuffers)
				if (hashed.get() == buffer)
					return;
			buffer->setContentHash(buffer->computeContentHash());
			hashedBuffers.push_back(core::smart_refctd_ptr<ICPUBuffer>(buffer));
		}

		inline void tryDefer(ICPUBuffer* const buffer)
		{
			if (!hashesDeferred() || !buffer)
				return;
			if (deferredThread.joinable())
				return;
			if (buffer->getContentHash() != IPreHashed::INVALID_HASH)
				return;
			auto keepAlive = core::smart_refctd_ptr<ICPUBuffer>(buffer);
			deferredThread = std::jthread([buffer = std::move(keepAlive)]() mutable {buffer->setContentHash(buffer->computeContentHash());});
		}

		inline void wait()
		{
			if (deferredThread.joinable())
				deferredThread.join();
		}
};

}

#endif
