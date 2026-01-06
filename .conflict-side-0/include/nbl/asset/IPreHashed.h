// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_PRE_HASHED_H_INCLUDED_
#define _NBL_ASSET_I_PRE_HASHED_H_INCLUDED_

#include "nbl/core/hash/blake.h"
#include "nbl/asset/IAsset.h"

namespace nbl::asset
{
//! Sometimes an asset is too complex or big to be hashed, so we need a hash to be set explicitly.
//! Meant to be inherited from in conjunction with `IAsset`
class IPreHashed : public IAsset
{
	public:
		constexpr static inline core::blake3_hash_t INVALID_HASH = { 0xaf,0x13,0x49,0xb9,0xf5,0xf9,0xa1,0xa6,0xa0,0x40,0x4d,0xea,0x36,0xdc,0xc9,0x49,0x9b,0xcb,0x25,0xc9,0xad,0xc1,0x12,0xb7,0xcc,0x9a,0x93,0xca,0xe4,0x1f,0x32,0x62 };
		//
		inline const core::blake3_hash_t& getContentHash() const {return m_contentHash;}
		//
		inline void setContentHash(const core::blake3_hash_t& hash)
		{
			if (!isMutable())
				return;
			m_contentHash = hash;
		}

		//
		virtual core::blake3_hash_t computeContentHash() const = 0;

		// One can free up RAM by discarding content, but keep the pointers and content hash around.
		// This is a good alternative to simply ejecting assets from the <path,asset> cache as it prevents repeated loads.
		// And you can still hash the asset DAG and find your already converted GPU objects.
		// NOTE: `missingContent` is only about this DAG node!
		virtual bool missingContent() const = 0;
		inline void discardContent()
		{
			if (isMutable() && !missingContent())
				discardContent_impl();
		}

        static inline void discardDependantsContents(const std::span<IAsset* const> roots)
        {
            core::vector<IAsset*> stack;
            core::unordered_set<IAsset*> alreadyVisited; // whether we have push the node to the stack
            auto push = [&stack,&alreadyVisited](IAsset* node) -> bool
            {
                const auto [dummy,inserted] = alreadyVisited.insert(node);
                if (inserted)
                    stack.push_back(node);
                return true;
            };
            for (const auto& root : roots)
                push(root);
            while (!stack.empty())
            {
                auto* entry = stack.back();
                stack.pop_back();
                entry->visitDependents(push);
                // pre order traversal does discard
                auto* isPrehashed = dynamic_cast<IPreHashed*>(entry);
                if (isPrehashed)
                    isPrehashed->discardContent();
            }
        }
        static inline bool anyDependantDiscardedContents(const IAsset* root)
        {
            core::vector<const IAsset*> stack;
            core::unordered_set<const IAsset*> alreadyVisited; // whether we have push the node to the stack
            bool result = false;
            auto push = [&stack,&alreadyVisited,&result](const IAsset* node) -> bool
            {
                const auto [dummy,inserted] = alreadyVisited.insert(node);
                if (inserted)
                {
                    auto* isPrehashed = dynamic_cast<const IPreHashed*>(node);
                    if (isPrehashed && isPrehashed->missingContent())
                    {
                        stack.clear();
                        result = true;
                        return false;
                    }
                    stack.push_back(node);
                }
                return true;
            };
            if (!push(root))
                return true;
            while (!stack.empty())
            {
                auto* entry = stack.back();
                stack.pop_back();
                entry->visitDependents(push);
            }
            return result;
        }

	protected:
		inline IPreHashed() = default;
		virtual inline ~IPreHashed() = default;

		virtual void discardContent_impl() = 0;

	private:
		// The initial value is a hash of an "as if" of a zero-length array
		core::blake3_hash_t m_contentHash = static_cast<core::blake3_hash_t>(core::blake3_hasher{});
};
}

#endif
