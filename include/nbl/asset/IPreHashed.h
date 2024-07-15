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
		virtual bool missingContent() const = 0;
		inline void discardContent()
		{
			if (isMutable() && !missingContent())
				discardContent_impl();
		}

		static inline void discardDependantsContents(IAsset* root)
		{
			struct stack_entry_t
			{
				IAsset* asset;
				size_t childrenVisited = 0;
			};
			core::stack<stack_entry_t> stack;
			stack.push({.asset=root});
			while (!stack.empty())
			{
				auto& entry = stack.top();
				if (entry.childrenVisited<entry.asset->getDependantCount())
				{
					auto* isPrehashed = dynamic_cast<IPreHashed*>(entry.asset);
					if (isPrehashed)
						isPrehashed->discardContent();
					stack.pop();
				}
				else
					stack.push({entry.asset->getDependant(entry.childrenVisited++)});
			}
		}
		static inline bool anyDependantDiscardedContents(IAsset* root)
		{
			struct stack_entry_t
			{
				IAsset* asset;
				size_t childrenVisited = 0;
			};
			core::stack<stack_entry_t> stack;
			stack.push({.asset=root});
			while (!stack.empty())
			{
				auto& entry = stack.top();
				if (entry.childrenVisited<entry.asset->getDependantCount())
				{
					auto* isPrehashed = dynamic_cast<IPreHashed*>(entry.asset);
					if (isPrehashed->missingContent())
						return true;
					stack.pop();
				}
				else
					stack.push({entry.asset->getDependant(entry.childrenVisited++)});
			}
			return false;
		}

	protected:
		inline IPreHashed() = default;
		virtual inline ~IPreHashed() = default;

		virtual void discardContent_impl() = 0;

	private:
		core::blake3_hash_t m_contentHash = {};
};
}

#endif
