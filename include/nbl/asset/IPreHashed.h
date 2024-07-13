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

	protected:
		inline IPreHashed() = default;
		virtual inline ~IPreHashed() = default;

	private:
		core::blake3_hash_t m_contentHash = {};
};
}

#endif
