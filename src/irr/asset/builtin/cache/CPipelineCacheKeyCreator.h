// Copyright (C) 2020 AnastaZIuk
// This file is part of the "Nabla Engine"
// For conditions of distribution and use, see copyright notice in LICENSE.md

#ifndef __C_PIPELINE_CACHE_KEY_CREATOR_H_INCLUDED__
#define __C_PIPELINE_CACHE_KEY_CREATOR_H_INCLUDED__

#include "irr/asset/builtin/cache/ICacheKeyCreator.h"

namespace irr
{
	namespace asset
	{
		template<typename CPipeline>
		class CPipelineCacheKeyCreator : public ICacheKeyCreator
		{
		public:
			bool createCacheKey(core::smart_refctd_ptr<IAsset> asset) override final;
		};
	}
}

#endif // __C_PIPELINE_CACHE_KEY_CREATOR_H_INCLUDED__