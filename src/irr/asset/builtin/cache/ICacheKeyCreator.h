// Copyright (C) 2020 AnastaZIuk
// This file is part of the "Nabla Engine"
// For conditions of distribution and use, see copyright notice in LICENSE.md

#ifndef __I_CACHE_KEY_CREATOR_H_INCLUDED__
#define __I_CACHE_KEY_CREATOR_H_INCLUDED__

#include "irr/asset/IAsset.h"

namespace irr
{
	namespace asset
	{
		class ICacheKeyCreator
		{
			public:
				std::string getCacheKey() { return cacheKey; }
				virtual bool createCacheKey(core::smart_refctd_ptr<IAsset> asset) = 0;

				template<typename Type>
				_IRR_STATIC_INLINE std::string getNewCommmaValue(const Type& value)
				{
					static_assert(std::numeric_limits<Type>::is_integer || std::is_enum<Type>::type, "The type must be an interger or enum!");

					if constexpr (std::numeric_limits<Type>::is_integer)
						return "_" + std::to_string(value);
					else if (std::is_enum<Type>::type)
						return "_" + std::to_string(static_cast<size_t>(value));
				}

			protected:
				std::string cacheKey;
		};
	}
}

#endif // __I_CACHE_KEY_CREATOR_H_INCLUDED__
