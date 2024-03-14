// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_IES_METADATA_H_INCLUDED__
#define __NBL_ASSET_C_IES_METADATA_H_INCLUDED__

#include "nbl/asset/metadata/IAssetMetadata.h"
#include "nbl/asset/utils/CIESProfile.h"

namespace nbl
{
	namespace asset
	{
		class CIESProfileMetadata final : public asset::IAssetMetadata 
		{
			public:
				CIESProfileMetadata(const CIESProfile& _profile)
					: IAssetMetadata(), profile(_profile) {}

				_NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CIESProfileLoader";
				const char* getLoaderName() const override { return LoaderName; }

				const CIESProfile profile;
		};
	}
}

#endif // __NBL_ASSET_C_IES_METADATA_H_INCLUDED__
