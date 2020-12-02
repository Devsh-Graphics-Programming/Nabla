// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_C_MITSUBA_METADATA_H_INCLUDED__
#define __NBL_C_MITSUBA_METADATA_H_INCLUDED__

#include "nbl/asset/IAssetMetadata.h"
#include "nbl/ext/MitsubaLoader/CGlobalMitsubaMetadata.h"

namespace irr
{
	namespace ext
	{
		namespace MitsubaLoader
		{
			//! A class to derive mitsuba mesh loader metadata objects from

			class CMitsubaMetadata : public asset::IAssetMetadata
			{
				public:

					CMitsubaMetadata(core::smart_refctd_ptr<CGlobalMitsubaMetadata> _mitsubaMetadata) : mitsubaMetadata(std::move(_mitsubaMetadata)) {}

					auto getMitsubaMetadata() const { return mitsubaMetadata; }

					_NBL_STATIC_INLINE_CONSTEXPR const char* loaderName = "CMitsubaLoader";
					const char* getLoaderName() const override { return loaderName; }

				private:
					core::smart_refctd_ptr<CGlobalMitsubaMetadata> mitsubaMetadata;
			};
		}
	}
}

#endif
