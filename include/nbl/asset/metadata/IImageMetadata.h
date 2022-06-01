// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_I_IMAGE_METADATA_H_INCLUDED_
#define _NBL_ASSET_I_IMAGE_METADATA_H_INCLUDED_

#include "nbl/asset/ICPUImage.h"
#include "nbl/asset/format/EColorSpace.h"

namespace nbl::asset
{

//! A class to derive loader-specific image metadata objects from
/**
	Images may sometimes require external inputs from outside of the resourced they were built with, for total flexibility
	we cannot standardise "conventions" of each image inputs,

	but we can provide useful metadata from the loader.
*/
class NBL_API IImageMetadata : public core::Interface
{
	public:
		struct ColorSemantic
		{
			E_COLOR_PRIMARIES colorSpace;
			ELECTRO_OPTICAL_TRANSFER_FUNCTION transferFunction;

			auto operator<=>(const ColorSemantic&) const = default;
		};

		inline IImageMetadata() : colorSemantic{ECP_COUNT,EOTF_UNKNOWN} {}
		inline IImageMetadata(const ColorSemantic& _colorSemantic) : colorSemantic(_colorSemantic) {}

		ColorSemantic colorSemantic;

		inline bool operator!=(const IImageMetadata& other) const
		{
			return colorSemantic != other.colorSemantic;
		}

	protected:
		virtual ~IImageMetadata() = default;

		inline IImageMetadata& operator=(IImageMetadata&& other)
		{
			std::swap(colorSemantic,other.colorSemantic);
			return *this;
		}
};

}

#endif
