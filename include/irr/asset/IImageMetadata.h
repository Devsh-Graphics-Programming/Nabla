#ifndef __IRR_I_IMAGE_METADATA_H_INCLUDED__
#define __IRR_I_IMAGE_METADATA_H_INCLUDED__

#include "irr/asset/IAssetMetadata.h"
#include "irr/asset/EColorSpace.h"

namespace irr
{
	namespace asset
	{
		//! A class to derive loader-specific image metadata objects from
		/**
			Images may sometimes require external inputs from outside of the resourced they were built with, for total flexibility
			we cannot standardise "conventions" of each image inputs,

			but we can provide useful metadata from the loader.
		*/
		class IImageMetadata : public IAssetMetadata
		{
			public:
				struct ColorSemantic
				{
					E_COLOR_PRIMARIES colorSpace;
					ELECTRO_OPTICAL_TRANSFER_FUNCTION transferFunction;
				};

                //! Returns the exact (or guessed) color semantic of the pixel data stored
				const ColorSemantic& getColorSemantic() const { return colorSemantic; }

			protected:
				inline IImageMetadata(const ColorSemantic& _colorSemantic) : colorSemantic(_colorSemantic) {}
				virtual ~IImageMetadata() = default;

				ColorSemantic colorSemantic;
		};
	}
}

#endif // __IRR_I_IMAGE_METADATA_H_INCLUDED__
