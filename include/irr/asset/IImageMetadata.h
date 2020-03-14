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
			protected:
				virtual ~IImageMetadata() = default;

			public:

                #include "irr/irrpack.h"
                struct TransferFunction
                {
                    ELECTRO_OPTICAL_TRANSFER_FUNCTION eotf;
                } PACK_STRUCT;

                struct ImageInputSemantic
                {
                    E_COLOR_SPACE colorSpace;
                    TransferFunction transferFunction;
                } PACK_STRUCT;
                #include "irr/irrunpack.h"

                //! Returns list of "standard semenatics" as in the list of required inputs with meanings that are common in many images
                virtual core::SRange<const ImageInputSemantic> getCommonRequiredInputs() const = 0;
		};
	}
}

#endif // __IRR_I_IMAGE_METADATA_H_INCLUDED__
