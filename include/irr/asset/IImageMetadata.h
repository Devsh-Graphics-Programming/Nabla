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

                //! Data to linear value for images
                enum ELECTRO_OPTICAL_TRANSFER_FUNCTION
                {
                    EOTF_sRGB,
                    EOTF_DCI_P3_XYZ,
                    EOTF_SMPTE_170M,
                    EOTF_SMPTE_ST2084_PERCEPTUAL_QUANTIZER,
                    EOTF_SMPTE_ST2084,
                    EOTF_HDR10_HLG,
                    EOTF_GAMMA_2_2,

                    EOTF_UNKNOWN
                };

                //! Linear value to data for displays and swapchains
                enum OPTICO_ELECTRICAL_TRANSFER_FUNCTION 
                {
                    OETF_sRGB,
                    OETF_DCI_P3_XYZ,
                    OETF_SMPTE_170M,
                    OETF_SMPTE_ST2084_PERCEPTUAL_QUANTIZER,
                    OETF_SMPTE_ST2084,
                    OETF_HDR10_HLG,
                    OETF_GAMMA_2_2,

                    OETF_UNKNOWN
                };

                static_assert(EOTF_UNKNOWN == OETF_UNKNOWN, "Definitions of transfer functions don't match");

                #include "irr/irrpack.h"
                struct TransferFunction
                {
                    ELECTRO_OPTICAL_TRANSFER_FUNCTION eotf;
                    OPTICO_ELECTRICAL_TRANSFER_FUNCTION oetf;
                } PACK_STRUCT;

                struct ImageInputSemantic
                {
                    std::string imageName;
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
