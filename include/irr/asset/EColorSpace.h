#ifndef __IRR_E_COLOR_SPACE_H_INCLUDED__
#define __IRR_E_COLOR_SPACE_H_INCLUDED__

namespace irr
{
	namespace asset
	{
        //! Specifies a color space of an image
        enum E_COLOR_SPACE
        {
            //! Specifies support for the sRGB color space.
            ECS_SRGB,

            //! Specifies support for the Display-P3 color space to be displayed using an sRGB-like EOTF.
            ECS_DISPLAY_P3,

            //! Specifies support for the DCI-P3 color space to be displayed using the DCI-P3 EOTF. Note that values in such an image are interpreted as XYZ encoded color data by the presentation engine.
            ECS_DCI_P3,

            //! Specifies support for the BT709 color space to be displayed using a linear EOTF.
            ECS_BT709,

            //! Specifies support for the BT2020 color space to be displayed using a linear EOTF.
            ECS_BT2020,

            //! Specifies support for the HDR10 (BT2020 color) space to be displayed using the SMPTE ST2084 Perceptual Quantizer (PQ) EOTF.
            ECS_HDR10_ST2084,

            //! Specifies support for the Dolby Vision (BT2020 color space), proprietary encoding, to be displayed using the SMPTE ST2084 EOTF.
            ECS_DOLBYVISION,

            //! Specifies support for the HDR10 (BT2020 color space) to be displayed using the Hybrid Log Gamma (HLG) EOTF.
            ECS_HDR10_HLG,

            //! Specifies support for the AdobeRGB color space to be displayed.
            ECS_ADOBERGB,

            //! Specifies that color components are used �as is�. This is intended to allow applications to supply data for color spaces not described here.
            ECS_PASS_THROUGH,

            //! Specifies support for the display�s native color space. This matches the color space expectations of AMD�s FreeSync2 standard, for displays supporting it.
            ECS_DISPLAY_NATIVE_AMD,

            //! For internal 
            ECS_COUNT
        };

        //! Data to linear value for images
        enum ELECTRO_OPTICAL_TRANSFER_FUNCTION
        {
            EOTF_IDENTITY,
            EOTF_sRGB,
            EOTF_DCI_P3_XYZ,
            EOTF_SMPTE_170M,
            EOTF_SMPTE_ST2084,
            EOTF_HDR10_HLG,
            EOTF_GAMMA_2_2,

            EOTF_UNKNOWN
        };

        //! Linear value to data for displays and swapchains
        enum OPTICO_ELECTRICAL_TRANSFER_FUNCTION
        {
            OETF_IDENTITY,
            OETF_sRGB,
            OETF_DCI_P3_XYZ,
            OETF_SMPTE_170M,
            OETF_SMPTE_ST2084,
            OETF_HDR10_HLG,
            OETF_GAMMA_2_2,

            OETF_UNKNOWN
        };

        static_assert(EOTF_UNKNOWN == static_cast<ELECTRO_OPTICAL_TRANSFER_FUNCTION>(OETF_UNKNOWN), "Definitions of transfer functions don't match");
	}
}

#endif //__IRR_E_COLOR_SPACE_H_INCLUDED__
