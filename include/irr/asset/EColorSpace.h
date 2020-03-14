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
            ECS_SRGB_NONLINEAR_KHR,

            //! Specifies support for the Display-P3 color space to be displayed using an sRGB-like EOTF.
            ECS_DISPLAY_P3_NONLINEAR_EXT,

            //! Specifies support for the extended sRGB color space to be displayed using a linear EOTF.
            ECS_EXTENDED_SRGB_LINEAR_EXT,

            //! Specifies support for the Display-P3 color space to be displayed using a linear EOTF.
            ECS_DISPLAY_P3_LINEAR_EXT,

            //! Specifies support for the DCI-P3 color space to be displayed using the DCI-P3 EOTF. Note that values in such an image are interpreted as XYZ encoded color data by the presentation engine.
            ECS_DCI_P3_NONLINEAR_EXT,

            //! Specifies support for the BT709 color space to be displayed using a linear EOTF.
            ECS_BT709_LINEAR_EXT,

            //! Specifies support for the BT709 color space to be displayed using the SMPTE 170M EOTF.
            ECS_BT709_NONLINEAR_EXT,

            //! Specifies support for the BT2020 color space to be displayed using a linear EOTF.
            ECS_BT2020_LINEAR_EXT,

            //! Specifies support for the HDR10 (BT2020 color) space to be displayed using the SMPTE ST2084 Perceptual Quantizer (PQ) EOTF.
            ECS_HDR10_ST2084_EXT,

            //! Specifies support for the Dolby Vision (BT2020 color space), proprietary encoding, to be displayed using the SMPTE ST2084 EOTF.
            ECS_DOLBYVISION_EXT,

            //! Specifies support for the HDR10 (BT2020 color space) to be displayed using the Hybrid Log Gamma (HLG) EOTF.
            ECS_HDR10_HLG_EXT,

            //! Specifies support for the AdobeRGB color space to be displayed using a linear EOTF.
            ECS_ADOBERGB_LINEAR_EXT,

            //! Specifies support for the AdobeRGB color space to be displayed using the Gamma 2.2 EOTF.
            ECS_ADOBERGB_NONLINEAR_EXT,

            //! Specifies that color components are used “as is”. This is intended to allow applications to supply data for color spaces not described here.
            ECS_PASS_THROUGH_EXT,

            //! Specifies support for the display’s native color space. This matches the color space expectations of AMD’s FreeSync2 standard, for displays supporting it.
            ECS_DISPLAY_NATIVE_AMD,

            //! For internal 
            ECS_COUNT
        };

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
	}
}

#endif //__IRR_E_COLOR_SPACE_H_INCLUDED__
