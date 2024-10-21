// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_E_COLOR_SPACE_H_INCLUDED_
#define _NBL_ASSET_E_COLOR_SPACE_H_INCLUDED_

// TODO: move to HLSL and turn into enum classes (when 8bit int comes)
namespace nbl::asset
{
//! Specifies a color space of an image
enum E_COLOR_PRIMARIES : uint8_t
{
    //! Specifies support for the sRGB color space. The primaries are the same for scRGB and BT709, only EOTFs differ.
    ECP_SRGB,

    //! Specifies support for the Display-P3 color space to be displayed using an sRGB-like EOTF.
    ECP_DISPLAY_P3,

    //! Specifies support for the DCI-P3 color space to be displayed using the DCI-P3 EOTF. Note that values in such an image are interpreted as XYZ encoded color data by the presentation engine.
    ECP_DCI_P3,

    //! Specifies support for the BT2020 color space to be displayed using a linear EOTF. Same primaries are used for HDR10 and DolbyVision
    ECP_BT2020,

    //! Specifies support for the AdobeRGB color space to be displayed.
    ECP_ADOBERGB,

    //! The reference ACES color space, not really supported by any graphics API for display/swapchain
    ECP_ACES,

    //! The slightly different primary space for ACES with quantization (ACEScc and ACEScct use same primaries)
    ECP_ACES_CC_T,

    //! Specifies that color components are used �as is�. This is intended to allow applications to supply data for color spaces not described here.
    ECP_PASS_THROUGH,

    //! For internal 
    ECP_COUNT
};

//! Data to linear value for images
enum ELECTRO_OPTICAL_TRANSFER_FUNCTION : uint8_t
{
    EOTF_IDENTITY,
    EOTF_sRGB,
    EOTF_DCI_P3_XYZ,
    EOTF_SMPTE_170M,
    EOTF_SMPTE_ST2084,
    EOTF_HDR10_HLG,
    EOTF_GAMMA_2_2,
    EOTF_ACEScc,
    EOTF_ACEScct,

    EOTF_UNKNOWN
};

//! Linear value to data for displays and swapchains
enum OPTICO_ELECTRICAL_TRANSFER_FUNCTION : uint8_t
{
    OETF_IDENTITY,
    OETF_sRGB,
    OETF_DCI_P3_XYZ,
    OETF_SMPTE_170M,
    OETF_SMPTE_ST2084,
    OETF_HDR10_HLG,
    OETF_GAMMA_2_2,
    OETF_ACEScc,
    OETF_ACEScct,

    OETF_UNKNOWN
};

static_assert(EOTF_UNKNOWN == static_cast<ELECTRO_OPTICAL_TRANSFER_FUNCTION>(OETF_UNKNOWN), "Definitions of transfer functions don't match");
}

#endif
