// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_COLOR_SPACE_COLORSPACE_INCLUDED_
#define _NBL_BUILTIN_HLSL_COLOR_SPACE_COLORSPACE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>
#include <nbl/builtin/hlsl/colorspace/decodeCIEXYZ.hlsl>

namespace nbl
{
namespace hlsl
{
namespace colorspace
{

struct colorspace_base
{
    // default CIE RGB primaries wavelengths
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_R = 700.0f;
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_G = 546.1f;
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_B = 435.8f;
};

struct scRGB : colorspace_base
{
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_R = 611.4f;
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_G = 549.1f;
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_B = 464.2f;

    static float32_t3x3 FromXYZ()
    {
        return XYZtoscRGB;
    }
    static float32_t3 FromXYZ(float32_t3 val) { return hlsl::mul(XYZtoscRGB, val); }

    static float32_t3x3 ToXYZ()
    {
        return scRGBtoXYZ;
    }
    static float32_t3 ToXYZ(float32_t3 val) { return hlsl::mul(scRGBtoXYZ, val); }
};

struct sRGB : scRGB {};
struct BT709 : scRGB {};

struct Display_P3 : colorspace_base
{
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_R = 614.9f;
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_G = 544.2f;
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_B = 464.2f;

    static float32_t3x3 FromXYZ()
    {
        return XYZtoDisplay_P3;
    }
    static float32_t3 FromXYZ(float32_t3 val) { return hlsl::mul(XYZtoDisplay_P3, val); }

    static float32_t3x3 ToXYZ()
    {
        return Display_P3toXYZ;
    }
    static float32_t3 ToXYZ(float32_t3 val) { return hlsl::mul(Display_P3toXYZ, val); }
};

struct DCI_P3 : colorspace_base
{
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_R = 614.9f;
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_G = 544.2f;
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_B = 464.2f;

    static float32_t3x3 FromXYZ()
    {
        return XYZtoDCI_P3;
    }
    static float32_t3 FromXYZ(float32_t3 val) { return hlsl::mul(XYZtoDCI_P3, val); }

    static float32_t3x3 ToXYZ()
    {
        return DCI_P3toXYZ;
    }
    static float32_t3 ToXYZ(float32_t3 val) { return hlsl::mul(DCI_P3toXYZ, val); }
};

struct BT2020 : colorspace_base
{
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_R = 630.0f;
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_G = 532.0f;
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_B = 467.0f;

    static float32_t3x3 FromXYZ()
    {
        return XYZtoBT2020;
    }
    static float32_t3 FromXYZ(float32_t3 val) { return hlsl::mul(XYZtoBT2020, val); }

    static float32_t3x3 ToXYZ()
    {
        return BT2020toXYZ;
    }
    static float32_t3 ToXYZ(float32_t3 val) { return hlsl::mul(BT2020toXYZ, val); }
};

struct HDR10_ST2084 : BT2020 {};
struct DOLBYIVISION : BT2020 {};
struct HDR10_HLG : BT2020 {};

struct AdobeRGB : colorspace_base
{
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_R = 611.4f;
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_G = 534.7f;
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_B = 464.2f;

    static float32_t3x3 FromXYZ()
    {
        return XYZtoAdobeRGB;
    }
    static float32_t3 FromXYZ(float32_t3 val) { return hlsl::mul(XYZtoAdobeRGB, val); }

    static float32_t3x3 ToXYZ()
    {
        return AdobeRGBtoXYZ;
    }
    static float32_t3 ToXYZ(float32_t3 val) { return hlsl::mul(AdobeRGBtoXYZ, val); }
};

struct ACES2065_1 : colorspace_base
{
    static float32_t3x3 FromXYZ()
    {
        return XYZtoACES2065_1;
    }
    static float32_t3 FromXYZ(float32_t3 val) { return hlsl::mul(XYZtoACES2065_1, val); }

    static float32_t3x3 ToXYZ()
    {
        return ACES2065_1toXYZ;
    }
    static float32_t3 ToXYZ(float32_t3 val) { return hlsl::mul(ACES2065_1toXYZ, val); }
};

struct ACEScc : colorspace_base
{
    static float32_t3x3 FromXYZ()
    {
        return XYZtoACEScc;
    }
    static float32_t3 FromXYZ(float32_t3 val) { return hlsl::mul(XYZtoACEScc, val); }

    static float32_t3x3 ToXYZ()
    {
        return ACEScctoXYZ;
    }
    static float32_t3 ToXYZ(float32_t3 val) { return hlsl::mul(ACEScctoXYZ, val); }
};

struct ACEScct : ACEScc {};

}
}
}

#endif
