// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_COLOR_SPACE_COLORSPACE_INCLUDED_
#define _NBL_BUILTIN_HLSL_COLOR_SPACE_COLORSPACE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
namespace colorspace
{

struct colorspace_base
{
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_R = 580.0f;
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_G = 550.0f;
    NBL_CONSTEXPR_STATIC_INLINE float32_t wavelength_B = 450.0f;
};

struct scRGB : colorspace_base
{
    static float32_t3x3 FromXYZ()
    {
        return float32_t3x3(
            float32_t3( 3.240970f, -1.537383f, -0.498611f),
            float32_t3(-0.969244f,  1.875968f,  0.041555f),
            float32_t3( 0.055630f, -0.203977f,  1.056972f)
        );
    }
    static float32_t3 FromXYZ(float32_t3 val) { return hlsl::mul(FromXYZ(), val); }

    static float32_t3x3 ToXYZ()
    {
        return float32_t3x3(
            float32_t3(0.412391f, 0.357584f, 0.180481f),
            float32_t3(0.212639f, 0.715169f, 0.072192f),
            float32_t3(0.019331f, 0.119195f, 0.950532f)
        );
    }
    static float32_t3 ToXYZ(float32_t3 val) { return hlsl::mul(ToXYZ(), val); }
};

struct sRGB : scRGB {};
struct BT709 : scRGB {};

struct Display_P3 : colorspace_base
{
    static float32_t3x3 FromXYZ()
    {
        return float32_t3x3(
            float32_t3( 2.4934969119f, -0.9313836179f, -0.4027107845f),
            float32_t3(-0.8294889696f,  1.7626640603f,  0.0236246858f),
            float32_t3( 0.0358458302f, -0.0761723893f,  0.9568845240f)
        );
    }
    static float32_t3 FromXYZ(float32_t3 val) { return hlsl::mul(FromXYZ(), val); }

    static float32_t3x3 ToXYZ()
    {
        return float32_t3x3(
            float32_t3(0.4865709486f, 0.2656676932f, 0.1982172852f),
            float32_t3(0.2289745641f, 0.6917385218f, 0.0792869141f),
            float32_t3(0.0000000000f, 0.0451133819f, 1.0439443689f)
        );
    }
    static float32_t3 ToXYZ(float32_t3 val) { return hlsl::mul(ToXYZ(), val); }
};

struct DCI_P3 : colorspace_base
{
    static float32_t3x3 FromXYZ()
    {
        return float32_t3x3(
            float32_t3(1.0f, 0.0f, 0.0f),
            float32_t3(0.0f, 1.0f, 0.0f),
            float32_t3(0.0f, 0.0f, 1.0f)
        );
    }
    static float32_t3 FromXYZ(float32_t3 val) { return hlsl::mul(FromXYZ(), val); }

    static float32_t3x3 ToXYZ()
    {
        return float32_t3x3(
            float32_t3(1.0f, 0.0f, 0.0f),
            float32_t3(0.0f, 1.0f, 0.0f),
            float32_t3(0.0f, 0.0f, 1.0f)
        );
    }
    static float32_t3 ToXYZ(float32_t3 val) { return hlsl::mul(ToXYZ(), val); }
};

struct BT2020 : colorspace_base
{
    static float32_t3x3 FromXYZ()
    {
        return float32_t3x3(
            float32_t3( 1.716651f, -0.355671f, -0.253366f),
            float32_t3(-0.666684f,  1.616481f,  0.015769f),
            float32_t3( 0.017640f, -0.042771f,  0.942103f)
        );
    }
    static float32_t3 FromXYZ(float32_t3 val) { return hlsl::mul(FromXYZ(), val); }

    static float32_t3x3 ToXYZ()
    {
        return float32_t3x3(
            float32_t3(0.636958f, 0.144617f, 0.168881f),
            float32_t3(0.262700f, 0.677998f, 0.059302f),
            float32_t3(0.000000f, 0.028073f, 1.060985f)
        );
    }
    static float32_t3 ToXYZ(float32_t3 val) { return hlsl::mul(ToXYZ(), val); }
};

struct HDR10_ST2084 : BT2020 {};
struct DOLBYIVISION : BT2020 {};
struct HDR10_HLG : BT2020 {};

struct AdobeRGB : colorspace_base
{
    static float32_t3x3 FromXYZ()
    {
        return float32_t3x3(
            float32_t3( 2.0415879038f, -0.5650069743f, -0.3447313508f),
            float32_t3(-0.9692436363f,  1.8759675015f,  0.0415550574f),
            float32_t3( 0.0134442806f, -0.1183623922f,  1.0151749944f)
        );
    }
    static float32_t3 FromXYZ(float32_t3 val) { return hlsl::mul(FromXYZ(), val); }

    static float32_t3x3 ToXYZ()
    {
        return float32_t3x3(
            float32_t3(0.5766690429f, 0.1855582379f, 0.1882286462f),
            float32_t3(0.2973449753f, 0.6273635663f, 0.0752914585f),
            float32_t3(0.0270313614f, 0.0706888525f, 0.9913375368f)
        );
    }
    static float32_t3 ToXYZ(float32_t3 val) { return hlsl::mul(ToXYZ(), val); }
};

struct ACES2065_1 : colorspace_base
{
    static float32_t3x3 FromXYZ()
    {
        return float32_t3x3(
            float32_t3( 1.0498110175f, 0.0000000000f, -0.0000974845f),
            float32_t3(-0.4959030231f, 1.3733130458f,  0.0982400361f),
            float32_t3( 0.0000000000f, 0.0000000000f,  0.9912520182f)
        );
    }
    static float32_t3 FromXYZ(float32_t3 val) { return hlsl::mul(FromXYZ(), val); }

    static float32_t3x3 ToXYZ()
    {
        return float32_t3x3(
            float32_t3(0.9525523959f, 0.0000000000f,  0.0000936786f),
            float32_t3(0.3439664498f, 0.7281660966f, -0.0721325464f),
            float32_t3(0.0000000000f, 0.0000000000f,  1.0088251844f)
        );
    }
    static float32_t3 ToXYZ(float32_t3 val) { return hlsl::mul(ToXYZ(), val); }
};

struct ACEScc : colorspace_base
{
    static float32_t3x3 FromXYZ()
    {
        return float32_t3x3(
            float32_t3( 1.6410233797f, -0.3248032942f, -0.2364246952f),
            float32_t3(-0.6636628587f,  1.6153315917f,  0.0167563477f),
            float32_t3( 0.0117218943f, -0.0082844420f,  0.9883948585f)
        );
    }
    static float32_t3 FromXYZ(float32_t3 val) { return hlsl::mul(FromXYZ(), val); }

    static float32_t3x3 ToXYZ()
    {
        return float32_t3x3(
            float32_t3( 0.6624541811f, 0.1340042065f, 0.1561876870f),
            float32_t3( 0.2722287168f, 0.6740817658f, 0.0536895174f),
            float32_t3(-0.0055746495f, 0.0040607335f, 1.0103391003f)
        );
    }
    static float32_t3 ToXYZ(float32_t3 val) { return hlsl::mul(ToXYZ(), val); }
};

struct ACEScct : ACEScc {};

}
}
}

#endif
