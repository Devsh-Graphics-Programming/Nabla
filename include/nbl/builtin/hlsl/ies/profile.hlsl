// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_IES_PROFILE_INCLUDED_
#define _NBL_BUILTIN_HLSL_IES_PROFILE_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

namespace nbl 
{
namespace hlsl 
{
namespace ies 
{

struct ProfileProperties
{
    //! max 16K resolution
    NBL_CONSTEXPR_STATIC_INLINE uint32_t CDC_MAX_TEXTURE_WIDTH = 15360u;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t CDC_MAX_TEXTURE_HEIGHT = 8640u;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t CDC_DEFAULT_TEXTURE_WIDTH = 1024u;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t CDC_DEFAULT_TEXTURE_HEIGHT = 1024u;

    NBL_CONSTEXPR_STATIC_INLINE float32_t MAX_VANGLE = 180.f;
    NBL_CONSTEXPR_STATIC_INLINE float32_t MAX_HANGLE = 360.f;

    enum Version : uint16_t
    {
        V_1995,
        V_2002,
        V_SIZE
    };

    enum PhotometricType : uint16_t
    {
        TYPE_NONE,
        TYPE_C,
        TYPE_B,
        TYPE_A
    };

    enum LuminairePlanesSymmetry : uint16_t
    {
        ISOTROPIC,                  //! Only one horizontal angle present and a luminaire is assumed to be laterally axial symmetric
        QUAD_SYMETRIC,              //! The luminaire is assumed to be symmetric in each quadrant
        HALF_SYMETRIC,              //! The luminaire is assumed to be symmetric about the 0 to 180 degree plane
        OTHER_HALF_SYMMETRIC,       //! HALF_SYMETRIC case for legacy V_1995 version where horizontal angles are in range [90, 270], in that case the parser patches horizontal angles to be HALF_SYMETRIC
        NO_LATERAL_SYMMET           //! The luminaire is assumed to exhibit no lateral symmet
    };

    PhotometricType type;
    Version version;
    LuminairePlanesSymmetry symmetry;

    float32_t maxCandelaValue;            //! Max scalar value from candela data vector    
    float32_t totalEmissionIntegral;      //! Total energy emitted
    float32_t avgEmmision;                //! totalEmissionIntegral / <size of the emission domain where non zero emission values>
};

}
}
}

#endif // _NBL_BUILTIN_HLSL_IES_PROFILE_INCLUDED_
