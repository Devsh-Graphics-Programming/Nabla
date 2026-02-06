// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_IES_PROFILE_INCLUDED_
#define _NBL_BUILTIN_HLSL_IES_PROFILE_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/cpp_compat/basic.h"

namespace nbl 
{
namespace hlsl 
{
namespace ies 
{

struct ProfileProperties
{
    NBL_CONSTEXPR_STATIC_INLINE float32_t MaxVAngleDegrees = 180.f;
    NBL_CONSTEXPR_STATIC_INLINE float32_t MaxHAngleDegrees = 360.f;

	// TODO: could change to uint8_t once we get implemented
    // https://github.com/microsoft/hlsl-specs/pull/538
	using packed_flags_t = uint32_t;

    NBL_CONSTEXPR_STATIC_INLINE packed_flags_t VersionBits  = 2u;
    NBL_CONSTEXPR_STATIC_INLINE packed_flags_t TypeBits     = 2u;
    NBL_CONSTEXPR_STATIC_INLINE packed_flags_t SymmetryBits = 3u;
    NBL_CONSTEXPR_STATIC_INLINE packed_flags_t VersionMask  = (packed_flags_t(1u) << VersionBits) - packed_flags_t(1u);
    NBL_CONSTEXPR_STATIC_INLINE packed_flags_t TypeMask     = (packed_flags_t(1u) << TypeBits) - packed_flags_t(1u);
    NBL_CONSTEXPR_STATIC_INLINE packed_flags_t SymmetryMask = (packed_flags_t(1u) << SymmetryBits) - packed_flags_t(1u);
    NBL_CONSTEXPR_STATIC_INLINE packed_flags_t TypeShift    = VersionBits;
    NBL_CONSTEXPR_STATIC_INLINE packed_flags_t SymmetryShift = VersionBits + TypeBits;

    enum Version : packed_flags_t
    {
        V_1995,
        V_2002,
        V_SIZE
    };

    enum PhotometricType : packed_flags_t
    {
        TYPE_NONE,
        TYPE_C,
        TYPE_B,
        TYPE_A
    };

    enum LuminairePlanesSymmetry : packed_flags_t
    {
        ISOTROPIC,                  //! Only one horizontal angle present and a luminaire is assumed to be laterally axial symmetric
        QUAD_SYMETRIC,              //! The luminaire is assumed to be symmetric in each quadrant
        HALF_SYMETRIC,              //! The luminaire is assumed to be symmetric about the 0 to 180 degree plane
        OTHER_HALF_SYMMETRIC,       //! HALF_SYMETRIC case for legacy V_1995 version where horizontal angles are in range [90, 270], in that case the parser patches horizontal angles to be HALF_SYMETRIC
        NO_LATERAL_SYMMET           //! The luminaire is assumed to exhibit no lateral symmet
    };

    Version getVersion() NBL_CONST_MEMBER_FUNC
    {
        return (Version)(packed & VersionMask);
    }

    PhotometricType getType() NBL_CONST_MEMBER_FUNC
    {
        return (PhotometricType)((packed >> TypeShift) & TypeMask);
    }

    LuminairePlanesSymmetry getSymmetry() NBL_CONST_MEMBER_FUNC
    {
        return (LuminairePlanesSymmetry)((packed >> SymmetryShift) & SymmetryMask);
    }

    void setVersion(Version v)
    {
        packed_flags_t vBits = (packed_flags_t)(v) & VersionMask;
        packed = (packed & ~VersionMask) | vBits;
    }

    void setType(PhotometricType t)
    {
        packed_flags_t tBits = ((packed_flags_t)(t) & TypeMask) << TypeShift;
        packed = (packed & ~(TypeMask << TypeShift)) | tBits;
    }

    void setSymmetry(LuminairePlanesSymmetry s)
    {
        packed_flags_t sBits = ((packed_flags_t)(s) & SymmetryMask) << SymmetryShift;
        packed = (packed & ~(SymmetryMask << SymmetryShift)) | sBits;
    }

	float32_t maxCandelaValue;        //! Max candela sample value
	float32_t totalEmissionIntegral;  //! Total emitted intensity (integral over full angular domain)
	float32_t fullDomainAvgEmission;  //! Mean intensity over full angular domain (including I == 0)
	float32_t avgEmmision;            //! Mean intensity over emitting solid angle (I > 0)
	packed_flags_t packed;			  //! Packed version, type and symmetry flags
};

}

}
}

#endif // _NBL_BUILTIN_HLSL_IES_PROFILE_INCLUDED_
