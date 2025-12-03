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

	// TODO: could change to uint8_t once we get implemented
    // https://github.com/microsoft/hlsl-specs/pull/538
	using packed_flags_t = uint16_t;

    NBL_CONSTEXPR_STATIC_INLINE packed_flags_t VERSION_BITS = 2u;
    NBL_CONSTEXPR_STATIC_INLINE packed_flags_t TYPE_BITS    = 2u;
    NBL_CONSTEXPR_STATIC_INLINE packed_flags_t SYMM_BITS    = 3u;
    NBL_CONSTEXPR_STATIC_INLINE packed_flags_t VERSION_MASK = (packed_flags_t(1u) << VERSION_BITS) - packed_flags_t(1u);
    NBL_CONSTEXPR_STATIC_INLINE packed_flags_t TYPE_MASK    = (packed_flags_t(1u) << TYPE_BITS)    - packed_flags_t(1u);
    NBL_CONSTEXPR_STATIC_INLINE packed_flags_t SYMM_MASK    = (packed_flags_t(1u) << SYMM_BITS)    - packed_flags_t(1u);

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

    Version getVersion() const
    {
        return static_cast<Version>( packed & VERSION_MASK );
    }

    PhotometricType getType() const
    {
        const packed_flags_t shift = VERSION_BITS;
        return static_cast<PhotometricType>( (packed >> shift) & TYPE_MASK );
    }

    LuminairePlanesSymmetry getSymmetry() const
    {
        const packed_flags_t shift = VERSION_BITS + TYPE_BITS;
        return static_cast<LuminairePlanesSymmetry>( (packed >> shift) & SYMM_MASK );
    }

    void setVersion(Version v)
    {
        packed_flags_t vBits = static_cast<packed_flags_t>(v) & VERSION_MASK;
        packed = (packed & ~VERSION_MASK) | vBits;
    }

    void setType(PhotometricType t)
    {
        const packed_flags_t shift = VERSION_BITS;
        packed_flags_t tBits = (static_cast<packed_flags_t>(t) & TYPE_MASK) << shift;
        packed = (packed & ~(TYPE_MASK << shift)) | tBits;
    }

    void setSymmetry(LuminairePlanesSymmetry s)
    {
        const packed_flags_t shift = VERSION_BITS + TYPE_BITS;
        packed_flags_t sBits = (static_cast<packed_flags_t>(s) & SYMM_MASK) << shift;
        packed = (packed & ~(SYMM_MASK << shift)) | sBits;
    }

	float32_t maxCandelaValue;        //! Max candela sample value
	float32_t totalEmissionIntegral;  //! Total emitted intensity (integral over full angular domain)
	float32_t fullDomainAvgEmission;  //! Mean intensity over full angular domain (including I == 0)
	float32_t avgEmmision;            //! Mean intensity over emitting solid angle (I > 0)
	packed_flags_t packed = 0u;		  //! Packed version, type and symmetry flags
};

}
}
}

#endif // _NBL_BUILTIN_HLSL_IES_PROFILE_INCLUDED_
