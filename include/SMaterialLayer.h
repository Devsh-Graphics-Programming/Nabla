// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __S_MATERIAL_LAYER_H_INCLUDED__
#define __S_MATERIAL_LAYER_H_INCLUDED__

#include "matrix4.h"
#include "irrAllocator.h"

namespace irr
{
namespace video
{
	class IVirtualTexture;

	//! Texture coord clamp mode outside [0.0, 1.0]
	enum E_TEXTURE_CLAMP
	{
		//! Texture repeats
		ETC_REPEAT = 0,
		//! Texture is clamped to the edge pixel
		ETC_CLAMP_TO_EDGE,
		//! Texture is clamped to the border pixel (if exists)
		ETC_CLAMP_TO_BORDER,
		//! Texture is alternatingly mirrored (0..1..0..1..0..)
		ETC_MIRROR,
		//! Texture is mirrored once and then clamped to edge
		ETC_MIRROR_CLAMP_TO_EDGE,
		//! Texture is mirrored once and then clamped to border
		ETC_MIRROR_CLAMP_TO_BORDER
	};
	static const char* const aTextureClampNames[] = {
			"texture_clamp_repeat",
			"texture_clamp_clamp_to_edge",
			"texture_clamp_mirror",
			"texture_clamp_mirror_clamp_to_edge", 0};

    //! KEEP ENUM ORDER, makes it easy to disable mip-maps
    enum E_TEXTURE_FILTER_TYPE
    {
        ETFT_NEAREST_NO_MIP = 0,
        ETFT_LINEAR_NO_MIP,
        //! the following use mip maps
        ETFT_NEAREST_NEARESTMIP, //just mip map level chooosing
        ETFT_LINEAR_NEARESTMIP, //
        ETFT_NEAREST_LINEARMIP,
        ETFT_LINEAR_LINEARMIP
    };
	static const char* const aMinFilterNames[] = {
			"NEAREST","LINEAR",
			"NEAREST_MIPMAP_NEAREST",
			"LINEAR_MIPMAP_NEAREST",
			"NEAREST_MIPMAP_LINEAR",
			"LINEAR_MIPMAP_LINEAR", 0};
	static const char* const aMaxFilterNames[] = {
			"NEAREST","LINEAR", 0};

    //! Things I have Purposefully Omitted:
    /** - Texture Border Colors (deprecated in openGL)
    **  - Texture lod range (not used yet)
    **  - Texture comparison mode (no depth texture support yet)
    **/
    class STextureSamplingParams
    {
        public:
        //! Texture Clamp Mode
		/** Values are taken from E_TEXTURE_CLAMP. */
		uint32_t TextureWrapU:3;
		uint32_t TextureWrapV:3;
		uint32_t TextureWrapW:3;

		//! filter type to use
		uint32_t MinFilter:3;
		uint32_t MaxFilter:1;

		//! quick setting
		uint32_t UseMipmaps:1;

		//! Is anisotropic filtering enabled? Default: 0, disabled
		/** In Irrlicht you can use anisotropic texture filtering
		in conjunction with bilinear or trilinear texture
		filtering to improve rendering results. Primitives
		will look less blurry with this flag switched on. The number gives
		the maximal anisotropy degree, and is often in the range 2-16.
		Value 1 is equivalent to 0, but should be avoided. */
		uint32_t AnisotropicFilter:5; //allows for 32x filter

		uint32_t SeamlessCubeMap:1;

		//! Bias for the mipmap choosing decision.
		/** This value can make the textures more or less blurry than with the
		default value of 0. The value is added to the mipmap level
		chosen initially, and thus takes a smaller mipmap for a region
		if the value is positive. */
		float LODBias;

        STextureSamplingParams()
        {
            TextureWrapU = ETC_REPEAT;
            TextureWrapV = ETC_REPEAT;
            TextureWrapW = ETC_REPEAT;

            MinFilter = ETFT_LINEAR_NEARESTMIP;
            MaxFilter = ETFT_LINEAR_NO_MIP;
            UseMipmaps = 1;
            AnisotropicFilter = 0;
            SeamlessCubeMap = 1;

            LODBias = 0.f;
        }

		inline uint64_t calculateHash() const
		{
            const uint64_t zero64 = 0;
		    STextureSamplingParams tmp = *((STextureSamplingParams*)&zero64);
		    tmp.TextureWrapU = 0x7u;
		    tmp.TextureWrapV = 0x7u;
		    tmp.TextureWrapW = 0x7u;
		    tmp.MinFilter = UseMipmaps ? 0xfu:0x1u;
		    tmp.MaxFilter = 0x1u;
		    tmp.UseMipmaps = 0;
		    tmp.AnisotropicFilter = 0x1fu;
		    tmp.SeamlessCubeMap = 0x1u;
            *((uint32_t*)&tmp.LODBias) = 0xffffffffu;

            uint64_t retval = *((uint64_t*)this);
            retval &= *((uint64_t*)&tmp);
            return retval;
		}

		uint64_t calculateHash(const IVirtualTexture* tex) const;
    };

	//! Struct for holding material parameters which exist per texture layer
	class SMaterialLayer
	{
	public:
		//! Default constructor
		SMaterialLayer()
			: Texture(0)
        {
        }

		//! Copy constructor
		/** \param other Material layer to copy from. */
		SMaterialLayer(const SMaterialLayer& other)
		{
			*this = other;
		}

		//! Destructor
		~SMaterialLayer()
		{
		}

		//! Assignment operator
		/** \param other Material layer to copy from.
		\return This material layer, updated. */
		SMaterialLayer& operator=(const SMaterialLayer& other)
		{
			// Check for self-assignment!
			if (this == &other)
				return *this;

			Texture = other.Texture;
			SamplingParams = other.SamplingParams;

			return *this;
		}

		//! Inequality operator
		/** \param b Layer to compare to.
		\return True if layers are different, else false. */
		inline bool operator!=(const SMaterialLayer& b) const
		{
			return Texture != b.Texture || SamplingParams.calculateHash(Texture) != b.SamplingParams.calculateHash(b.Texture);
		}

		//! Equality operator
		/** \param b Layer to compare to.
		\return True if layers are equal, else false. */
		inline bool operator==(const SMaterialLayer& b) const
		{ return !(b!=*this); }

		//! Texture
		IVirtualTexture* Texture;

        STextureSamplingParams SamplingParams;


	private:
		friend class SMaterial;
	};

} // end namespace video
} // end namespace irr

#endif // __S_MATERIAL_LAYER_H_INCLUDED__
