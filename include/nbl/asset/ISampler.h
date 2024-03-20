// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_SAMPLER_H_INCLUDED__
#define __NBL_ASSET_I_SAMPLER_H_INCLUDED__

#include "nbl/asset/IDescriptor.h"

namespace nbl
{
namespace asset
{

class ISampler : public virtual core::IReferenceCounted
{
	public:
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
			ETC_MIRROR_CLAMP_TO_BORDER,

			ETC_COUNT
		};

		enum E_TEXTURE_BORDER_COLOR
		{
			ETBC_FLOAT_TRANSPARENT_BLACK = 0,
			ETBC_INT_TRANSPARENT_BLACK,
			ETBC_FLOAT_OPAQUE_BLACK,
			ETBC_INT_OPAQUE_BLACK,
			ETBC_FLOAT_OPAQUE_WHITE,
			ETBC_INT_OPAQUE_WHITE,

			ETBC_COUNT
		};

		enum E_TEXTURE_FILTER
		{
			ETF_NEAREST = 0,
			ETF_LINEAR
		};

		enum E_SAMPLER_MIPMAP_MODE
		{
			ESMM_NEAREST = 0,
			ESMM_LINEAR
		};

		enum E_COMPARE_OP
		{
			ECO_NEVER = 0,
			ECO_LESS,
			ECO_EQUAL,
			ECO_LESS_OR_EQUAL,
			ECO_GREATER,
			ECO_NOT_EQUAL,
			ECO_GREATER_OR_EQUAL,
			ECO_ALWAYS
		};

	#include "nbl/nblpack.h"
		struct SParams
		{
			struct {
				//! Valeus taken from E_TEXTURE_CLAMP
				uint32_t TextureWrapU : 3 = ETC_REPEAT;
				//! Valeus taken from E_TEXTURE_CLAMP
				uint32_t TextureWrapV : 3 = ETC_REPEAT;
				//! Valeus taken from E_TEXTURE_CLAMP
				uint32_t TextureWrapW : 3 = ETC_REPEAT;
				//! Values taken from E_TEXTURE_BORDER_COLOR
				uint32_t BorderColor : 3 = ETBC_FLOAT_OPAQUE_BLACK;
				//! Values taken from E_TEXTURE_FILTER
				uint32_t MinFilter : 1 = ETF_LINEAR;
				//! Values taken from E_TEXTURE_FILTER
				uint32_t MaxFilter : 1 = ETF_LINEAR;
				//! Values taken from E_SAMPLER_MIPMAP_MODE
				uint32_t MipmapMode : 1 = ESMM_LINEAR;
				//! Encoded as power of two (so that if you need 16, Anisotropy should be 4); max value is 5
				uint32_t AnisotropicFilter : 3 = 3;
				//! Boolean, compare ref to texture
				uint32_t CompareEnable : 1 = false;
				//! Values taken from E_COMPARE_OP
				uint32_t CompareFunc : 3 = ECO_GREATER;
			};
			float LodBias = 0.f;
			float MinLod = -1000.f;
			float MaxLod = 1000.f;

			inline bool operator==(const SParams& rhs) const
			{
				return 
					TextureWrapU==rhs.TextureWrapU &&
					TextureWrapV==rhs.TextureWrapV &&
					TextureWrapW==rhs.TextureWrapW &&
					BorderColor==rhs.BorderColor &&
					MinFilter==rhs.MinFilter &&
					MaxFilter==rhs.MaxFilter &&
					MipmapMode==rhs.MipmapMode &&
					AnisotropicFilter==rhs.AnisotropicFilter &&
					CompareEnable==rhs.CompareEnable &&
					CompareFunc==rhs.CompareFunc &&
					LodBias==rhs.LodBias &&
					MinLod==rhs.MinLod &&
					MaxLod==rhs.MaxLod;
			}
			inline bool operator!=(const SParams& rhs) const { return !((*this)==rhs); }
		} PACK_STRUCT;
	#include "nbl/nblunpack.h"

	protected:
		ISampler(const SParams& _params) : m_params(_params) {}
		virtual ~ISampler() = default;

		SParams m_params;

	public:
		const SParams& getParams() const { return m_params; }
};

}
}

#endif 