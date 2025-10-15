// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_FILM_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_C_ELEMENT_FILM_H_INCLUDED_


#include "nbl/ext/MitsubaLoader/CElementRFilter.h"


namespace nbl::ext::MitsubaLoader
{

class CElementFilm final : public IElement
{
	public:
		enum Type : uint8_t
		{
			INVALID,
			HDR_FILM,
			TILED_HDR,
			LDR_FILM,
			MFILM
		};
		enum PixelFormat : uint8_t
		{
			LUMINANCE,
			LUMINANCE_ALPHA,
			RGB,
			RGBA,
			XYZ,
			XYZA,
			SPECTRUM,
			SPECTRUM_ALPHA
		};
		enum FileFormat : uint8_t
		{
			OPENEXR,
			RGBE,
			PFM,
			PNG,
			JPEG,
			MATLAB,
			MATHEMATICA,
			NUMPY
		};
		enum ComponentFormat : uint8_t
		{
			FLOAT16,
			FLOAT32,
			UINT32
		};
		struct HDR
		{
			bool attachLog = true;
		};
		struct LDR
		{
			enum TonemapMethod
			{
				GAMMA,
				REINHARD
			};
			TonemapMethod tonemapMethod = GAMMA;
			float gamma = -1.f; // should really be an OETF choice
			float exposure = 0.f;
			float key = 0.18;
			float burn = 0.f;
		};
		struct M
		{
			M() : digits(4)
			{
				variable[0] = 'd';
				variable[1] = 'a';
				variable[2] = 't';
				variable[3] = 'a';
				variable[4] = 0;
			}
			int32_t digits;
			constexpr static inline size_t MaxVarNameLen = 63; // matlab
			char variable[MaxVarNameLen+1];
		};

		inline CElementFilm(const char* id) : IElement(id), type(Type::HDR_FILM),
			width(768), height(576), cropOffsetX(0), cropOffsetY(0), cropWidth(INT_MAX), cropHeight(INT_MAX),
			fileFormat(OPENEXR), pixelFormat(RGB), componentFormat(FLOAT16),
			banner(true), highQualityEdges(false), rfilter("")
		{
			hdrfilm = HDR();
		}
		virtual ~CElementFilm()
		{
		}

		bool addProperty(SNamedPropertyElement&& _property, system::logger_opt_ptr logger) override;
		bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CMitsubaMetadata* globalMetadata) override;
		inline IElement::Type getType() const override { return IElement::Type::FILM; }
		inline std::string getLogName() const override { return "film"; }

		inline bool processChildData(IElement* _child, const std::string& name) override
		{
			if (!_child)
				return true;
			if (_child->getType() != IElement::Type::RFILTER)
				return false;
			auto _rfilter = static_cast<CElementRFilter*>(_child);
			if (_rfilter->type == CElementRFilter::Type::INVALID)
				return false;
			rfilter = *_rfilter;
			return true;
		}

		// make these public
		Type			type;
		int32_t			width,height;
		int32_t			cropOffsetX,cropOffsetY,cropWidth,cropHeight;
		FileFormat		fileFormat;
		PixelFormat		pixelFormat;
		ComponentFormat	componentFormat;
		bool banner;
		bool highQualityEdges;
		CElementRFilter rfilter;
		union
		{
			HDR hdrfilm;
			LDR ldrfilm;
			M	mfilm;
		};

		constexpr static inline size_t MaxPathLen = 256;
		char outputFilePath[MaxPathLen+1] = {0};
		char denoiserBloomFilePath[MaxPathLen+1] = {0};
		int32_t cascadeCount = 1;
		float cascadeLuminanceBase = core::nan<float>();
		float cascadeLuminanceStart = core::nan<float>();
		float denoiserBloomScale = 0.0f;
		float denoiserBloomIntensity = 0.0f;
		constexpr static inline size_t MaxTonemapperArgsLen = 128;
		char denoiserTonemapperArgs[MaxTonemapperArgsLen+1] = {0};
		float envmapRegularizationFactor = 0.5f; // 1.0f means based envmap luminance, 0.0f means uniform
};


}
#endif