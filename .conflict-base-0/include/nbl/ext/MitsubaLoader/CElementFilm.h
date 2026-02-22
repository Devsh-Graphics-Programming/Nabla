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
		//
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
			constexpr static inline Type VariantType = Type::HDR_FILM;

			bool attachLog = true;
		};
		struct TiledHDR : HDR
		{
			constexpr static inline Type VariantType = Type::TILED_HDR;

			// TODO: sure we don't have more members?
		};
		struct LDR
		{
			constexpr static inline Type VariantType = Type::LDR_FILM;

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
			constexpr static inline Type VariantType = Type::MFILM;

			inline M() : digits(4)
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

		//
		using variant_list_t = core::type_list<
			HDR,
			TiledHDR,
			LDR,
			M
		>;
		static inline core::unordered_map<core::string,Type,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> compStringToTypeMap()
		{
			return {
				{"hdrfilm",		Type::HDR_FILM},
				{"tiledhdrfilm",Type::TILED_HDR},
				{"ldrfilm",		Type::LDR_FILM},
				{"mfilm",		Type::MFILM}
			};
		}
		static AddPropertyMap<CElementFilm> compAddPropertyMap();

		inline CElementFilm(const char* id) : IElement(id), type(Type::HDR_FILM),
			width(768), height(576), cropOffsetX(0), cropOffsetY(0), cropWidth(INT_MAX), cropHeight(INT_MAX),
			fileFormat(OPENEXR), pixelFormat(RGB), componentFormat(FLOAT16),
			banner(true), highQualityEdges(false), rfilter("")
		{
			hdrfilm = HDR();
		}
		virtual inline ~CElementFilm()
		{
		}

		inline void initialize()
		{
			switch (type)
			{
				case CElementFilm::Type::LDR_FILM:
					fileFormat = CElementFilm::FileFormat::PNG;
					//componentFormat = UINT8;
					ldrfilm = CElementFilm::LDR();
					break;
				case CElementFilm::Type::MFILM:
					width = 1;
					height = 1;
					fileFormat = CElementFilm::FileFormat::MATLAB;
					pixelFormat = CElementFilm::PixelFormat::LUMINANCE;
					mfilm = CElementFilm::M();
					break;
				default:
					break;
			}
		}

		template<typename Visitor>
		inline void visit(Visitor&& visitor)
		{
			switch (type)
			{
				case CElementFilm::Type::LDR_FILM:
					visitor(ldrfilm);
					break;
				case CElementFilm::Type::MFILM:
					visitor(mfilm);
					break;
				default:
					visitor(hdrfilm);
					break;
			}
		}
		template<typename Visitor>
		inline void visit(Visitor&& visitor) const
		{
			const_cast<CElementFilm*>(this)->visit([&]<typename T>(T& var)->void
				{
					visitor(const_cast<const T&>(var));
				}
			);
		}

		bool onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger) override;

		constexpr static inline auto ElementType = IElement::Type::FILM;
		inline IElement::Type getType() const override { return ElementType; }
		inline std::string getLogName() const override { return "film"; }

		inline bool processChildData(IElement* _child, const std::string& name, system::logger_opt_ptr logger) override
		{
			if (!_child)
				return true;
			if (_child->getType() != IElement::Type::RFILTER)
			{
				logger.log("CElementFilm only expects type %d children, is %d instead",system::ILogger::ELL_ERROR,IElement::Type::RFILTER,_child->getType());
				return false;
			}
			auto _rfilter = static_cast<CElementRFilter*>(_child);
			if (_rfilter->type == CElementRFilter::Type::INVALID)
			{
				logger.log("CElementRFilter::Type::INVALID used as child in CElementFilm",system::ILogger::ELL_ERROR);
				return false;
			}
			rfilter = *_rfilter;
			return true;
		}

		// make these public
		Type			type;
		int32_t			width,height;
		int32_t			cropOffsetX,cropOffsetY,cropWidth,cropHeight;
		FileFormat		fileFormat = OPENEXR;
		PixelFormat		pixelFormat;
		ComponentFormat	componentFormat;
		bool banner;
		bool highQualityEdges;
		CElementRFilter rfilter;
		union
		{
			HDR			hdrfilm;
			LDR			ldrfilm;
			M			mfilm;
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