// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/ext/MitsubaLoader/CElementFilm.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"


#include <functional>

namespace nbl::ext::MitsubaLoader
{

bool CElementFilm::addProperty(SNamedPropertyElement&& _property, system::logger_opt_ptr logger)
{
#if 0
	bool error = type==Type::INVALID;
#define SET_PROPERTY(MEMBER,PROPERTY_TYPE)		[&]() -> void { \
		if (_property.type!=PROPERTY_TYPE) { \
			error = true; \
			return; \
		} \
		MEMBER = _property.getProperty<PROPERTY_TYPE>(); \
	}
	auto setWidth			= SET_PROPERTY(width,SNamedPropertyElement::Type::INTEGER);
	auto setHeight			= SET_PROPERTY(height,SNamedPropertyElement::Type::INTEGER);
	auto setCropOffsetX		= SET_PROPERTY(cropOffsetX,SNamedPropertyElement::Type::INTEGER);
	auto setCropOffsetY		= SET_PROPERTY(cropOffsetY,SNamedPropertyElement::Type::INTEGER);
	auto setCropWidth		= SET_PROPERTY(cropWidth,SNamedPropertyElement::Type::INTEGER);
	auto setCropHeight		= SET_PROPERTY(cropHeight,SNamedPropertyElement::Type::INTEGER);
	auto setFileFormat = [&]() -> void
	{
		if (_property.type!=SNamedPropertyElement::Type::STRING)
		{
			error = true;
			return;
		}
		static const core::unordered_map<std::string, FileFormat, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> StringToType =
		{
			{"openexr",		OPENEXR},
			{"png",			PNG},
			{"rgbe",		RGBE},
			{"pfm",			PFM},
			{"matlab",		MATLAB},
			{"mathematica",	MATHEMATICA},
			{"numpy",		NUMPY}
		};
		auto found = StringToType.find(_property.svalue);
		if (found==StringToType.end())
		{
			error = true;
			return;
		}
		fileFormat = found->second;
	};
	auto setPixelFormat = [&]() -> void
	{
		if (_property.type!=SNamedPropertyElement::Type::STRING)
		{
			error = true;
			return;
		}
		static const core::unordered_map<std::string, PixelFormat, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> StringToType =
		{
			{"luminance",		LUMINANCE},
			{"luminanceAlpha",	LUMINANCE_ALPHA},
			{"rgb",				RGB},
			{"rgba",			RGBA},
			{"xyz",				XYZ},
			{"xyza",			XYZA},
			{"spectrum",		SPECTRUM},
			{"spectrumAlpha",	SPECTRUM_ALPHA}
		};
		auto found = StringToType.find(_property.svalue);
		if (found==StringToType.end())
		{
			error = true;
			return;
		}
		pixelFormat = found->second;
	};
	auto setComponentFormat = [&]() -> void
	{
		if (_property.type!=SNamedPropertyElement::Type::STRING || type==Type::LDR_FILM || type==Type::MFILM)
		{
			error = true;
			return;
		}
		static const core::unordered_map<std::string, ComponentFormat, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> StringToType =
		{
			{"float16",	FLOAT16},
			{"float32",	FLOAT32},
			{"uint32",	UINT32}
		};
		auto found = StringToType.find(_property.svalue);
		if (found==StringToType.end())
		{
			error = true;
			return;
		}
		componentFormat = found->second;
	};
	auto setBanner			= SET_PROPERTY(banner,SNamedPropertyElement::Type::BOOLEAN);
	auto setHighQualityEdges= SET_PROPERTY(highQualityEdges,SNamedPropertyElement::Type::BOOLEAN);
	

	auto dispatch = [&](auto func) -> void
	{
		switch (type)
		{
			case CElementFilm::Type::HDR_FILM:
				func(hdrfilm);
				break;
			case CElementFilm::Type::LDR_FILM:
				func(ldrfilm);
				break;
			case CElementFilm::Type::MFILM:
				func(mfilm);
				break;
			default:
				error = true;
				break;
		}
	};
#define SET_PROPERTY_TEMPLATE(MEMBER,PROPERTY_TYPE, ... )		[&]() -> void { \
		dispatch([&](auto& state) -> void { \
			if constexpr (is_any_of<std::remove_reference<decltype(state)>::type,__VA_ARGS__>::value) \
			{ \
				if (_property.type!=PROPERTY_TYPE) { \
					error = true; \
					return; \
				} \
				state. ## MEMBER = _property.getProperty<PROPERTY_TYPE>(); \
			} \
		}); \
	}

	auto setAttachLog = SET_PROPERTY_TEMPLATE(attachLog, SNamedPropertyElement::Type::BOOLEAN, HDR);
	auto setTonemapMethod = [&]() -> void
	{
		if (_property.type != SNamedPropertyElement::Type::STRING || type == Type::LDR_FILM)
		{
			error = true;
			return;
		}
		static const core::unordered_map<std::string, LDR::TonemapMethod, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> StringToType =
		{
			{"gamma",	LDR::GAMMA},
			{"reinhard",LDR::REINHARD}
		};
		auto found = StringToType.find(_property.svalue);
		if (found != StringToType.end())
		{
			error = true;
			return;
		}
		ldrfilm.tonemapMethod = found->second;
	};
	auto setGamma = SET_PROPERTY_TEMPLATE(gamma, SNamedPropertyElement::Type::FLOAT, LDR);
	auto setExposure = SET_PROPERTY_TEMPLATE(exposure, SNamedPropertyElement::Type::FLOAT, LDR);
	auto setKey = SET_PROPERTY_TEMPLATE(key, SNamedPropertyElement::Type::FLOAT, LDR);
	auto setBurn = SET_PROPERTY_TEMPLATE(burn, SNamedPropertyElement::Type::FLOAT, LDR);
	auto setDigits = SET_PROPERTY_TEMPLATE(digits, SNamedPropertyElement::Type::INTEGER, M);
	auto setVariable = [&]() -> void
	{
		if (_property.type != SNamedPropertyElement::Type::STRING || type == Type::MFILM)
		{
			error = true;
			return;
		}
		size_t len = std::min(strlen(_property.svalue),M::MaxVarNameLen);
		memcpy(mfilm.variable,_property.svalue,len);
		mfilm.variable[len] = 0;
	};
	auto setOutputFilePath = [&]() -> void
	{
		if (_property.type != SNamedPropertyElement::Type::STRING)
		{
			error = true;
			return;
		}

		size_t len = std::min(strlen(_property.svalue),MaxPathLen);
		memcpy(outputFilePath,_property.svalue,len);
		outputFilePath[len] = 0;
	};
	
	auto setBloomFilePath = [&]() -> void
	{
		if (_property.type != SNamedPropertyElement::Type::STRING)
		{
			error = true;
			return;
		}

		size_t len = std::min(strlen(_property.svalue),MaxPathLen);
		memcpy(denoiserBloomFilePath,_property.svalue,len);
		denoiserBloomFilePath[len] = 0;
	};
	
	auto setTonemapperArgs = [&]() -> void
	{
		if (_property.type != SNamedPropertyElement::Type::STRING)
		{
			error = true;
			return;
		}

		size_t len = std::min(strlen(_property.svalue),MaxTonemapperArgsLen);
		memcpy(denoiserTonemapperArgs,_property.svalue,len);
		denoiserTonemapperArgs[len] = 0;
	};

	auto setCascadeCount			= SET_PROPERTY(cascadeCount,SNamedPropertyElement::Type::INTEGER);
	auto setCascadeLuminanceBase	= SET_PROPERTY(cascadeLuminanceBase,SNamedPropertyElement::Type::FLOAT);
	auto setCascadeLuminanceStart	= SET_PROPERTY(cascadeLuminanceStart,SNamedPropertyElement::Type::FLOAT);

	auto setBloomScale				= SET_PROPERTY(denoiserBloomScale,SNamedPropertyElement::Type::FLOAT);
	auto setBloomIntensity			= SET_PROPERTY(denoiserBloomIntensity,SNamedPropertyElement::Type::FLOAT);

	auto setEnvmapRegularizationFactor = SET_PROPERTY(envmapRegularizationFactor,SNamedPropertyElement::Type::FLOAT);

	const core::unordered_map<std::string, std::function<void()>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> SetPropertyMap =
	{
		{"width",					setWidth},
		{"height",					setHeight},
		{"cropOffsetX",				setCropOffsetX},
		{"cropOffsetY",				setCropOffsetY},
		{"cropWidth",				setCropWidth},
		{"cropHeight",				setCropHeight},
		{"fileFormat",				setFileFormat},
		{"pixelFormat",				setPixelFormat},
		{"componentFormat",			setComponentFormat},
		{"banner",					setBanner},
		{"highQualityEdges",		setHighQualityEdges},
		{"attachLog",				setAttachLog},
		{"tonemapMethod",			setTonemapMethod},
		{"gamma",					setGamma},
		{"exposure",				setExposure},
		{"key",						setKey},
		{"burn",					setBurn},
		{"digits",					setDigits},
		{"variable",				setVariable},
		{"outputFilePath",			setOutputFilePath},
		{"bloomFilePath",			setBloomFilePath},
		{"cascadeCount",			setCascadeCount},
		{"cascadeLuminanceBase",	setCascadeLuminanceBase},
		{"cascadeLuminanceStart",	setCascadeLuminanceStart},
		{"bloomScale",				setBloomScale},
		{"bloomIntensity",			setBloomIntensity},
		{"tonemapper",				setTonemapperArgs},
		{"envmapRegularizationFactor", setEnvmapRegularizationFactor}
	};

	auto found = SetPropertyMap.find(_property.name);
	if (found == SetPropertyMap.end())
	{
		
		invalidXMLFileStructure(logger,"No Film can have such property set with name: " + _property.name+"\nRemember we don't support \"render-time annotations\"");
		return false;
	}

	found->second();
	return !error;
#endif
	assert(false);
	return false;
}

bool CElementFilm::onEndTag(CMitsubaMetadata* globalMetadata, system::logger_opt_ptr logger)
{
	cropOffsetX = std::max(cropOffsetX,0);
	cropOffsetY = std::max(cropOffsetY,0);
	cropWidth = std::min(cropWidth,width-cropOffsetX);
	cropHeight = std::min(cropHeight,height-cropOffsetY);

	switch (type)
	{
		case Type::HDR_FILM:
			switch (fileFormat)
			{
				case OPENEXR:
					[[fallthrough]];
				case RGBE:
					[[fallthrough]];
				case PFM:
					break;
				default:
					invalidXMLFileStructure(logger,getLogName() + ": film type does not support this file format");
					
					return false;
			};
			break;
		case Type::TILED_HDR:
			switch (fileFormat)
			{
				case OPENEXR:
					break;
				default:
					invalidXMLFileStructure(logger,getLogName() + ": film type does not support this file format");
					
					return false;
			};
			break;
		case Type::LDR_FILM:
			switch (fileFormat)
			{
				case PNG:
					[[fallthrough]];
				case JPEG:
					break;
				default:
					invalidXMLFileStructure(logger,getLogName() + ": film type does not support this file format");
					
					return false;
			};
			switch (pixelFormat)
			{
				case LUMINANCE_ALPHA:
					[[fallthrough]];
				case RGBA:
					if (type==PNG)
						break;
					[[fallthrough]];
				case XYZ:
					[[fallthrough]];
				case XYZA:
					invalidXMLFileStructure(logger,getLogName() + ": film type does not support this pixel format");
					
					return false;
					break;
				default:
					break;
			};
			break;
		case Type::MFILM:
			switch (fileFormat)
			{
				case MATLAB:
					[[fallthrough]];
				case MATHEMATICA:
					[[fallthrough]];
				case NUMPY:
					break;
				default:
					invalidXMLFileStructure(logger,getLogName() + ": film type does not support this file format");
					
					return false;
			};
			switch (pixelFormat)
			{
				case XYZ:
					[[fallthrough]];
				case XYZA:
					invalidXMLFileStructure(logger,getLogName() + ": film type does not support this pixel format");
					
					return false;
					break;
				default:
					break;
			};
			switch (componentFormat)
			{
				case FLOAT32:
					break;
				default:
					invalidXMLFileStructure(logger,getLogName() + ": film type does not support this component format");
					
					return false;
			};
			break;
		default:
			invalidXMLFileStructure(logger,getLogName() + ": type not specified");
			
			return false;
	}

	return true;
}


}