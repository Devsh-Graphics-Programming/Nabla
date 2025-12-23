// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/ext/MitsubaLoader/CElementFilm.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"

#include "nbl/ext/MitsubaLoader/ElementMacros.h"

#include <functional>

namespace nbl::ext::MitsubaLoader
{

auto CElementFilm::compAddPropertyMap() -> AddPropertyMap<CElementFilm>
{
	using this_t = CElementFilm;
	AddPropertyMap<CElementFilm> retval;
	
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY(width,INTEGER);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY(height,INTEGER);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY(cropOffsetX,INTEGER);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY(cropOffsetY,INTEGER);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY(cropWidth,INTEGER);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY(cropHeight,INTEGER);
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY("fileFormat",STRING)
		{
			static const core::unordered_map<std::string,FileFormat,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> StringToType =
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
				return false;
			_this->fileFormat = found->second;
			return true;
		}
	});
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY("pixelFormat",STRING)
		{
			static const core::unordered_map<std::string,PixelFormat,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> StringToType =
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
				return false;
			_this->pixelFormat = found->second;
			return true;
		}
	});
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY("setComponentFormat",STRING)
		{
			static const core::unordered_map<std::string,ComponentFormat,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> StringToType =
			{
				{"float16",	FLOAT16},
				{"float32",	FLOAT32},
				{"uint32",	UINT32}
			};
			auto found = StringToType.find(_property.svalue);
			if (found==StringToType.end())
				return false;
			_this->componentFormat = found->second;
			return true;
		}
	});
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY(banner,BOOLEAN);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY(highQualityEdges,BOOLEAN);
	
	
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(attachLog,BOOLEAN,std::is_same,HDR);
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED("tonemapMethod",STRING,std::is_same,LDR)
		{
			static const core::unordered_map<std::string,LDR::TonemapMethod,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> StringToType =
			{
				{"gamma",	LDR::GAMMA},
				{"reinhard",LDR::REINHARD}
			};
			auto found = StringToType.find(_property.svalue);
			if (found==StringToType.end())
				return false;
			_this->ldrfilm.tonemapMethod = found->second;
			return true;
		}
	);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(gamma,FLOAT,std::is_same,LDR);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(exposure,FLOAT,std::is_same,LDR);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(key,FLOAT,std::is_same,LDR);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(burn,FLOAT,std::is_same,LDR);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_VARIANT_PROPERTY_CONSTRAINED(attachLog,INTEGER,std::is_same,HDR);

	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY_CONSTRAINED("variable",STRING,std::is_same,M)
		{
			setLimitedString("variable",_this->outputFilePath,_property,logger); return true;
		}
	);
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY("outputFilePath",STRING)
		{
			setLimitedString("outputFilePath",_this->outputFilePath,_property,logger); return true;
		}
	});
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY("bloomFilePath",STRING)
		{
			setLimitedString("bloomFilePath",_this->denoiserTonemapperArgs,_property,logger); return true;
		}
	});
	NBL_EXT_MITSUBA_LOADER_REGISTER_ADD_PROPERTY("tonemapper",STRING)
		{
			setLimitedString("tonemapper",_this->denoiserTonemapperArgs,_property,logger); return true;
		}
	});

	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY(cascadeCount,INTEGER);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY(cascadeLuminanceBase,FLOAT);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY(cascadeLuminanceStart,FLOAT);

	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY(denoiserBloomScale,FLOAT);
	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY(denoiserBloomIntensity,FLOAT);

	NBL_EXT_MITSUBA_LOADER_REGISTER_SIMPLE_ADD_PROPERTY(envmapRegularizationFactor,FLOAT);


	return retval;
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