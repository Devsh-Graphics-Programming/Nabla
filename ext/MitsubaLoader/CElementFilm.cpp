#include "../../ext/MitsubaLoader/CElementFilm.h"
#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/PropertyElement.h"

#include <functional>

namespace irr { namespace ext { namespace MitsubaLoader {

bool CElementFilm::processAttributes(const char** _atts)
{
	static const core::unordered_map<std::string, EFilmType> acceptableTypes = {
		std::make_pair("hdrfilm", EFilmType::HDR_FILM),
		std::make_pair("tiledhdrfilm", EFilmType::TILED_HDR_FILM),
		std::make_pair("ldrfilm", EFilmType::LDR_FILM),
		std::make_pair("mfilm", EFilmType::M_FILM)
	};

	//only type is an acceptable argument
	for (int i = 0; _atts[i]; i += 2)
	{
		if (std::strcmp(_atts[i], "type"))
		{
			ParserLog::invalidXMLFileStructure(std::string(_atts[i]) + " is not attribute of shape element.");
			return false;
		}
		else
		{
			auto samplerType = acceptableTypes.find(_atts[i + 1]);
			if (samplerType == acceptableTypes.end())
			{
				ParserLog::invalidXMLFileStructure("unknown type");
				_IRR_DEBUG_BREAK_IF(false);
				return false;
			}

			data.type = samplerType->second;

			//set default values
			switch (data.type)
			{
			case EFilmType::HDR_FILM:
			{
				data.hdrFilmData.fileFormat = EHDRFileFormat::OPENEXR;
				data.hdrFilmData.componentFormat = EComponentFormat::FLOAT16;
				data.hdrFilmData.attachLog = true;
				data.hdrFilmData.banner = true;
				data.hdrFilmData.highQualityEdges = false;
			}
			break;
			case EFilmType::TILED_HDR_FILM:
			{
				data.tiledHdrFilmData.componentFormat = EComponentFormat::FLOAT16;
			}
			break;
			case EFilmType::LDR_FILM:
			{
				data.ldrFilmData.fileFormat = ELDRFileFormat::PNG;
				data.ldrFilmData.tonemapMethod = ETonemapMethod::GAMMA;
				data.ldrFilmData.gamma = -1.0f;
				data.ldrFilmData.exposure = 0;
				data.ldrFilmData.key = 0.18f;
				data.ldrFilmData.burn = 0.0f;
				data.ldrFilmData.banner = true;
				data.ldrFilmData.highQualityEdges = false;
			}
			break;
			case EFilmType::M_FILM:
			{
				data.width = 1;
				data.height = 1;
				data.pixelFormat = EPixelFormat::LUMINANCE;

				data.mFilmData.fileFormat = EMFileFormat::MATLAB;
				data.mFilmData.digits = 4;
				data.mFilmData.variable = "data";
				data.mFilmData.highQualityEdges = true;
			}
			break;
			}
		}
	}

	return true;
}

bool CElementFilm::onEndTag(asset::IAssetManager* _assetManager)
{
	switch (data.type)
	{
	case EFilmType::HDR_FILM:
		return processHDRFilmProperties();

	case EFilmType::TILED_HDR_FILM:
		return processTiledHDRFilmProperties();

	case EFilmType::LDR_FILM:
		return processLDRFilmProperties();

	case EFilmType::M_FILM:
		return processMFilmProperties();

	default:
		_IRR_DEBUG_BREAK_IF(true);
		return false;
	}
}

bool CElementFilm::processSharedDataProperty(const SPropertyElementData& _property)
{
	if (_property.type == SPropertyElementData::Type::INTEGER)
	{
		if (_property.name == "width")
		{
			data.width = CPropertyElementManager::retriveIntValue(_property.value);
		}
		else if (_property.name == "height")
		{
			data.height = CPropertyElementManager::retriveIntValue(_property.value);
		}
		else if (_property.name == "cropOffsetX")
		{
			data.isCropUsed = true;
			data.height = CPropertyElementManager::retriveIntValue(_property.value);
		}
		else if (_property.name == "cropOffsetY")
		{
			data.isCropUsed = true;
			data.height = CPropertyElementManager::retriveIntValue(_property.value);
		}
		else if (_property.name == "cropWidth")
		{
			data.isCropUsed = true;
			data.height = CPropertyElementManager::retriveIntValue(_property.value);
		}
		else if (_property.name == "cropHeight")
		{
			data.isCropUsed = true;
			data.height = CPropertyElementManager::retriveIntValue(_property.value);
		}
		else
		{
			ParserLog::invalidXMLFileStructure("unknown property");
			_IRR_DEBUG_BREAK_IF(true);
			return false;
		}
	}
	else
	if (_property.type == SPropertyElementData::Type::STRING &&
		_property.name == "pixelFormat")
	{
		if (_property.value == "luminance")      data.pixelFormat = EPixelFormat::LUMINANCE; else
		if (_property.value == "luminanceAlpha") data.pixelFormat = EPixelFormat::LUMINANCE_ALPHA; else
		if (_property.value == "rgb")            data.pixelFormat = EPixelFormat::RGB; else
		if (_property.value == "rgba")           data.pixelFormat = EPixelFormat::RGBA; else
		if (_property.value == "xyz")            data.pixelFormat = EPixelFormat::XYZ; else
		if (_property.value == "xyza")           data.pixelFormat = EPixelFormat::XYZA; else
		if (_property.value == "spectrum")       data.pixelFormat = EPixelFormat::SPECTRUM; else
		if (_property.value == "spectrumAlpha")  data.pixelFormat = EPixelFormat::SPECTRUM_ALPHA; else
		{
			ParserLog::invalidXMLFileStructure("unknown property");
			_IRR_DEBUG_BREAK_IF(true);
			return false;
		}
	}
	else
	{
		ParserLog::invalidXMLFileStructure("unkown property");
		_IRR_DEBUG_BREAK_IF(true);
		return false;
	}
	
	return true;
}

bool CElementFilm::processHDRFilmProperties()
{
	for (const SPropertyElementData& property : properties)
	{
		if (property.type == SPropertyElementData::Type::STRING &&
			property.name == "fileFormat")
		{
			if (property.value == "openexr") data.hdrFilmData.fileFormat = EHDRFileFormat::OPENEXR; else
			if (property.value == "rgbe")    data.hdrFilmData.fileFormat = EHDRFileFormat::RGBE; else
			if (property.value == "pfm")     data.hdrFilmData.fileFormat = EHDRFileFormat::PFM; else
			{
				ParserLog::invalidXMLFileStructure("unknown property");
				_IRR_DEBUG_BREAK_IF(true);
				return false;
			}
		}
		else
		if (property.type == SPropertyElementData::Type::STRING &&
			property.name == "componentFormat")
		{
			if (property.value == "float16")	data.hdrFilmData.componentFormat = EComponentFormat::FLOAT16; else
			if (property.value == "float32")    data.hdrFilmData.componentFormat = EComponentFormat::FLOAT32; else
			if (property.value == "uint32")     data.hdrFilmData.componentFormat = EComponentFormat::UINT32; else
			{
				ParserLog::invalidXMLFileStructure("unknown property");
				_IRR_DEBUG_BREAK_IF(true);
				return false;
			}
		}
		else
		if (property.type == SPropertyElementData::Type::BOOLEAN)
		{
			if (property.name == "attachLog")
			{
				data.hdrFilmData.attachLog = CPropertyElementManager::retriveBooleanValue(property.value);
			}
			else if (property.name == "banner")
			{
				data.hdrFilmData.banner = CPropertyElementManager::retriveBooleanValue(property.value);
			}
			else if (property.name == "hightQualityEdges")
			{
				data.hdrFilmData.highQualityEdges = CPropertyElementManager::retriveBooleanValue(property.value);
			}
			else
			{
				ParserLog::invalidXMLFileStructure("unknown property");
				_IRR_DEBUG_BREAK_IF(true);
				return false;
			}
		}
		else if(!processSharedDataProperty(property))
		{
			return false;
		}
	}

	return true;
}

bool CElementFilm::processTiledHDRFilmProperties()
{
	for (const SPropertyElementData& property : properties)
	{
		if (property.type == SPropertyElementData::Type::STRING &&
			property.name == "componentFormat")
		{
			if (property.value == "float16")	data.tiledHdrFilmData.componentFormat = EComponentFormat::FLOAT16; else
			if (property.value == "float32")    data.tiledHdrFilmData.componentFormat = EComponentFormat::FLOAT32; else
			if (property.value == "uint32")     data.tiledHdrFilmData.componentFormat = EComponentFormat::UINT32; 
			else
			{
				ParserLog::invalidXMLFileStructure("unknown property");
				_IRR_DEBUG_BREAK_IF(true);
				return false;
			}
		}
		else if (!processSharedDataProperty(property))
		{
			return false;
		}
	}

	return true;
}

bool CElementFilm::processLDRFilmProperties()
{
	for (const SPropertyElementData& property : properties)
	{
		if (property.type == SPropertyElementData::Type::STRING &&
			property.name == "fileFormat")
		{
			if (property.value == "png")  data.ldrFilmData.fileFormat = ELDRFileFormat::PNG; else
			if (property.value == "jpeg") data.ldrFilmData.fileFormat = ELDRFileFormat::JPEG; else
			{
				ParserLog::invalidXMLFileStructure("unknown property");
				_IRR_DEBUG_BREAK_IF(true);
				return false;
			}
		}
		else
		if (property.type == SPropertyElementData::Type::STRING &&
			property.name == "tonemapMethod")
		{
			if (property.value == "gamma")    data.ldrFilmData.tonemapMethod = ETonemapMethod::GAMMA; else
			if (property.value == "reinhard") data.ldrFilmData.tonemapMethod = ETonemapMethod::REINHARD; else
			{
				ParserLog::invalidXMLFileStructure("unknown property");
				_IRR_DEBUG_BREAK_IF(true);
				return false;
			}
		}
		else
		if (property.type == SPropertyElementData::Type::BOOLEAN)
		{
			if (property.name == "banner")
			{
				data.ldrFilmData.banner = CPropertyElementManager::retriveBooleanValue(property.value);
			}
			else if (property.name == "hightQualityEdges")
			{
				data.ldrFilmData.highQualityEdges = CPropertyElementManager::retriveBooleanValue(property.value);
			}
			else
			{
				ParserLog::invalidXMLFileStructure("unknown property");
				_IRR_DEBUG_BREAK_IF(true);
				return false;
			}
		}
		else
		if (property.type == SPropertyElementData::Type::FLOAT)
		{
			if (property.name == "gamma")
			{
				data.ldrFilmData.gamma = CPropertyElementManager::retriveFloatValue(property.value);
			}
			else if (property.name == "exposure")
			{
				data.ldrFilmData.exposure = CPropertyElementManager::retriveFloatValue(property.value);
			}
			else if (property.name == "key")
			{
				data.ldrFilmData.key = CPropertyElementManager::retriveFloatValue(property.value);
			}
			else if (property.name == "burn")
			{
				data.ldrFilmData.burn = CPropertyElementManager::retriveFloatValue(property.value);
			}
			else
			{
				ParserLog::invalidXMLFileStructure("unknown property");
				_IRR_DEBUG_BREAK_IF(true);
				return false;
			}
		}
		else if(!processSharedDataProperty(property))
		{
			return false;
		}
	}

	return true;
}

bool CElementFilm::processMFilmProperties()
{
	for (const SPropertyElementData& property : properties)
	{
		if (property.type == SPropertyElementData::Type::STRING)
		{
			if (property.name == "fileFormat")
			{
				if (property.value == "matlab")      data.mFilmData.fileFormat = EMFileFormat::MATLAB; else
				if (property.value == "mathematica") data.mFilmData.fileFormat = EMFileFormat::MATHEMATICA; else
				if (property.value == "numpy")       data.mFilmData.fileFormat = EMFileFormat::NUMPY; else
				{
					ParserLog::invalidXMLFileStructure("unknown property");
					_IRR_DEBUG_BREAK_IF(true);
					return false;
				}
			}
			else if (property.name == "variable")
			{
				data.mFilmData.variable = property.value;
			}
			else if(property.name != "pixelFormat")
			{
				ParserLog::invalidXMLFileStructure("unknown property");
				_IRR_DEBUG_BREAK_IF(true);
				return false;
			}
		}
		else
		if (property.type == SPropertyElementData::Type::INTEGER &&
			property.name == "digits")
		{
			data.mFilmData.digits = CPropertyElementManager::retriveIntValue(property.value);
		}
		else
		if (property.type == SPropertyElementData::Type::BOOLEAN &&
			property.name == "highQualityEdges")
		{
			data.mFilmData.highQualityEdges = CPropertyElementManager::retriveBooleanValue(property.value);
		}
		else if(!processSharedDataProperty(property))
		{
			return false;
		}
	}

	return true;
}

}
}
}