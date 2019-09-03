#include "../../ext/MitsubaLoader/CElementEmitter.h"
#include "../../ext/MitsubaLoader/CElementTransform.h"
#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/PropertyElement.h"


#include <functional>

namespace irr { namespace ext { namespace MitsubaLoader {

bool CElementEmitter::processAttributes(const char** _atts)
{
	static const core::unordered_map<std::string, EEmitterType> acceptableTypes = {
		std::make_pair("point", EEmitterType::POINT),
		std::make_pair("area", EEmitterType::AREA),
		std::make_pair("spot", EEmitterType::SPOT),
		std::make_pair("directional", EEmitterType::DIRECTIONAL),
		std::make_pair("collimated", EEmitterType::COLLIMATED),
		std::make_pair("sky", EEmitterType::SKY),
		std::make_pair("sunsky", EEmitterType::SUNSKY),
		std::make_pair("envmap", EEmitterType::ENVMAP),
		std::make_pair("constant", EEmitterType::CONSTANT)
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
			case EEmitterType::POINT:
			{
				data.pointData.position = core::vector3df_SIMD(0.0f);
				data.pointData.intensity = video::SColorf(1.0f);
				data.pointData.samplingWeight = 1.0f;

			}
			break;
			case EEmitterType::AREA:
			{
				data.areaData.radiance = video::SColorf(1.0f);
				data.areaData.samplingWeight = 1.0f;
			}
			break;
			case EEmitterType::SPOT:
			{
				data.spotData.intensity = video::SColorf(1.0f);
				data.spotData.cutoffAngle = 20.0f;
				data.spotData.beamWidth = 20.0f * (3.0f / 4.0f);
				data.spotData.samplingWeight = 1.0f;
			}
			break;
			case EEmitterType::DIRECTIONAL:
			{
				data.directionalData.direction = core::vector3df_SIMD(0.0f,-1.0f, 0.0f);
				data.directionalData.irradiance = video::SColorf(1.0f);
				data.directionalData.samplingWeight = 1.0f;
			}
			break;
			case EEmitterType::COLLIMATED:
			{
				data.collimatedData.power = video::SColorf(1.0f);
				data.collimatedData.samplingWeight = 1.0f;
			}
			break;
			case EEmitterType::SKY:
			case EEmitterType::SUN:
			case EEmitterType::SUNSKY:
			case EEmitterType::ENVMAP:
			{
				ParserLog::invalidXMLFileStructure("not supported yet.");
				_IRR_DEBUG_BREAK_IF(true);
				return false;
			}
			break;
			case EEmitterType::CONSTANT:
			{
				data.constantData.radiance = video::SColorf(1.0f);
				data.constantData.samplingWeight = 1.0f;
			}
			break;
			default:
				assert(false);
			}
		}
	}

	return true;
}

bool CElementEmitter::processChildData(IElement* _child)
{
	switch (_child->getType())
	{
	case IElement::Type::TRANSFORM:
	{
		transform = static_cast<CElementTransform*>(_child)->getMatrix();
		return true;
	}
	default:
	{
		ParserLog::invalidXMLFileStructure(_child->getLogName() + "is not a child of sensor element");
		_IRR_DEBUG_BREAK_IF(true);
		return false;
	}
	}
}

bool CElementEmitter::onEndTag(asset::IAssetManager& _assetManager)
{
	switch (data.type)
	{
	case EEmitterType::POINT:
		return processPointEmitterProperties();

	case EEmitterType::AREA:
		return processAreaEmitterProperties();

	case EEmitterType::SPOT:
		return processSpotEmitterProperties();

	case EEmitterType::DIRECTIONAL:
		return processDirectionalEmitterProperties();

	case EEmitterType::COLLIMATED:
		return processCollimatedEmitterProperties();

	case EEmitterType::SKY:
	case EEmitterType::SUN:
	case EEmitterType::SUNSKY:
	case EEmitterType::ENVMAP:
		ParserLog::invalidXMLFileStructure("not supported yet.");
		_IRR_DEBUG_BREAK_IF(true);
		return false;

	case EEmitterType::CONSTANT:
		return processConstantEmitterProperties();

	default:
		_IRR_DEBUG_BREAK_IF(true);
		return false;
	}
}

bool CElementEmitter::processSharedDataProperty(const SPropertyElementData& _property)
{
	if (_property.type == SPropertyElementData::Type::FLOAT)
	{
		if (_property.name == "shutterOpen")
		{
			data.shutterOpen = CPropertyElementManager::retriveIntValue(_property.value);
		}
		else if (_property.name == "shutterClose")
		{
			data.shutterClose = CPropertyElementManager::retriveIntValue(_property.value);
		}
		else
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

bool CElementEmitter::processPointEmitterProperties()
{
	for (const SPropertyElementData& property : properties)
	{
		if (property.type == SPropertyElementData::Type::POINT &&
			property.name == "position")
		{
			data.pointData.position = CPropertyElementManager::retriveVector(property.value);
		}
		else
		if ((property.type == SPropertyElementData::Type::RGB ||
			property.type == SPropertyElementData::Type::SRGB) &&
			property.name == "intensity")
		{
			_IRR_DEBUG_BREAK_IF(true);
			return false;	
		}
		else
		if (property.type == SPropertyElementData::Type::FLOAT &&
			property.name == "samplingWeight")
		{
			data.pointData.samplingWeight = CPropertyElementManager::retriveFloatValue(property.value);
		}
		else if(!processSharedDataProperty(property))
		{
			return false;
		}
	}

	return true;
}

bool CElementEmitter::processAreaEmitterProperties()
{
	_IRR_DEBUG_BREAK_IF(true);
	return true;
}

bool CElementEmitter::processSpotEmitterProperties()
{
	_IRR_DEBUG_BREAK_IF(true);
	return true;
}

bool CElementEmitter::processDirectionalEmitterProperties()
{
	_IRR_DEBUG_BREAK_IF(true);
	return true;
}


bool CElementEmitter::processCollimatedEmitterProperties()
{
	_IRR_DEBUG_BREAK_IF(true);
	return true;
}

bool CElementEmitter::processConstantEmitterProperties()
{
	for (const SPropertyElementData& property : properties)
	{
		if ((property.type == SPropertyElementData::Type::RGB ||
			property.type == SPropertyElementData::Type::SRGB) &&
			property.name == "radiance")
		{
			_IRR_DEBUG_BREAK_IF(true);
			return false;	
		}
		else
		if (property.type == SPropertyElementData::Type::FLOAT &&
			property.name == "samplingWeight")
		{
			data.constantData.samplingWeight = CPropertyElementManager::retriveFloatValue(property.value);
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