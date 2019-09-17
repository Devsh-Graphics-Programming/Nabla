#include "../../ext/MitsubaLoader/CElementSensor.h"
#include "../../ext/MitsubaLoader/CElementTransform.h"
#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../../ext/MitsubaLoader/PropertyElement.h"


#include <functional>

namespace irr { namespace ext { namespace MitsubaLoader {

bool CElementSensor::processAttributes(const char** _atts)
{
	static const core::unordered_map<std::string, ESensorType> acceptableTypes = {
		std::make_pair("perspective", ESensorType::PERSPECTIVE),
		std::make_pair("thinlens", ESensorType::THINLENS),
		std::make_pair("orthographic", ESensorType::ORTHOGRAPHIC),
		std::make_pair("telecentric", ESensorType::TELECENTRIC),
		std::make_pair("spherical", ESensorType::SPHERICAL),
		std::make_pair("irradiancemeter", ESensorType::IRRADIANCEMETER),
		std::make_pair("radiancemeter", ESensorType::RADIANCEMETER),
		std::make_pair("fluencemeter", ESensorType::FLUENCEMETER)
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
			case ESensorType::PERSPECTIVE:
			{
				data.perspectiveData.focalLength = "50mm";
				data.perspectiveData.fovAxis = EFOVAxis::X;
				data.perspectiveData.fov = 0.0f;
				data.perspectiveData.nearClip = 0.01f;
				data.perspectiveData.nearClip = 10000.0f;

			}
			break;
			case ESensorType::THINLENS:
			{
				data.thinlensData.focalLength = "50mm";
				data.thinlensData.fovAxis = EFOVAxis::X;
				data.thinlensData.fov = 0.0f;
				data.thinlensData.nearClip = 0.01f;
				data.thinlensData.nearClip = 10000.0f;
				data.thinlensData.apertureRadius = 0.0f;
				data.thinlensData.focusDistance = 0.0f;
			}
			break;
			case ESensorType::ORTHOGRAPHIC:
			{
				data.thinlensData.nearClip = 0.01f;
				data.thinlensData.nearClip = 10000.0f;
			}
			break;
			case ESensorType::TELECENTRIC:
			{
				data.thinlensData.nearClip = 0.01f;
				data.thinlensData.nearClip = 10000.0f;
				data.thinlensData.apertureRadius = 0.0f;
				data.thinlensData.focusDistance = 0.0f;
			}
			break;
			default:
				assert(true);
			}
		}
	}

	return true;
}

bool CElementSensor::processChildData(IElement* _child)
{
	switch (_child->getType())
	{
	case IElement::Type::FILM:	
	{
		data.filmData = static_cast<CElementFilm*>(_child)->getMetadata();
		return true;
	}

	case IElement::Type::SAMPLER:
	{
		data.samperData = static_cast<CElementSampler*>(_child)->getMetadata();
		return true;
	}
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

bool CElementSensor::onEndTag(asset::IAssetManager* _assetManager)
{
	switch (data.type)
	{
	case ESensorType::PERSPECTIVE:
		data.perspectiveData.toWorld = transform;
		return processPerspectiveSensorProperties();

	case ESensorType::THINLENS:
		data.thinlensData.toWorld = transform;
		return processThinlensSensorProperties();

	case ESensorType::ORTHOGRAPHIC:
		data.orthographicData.toWorld = transform;
		return processOrthographicSensorProperties();

	case ESensorType::TELECENTRIC:
		data.telecentricData.toWorld = transform;
		return processTelecentricSensorProperties();

	case ESensorType::SPHERICAL:
		data.sphericalData.toWorld = transform;
		return true;

	case ESensorType::IRRADIANCEMETER:
		return true;

	case ESensorType::RADIANCEMETER:
		data.telecentricData.toWorld = transform;
		return true;

	case ESensorType::FLUENCEMETER:
		data.telecentricData.toWorld = transform;
		return true;

	default:
		_IRR_DEBUG_BREAK_IF(true);
		return false;
	}
}

bool CElementSensor::processSharedDataProperty(const SPropertyElementData& _property)
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

bool CElementSensor::processPerspectiveSensorProperties()
{
	for (const SPropertyElementData& property : properties)
	{
		if (property.type == SPropertyElementData::Type::STRING &&
			property.name == "focalLength")
		{
			data.perspectiveData.focalLength = property.value;
		}
		else
		if (property.type == SPropertyElementData::Type::STRING &&
			property.name == "fovAxis")
		{
			if (property.value == "x")	      data.perspectiveData.fovAxis = EFOVAxis::X; else
			if (property.value == "y")        data.perspectiveData.fovAxis = EFOVAxis::Y; else
			if (property.value == "diagonal") data.perspectiveData.fovAxis = EFOVAxis::DIAGONAL; else
			if (property.value == "smaller")  data.perspectiveData.fovAxis = EFOVAxis::SMALLER; else
			if (property.value == "larger")   data.perspectiveData.fovAxis = EFOVAxis::LARGER; else
			{
				ParserLog::invalidXMLFileStructure("unknown property");
				_IRR_DEBUG_BREAK_IF(true);
				return false;
			}
		}
		else
		if (property.type == SPropertyElementData::Type::FLOAT)
		{
			if (property.name == "fov")
			{
				data.perspectiveData.fov = CPropertyElementManager::retriveFloatValue(property.value);
			}
			else if (property.name == "nearClip")
			{
				data.perspectiveData.nearClip = CPropertyElementManager::retriveFloatValue(property.value);
			}
			else if (property.name == "farClip")
			{
				data.perspectiveData.farClip = CPropertyElementManager::retriveFloatValue(property.value);
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

bool CElementSensor::processThinlensSensorProperties()
{
	for (const SPropertyElementData& property : properties)
	{
		if (property.type == SPropertyElementData::Type::STRING &&
			property.name == "focalLength")
		{
			data.perspectiveData.focalLength = property.value;
		}
		else
		if (property.type == SPropertyElementData::Type::STRING &&
			property.name == "fovAxis")
		{
			if (property.value == "x")	      data.thinlensData.fovAxis = EFOVAxis::X; else
			if (property.value == "y")        data.thinlensData.fovAxis = EFOVAxis::Y; else
			if (property.value == "diagonal") data.thinlensData.fovAxis = EFOVAxis::DIAGONAL; else
			if (property.value == "smaller")  data.thinlensData.fovAxis = EFOVAxis::SMALLER; else
			if (property.value == "larger")   data.thinlensData.fovAxis = EFOVAxis::LARGER; else
			{
				ParserLog::invalidXMLFileStructure("unknown property");
				_IRR_DEBUG_BREAK_IF(true);
				return false;
			}
		}
		else
		if (property.type == SPropertyElementData::Type::FLOAT)
		{
			if (property.name == "fov")
			{
				data.thinlensData.fov = CPropertyElementManager::retriveFloatValue(property.value);
			}
			else if (property.name == "nearClip")
			{
				data.thinlensData.nearClip = CPropertyElementManager::retriveFloatValue(property.value);
			}
			else if (property.name == "farClip")
			{
				data.thinlensData.farClip = CPropertyElementManager::retriveFloatValue(property.value);
			}
			else if (property.name == "apertureRadius")
			{
				data.thinlensData.apertureRadius = CPropertyElementManager::retriveFloatValue(property.value);
			}
			else if (property.name == "focusDistance")
			{
				data.thinlensData.focusDistance = CPropertyElementManager::retriveFloatValue(property.value);
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

bool CElementSensor::processOrthographicSensorProperties()
{
	for (const SPropertyElementData& property : properties)
	{
		if (property.type == SPropertyElementData::Type::FLOAT)
		{
			
			if (property.name == "nearClip")
			{
				data.perspectiveData.nearClip = CPropertyElementManager::retriveFloatValue(property.value);
			}
			else if (property.name == "farClip")
			{
				data.perspectiveData.farClip = CPropertyElementManager::retriveFloatValue(property.value);
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

bool CElementSensor::processTelecentricSensorProperties()
{
	for (const SPropertyElementData& property : properties)
	{
		if (property.type == SPropertyElementData::Type::FLOAT)
		{
			
			if (property.name == "nearClip")
			{
				data.perspectiveData.nearClip = CPropertyElementManager::retriveFloatValue(property.value);
			}
			else if (property.name == "farClip")
			{
				data.perspectiveData.farClip = CPropertyElementManager::retriveFloatValue(property.value);
			}
			else if (property.name == "apertureRadius")
			{
				data.telecentricData.apertureRadius = CPropertyElementManager::retriveFloatValue(property.value);
			}
			else if (property.name == "focusDistance")
			{
				data.telecentricData.focusDistance = CPropertyElementManager::retriveFloatValue(property.value);
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


}
}
}