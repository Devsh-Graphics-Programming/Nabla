#include "../../ext/MitsubaLoader/PropertyElement.h"
#include "../../ext/MitsubaLoader/ParserUtil.h"

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

const core::unordered_map<std::string,SPropertyElementData::Type,CaseInsensitiveHash,CaseInsensitiveEquals> SPropertyElementData::StringToType = {
	{"float",		SPropertyElementData::Type::FLOAT},
	{"integer",		SPropertyElementData::Type::INTEGER},
	{"boolean",		SPropertyElementData::Type::BOOLEAN},
	{"string",		SPropertyElementData::Type::STRING},
	{"rgb",			SPropertyElementData::Type::RGB},
	{"srgb",		SPropertyElementData::Type::SRGB},
	{"spectrum",	SPropertyElementData::Type::SPECTRUM},
	{"matrix",		SPropertyElementData::Type::MATRIX},
	{"translate",	SPropertyElementData::Type::TRANSLATE},
	{"rotate",		SPropertyElementData::Type::ROTATE},
	{"scale",		SPropertyElementData::Type::SCALE},
	{"lookat",		SPropertyElementData::Type::LOOKAT},
	{"point",		SPropertyElementData::Type::POINT},
	{"vector",		SPropertyElementData::Type::VECTOR}
};

std::pair<bool, SPropertyElementData> CPropertyElementManager::createPropertyData(const char* _el, const char** _atts)
{
	SPropertyElementData result(_el);
	const char* value = result.initialize(_atts);
	if (!value)
	{
		_IRR_DEBUG_BREAK_IF(true);
		return std::make_pair(false, SPropertyElementData());
	}

	bool success = true;
	switch (result.type)
	{
		case SPropertyElementData::Type::FLOAT:
			result.fvalue = atof(value);
			break;
		case SPropertyElementData::Type::INTEGER:
			result.ivalue = atoi(value);
			break;
		case SPropertyElementData::Type::BOOLEAN:
			result.bvalue = retriveBooleanValue(value);
			break;
		case SPropertyElementData::Type::STRING:
			auto len = strlen(value);
			auto* tmp = (char*)_IRR_ALIGNED_MALLOC(len + 1u, 64u);
			strcpy(tmp, value); tmp[len] = 0;
			result.svalue = tmp;
			break;
		case SPropertyElementData::Type::RGB:
		case SPropertyElementData::Type::SRGB:
		case SPropertyElementData::Type::VECTOR:
		case SPropertyElementData::Type::POINT:
			vvalue = other.vvalue;
			break;
		case SPropertyElementData::Type::SPECTRUM:
			assert(false);
			break;
		case SPropertyElementData::Type::MATRIX:
		case SPropertyElementData::Type::TRANSLATE:
		case SPropertyElementData::Type::ROTATE:
		case SPropertyElementData::Type::SCALE:
		case SPropertyElementData::Type::LOOKAT:
			mvalue = other.mvalue;
			break;
		default:
			success = false;
			break;
	}

	_IRR_DEBUG_BREAK_IF(!success);
	if (success)
		return std::make_pair(true, result);

	ParserLog::invalidXMLFileStructure("invalid element, name:\'" + result.name + "\' value:\'" + value + "\'");
	return std::make_pair(false, SPropertyElementData());

	if (elName == "translate")
	{
		result.type = SPropertyElementData::Type::TRANSLATE;

		result.value = findAndConvertXYZAttsToSingleString(_atts, errorOccurred, { "name" });
		return std::make_pair(true, result);
	}
	else if (elName == "rotate")
	{
		result.type = SPropertyElementData::Type::ROTATE;

		bool errorOccured = false;
		const std::string axisStr = findAndConvertXYZAttsToSingleString(_atts, errorOccured, { "name", "angle" });

		if (errorOccured)
			return std::make_pair(false, SPropertyElementData());

		const core::vector3df_SIMD rotationAxis = core::normalize(retriveVector(axisStr));

		if (rotationAxis.getLengthAsFloat() == 0.0f)
		{
			ParserLog::invalidXMLFileStructure("invalid rotation axis");
			return std::make_pair(false, SPropertyElementData());
		}

		float angle = 0.0f;

		for (int i = 0; _atts[i]; i += 2)
		{
			if (!std::strcmp(_atts[i], "angle"))
			{
				angle = retriveFloatValue(_atts[i + 1]);
			}
		}
		
		core::quaternion resultingQuaternion = core::quaternion::fromAngleAxis(angle, rotationAxis);

		std::stringstream ss;
		float tmpCoord;

		for (int i = 0; i < 4; i++)
		{
			tmpCoord = resultingQuaternion.getPointer()[i];
			ss << tmpCoord << ' ';
		}
			
		
		result.value = ss.str();

		return std::make_pair(true, result);
	}
	else if (elName == "scale")
	{
		result.type = SPropertyElementData::Type::SCALE;

		bool errorOccured = false;
		//value should contain only one number here!
		const std::string val = findStandardValue(_atts, errorOccured, { "name", "x", "y", "z" });

		if (errorOccured)
		{
			_IRR_DEBUG_BREAK_IF(true);
			return std::make_pair(false, SPropertyElementData());
		}

		if (val != "not set")
		{
			result.value = val + ' ' + val + ' ' + val;
			return std::make_pair(true, result);
		}

		result.value = findAndConvertXYZAttsToSingleString(_atts, errorOccured, { "name" });

		if (errorOccured)
		{
			_IRR_DEBUG_BREAK_IF(true);
			return std::make_pair(false, SPropertyElementData());
		}

		return std::make_pair(true, result);

	}
	else if (elName == "lookat")
	{
		result.type = SPropertyElementData::Type::LOOKAT;

		std::string originStr = "0.0 0.0 0.0";
		std::string targetStr = "0.0 0.0 -1.0";
		std::string upStr = "0.0 1.0 0.0";

		for (int i = 0; _atts[i]; i += 2)
		{
			if (!std::strcmp(_atts[i], "origin"))
			{
				originStr = _atts[i + 1];
			}
			else if (!std::strcmp(_atts[i], "target"))
			{
				targetStr = _atts[i + 1];
			}
			else if (!std::strcmp(_atts[i], "up"))
			{
				upStr = _atts[i + 1];
			}
		}

		core::vector3df_SIMD origin = retriveVector(originStr);
		core::vector3df_SIMD target = retriveVector(targetStr);
		core::vector3df_SIMD up = retriveVector(upStr);

		core::matrix4SIMD lookAt = core::matrix4SIMD::buildCameraLookAtMatrixRH(origin, target, up);

		std::stringstream ss;
		
		for (int i = 0; i < 16; i++)
		{
			ss << lookAt.pointer()[i] << ' ';
		}

		result.value = ss.str();

		return std::make_pair(true, result);
	}
	else if (elName == "point")
	{
		result.type = SPropertyElementData::Type::POINT;

		bool errorOccurred = false;
		result.value = findStandardValue(_atts, errorOccurred, { "name", "x", "y", "z" });

		if (errorOccurred)
			return std::make_pair(false, SPropertyElementData());

		if (result.value != "not set")
			return std::make_pair(true, result);

		result.value = findAndConvertXYZAttsToSingleString(_atts, errorOccurred, { "name" });

		if (errorOccurred)
			return std::make_pair(false, SPropertyElementData());

		return std::make_pair(true, result);
		
	}
	else if (elName == "vector")
	{
		result.type = SPropertyElementData::Type::VECTOR;
		_IRR_DEBUG_BREAK_IF(true);
	}

	bool errorOccurred;
	result.value = findStandardValue(_atts, errorOccurred, { "name" });
}

bool CPropertyElementManager::retrieveBooleanValue(const std::string& _data, bool& success)
{
	if (_data == "true")
	{
		return true;
	}
	else if (_data == "false")
	{
		return false;
	}
	else
	{
		_IRR_DEBUG_BREAK_IF(true);
		ParserLog::invalidXMLFileStructure("Invalid boolean specified.");
		success = false;
	}
}

core::matrix4SIMD CPropertyElementManager::retrieveMatrix(const std::string& _data, bool& success)
{
	std::string str = _data;
	std::replace(str.begin(), str.end(), ',', ' ');

	float matrixData[16];
	std::stringstream ss;
	ss << str;

	for (int i = 0; i < 16; i++)
	{
		float f = std::numeric_limits<float>::quiet_NaN();
		ss >> f;

		matrixData[i] = f;

		if (isnan(f))
		{
			_IRR_DEBUG_BREAK_IF(true);
			ParserLog::invalidXMLFileStructure("Invalid matrix specified.");
			success = false;
			return core::matrix4SIMD();
		}
	}

	return core::matrix4SIMD(matrixData);
}

core::vectorSIMDf CPropertyElementManager::retrieveVector(const std::string& _data, bool& success)
{
	std::string str = _data;
	std::replace(str.begin(), str.end(), ',', ' ');

	float vectorData[4];
	std::stringstream ss;
	ss << str;

	for (int i = 0; i < 4; i++)
	{
		float f = std::numeric_limits<float>::quiet_NaN();
		ss >> f;

		vectorData[i] = f;

		if (isnan(f))
		{
			if (i == 3)
			{
				vectorData[3] = 0.0f;
				break;
			}

			_IRR_DEBUG_BREAK_IF(true);
			ParserLog::invalidXMLFileStructure("Invalid vector specified.");
			success = false;
			return core::vectorSIMDf();
		}
	}

	return core::vectorSIMDf(vectorData);
}

std::string CPropertyElementManager::findStandardValue(const char** _atts, bool& _errorOccurred, const core::vector<std::string>& _acceptableAttributes)
{
	std::string value = "not set";

	for (int i = 0; _atts[i]; i += 2)
	{
		if (!std::strcmp(_atts[i], "value"))
		{
			value = _atts[i + 1];
		}
		else
		{
			bool acceptableAttrFound = false;
			const std::string currAttr = _atts[i];
			for (const std::string& attr : _acceptableAttributes)
			{
				if (currAttr == attr)
				{
					acceptableAttrFound = true;
					break;
				}
			}

			if (acceptableAttrFound)
				continue;

			ParserLog::invalidXMLFileStructure(std::string(_atts[i]) + "is not an attribute of the property element");
			_IRR_DEBUG_BREAK_IF(true);
			_errorOccurred = true;
			return value;
		}
	}

	_errorOccurred = false;
	return value;
}

std::string CPropertyElementManager::findAndConvertXYZAttsToSingleString(const char** _atts, bool& _errorOccurred, const core::vector<std::string>& _acceptableAttributes)
{
	std::string values[3] = { "0.0", "0.0", "0.0" };

	for (int i = 0; _atts[i]; i += 2)
	{
		if (!std::strcmp(_atts[i], "x"))
		{
			values[0] = _atts[i + 1];
		}
		else if (!std::strcmp(_atts[i], "y"))
		{
			values[1] = _atts[i + 1];
		}
		else if (!std::strcmp(_atts[i], "z"))
		{
			values[2] = _atts[i + 1];
		}
		else
		{
			bool acceptableAttrFound = false;
			const std::string currAttr = _atts[i];
			for (const std::string& attr : _acceptableAttributes)
			{
				if (currAttr == attr)
				{
					acceptableAttrFound = true;
					break;
				}
			}

			if (acceptableAttrFound)
				break;

			ParserLog::invalidXMLFileStructure(std::string(_atts[i]) + "is not an attribute of the property element");
			_IRR_DEBUG_BREAK_IF(true);
			_errorOccurred = true;
			return "";
		}
	}

	_errorOccurred = false;
	std::string result = values[0] + ' ' + values[1] + ' ' + values[2];
	return result;
}

}
}
}