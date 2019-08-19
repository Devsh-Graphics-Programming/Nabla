#include "../../ext/MitsubaLoader/PropertyElement.h"
#include "../../ext/MitsubaLoader/ParserUtil.h"

namespace irr { namespace ext { namespace MitsubaLoader {

std::pair<bool, SPropertyElementData> CPropertyElementManager::createPropertyData(const char* _el, const char** _atts)
{
	SPropertyElementData result;
	std::string elName = _el;

	for (int i = 0; _atts[i]; i += 2)
	{
		if (!std::strcmp(_atts[i], "name"))
		{
			result.name = _atts[i + 1];
			break;
		}
	}

	if (elName == "float")
	{
		result.type = SPropertyElementData::Type::FLOAT;
	}
	else if (elName == "integer")
	{
		result.type = SPropertyElementData::Type::INTEGER;
	}
	else if (elName == "boolean")
	{
		result.type = SPropertyElementData::Type::BOOLEAN;
	}
	else if (elName == "string")
	{
		result.type = SPropertyElementData::Type::STRING;
	}
	else if (elName == "rgb")
	{
		result.type = SPropertyElementData::Type::RGB;
	}
	else if (elName == "srgb")
	{
		result.type = SPropertyElementData::Type::SRGB;
	}
	else if (elName == "spectrum")
	{
		result.type = SPropertyElementData::Type::SPECTRUM;
	}
	else if (elName == "matrix")
	{
		result.type = SPropertyElementData::Type::MATRIX;
	}
	else if (elName == "translate")
	{
		result.type = SPropertyElementData::Type::TRANSLATE;
	}
	else if (elName == "rotate")
	{
		result.type = SPropertyElementData::Type::ROTATE;
	}
	else if (elName == "scale")
	{
		result.type = SPropertyElementData::Type::SCALE;
	}
	else if (elName == "point")
	{
		result.type = SPropertyElementData::Type::POINT;

		std::string values[4] = { "0.0", "0.0", "0.0", "0.0" };

		for (int i = 0; _atts[i]; i += 2)
		{
			if (!std::strcmp(_atts[i], "name"))
			{
				continue;
			}
			if (!std::strcmp(_atts[i], "value"))
			{
				result.value = _atts[i + 1];
				return std::make_pair(true, result);
			}
			else if (!std::strcmp(_atts[i], "x"))
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
			else if (!std::strcmp(_atts[i], "w"))
			{
				values[3] = _atts[i + 1];
			}
			else
			{
				_IRR_DEBUG_BREAK_IF(false);
				return std::make_pair(false, SPropertyElementData());
			}
		}

		result.value = values[0] + ' ' + values[1] + ' ' + values[2] + ' ' + values[3];

		return std::make_pair(true, result);
	}
	else if (elName == "vector")
	{
		result.type = SPropertyElementData::Type::VECTOR;
		_IRR_DEBUG_BREAK_IF(true);
	}

	bool errorOccurred;
	result.value = findStandardValue(_atts, errorOccurred);

	if (errorOccurred)
		return std::make_pair(false, SPropertyElementData());

	return std::make_pair(true, result);		
}

float CPropertyElementManager::retriveFloatValue(const std::string& _data)
{
	std::stringstream ss;
	ss << _data;

	float result = std::numeric_limits<float>::quiet_NaN();

	ss >> result;
	return result;
}

int CPropertyElementManager::retriveIntValue(const std::string& _data)
{
	std::stringstream ss;
	ss << _data;

	int result = std::numeric_limits<int>::quiet_NaN();

	ss >> result;
	return result;
}

bool CPropertyElementManager::retriveBooleanValue(const std::string& _data)
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
		ParserLog::invalidXMLFileStructure(_data + "is not an attribute of boolean element");
	}
}

core::matrix4SIMD CPropertyElementManager::retriveMatrix(const std::string& _data)
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
			return core::matrix4SIMD();
		}
	}

	return core::matrix4SIMD(matrixData);
}

core::vectorSIMDf CPropertyElementManager::retriveVector(const std::string& _data)
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
			return core::vectorSIMDf();
		}
	}

	return core::vectorSIMDf(vectorData);
}

std::string CPropertyElementManager::findStandardValue(const char** _atts, bool& _errorOccurred)
{
	std::string value;
	bool isValueSet = false;

	for (int i = 0; _atts[i]; i += 2)
	{
		if (!std::strcmp(_atts[i], "value"))
		{
			value = _atts[i + 1];
			isValueSet = true;
		}
		else if (!std::strcmp(_atts[i], "name"))
		{
			continue;
		}
		else
		{
			ParserLog::invalidXMLFileStructure(std::string(_atts[i]) + "is not an attribute of the property element");
			_errorOccurred = true;
			return value;
		}
	}

	_errorOccurred = isValueSet ? false : true;
	return value;
}

}
}
}