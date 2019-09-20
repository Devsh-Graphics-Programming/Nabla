#include "../../ext/MitsubaLoader/PropertyElement.h"
#include "../../ext/MitsubaLoader/ParserUtil.h"

#include "irr/asset/format/decodePixels.h"

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
			result.bvalue = retrieveBooleanValue(value,success);
			break;
		case SPropertyElementData::Type::STRING:
			auto len = strlen(value);
			auto* tmp = (char*)_IRR_ALIGNED_MALLOC(len + 1u, 64u);
			strcpy(tmp, value); tmp[len] = 0;
			result.svalue = tmp;
			break;
		case SPropertyElementData::Type::RGB:
			result.vvalue = retrieveVector(value, success);
			break;
		case SPropertyElementData::Type::SRGB:
			{
				bool tryVec = true;
				result.vvalue = retrieveVector(value, tryVec);
				if (!tryVec)
					result.vvalue = retrieveHex(value, success);
				for (auto i=0; i<3u; i++)
					result.vvalue[i] = video::impl::srgb2lin(result.vvalue[i]);
			}
			break;
		case SPropertyElementData::Type::VECTOR:
		case SPropertyElementData::Type::POINT:
			// only x,y,z acceptable
			vvalue = other.vvalue;
			break;
		case SPropertyElementData::Type::SPECTRUM:
		case SPropertyElementData::Type::BLACKBODY:
			result.type = SPropertyElementData::Type::INVALID;
			break;
		case SPropertyElementData::Type::MATRIX:
			// value
		case SPropertyElementData::Type::TRANSLATE:
			// only x,y,z acceptable
		case SPropertyElementData::Type::ROTATE:
			// only x,y,z + angle
		case SPropertyElementData::Type::SCALE:
			// value or x,y,z
		case SPropertyElementData::Type::LOOKAT:
			// origin + target + optional up
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
		// find out if we go by "value" or "x y z"

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
			success = false;
			return core::vectorSIMDf();
		}
	}

	return core::vectorSIMDf(vectorData);
}

core::vectorSIMDf CPropertyElementManager::retrieveHex(const std::string& _data, bool& success)
{
	core::vectorSIMDf zero;
	auto ptr = _data.begin();
	if (_data.size()!=7u || *ptr!='#')
	{
		success = false;
		return zero;
	}

	core::vectorSIMDf retval(0.f, 0.f, 0.f, 255.f);
	for (auto i = 0; i < 3; i++)
	for (auto j = 4; j >=0;j-=4)
	{
		char c = *(++ptr);
		if (!isxdigit(c))
		{
			success = false;
			return zero;
		}
		int intval = (c >= 'A') ? (c - 'A' + 10) : (c - '0');
		retval[i] += float(intval <<j);
	}
	return retval/255.f;
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