// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "quaternion.h"
#include "matrix3x4SIMD.h"
#include "matrix4SIMD.h"
#include "nbl/asset/format/decodePixels.h"

#include "nbl/ext/MitsubaLoader/PropertyElement.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"

namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{

template<> const typename SPropertyElementData::get_typename<SPropertyElementData::Type::FLOAT>::type& SPropertyElementData::getProperty<SPropertyElementData::Type::FLOAT>() const
{ return fvalue; }
template<> const typename SPropertyElementData::get_typename<SPropertyElementData::Type::INTEGER>::type& SPropertyElementData::getProperty<SPropertyElementData::Type::INTEGER>() const
{ return ivalue; }
template<> const typename SPropertyElementData::get_typename<SPropertyElementData::Type::BOOLEAN>::type& SPropertyElementData::getProperty<SPropertyElementData::Type::BOOLEAN>() const
{ return bvalue; }
template<> const typename SPropertyElementData::get_typename<SPropertyElementData::Type::STRING>::type& SPropertyElementData::getProperty<SPropertyElementData::Type::STRING>() const
{ return svalue; }
template<> const typename SPropertyElementData::get_typename<SPropertyElementData::Type::RGB>::type& SPropertyElementData::getProperty<SPropertyElementData::Type::RGB>() const
{ return vvalue; }
template<> const typename SPropertyElementData::get_typename<SPropertyElementData::Type::SRGB>::type& SPropertyElementData::getProperty<SPropertyElementData::Type::SRGB>() const
{ return vvalue; }
template<> const typename SPropertyElementData::get_typename<SPropertyElementData::Type::SPECTRUM>::type& SPropertyElementData::getProperty<SPropertyElementData::Type::SPECTRUM>() const
{ return vvalue; }
template<> const typename SPropertyElementData::get_typename<SPropertyElementData::Type::VECTOR>::type& SPropertyElementData::getProperty<SPropertyElementData::Type::VECTOR>() const
{ return vvalue; }
template<> const typename SPropertyElementData::get_typename<SPropertyElementData::Type::POINT>::type& SPropertyElementData::getProperty<SPropertyElementData::Type::POINT>() const
{ return vvalue; }
template<> const typename SPropertyElementData::get_typename<SPropertyElementData::Type::MATRIX>::type& SPropertyElementData::getProperty<SPropertyElementData::Type::MATRIX>() const
{ return mvalue; }
template<> const typename SPropertyElementData::get_typename<SPropertyElementData::Type::TRANSLATE>::type& SPropertyElementData::getProperty<SPropertyElementData::Type::TRANSLATE>() const
{ return mvalue; }
template<> const typename SPropertyElementData::get_typename<SPropertyElementData::Type::ROTATE>::type& SPropertyElementData::getProperty<SPropertyElementData::Type::ROTATE>() const
{ return mvalue; }
template<> const typename SPropertyElementData::get_typename<SPropertyElementData::Type::SCALE>::type& SPropertyElementData::getProperty<SPropertyElementData::Type::SCALE>() const
{ return mvalue; }
template<> const typename SPropertyElementData::get_typename<SPropertyElementData::Type::LOOKAT>::type& SPropertyElementData::getProperty<SPropertyElementData::Type::LOOKAT>() const
{ return mvalue; }

const core::unordered_map<std::string,SPropertyElementData::Type,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> SPropertyElementData::StringToType = {
	{"float",		SPropertyElementData::Type::FLOAT},
	{"integer",		SPropertyElementData::Type::INTEGER},
	{"boolean",		SPropertyElementData::Type::BOOLEAN},
	{"string",		SPropertyElementData::Type::STRING},
	{"rgb",			SPropertyElementData::Type::RGB},
	{"srgb",		SPropertyElementData::Type::SRGB},
	{"spectrum",	SPropertyElementData::Type::SPECTRUM},
	{"blackbody",	SPropertyElementData::Type::BLACKBODY},
	{"matrix",		SPropertyElementData::Type::MATRIX},
	{"translate",	SPropertyElementData::Type::TRANSLATE},
	{"rotate",		SPropertyElementData::Type::ROTATE},
	{"scale",		SPropertyElementData::Type::SCALE},
	{"lookat",		SPropertyElementData::Type::LOOKAT},
	{"point",		SPropertyElementData::Type::POINT},
	{"vector",		SPropertyElementData::Type::VECTOR}
};
const char* SPropertyElementData::attributeStrings[SPropertyElementData::Type::INVALID][SPropertyElementData::MaxAttributes] = {
	{"value"}, // FLOAT
	{"value"}, // INTEGER
	{"value"}, // BOOLEAN
	{"value"}, // STRING
	{"value","intent"}, // RGB
	{"value","intent"}, // SRGB
	{"value","intent","filename"}, // SPECTRUM
	{"temperature","scale"}, // BLACKBODY
	{"value"}, // MATRIX
	{"x","y","z"}, // TRANSLATE
	{"angle","x","y","z"}, // ROTATE
	{"value","x","y","z"}, // SCALE
	{"origin","target","up"}, // LOOKAT
	{"x","y","z"}, // POINT
	{"x","y","z"} // VECTOR
};

std::pair<bool, SNamedPropertyElement> CPropertyElementManager::createPropertyData(const char* _el, const char** _atts)
{
	SNamedPropertyElement result(_el);

	const char* desiredAttributes[SPropertyElementData::MaxAttributes] = { nullptr };
	if (!result.initialize(_atts, desiredAttributes))
	{
		_NBL_DEBUG_BREAK_IF(true);
		return std::make_pair(false, SNamedPropertyElement());
	}

	bool success = true;
	#define FAIL_IF_ATTRIBUTE_NULL(N) if (!desiredAttributes[N]) {success = false; break;}
	switch (result.type)
	{
		case SPropertyElementData::Type::FLOAT:
			FAIL_IF_ATTRIBUTE_NULL(0u)
			result.fvalue = atof(desiredAttributes[0]);
			break;
		case SPropertyElementData::Type::INTEGER:
			FAIL_IF_ATTRIBUTE_NULL(0u)
			result.ivalue = atoi(desiredAttributes[0]);
			break;
		case SPropertyElementData::Type::BOOLEAN:
			FAIL_IF_ATTRIBUTE_NULL(0u)
			result.bvalue = retrieveBooleanValue(desiredAttributes[0],success);
			break;
		case SPropertyElementData::Type::STRING:
			FAIL_IF_ATTRIBUTE_NULL(0u)
			{
				auto len = strlen(desiredAttributes[0]);
				auto* tmp = (char*)_NBL_ALIGNED_MALLOC(len + 1u, 64u);
				strcpy(tmp, desiredAttributes[0]); tmp[len] = 0;
				result.svalue = tmp;
			}
			break;
		case SPropertyElementData::Type::RGB:
			FAIL_IF_ATTRIBUTE_NULL(0u)
			result.vvalue = retrieveVector(desiredAttributes[0], success);
			break;
		case SPropertyElementData::Type::SRGB:
			FAIL_IF_ATTRIBUTE_NULL(0u)
			{
				bool tryVec = true;
				result.vvalue = retrieveVector(desiredAttributes[0], tryVec);
				if (!tryVec)
					result.vvalue = retrieveHex(desiredAttributes[0], success);
				for (auto i=0; i<3u; i++)
					result.vvalue[i] = core::srgb2lin(result.vvalue[i]);
				result.type = SPropertyElementData::Type::RGB; // now its an RGB value
			}
			break;
		case SPropertyElementData::Type::VECTOR:
		case SPropertyElementData::Type::POINT:
			result.vvalue.set(0.f, 0.f, 0.f);
			for (auto i=0u; i<3u; i++)
			{
				if (desiredAttributes[i])
					result.vvalue[i] = atof(desiredAttributes[i]);
				else
				{
					success = false;
					break;
				}
			}
			break;
		case SPropertyElementData::Type::SPECTRUM:
			assert(!desiredAttributes[1]); // no intent, TODO
			assert(!desiredAttributes[2]); // does not come from a file
			{
				std::string data(desiredAttributes[0]);
				assert(data.find(':')==std::string::npos); // no hand specified wavelengths
				result.vvalue = retrieveVector(data,success); // TODO: convert between mitsuba spectral buckets and Rec. 709
			}
			break;
		case SPropertyElementData::Type::BLACKBODY:
			result.type = SPropertyElementData::Type::INVALID;
			break;
		case SPropertyElementData::Type::MATRIX:
			FAIL_IF_ATTRIBUTE_NULL(0u)
			result.mvalue = retrieveMatrix(desiredAttributes[0],success);
			break;
		case SPropertyElementData::Type::TRANSLATE:
			result.vvalue.set(0.f, 0.f, 0.f);
			for (auto i=0u; i<3u; i++)
			if (desiredAttributes[i])
				result.vvalue[i] = atof(desiredAttributes[i]);
			{
				core::matrix3x4SIMD m;
				m.setTranslation(result.vvalue);
				result.mvalue = core::matrix4SIMD(m);
			}
			break;
		case SPropertyElementData::Type::ROTATE:
			FAIL_IF_ATTRIBUTE_NULL(0u) // have to have an angle
			result.vvalue.set(0.f, 0.f, 0.f);
			for (auto i=0u; i<3u; i++)
			if (desiredAttributes[i+1])
				result.vvalue[i] = atof(desiredAttributes[i+1]);
			if ((core::vectorSIMDf(0.f) == result.vvalue).all())
			{
				success = false;
				break;
			}
			result.vvalue = core::normalize(result.vvalue);
			{
				core::matrix3x4SIMD m;
				m.setRotation(core::quaternion::fromAngleAxis(core::radians(atof(desiredAttributes[0])),result.vvalue));
				result.mvalue = core::matrix4SIMD(m);
			}
			break;
		case SPropertyElementData::Type::SCALE:
			result.vvalue.set(1.f, 1.f, 1.f);
			if (desiredAttributes[0])
			{
				float uniformScale = atof(desiredAttributes[0]);
				result.vvalue.set(uniformScale, uniformScale, uniformScale);
			}
			else
			for (auto i=0u; i<3u; i++)
			if (desiredAttributes[i+1u])
				result.vvalue[i] = atof(desiredAttributes[i+1u]);
			{
				core::matrix3x4SIMD m;
				m.setScale(result.vvalue);
				result.mvalue = core::matrix4SIMD(m);
			}
			break;
		case SPropertyElementData::Type::LOOKAT:
			FAIL_IF_ATTRIBUTE_NULL(0u)
			FAIL_IF_ATTRIBUTE_NULL(1u)
			{
				core::vectorSIMDf origin,target,up;
				origin = retrieveVector(desiredAttributes[0u], success);
				target = retrieveVector(desiredAttributes[1u], success);
				if (desiredAttributes[2u])
					up = retrieveVector(desiredAttributes[2u],success);
				else
				{
					auto viewDirection = target - origin;
					float maxDot = viewDirection[0];
					uint32_t index = 0u;
					for (auto i = 1u; i < 3u; i++)
					if (viewDirection[i] < maxDot)
					{
						maxDot = viewDirection[i];
						index = i;
					}
					up[index] = 1.f;
				}
				// mitsuba understands look-at and right-handed camera little bit differently than I do
				core::matrix4SIMD(core::matrix3x4SIMD::buildCameraLookAtMatrixLH(origin,target,up)).getInverseTransform(result.mvalue);
			}
			break;
		default:
			success = false;
			break;
	}

	_NBL_DEBUG_BREAK_IF(!success);
	if (success)
		return std::make_pair(true, std::move(result));

	ParserLog::invalidXMLFileStructure("invalid element, name:\'" + result.name + "\'"); // in the future print values
	return std::make_pair(false, SNamedPropertyElement());
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
		_NBL_DEBUG_BREAK_IF(true);
		ParserLog::invalidXMLFileStructure("Invalid boolean specified.");
		success = false;
		return false; // so GCC doesn't moan
	}
}

core::matrix4SIMD CPropertyElementManager::retrieveMatrix(const std::string& _data, bool& success)
{
	std::string str = _data;
	std::replace(str.begin(), str.end(), ',', ' ');

	core::matrix4SIMD matrixData;
	std::stringstream ss;
	ss << str;

	for (auto i=0u; i<16u; i++)
	{
		float f = std::numeric_limits<float>::quiet_NaN();
		ss >> f;

		if (isnan(f))
		{
			_NBL_DEBUG_BREAK_IF(true);
			ParserLog::invalidXMLFileStructure("Invalid matrix specified.");
			success = false;
			return core::matrix4SIMD();
		}
		matrixData.pointer()[i] = f;
	}

	return matrixData;
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
			if (i == 1)
			{
				vectorData[2] = vectorData[1] = vectorData[0];
				vectorData[3] = 0.0f;
				break;
			}
			else if (i == 3)
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

}
}
}