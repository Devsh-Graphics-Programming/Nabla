// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


//#include "quaternion.h"
//#include "matrix3x4SIMD.h"
//#include "matrix4SIMD.h"
//#include "nbl/asset/format/decodePixels.h"

#include "nbl/ext/MitsubaLoader/PropertyElement.h"
#include "nbl/ext/MitsubaLoader/ParserUtil.h"

#include "nbl/builtin/hlsl/math/linalg/transform.hlsl"
#include "glm/gtc/matrix_transform.hpp"


namespace nbl::ext::MitsubaLoader
{

CPropertyElementManager::CPropertyElementManager() : StringToType({
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
}) {}


std::optional<SNamedPropertyElement> CPropertyElementManager::createPropertyData(const char* _el, const char** _atts, system::logger_opt_ptr logger) const
{
	SNamedPropertyElement result = {};
	auto found = StringToType.find(_el);
	if (found!=StringToType.end())
		result.type = found->second;

	// initialization returns strings from `_atts` which match expected attributes
	const char* desiredAttributes[SPropertyElementData::MaxAttributes] = { nullptr };
	if (!result.initialize(_atts,desiredAttributes))
	{
		invalidXMLFileStructure(logger,"Failed to Intialize Named Property Element.");
		return {};
	}

    auto printFailure = [&](const uint8_t attrId)->void{invalidXMLFileStructure(logger,"invalid element, name:\'"+result.name+"\' value:\'"+desiredAttributes[attrId]+"\'");};

	#define FAIL_IF_ATTRIBUTE_NULL(N) if (!desiredAttributes[N]) \
	{ \
		invalidXMLFileStructure(logger,"Invalid element, name:\'"+result.name+"\' Attribute #"+std::to_string(N)+"not found"); \
		return {}; \
	}
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
			if (auto ret=retrieveBooleanValue(desiredAttributes[0],logger); ret.has_value())
				result.bvalue = ret.value();
			else
			{
				printFailure(0);
				return {};
			}
			break;
		case SPropertyElementData::Type::STRING:
			FAIL_IF_ATTRIBUTE_NULL(0u)
			{
				auto len = strlen(desiredAttributes[0]);
				auto* tmp = (char*)_NBL_ALIGNED_MALLOC(len+1u,64u);
				strcpy(tmp,desiredAttributes[0]); tmp[len]=0;
				result.svalue = tmp;
			}
			break;
		case SPropertyElementData::Type::RGB:
			FAIL_IF_ATTRIBUTE_NULL(0u)
			result.vvalue = retrieveVector(desiredAttributes[0],logger);
			if (core::isnan(result.vvalue[0]))
			{
				printFailure(0);
				return {};
			}
			break;
		case SPropertyElementData::Type::SRGB:
			FAIL_IF_ATTRIBUTE_NULL(0u)
			{
				result.vvalue = retrieveVector(desiredAttributes[0],logger);
				if (core::isnan(result.vvalue[0]))
				{
					result.vvalue = retrieveHex(desiredAttributes[0],logger);
					if (core::isnan(result.vvalue[0]))
					{
						printFailure(0);
						return {};
					}
				}
				for (auto i=0; i<3u; i++)
					result.vvalue[i] = core::srgb2lin(result.vvalue[i]);
				result.type = SPropertyElementData::Type::RGB; // now its an RGB value
			}
			break;
		case SPropertyElementData::Type::VECTOR:
			result.vvalue = hlsl::float32_t4(core::nan<float>());
			for (auto i=0u; i<4u; i++)
			{
				if (desiredAttributes[i])
					result.vvalue[i] = atof(desiredAttributes[i]);
				else
				{
					// once a component is missing, the rest need to be missing too
					for (auto j=i+1; j<4u; j++)
					if (desiredAttributes[j])
					{
						printFailure(0);
						return {};
					}
					break;
				}
			}
			break;
		case SPropertyElementData::Type::POINT:
			result.vvalue = hlsl::float32_t4(0.f,0.f,0.f,core::nan<float>());
			for (auto i=0u; i<3u; i++)
			{
				if (desiredAttributes[i])
					result.vvalue[i] = atof(desiredAttributes[i]);
				else
				{
					printFailure(0);
					return {};
				}
			}
			break;
		case SPropertyElementData::Type::SPECTRUM:
			if (desiredAttributes[1]||desiredAttributes[2])
			{
				invalidXMLFileStructure(logger,"Spectrum intent and loading from file unsupported!");
				return {};
			}
			{
				std::string_view data(desiredAttributes[0]);
				if (data.find(':')!=std::string::npos)
				{
					invalidXMLFileStructure(logger,"Manually specified wavelengths for spectral curve knots are unsupported!");
					return {};
				}
				result.vvalue = retrieveVector(data,logger); // TODO: convert between mitsuba spectral buckets and Rec. 709
				if (core::isnan(result.vvalue[0]))
				{
					printFailure(0);
					return {};
				}
			}
			break;
		case SPropertyElementData::Type::BLACKBODY:
			result.type = SPropertyElementData::Type::INVALID;
			break;
		case SPropertyElementData::Type::MATRIX:
			FAIL_IF_ATTRIBUTE_NULL(0u)
			result.mvalue = retrieveMatrix(desiredAttributes[0],logger);
			if (core::isnan(result.mvalue[0][0]))
			{
				printFailure(0);
				return {};
			}
			break;
		case SPropertyElementData::Type::TRANSLATE:
			result.mvalue = hlsl::float32_t4x4(1.f);
			// we're a bit more lax about what items we need present
			for (auto i=0u; i<3u; i++)
			if (desiredAttributes[i])
				result.mvalue[i][3] = atof(desiredAttributes[i]);
			break;
		case SPropertyElementData::Type::ROTATE:
			FAIL_IF_ATTRIBUTE_NULL(0u) // have to have an angle
			result.mvalue = hlsl::float32_t4x4(1.f);
			{
				auto axis = hlsl::float32_t3(0.f);
				// again some laxness
				for (auto i=0u; i<3u; i++)
				if (desiredAttributes[i+1])
					axis[i] = atof(desiredAttributes[i+1]);
				axis = hlsl::normalize(axis);
				if (core::isnan(axis.x))
				{
					invalidXMLFileStructure(logger,"Invalid element, name:\'"+result.name+"\' Axis can't be (0,0,0)");
					return {};
				}
				// TODO: quaternion after the rework
				using namespace nbl::hlsl::math;//::linalg;
				result.mvalue = linalg::promote_affine<4,4>(linalg::rotation_mat<float>(hlsl::radians(atof(desiredAttributes[0])),axis));
			}
			break;
		case SPropertyElementData::Type::SCALE:
			result.mvalue = hlsl::float32_t4x4(1.f);
			if (desiredAttributes[0]) // you either get this one attribute
			{
				const float uniformScale = atof(desiredAttributes[0]);
				for (auto i=0u; i<3u; i++)
					result.mvalue[i][i] = uniformScale;
			}
			else // or x,y,z
			{
				for (auto i=0u; i<3u; i++)
				if (desiredAttributes[i+1u])
					result.mvalue[i][i] = atof(desiredAttributes[i+1u]);
			}
			break;
		case SPropertyElementData::Type::LOOKAT:
			FAIL_IF_ATTRIBUTE_NULL(0u)
			FAIL_IF_ATTRIBUTE_NULL(1u)
			result.mvalue = hlsl::float32_t4x4(1.f);
			{
				const hlsl::float32_t3 origin = retrieveVector(desiredAttributes[0u],logger).xyz;
				if (core::isnan(origin.x))
				{
					printFailure(0);
					return {};
				}
				const hlsl::float32_t3 target = retrieveVector(desiredAttributes[1u],logger).xyz;
				if (core::isnan(target.x))
				{
					printFailure(1);
					return {};
				}
				auto up = hlsl::float32_t3(core::nan<float>());
				if (desiredAttributes[2u])
					up = retrieveVector(desiredAttributes[2u],logger).xyz;
				if (core::isnan(up.x))
				{
					up = hlsl::float32_t3(0.f);
					const auto viewDirection = target - origin;
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
				// TODO: after the rm-core matrix PR we need to get rid of the tranpose (I transpose only because of GLM and HLSL mixup)
				const auto lookAtGLM = reinterpret_cast<const hlsl::float32_t4x4&>(glm::lookAtLH<float>(origin,target,up));
				const auto lookAt = hlsl::transpose(lookAtGLM);
				// mitsuba understands look-at and right-handed camera little bit differently than I do
				const auto rotation = hlsl::inverse<hlsl::float32_t3x3>(hlsl::float32_t3x3(lookAt));
				// set the origin to avoid numerical issues
				for (auto r=0; r<3; r++)
				{
					result.mvalue[r][3] = origin[r];
				}
			}
			break;
		default:
			invalidXMLFileStructure(logger,"Unsupported element type, name:\'"+result.name+"\'");
			return {};
	}
	#undef FAIL_IF_ATTRIBUTE_NULL

	return result;
}

std::optional<bool> CPropertyElementManager::retrieveBooleanValue(const std::string_view& _data, system::logger_opt_ptr logger)
{
	if (_data=="true")
		return true;
	else if (_data=="false")
		return false;
	else
	{
		invalidXMLFileStructure(logger,"Invalid boolean specified.");
		return {};
	}
}

hlsl::float32_t4x4 CPropertyElementManager::retrieveMatrix(const std::string_view& _data, system::logger_opt_ptr logger)
{
	std::string str(_data);
	std::replace(str.begin(),str.end(),',',' ');

	hlsl::float32_t4x4 matrixData;
	std::stringstream ss;
	ss << str;

	for (auto r=0u; r<4u; r++)
	for (auto c=0u; c<4u; c++)
	{
		float f = std::numeric_limits<float>::quiet_NaN();
		ss >> f;

		if (core::isnan(f))
		{
			invalidXMLFileStructure(logger,"Invalid matrix specified.");
			matrixData[0][0] = f;
			return matrixData;
		}
		matrixData[r][c] = f;
	}

	return matrixData;
}

hlsl::float32_t4 CPropertyElementManager::retrieveVector(const std::string_view& _data, system::logger_opt_ptr logger)
{
	std::string str(_data);
	std::replace(str.begin(), str.end(), ',', ' ');

	hlsl::float32_t4 retval;
	std::stringstream ss;
	ss << str;

	for (int i = 0; i < 4; i++)
	{
		float f = std::numeric_limits<float>::quiet_NaN();
		ss >> f;

		retval[i] = f;

		if (isnan(f))
		{
			if (i==1) // second not present
			{
				// make monochrome RGB or scalar XYZ
				retval[2] = retval[1] = retval[0];
				retval[3] = 0.0f;
			}
			else if (i==3) // last not present
			{
				// allow last coordinate to be 0
				retval[3] = 0.0f;
			}
			return retval;
		}
	}

	return retval;
}

hlsl::float32_t4 CPropertyElementManager::retrieveHex(const std::string_view& _data, system::logger_opt_ptr logger)
{
	auto ptr = _data.begin();
	const auto invalid = hlsl::float32_t4(std::numeric_limits<float>::quiet_NaN());
	// not a hex
	if (_data.size()!=7u || *ptr!='#')
		return invalid;

	hlsl::float32_t4 retval(0.f, 0.f, 0.f, 255.f);
	for (auto i=0; i<3; i++)
	for (auto j=4; j>=0;j-=4)
	{
		char c = *(++ptr);
		if (!isxdigit(c))
			return invalid;
		// case insensitiveness
		int intval = (c >= 'A') ? (c - 'A' + 10) : (c - '0');
		// written form of hex is obviously big endian
		retval[i] += float(intval<<j);
	}
	return retval/255.f;
}

}