#ifndef __PROPERTY_ELEMENT_H_INCLUDED__
#define __PROPERTY_ELEMENT_H_INCLUDED__

#include "irr/core/core.h"
#include "matrix4SIMD.h"
#include <string>

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

struct SPropertyElementData
{
	enum Type
	{
		FLOAT,
		INTEGER,
		BOOLEAN,
		STRING,
		RGB,
		SRGB,
		SPECTRUM, // not supported, provided for completeness
		BLACKBODY, // not supported, provided for completeness
		MATRIX,
		TRANSLATE,
		ROTATE,
		SCALE,
		LOOKAT,
		POINT,
		VECTOR,
		INVALID
	};

	static const core::unordered_map<std::string,Type,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> StringToType;
	_IRR_STATIC_INLINE_CONSTEXPR uint32_t MaxAttributes = 4u;
	static const char* attributeStrings[Type::INVALID][MaxAttributes];

	SPropertyElementData() : type(Type::INVALID), name("")
	{
		std::fill(mvalue.pointer(), mvalue.pointer() + 16, 0.f);
	}
	SPropertyElementData(const std::string& _type) : SPropertyElementData()
	{
		auto found = StringToType.find(_type);
		if (found != StringToType.end())
			type = found->second;
	}
	SPropertyElementData(const SPropertyElementData& other) : SPropertyElementData()
	{
		operator=(other);
	}
	SPropertyElementData(SPropertyElementData&& other) : SPropertyElementData()
	{
		operator=(std::move(other));
	}
	~SPropertyElementData()
	{
		if (type == Type::STRING)
			_IRR_ALIGNED_FREE((void*)svalue);
	}

	bool initialize(const char** _atts, const char** outputMatch)
	{
		if (type==Type::INVALID || !_atts)
			return false;

		for (auto it = _atts; *it; it++)
		{
			if (core::strcmpi(*it, "name"))
			{
				it++;
				if (*it)
				{
					name = *it;
					continue;
				}
				else
					return false;
			}

			for (auto i=0u; i<MaxAttributes; i++)
			if (core::strcmpi(*it, attributeStrings[type][i]))
			{
				it++;
				if (!outputMatch[i] && *it)
				{
					outputMatch[i] = *it;
					break;
				}
				else
					return false;
			}
		}
		return true;
	}

	inline SPropertyElementData& operator=(const SPropertyElementData& other)
	{
		type = other.type;
		name = other.name;
		switch (other.type)
		{
			case Type::FLOAT:
				fvalue = other.fvalue;
				break;
			case Type::INTEGER:
				ivalue = other.ivalue;
				break;
			case Type::BOOLEAN:
				bvalue = other.bvalue;
				break;
			case Type::STRING:
			{
				auto len = strlen(other.svalue);
				auto* tmp = (char*)_IRR_ALIGNED_MALLOC(len+1u,64u);
				memcpy(tmp,other.svalue,len);
				tmp[len] = 0;
				svalue = tmp;
			}
				break;
			case Type::RGB:
			case Type::SRGB:
			case Type::VECTOR:
			case Type::POINT:
				vvalue = other.vvalue;
				break;
			case Type::SPECTRUM:
			case Type::BLACKBODY:
				assert(false);
				break;
			case Type::MATRIX:
			case Type::TRANSLATE:
			case Type::ROTATE:
			case Type::SCALE:
			case Type::LOOKAT:
				mvalue = other.mvalue;
				break;
			default:
				std::fill(mvalue.pointer(), mvalue.pointer()+16, 0.f);
				break;
		}
		return *this;
	}
	inline SPropertyElementData& operator=(SPropertyElementData&& other)
	{
		std::swap(type,other.type);
		std::swap(name,other.name);
		switch (other.type)
		{
		case Type::FLOAT:
			fvalue = other.fvalue;
			break;
		case Type::INTEGER:
			ivalue = other.ivalue;
			break;
		case Type::BOOLEAN:
			bvalue = other.bvalue;
			break;
		case Type::STRING:
			svalue = other.svalue;
			break;
		case Type::RGB:
		case Type::SRGB:
		case Type::VECTOR:
		case Type::POINT:
			vvalue = other.vvalue;
			break;
		case Type::SPECTRUM:
		case Type::BLACKBODY:
			assert(false);
			break;
		case Type::MATRIX:
		case Type::TRANSLATE:
		case Type::ROTATE:
		case Type::SCALE:
		case Type::LOOKAT:
			mvalue = other.mvalue;
			break;
		default:
			break;
		}
		std::fill(other.mvalue.pointer(), other.mvalue.pointer() + 16, 0.f);
		return *this;
	}


	template<uint32_t property_type>
	struct get_typename;
	template<uint32_t property_type>
	const typename get_typename<property_type>::type& getProperty() const;


	SPropertyElementData::Type type;
	std::string name;
	union
	{
		float				fvalue;
		int32_t				ivalue;
		bool				bvalue;
		const char*			svalue;
		core::vectorSIMDf	vvalue; // rgb, srgb, vector, point
		core::matrix4SIMD	mvalue; // matrix, translate, rotate, scale, lookat
	};
};

template<> struct SPropertyElementData::get_typename<SPropertyElementData::Type::FLOAT>
{ using type = float; };
template<> struct SPropertyElementData::get_typename<SPropertyElementData::Type::INTEGER>
{ using type = int32_t; };
template<> struct SPropertyElementData::get_typename<SPropertyElementData::Type::BOOLEAN>
{ using type = bool; };
template<> struct SPropertyElementData::get_typename<SPropertyElementData::Type::STRING>
{ using type = const char*; };
template<> struct SPropertyElementData::get_typename<SPropertyElementData::Type::RGB>
{ using type = core::vectorSIMDf; };
template<> struct SPropertyElementData::get_typename<SPropertyElementData::Type::SRGB>
{ using type = core::vectorSIMDf; };
template<> struct SPropertyElementData::get_typename<SPropertyElementData::Type::VECTOR>
{ using type = core::vectorSIMDf; };
template<> struct SPropertyElementData::get_typename<SPropertyElementData::Type::POINT>
{ using type = core::vectorSIMDf; };
template<> struct SPropertyElementData::get_typename<SPropertyElementData::Type::SPECTRUM>
{ using type = void; };
template<> struct SPropertyElementData::get_typename<SPropertyElementData::Type::BLACKBODY>
{ using type = void; };
template<> struct SPropertyElementData::get_typename<SPropertyElementData::Type::MATRIX>
{ using type = core::matrix4SIMD; };
template<> struct SPropertyElementData::get_typename<SPropertyElementData::Type::TRANSLATE>
{ using type = core::matrix4SIMD; };
template<> struct SPropertyElementData::get_typename<SPropertyElementData::Type::ROTATE>
{ using type = core::matrix4SIMD; };
template<> struct SPropertyElementData::get_typename<SPropertyElementData::Type::SCALE>
{ using type = core::matrix4SIMD; };
template<> struct SPropertyElementData::get_typename<SPropertyElementData::Type::LOOKAT>
{ using type = core::matrix4SIMD; };
template<> struct SPropertyElementData::get_typename<SPropertyElementData::Type::INVALID>
{ using type = void; };


class CPropertyElementManager
{
	public:
		static std::pair<bool, SPropertyElementData> createPropertyData(const char* _el, const char** _atts);

		static bool retrieveBooleanValue(const std::string& _data, bool& success);
		static core::matrix4SIMD retrieveMatrix(const std::string& _data, bool& success);
		static core::vectorSIMDf retrieveVector(const std::string& _data, bool& success);
		static core::vectorSIMDf retrieveHex(const std::string& _data, bool& success);

};

}
}
}

#endif