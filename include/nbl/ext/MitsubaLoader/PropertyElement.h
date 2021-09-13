// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __PROPERTY_ELEMENT_H_INCLUDED__
#define __PROPERTY_ELEMENT_H_INCLUDED__

//#include "nbl/core/core.h"
#include "matrix4SIMD.h"
#include <string>

namespace nbl
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
	_NBL_STATIC_INLINE_CONSTEXPR uint32_t MaxAttributes = 4u;
	static const char* attributeStrings[Type::INVALID][MaxAttributes];

	inline SPropertyElementData() : type(Type::INVALID)
	{
		std::fill(mvalue.pointer(), mvalue.pointer() + 16, 0.f);
	}
	inline SPropertyElementData(const SPropertyElementData& other) : SPropertyElementData()
	{
		operator=(other);
	}
	inline SPropertyElementData(SPropertyElementData&& other) : SPropertyElementData()
	{
		operator=(std::move(other));
	}
	inline SPropertyElementData(const std::string& _type) : SPropertyElementData()
	{
		auto found = StringToType.find(_type);
		if (found != StringToType.end())
			type = found->second;
	}
	inline explicit SPropertyElementData(float value)								: type(FLOAT)	{ fvalue = value; }
	inline explicit SPropertyElementData(int32_t value)								: type(INTEGER) { ivalue = value; }
	inline explicit SPropertyElementData(bool value)								: type(BOOLEAN) { bvalue = value; }
	//explicit SPropertyElementData(const std::string& value)						: type(STRING) { #error }
	inline explicit SPropertyElementData(Type _type, const core::vectorSIMDf& value): type(INVALID)
	{
		switch (_type)
		{
			case Type::RGB:
			case Type::SRGB:
			case Type::VECTOR:
			case Type::POINT:
				type = _type;
				vvalue = value;
				break;
			default:
				assert(false);
				break;
		};
	}
	~SPropertyElementData()
	{
		if (type == Type::STRING)
			_NBL_ALIGNED_FREE((void*)svalue);
	}

	inline SPropertyElementData& operator=(const SPropertyElementData& other)
	{
		type = other.type;
		switch (type)
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
				auto* tmp = (char*)_NBL_ALIGNED_MALLOC(len+1u,64u);
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
				vvalue = other.vvalue;
				break;
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
		switch (type)
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
				std::swap(svalue,other.svalue);
				break;
			case Type::RGB:
			case Type::SRGB:
			case Type::VECTOR:
			case Type::POINT:
				vvalue = other.vvalue;
				break;
			case Type::SPECTRUM:
				vvalue = other.vvalue;
				break;
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
				std::fill(other.mvalue.pointer(), other.mvalue.pointer() + 16, 0.f);
				break;
		}
		return *this;
	}


	template<uint32_t property_type>
	struct get_typename;
	template<uint32_t property_type>
	const typename get_typename<property_type>::type& getProperty() const;


	SPropertyElementData::Type type;
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

struct SNamedPropertyElement : SPropertyElementData
{
	SNamedPropertyElement() : SPropertyElementData(), name("")
	{
	}
	SNamedPropertyElement(const std::string& _type) : SNamedPropertyElement()
	{
		auto found = SPropertyElementData::StringToType.find(_type);
		if (found != SPropertyElementData::StringToType.end())
			type = found->second;
	}
	SNamedPropertyElement(const SNamedPropertyElement& other) : SNamedPropertyElement()
	{
		SNamedPropertyElement::operator=(other);
	}
	SNamedPropertyElement(SNamedPropertyElement&& other) : SNamedPropertyElement()
	{
		SNamedPropertyElement::operator=(std::move(other));
	}

	bool initialize(const char** _atts, const char** outputMatch)
	{
		if (type == Type::INVALID || !_atts)
			return false;

		for (auto it = _atts; *it; it++)
		{
			if (core::strcmpi(*it, "name") == 0)
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

			for (auto i = 0u; i < SPropertyElementData::MaxAttributes; i++)
				if (core::strcmpi(*it, SPropertyElementData::attributeStrings[type][i]) == 0)
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

	inline SNamedPropertyElement& operator=(const SNamedPropertyElement& other)
	{
		SPropertyElementData::operator=(other);
		name = other.name;
		return *this;
	}
	inline SNamedPropertyElement& operator=(SNamedPropertyElement&& other)
	{
		SPropertyElementData::operator=(std::move(other));
		std::swap(name, other.name);
		return *this;
	}

	std::string name;
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
{ using type = core::vectorSIMDf; };
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
		static std::pair<bool, SNamedPropertyElement> createPropertyData(const char* _el, const char** _atts);

		static bool retrieveBooleanValue(const std::string& _data, bool& success);
		static core::matrix4SIMD retrieveMatrix(const std::string& _data, bool& success);
		static core::vectorSIMDf retrieveVector(const std::string& _data, bool& success);
		static core::vectorSIMDf retrieveHex(const std::string& _data, bool& success);

};

}
}
}

#endif