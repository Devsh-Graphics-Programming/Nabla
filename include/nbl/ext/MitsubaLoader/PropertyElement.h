// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_MISTUBA_LOADER_PROPERTY_ELEMENT_H_INCLUDED_
#define _NBL_EXT_MISTUBA_LOADER_PROPERTY_ELEMENT_H_INCLUDED_


#include "nbl/core/declarations.h"
#include "nbl/builtin/hlsl/cpp_compat.hlsl"


namespace nbl::ext::MitsubaLoader
{
// maybe move somewhere
inline void invalidXMLFileStructure(system::logger_opt_ptr logger, const std::string& errorMessage)
{
	// TODO: print the line in the XML or something
	logger.log("Mitsuba loader error - Invalid .xml file structure: \'%s\'",system::ILogger::E_LOG_LEVEL::ELL_ERROR,errorMessage.c_str());
	_NBL_DEBUG_BREAK_IF(true);
}

struct SPropertyElementData
{
	// TODO: enum class, and smaller type
	enum Type : uint32_t
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
	//
	constexpr static inline uint32_t MaxAttributes = 5u;

	inline SPropertyElementData() : type(Type::INVALID)
	{
		memset(&fvalue,0,sizeof(mvalue));
	}
	inline SPropertyElementData(const SPropertyElementData& other) : SPropertyElementData()
	{
		operator=(other);
	}
	inline SPropertyElementData(SPropertyElementData&& other) : SPropertyElementData()
	{
		operator=(std::move(other));
	}
	inline explicit SPropertyElementData(float value)								: type(FLOAT)	{ fvalue = value; }
	inline explicit SPropertyElementData(int32_t value)								: type(INTEGER) { ivalue = value; }
	inline explicit SPropertyElementData(bool value)								: type(BOOLEAN) { bvalue = value; }
	//explicit SPropertyElementData(const std::string& value)						: type(STRING) { #error }
	inline explicit SPropertyElementData(Type _type, const hlsl::float32_t4& value): type(INVALID)
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
	inline ~SPropertyElementData()
	{
		if (type==Type::STRING)
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
				memset(&fvalue,0,sizeof(mvalue));
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
				memset(&fvalue,0,sizeof(mvalue));
				break;
		}
		return *this;
	}

	// TODO: enum class on the template param
	template<uint32_t property_type>
	struct get_type;
	template<uint32_t property_type>
	using get_type_t = typename get_type<property_type>::type;
	template<uint32_t property_type>
	const get_type_t<property_type>& getProperty() const;

	inline uint8_t getVectorDimension() const
	{
		uint8_t i = 0u;
		if (type==Type::VECTOR)
		while (i<4u)
		if (core::isnan(vvalue[i]))
			break;
		return i;
	}


	SPropertyElementData::Type type;
	union
	{
		float				fvalue;
		int32_t				ivalue;
		bool				bvalue;
		const char*			svalue;
		hlsl::float32_t4	vvalue; // rgb, srgb, vector, point
		hlsl::float32_t4x4	mvalue; // matrix, translate, rotate, scale, lookat
	};
};

struct SNamedPropertyElement : SPropertyElementData
{
	inline SNamedPropertyElement() : SPropertyElementData(), name("")
	{
	}
	inline SNamedPropertyElement(const SNamedPropertyElement& other) : SNamedPropertyElement()
	{
		SNamedPropertyElement::operator=(other);
	}
	inline SNamedPropertyElement(SNamedPropertyElement&& other) : SNamedPropertyElement()
	{
		SNamedPropertyElement::operator=(std::move(other));
	}

	inline bool initialize(const char** _atts, const char** outputMatch)
	{
		if (type==Type::INVALID || !_atts)
			return false;

		constexpr const char* AttributeStrings[SPropertyElementData::Type::INVALID][SPropertyElementData::MaxAttributes] = {
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
			{"x","y","z","w"} // VECTOR
		};
		// TODO: some magical constexpr thing to count up 
		//constexpr size_t AttributeCount[SPropertyElementData::Type::INVALID][SPropertyElementData::MaxAttributes] = {};

		for (auto it=_atts; *it; it++)
		{
			// found the name attribute
			if (core::strcmpi(*it,"name") == 0)
			{
				// value follows the attribute name
				it++;
				if (*it)
				{
					// next attribute is the actual name, first is just the `name=`
					name = *it;
					continue;
				}
				else // no name present e.g. `name=""`
					return false;
			}

			// now go through the expected attributes
			for (auto i=0u; i<SPropertyElementData::MaxAttributes; i++)
			{
				// the list of strings is terminated by nullptr (could cache this btw) and shorten the loop
				if (AttributeStrings[type][i]==nullptr)
					continue;
				// if we match any expected in the `attributeStrings[type]` row, we look at the next attribute
				if (core::strcmpi(*it,AttributeStrings[type][i]) == 0)
				{
					// value follows the attribute name
					it++;
					// output will be ordered same as `attributeStrings[type]`
					if (!outputMatch[i] && *it)
					{
						outputMatch[i] = *it;
						break;
					}
					else // no value to assign (empty string), or already matched this attribute once before
						return false;
				}
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

template<> struct SPropertyElementData::get_type<SPropertyElementData::Type::FLOAT>
{ using type = float; };
template<> struct SPropertyElementData::get_type<SPropertyElementData::Type::INTEGER>
{ using type = int32_t; };
template<> struct SPropertyElementData::get_type<SPropertyElementData::Type::BOOLEAN>
{ using type = bool; };
template<> struct SPropertyElementData::get_type<SPropertyElementData::Type::STRING>
{ using type = const char*; };
template<> struct SPropertyElementData::get_type<SPropertyElementData::Type::RGB>
{ using type = hlsl::float32_t4; };
template<> struct SPropertyElementData::get_type<SPropertyElementData::Type::SRGB>
{ using type = hlsl::float32_t4; };
template<> struct SPropertyElementData::get_type<SPropertyElementData::Type::VECTOR>
{ using type = hlsl::float32_t4; };
template<> struct SPropertyElementData::get_type<SPropertyElementData::Type::POINT>
{ using type = hlsl::float32_t4; };
template<> struct SPropertyElementData::get_type<SPropertyElementData::Type::SPECTRUM>
{ using type = hlsl::float32_t4; };
template<> struct SPropertyElementData::get_type<SPropertyElementData::Type::BLACKBODY>
{ using type = void; };
template<> struct SPropertyElementData::get_type<SPropertyElementData::Type::MATRIX>
{ using type = hlsl::float32_t4x4; };
template<> struct SPropertyElementData::get_type<SPropertyElementData::Type::TRANSLATE>
{ using type = hlsl::float32_t4x4; };
template<> struct SPropertyElementData::get_type<SPropertyElementData::Type::ROTATE>
{ using type = hlsl::float32_t4x4; };
template<> struct SPropertyElementData::get_type<SPropertyElementData::Type::SCALE>
{ using type = hlsl::float32_t4x4; };
template<> struct SPropertyElementData::get_type<SPropertyElementData::Type::LOOKAT>
{ using type = hlsl::float32_t4x4; };
template<> struct SPropertyElementData::get_type<SPropertyElementData::Type::INVALID>
{ using type = void; };
// TODO: rewrite rest to be less `::` verbose
template<> inline auto SPropertyElementData::getProperty<SPropertyElementData::Type::FLOAT>() const -> const get_type_t<Type::FLOAT>&
{ return fvalue; }
template<> inline auto SPropertyElementData::getProperty<SPropertyElementData::Type::INTEGER>() const -> const get_type_t<Type::INTEGER>&
{ return ivalue; }
template<> inline auto SPropertyElementData::getProperty<SPropertyElementData::Type::BOOLEAN>() const -> const get_type_t<Type::BOOLEAN>&
{ return bvalue; }
template<> inline auto SPropertyElementData::getProperty<SPropertyElementData::Type::STRING>() const -> const get_type_t<Type::STRING>&
{ return svalue; }
template<> inline auto SPropertyElementData::getProperty<SPropertyElementData::Type::RGB>() const -> const get_type_t<Type::RGB>&
{ return vvalue; }
template<> inline auto SPropertyElementData::getProperty<SPropertyElementData::Type::SRGB>() const -> const get_type_t<Type::SRGB>&
{ return vvalue; }
template<> inline auto SPropertyElementData::getProperty<SPropertyElementData::Type::SPECTRUM>() const -> const get_type_t<Type::SPECTRUM>&
{ return vvalue; }
template<> inline auto SPropertyElementData::getProperty<SPropertyElementData::Type::VECTOR>() const -> const get_type_t<Type::VECTOR>&
{ return vvalue; }
template<> inline auto SPropertyElementData::getProperty<SPropertyElementData::Type::POINT>() const -> const get_type_t<Type::POINT>&
{ return vvalue; }
template<> inline auto SPropertyElementData::getProperty<SPropertyElementData::Type::MATRIX>() const -> const get_type_t<Type::MATRIX>&
{ return mvalue; }
template<> inline auto SPropertyElementData::getProperty<SPropertyElementData::Type::TRANSLATE>() const -> const get_type_t<Type::TRANSLATE>&
{ return mvalue; }
template<> inline auto SPropertyElementData::getProperty<SPropertyElementData::Type::ROTATE>() const -> const get_type_t<Type::ROTATE>&
{ return mvalue; }
template<> inline auto SPropertyElementData::getProperty<SPropertyElementData::Type::SCALE>() const -> const get_type_t<Type::SCALE>&
{ return mvalue; }
template<> inline auto SPropertyElementData::getProperty<SPropertyElementData::Type::LOOKAT>() const -> const get_type_t<Type::LOOKAT>&
{ return mvalue; }


class CPropertyElementManager final : public core::Unmovable
{
		const core::unordered_map<std::string,SPropertyElementData::Type,core::CaseInsensitiveHash,core::CaseInsensitiveEquals> StringToType;

		static std::optional<bool> retrieveBooleanValue(const std::string_view& _data, system::logger_opt_ptr logger);
		static hlsl::float32_t4x4 retrieveMatrix(const std::string_view& _data, system::logger_opt_ptr logger);
		static hlsl::float32_t4 retrieveVector(const std::string_view& _data, system::logger_opt_ptr logger);
		static hlsl::float32_t4 retrieveHex(const std::string_view& _data, system::logger_opt_ptr logger);

	public:
		CPropertyElementManager();
#if 0
		inline SPropertyElementData(const std::string& _type) : SPropertyElementData()
		{
			auto found = StringToType.find(_type);
			if (found != StringToType.end())
				type = found->second;
		}
		SNamedPropertyElement(const std::string& _type) : SNamedPropertyElement()
		{
			auto found = SPropertyElementData::StringToType.find(_type);
			if (found != SPropertyElementData::StringToType.end())
				type = found->second;
		}
#endif
		std::optional<SNamedPropertyElement> createPropertyData(const char* _el, const char** _atts, system::logger_opt_ptr logger) const;

};

}
#endif