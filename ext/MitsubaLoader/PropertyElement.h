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

struct CaseInsensitiveHash
{
	inline std::size_t operator()(const std::string& val) const
	{
		std::size_t seed = 0;
		for (auto it=val.begin(); it!=val.end(); it++)
		{
			seed ^= ~std::size_t(std::tolower(*it)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
		return seed;
	}
};
struct CaseInsensitiveEquals
{
	inline bool operator()(const std::string& A, const std::string& B) const
	{
		return core::strcmpi(A,B)!=0;
	}
};

struct SPropertyElementData
{
	enum class Type
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

	static const core::unordered_map<std::string,Type,CaseInsensitiveHash,CaseInsensitiveEquals> StringToType;

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

	const char* initialize(const char** _atts)
	{
		if (!_atts)
			return nullptr;

		const char* value = nullptr;
		for (auto it = _atts; *it; it++)
		{
			if (core::strcmpi(*it, "name"))
			{
				it++;
				if (*it)
					name = *it;
				else
					break;
			}
			else if (core::strcmpi(*it, "value"))
			{
				it++;
				if (*it)
					value = *it;
				break;
			}
		}
		return value;
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
				auto len = strlen(other.svalue);
				auto* tmp = (char*)_IRR_ALIGNED_MALLOC(len+1u,64u);
				memcpy(tmp,other.svalue,len);
				tmp[len] = 0;
				svalue = tmp;
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

class CPropertyElementManager
{
	public:
		static std::pair<bool, SPropertyElementData> createPropertyData(const char* _el, const char** _atts);

		static bool retrieveBooleanValue(const std::string& _data, bool& success);
		static core::matrix4SIMD retrieveMatrix(const std::string& _data, bool& success);
		static core::vectorSIMDf retrieveVector(const std::string& _data, bool& success);
		static core::vectorSIMDf retrieveHex(const std::string& _data, bool& success);

	private:
		static std::string findStandardValue(const char** _atts, bool& _errorOccurred, const core::vector<std::string>& _acceptableAttributes);
		static std::string findAndConvertXYZAttsToSingleString(const char** _atts, bool& _errorOccurred, const core::vector<std::string>& _acceptableAttributes);

};

}
}
}

#endif