#ifndef __C_SIMPLE_ELEMENT_H_INCLUDED__
#define __C_SIMPLE_ELEMENT_H_INCLUDED__

#include "../../ext/MitsubaLoader/IElement.h"
#include "../../ext/MitsubaLoader/ParserUtil.h"
#include "../include/irr/static_if.h"

#include "irrlicht.h"

namespace irr { namespace ext { namespace MitsubaLoader {

template<typename T>
class CSimpleElement : public IElement
{
	static_assert(
		std::is_same<T, float>::value ||
		std::is_same<T, bool>::value ||
		std::is_same<T, int>::value ||
		std::is_same<T, core::vector3df_SIMD>::value ||
		std::is_same<T, std::string>::value,
		"assertion failed: cannot use this type");

public:
	CSimpleElement()
	{
		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<T, float>::value) { logName = "float"; type = IElement::Type::FLOAT; }
		IRR_PSEUDO_IF_CONSTEXPR_END

		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<T, bool>::value) { logName = "boolean"; type = IElement::Type::BOOLEAN; }
		IRR_PSEUDO_IF_CONSTEXPR_END

		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<T, core::vector3df_SIMD>::value) { logName = "point"; type = IElement::Type::POINT; }
		IRR_PSEUDO_IF_CONSTEXPR_END

		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<T, int>::value) { logName = "integer"; type = IElement::Type::INTEGER; }
		IRR_PSEUDO_IF_CONSTEXPR_END

		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<T, std::string>::value) { logName = "string"; type = IElement::Type::STRING; }
		IRR_PSEUDO_IF_CONSTEXPR_END
	};

	virtual bool processAttributes(const char** _atts) override;
	virtual bool onEndTag(asset::IAssetManager& _assetManager, IElement* _parent) override;
	virtual IElement::Type getType() const  override { return type; };
	virtual std::string getLogName() const  override { return logName; };

	inline std::string getNameAttribute() const { return nameAttr; }
	inline T getValueAttribute() const { return value; }

private:
	IElement::Type type;
	std::string logName;
	
	std::string nameAttr;
	T value;
};

using CElementFloat = CSimpleElement<float>;
using CElementBoolean = CSimpleElement<bool>;
using CElementPoint = CSimpleElement<irr::core::vector3df_SIMD>;
using CElementString = CSimpleElement<std::string>;

template<typename T>
inline bool CSimpleElement<T>::processAttributes(const char** _atts)
{
	static_assert(false);
}

template<>
inline bool CSimpleElement<float>::processAttributes(const char** _atts)
{
	bool isNameSet = false;

	for (int i = 0; _atts[i]; i += 2)
	{
		if (!std::strcmp(_atts[i], "name"))
		{
			nameAttr = _atts[i + 1];
			isNameSet = true;
		}
		else if (!std::strcmp(_atts[i], "value"))
		{
			value = static_cast<float>(atof(_atts[i + 1]));
		}
		else
		{
			//print warning (only attributes float has are name and value)
		}
			
	}

	if (!isNameSet)
		;//print error

	return isNameSet;
}

template<>
inline bool CSimpleElement<bool>::processAttributes(const char** _atts)
{
	bool isNameSet = false;

	for (int i = 0; _atts[i]; i += 2)
	{
		if (!std::strcmp(_atts[i], "name"))
		{
			nameAttr = _atts[i + 1];
			isNameSet = true;
		}


		if (!std::strcmp(_atts[i], "value"))
		{
			if (!std::strcmp(_atts[i + 1], "true"))
				value = true;
			else if (!std::strcmp(_atts[i + 1], "false"))
				value = false;
			else
			{
				//print warning (only true or false)
			}
		}
	}

	return isNameSet;
}

template<>
inline bool CSimpleElement<std::string>::processAttributes(const char** _atts)
{
	bool isNameSet = false;

	for (int i = 0; _atts[i]; i += 2)
	{
		if (!std::strcmp(_atts[i], "name"))
		{
			nameAttr = _atts[i + 1];
			isNameSet = true;
		}
		else if (!std::strcmp(_atts[i], "value"))
		{
			value = _atts[i + 1];
		}
		else
		{
			//print warning (only attributes string has are name and value)
		}

	}

	if (!isNameSet)
		;//print error

	return isNameSet;
}

template<>
inline bool CSimpleElement<core::vector3df_SIMD>::processAttributes(const char** _atts)
{
	bool isNameSet = false;

	for (int i = 0; _atts[i]; i += 2)
	{
		if (!std::strcmp(_atts[i], "name"))
		{
			nameAttr = _atts[i + 1];
			isNameSet = true;
		}
		else if (!std::strcmp(_atts[i], "x"))
		{
			value.x = static_cast<float>(atof(_atts[i + 1]));;
		}
		else if (!std::strcmp(_atts[i], "y"))
		{
			value.y = static_cast<float>(atof(_atts[i + 1]));;
		}
		else if (!std::strcmp(_atts[i], "z"))
		{
			value.z = static_cast<float>(atof(_atts[i + 1]));;
		}
		else if (!std::strcmp(_atts[i], "w"))
		{
			value.w = static_cast<float>(atof(_atts[i + 1]));;
		}
		else
		{
			//print warning (only attributes string has are name and value)
		}

	}

	if (!isNameSet)
		;//print error

	return isNameSet;
}

template<>
inline bool CSimpleElement<int>::processAttributes(const char** _atts)
{
	bool isNameSet = false;

	return isNameSet;
}

template<typename T>
bool CSimpleElement<T>::onEndTag(asset::IAssetManager& _assetManager, IElement* _parent)
{
	if (_parent)
		return _parent->processChildData(this);

	return true;
}

}
}
}

#endif