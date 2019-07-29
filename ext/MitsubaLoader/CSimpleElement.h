#ifndef __C_SIMPLE_ELEMENT_H_INCLUDED__
#define __C_SIMPLE_ELEMENT_H_INCLUDED__

#include "IElement.h"
#include "../include/irr/static_if.h"
#include "ParserUtil.h"

#include "irrlicht.h"

namespace irr { namespace ext { namespace MitsubaLoader {

template<typename T>
class CSimpleElement : public IElement
{
	static_assert(
		std::is_same<T, float>::value ||
		std::is_same<T, bool>::value ||
		std::is_same<T, core::vector3df_SIMD>::value,
		"assertion failed: cannot use this type");

public:
	CSimpleElement()
		:type(IElement::Type::NONE),
		value(0.0f) 
	{
		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<T, float>::value)
		{
			logName = "float";
			return;
		}
		IRR_PSEUDO_IF_CONSTEXPR_END

		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<T, bool>::value)
		{
			logName = "boolean";
			return;
		}
		IRR_PSEUDO_IF_CONSTEXPR_END

		IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<T, core::vector3df_SIMD>::value)
		{
			logName = "point";
			return;
		}
		IRR_PSEUDO_IF_CONSTEXPR_END
	};

	virtual bool processAttributes(const char** _atts) override;
	virtual bool onEndTag(asset::IAssetManager& _assetManager, IElement* _parent) override 
	{ 
		//so far it is only for testing purpose

		os::Printer::print("\t" + getLogName() + " name: " + nameAttr + " value: ");
		//os::Printer::print(value);

		return true; 
	}
	virtual IElement::Type getType() const  override { return type; };
	virtual std::string getLogName() const  override { return logName; };

	inline std::string getNameAttribute() const { return nameAttr; }
	inline T getValueAttribute() const { return valueAttr; }

private:
	T value;
	IElement::Type type;
	std::string logName;
	
	std::string nameAttr;
	T valueAttr;
};

using CElementFloat = CSimpleElement<float>;
using CElementBoolean = CSimpleElement<bool>;
using CElementPoint = CSimpleElement<irr::core::vector3df_SIMD>;

template<typename T>
bool CSimpleElement<T>::processAttributes(const char** _atts)
{
	static_assert(false);
}

template<>
bool CSimpleElement<float>::processAttributes(const char** _atts)
{
	bool isNameSet = false;

	for (int i = 0; _atts[i]; i += 2)
	{
		if (!std::strcmp(_atts[i], "name"))
		{
			if (!std::strcmp(_atts[i + 1], "radius") ||
				!std::strcmp(_atts[i + 1], "maxSmoothAngle"))
			{
				nameAttr = _atts[i + 1];
				isNameSet = true;
			}
			else
			{
				//print warning
			}
		}
		else if (!std::strcmp(_atts[i], "value"))
		{
			value = static_cast<float>(atof(_atts[i + 1]));
		}
		else
		{
			//print warning
		}
			
	}

	if (!isNameSet)
		;//print error

	return isNameSet;
}

template<>
bool CSimpleElement<bool>::processAttributes(const char** _atts)
{
	bool isNameSet = false;

	for (int i = 0; _atts[i]; i += 2)
	{
		if (!std::strcmp(_atts[i], "name"))
		{
			if (!std::strcmp(_atts[i + 1], "flipNormals") ||
				!std::strcmp(_atts[i + 1], "flipTexCoords") ||
				!std::strcmp(_atts[i + 1], "faceNormals") ||
				!std::strcmp(_atts[i + 1], "srgb"))
			{
				nameAttr = _atts[i + 1];
				isNameSet = true;
			}
			else
			{
				//print warning
			}
		}


		if (!std::strcmp(_atts[i], "value"))
		{
			if (!std::strcmp(_atts[i + 1], "true"))
				value = true;
			else if (!std::strcmp(_atts[i + 1], "true"))
				value = false;
			else
			{
				//print warning
			}
		}
	}

	return isNameSet;
}

template<>
bool CSimpleElement<core::vector3df_SIMD>::processAttributes(const char** _atts)
{
	bool isNameSet = false;

	return isNameSet;
}

}
}
}

#endif