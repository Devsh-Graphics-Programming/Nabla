#ifndef __I_ELEMENT_H_INCLUDED__
#define __I_ELEMENT_H_INCLUDED__

#include "../../ext/MitsubaLoader/PropertyElement.h"

#include "irr/asset/IAssetManager.h"
#include <iostream>


namespace irr { namespace ext { namespace MitsubaLoader {

//TODO: Provide elements with getName member function which will return std::string. 

class IElement
{
public:
	enum class Type
	{
		NONE,
		SCENE,
		SAMPLER,
		FILM,
		SENSOR,

		//shapes
		SHAPE,

		//other
		TRANSFORM,
		TEXTURE,
		MATERIAL,
		MEDIUM
	};

public:
	//! default implementation for elements that doesnt have any attributes
	virtual bool processAttributes(const char** _atts)
	{
		if (_atts[0])
		{
			std::cout << "Invalid .xml file structure: element " << getLogName().c_str() << " doesn't take any attributes \n";
			return false;
		}

		return true;
	}
	//! default implementation for elements that doesnt have any children
	virtual bool processChildData(IElement* _child)
	{
		if (_child != nullptr)
		{
			//ParserLog::wrongChildElement(getLogName(), _child->getName());
			return false;
		}
		return true;
	}
	//
	
	virtual bool onEndTag(asset::IAssetManager& _assetManager) = 0;
	virtual IElement::Type getType() const = 0;
	virtual std::string getLogName() const = 0;
	virtual ~IElement() = default;

	void addProperty(const SPropertyElementData& _property)
	{
		properties.emplace_back(_property);
	}

	void addProperty(SPropertyElementData&& _property)
	{
		properties.emplace_back(std::move(_property));
	}

protected:
	core::vector<SPropertyElementData> properties;

};

}
}
}

#endif