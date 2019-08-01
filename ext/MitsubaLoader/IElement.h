#ifndef __I_ELEMENT_H_INCLUDED__
#define __I_ELEMENT_H_INCLUDED__

#include "irr/asset/IAssetManager.h"
#include <iostream>


namespace irr { namespace ext { namespace MitsubaLoader {

//TODO: Provide elements with getName member function which will return std::string. 

struct ParserData;

class IElement
{
public:
	enum class Type
	{
		NONE,
		SCENE,

		//shapes
		SHAPE_OBJ,
		SHAPE_PLY,
		SHAPE_SERIALIZED,
		SHAPE_CUBE,
		SHAPE_SPHERE,
		SHAPE_CYLINDER,
		SHAPE_DISK,
		SHAPE_RECTANGLE,

		//vectors, points, scalars
		FLOAT,
		BOOLEAN,
		INTEGER,
		POINT,

		//other
		TRANSFORM,
		TEXTURE,
		STRING,
		MATRIX
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
	virtual bool onEndTag(asset::IAssetManager& _assetManager, IElement* _parent) = 0;
	virtual IElement::Type getType() const = 0;
	virtual std::string getLogName() const = 0;
	

};

}
}
}

#endif