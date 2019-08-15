#ifndef __C_ELEMENT_SHAPE_RECTANGLE_H_INCLUDED__
#define __C_ELEMENT_SHAPE_RECTANGLE_H_INCLUDED__

#include "../../ext/MitsubaLoader/IElement.h"
#include "../../ext/MitsubaLoader/IShape.h"
#include "irrlicht.h"

namespace irr { namespace ext { namespace MitsubaLoader {

/*representation of cube shape: <shape type="cube"> .. </shape>*/
class CElementShapeRectangle : public IElement, public IShape
{
public:
	virtual bool processAttributes(const char** _args) override;
	virtual bool onEndTag(asset::IAssetManager& _assetManager, IElement* _parent) override;
	virtual IElement::Type getType() const override { return IElement::Type::SHAPE;  };
	virtual std::string getLogName() const override { return "shape rectangle"; };
	virtual bool processChildData(IElement* child) override;

};

}
}
}

#endif