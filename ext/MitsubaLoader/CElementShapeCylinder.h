#ifndef __C_ELEMENT_SHAPE_CYLINDER_H_INCLUDED__
#define __C_ELEMENT_SHAPE_CYLINDER_H_INCLUDED__

#include "../../ext/MitsubaLoader/IElement.h"
#include "../../ext/MitsubaLoader/IShape.h"
#include "irrlicht.h"

namespace irr { namespace ext { namespace MitsubaLoader {

/*representation of cube shape: <shape type="cube"> .. </shape>*/
class CElementShapeCylinder : public IElement, public IShape
{
public:
	CElementShapeCylinder()
		:p1(0.0f, 0.0f, 1.0f), radius(1.0f) {};

	virtual bool processAttributes(const char** _args) override;
	virtual bool onEndTag(asset::IAssetManager& _assetManager, IElement* _parent) override;
	virtual IElement::Type getType() const override { return IElement::Type::SHAPE;  };
	virtual std::string getLogName() const override { return "shape cube"; };
	virtual bool processChildData(IElement* child) override;

private:
	core::vector3df_SIMD p0;
	core::vector3df_SIMD p1;
	float radius;

};

}
}
}

#endif