#ifndef __C_ELEMENT_SHAPE_CUBE_H_INCLUDED__
#define __C_ELEMENT_SHAPE_CUBE_H_INCLUDED__

#include "IElement.h"
#include "irrlicht.h"

namespace irr { namespace ext { namespace MitsubaLoader {

/*representation of cube shape: <shape type="cube"> .. </shape>*/
class CElementShapeCube : public IElement
{
public:
	CElementShapeCube();

	virtual bool processAttributes(const char** _args) override;
	virtual bool onEndTag(asset::IAssetManager& _assetManager, IElement* _parent) override;
	virtual IElement::Type getType() const override { return IElement::Type::SHAPE_CUBE;  };
	virtual std::string getName() const override { return "shape cube"; };
	virtual bool processChildData(IElement* child) override;

private:
	core::matrix4SIMD transform;

	/* 
	From mitsuba documentation:
	This shape plugin describes a simple cube/cuboid intersection primitive. 
	By default, it creates a cube between the world-space positions (−1, −1, −1) and (1, 1, 1). 
	However, an arbitrary linear transformation may be specified to translate, rotate, 
	scale or skew it as desired. The parameterization of
	this shape maps every face onto the rectangle [0, 1]^2
	in uv space.
	*/
};

}
}
}

#endif