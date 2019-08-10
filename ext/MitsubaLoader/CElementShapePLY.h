#ifndef __C_ELEMENT_SHAPE_PLY_H_INCLUDED__
#define __C_ELEMENT_SHAPE_PLY_H_INCLUDED__

#include "../../ext/MitsubaLoader/IElement.h"
#include "../../ext/MitsubaLoader/IShape.h"
#include "irrlicht.h"

namespace irr { namespace ext { namespace MitsubaLoader {

class CElementShapePLY : public IElement, public IShape
{
public:
	CElementShapePLY()
		:faceNormals(false),
		smoothNormals(false),
		maxSmoothAngle(0.0f),
		srgb(true) {};

	~CElementShapePLY();

	virtual bool processAttributes(const char** _args) override;
	virtual bool onEndTag(asset::IAssetManager& _assetManager, IElement* _parent) override;
	virtual IElement::Type getType() const override { return IElement::Type::SHAPE;  };
	virtual std::string getLogName() const override { return "shape_obj"; };
	virtual bool processChildData(IElement* child) override;

private:
	std::string fileName;
	bool faceNormals;
	//indicates if normals should be smoothed (false by default)
	bool smoothNormals;
	float maxSmoothAngle;
	bool srgb;
};

}
}
}

#endif