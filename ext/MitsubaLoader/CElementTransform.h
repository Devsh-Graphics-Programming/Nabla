#ifndef __C_ELEMENT_TRANSFORM_H_INCLUDED__
#define __C_ELEMENT_TRANSFORM_H_INCLUDED__

#include "../../ext/MitsubaLoader/IElement.h"
#include "../../ext/MitsubaLoader/CElementMatrix.h"
#include "irrlicht.h"

namespace irr { namespace ext { namespace MitsubaLoader {

class CElementTransform : public IElement
{
public:
	virtual bool processAttributes(const char** _args) override;
	virtual bool onEndTag(asset::IAssetManager& _assetManager, IElement* _parent) override;
	virtual IElement::Type getType() const override { return IElement::Type::TRANSFORM; };
	virtual std::string getLogName() const override { return "transform"; };
	virtual bool processChildData(IElement* _child) override;

	inline const core::matrix4SIMD getMatrix() const { return resultMatrix; }
	inline const std::string getName() const { return name; }

private:
	core::vector<core::matrix4SIMD> matrices;
	core::matrix4SIMD resultMatrix;
	std::string name;

};

}
}
}

#endif