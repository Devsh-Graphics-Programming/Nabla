#ifndef __C_ELEMENT_TRANSFORM_H_INCLUDED__
#define __C_ELEMENT_TRANSFORM_H_INCLUDED__

#include "../../ext/MitsubaLoader/IElement.h"
#include "irrlicht.h"

namespace irr { namespace ext { namespace MitsubaLoader {

class CElementTransform : public IElement
{
public:
	virtual bool processAttributes(const char** _args) override;
	virtual bool onEndTag(asset::IAssetManager& _assetManager) override;
	virtual IElement::Type getType() const override { return IElement::Type::TRANSFORM; };
	virtual std::string getLogName() const override { return "transform"; };

	inline const core::matrix4SIMD getMatrix() const { return matrix; }
	inline const std::string getName() const { return name; }

private:
	core::matrix4SIMD matrix;
	std::string name;

};

}
}
}

#endif