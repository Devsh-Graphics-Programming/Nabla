#ifndef __C_ELEMENT_COLOR_H_INCLUDED__
#define __C_ELEMENT_COLOR_H_INCLUDED__

#include "../../ext/MitsubaLoader/IElement.h"
#include "../../ext/MitsubaLoader/IShape.h"
#include "irrlicht.h"

namespace irr { namespace ext { namespace MitsubaLoader {


class CElementColor : public IElement
{
public:
	CElementColor(bool _srgb = false)
		:srgb(_srgb) {};

	virtual bool processAttributes(const char** _args) override;
	virtual bool onEndTag(asset::IAssetManager& _assetManager, IElement* _parent) override;
	virtual IElement::Type getType() const override { return IElement::Type::COLOR;  };
	virtual std::string getLogName() const override { return "color"; };

	inline const std::string getName() const { return nameAttr; }
	inline const core::vector3df_SIMD getColor() const { return color; }

private:
	static std::pair<bool, core::vector3df_SIMD> retriveColorFromValueAttribute(std::string value);

private:
	std::string nameAttr;
	core::vector3df_SIMD color;

	bool srgb;
};

}
}
}

#endif