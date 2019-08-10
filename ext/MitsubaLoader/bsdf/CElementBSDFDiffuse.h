#ifndef __C_ELEMENT_BSDF_DIFFUSE_H_INCLUDED__
#define __C_ELEMENT_BSDF_DIFFUSE_H_INCLUDED__

#include "../../../ext/MitsubaLoader/CSimpleElement.h"



namespace irr { namespace ext { namespace MitsubaLoader {

class CElementBSDFDiffuse : public IElement
{
public:
	CElementBSDFDiffuse()
		:usesTexture(false) {};

	virtual bool processAttributes(const char** _args) override;
	virtual bool onEndTag(asset::IAssetManager& _assetManager, IElement* _parent) override;
	virtual IElement::Type getType() const override { return IElement::Type::MATERIAL; };
	virtual std::string getLogName() const override { return "BSDF_diffuse"; };
	virtual bool processChildData(IElement* child) override;

private:
	bool usesTexture;
	union
	{
		//TODO: texture
		core::vectorSIMDf color;
	};

};


}
}
}

#endif