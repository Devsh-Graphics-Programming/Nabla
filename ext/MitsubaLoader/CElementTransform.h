#ifndef __C_ELEMENT_TRANSFORM_H_INCLUDED__
#define __C_ELEMENT_TRANSFORM_H_INCLUDED__

#include "../../ext/MitsubaLoader/IElement.h"


namespace irr
{
namespace ext
{
namespace MitsubaLoader
{


class CElementTransform : public IElement
{
	public:
		CElementTransform(std::string&& _name) : IElement(""), name(_name), matrix() {}
		virtual ~CElementTransform() {}

		bool addProperty(SPropertyElementData&& _property) override;
		bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override, CGlobalMitsubaMetadata* globalMetadata) override { return true; }
		IElement::Type getType() const override { return IElement::Type::TRANSFORM; }
		std::string getLogName() const override { return "transform"; }

		std::string name;
		core::matrix4SIMD matrix;
};

}
}
}

#endif