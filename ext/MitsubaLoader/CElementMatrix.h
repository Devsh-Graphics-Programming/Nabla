#ifndef __C_ELEMENT_MATRIX_H_INCLUDED__
#define __C_ELEMENT_MATRIX_H_INCLUDED__

#include "../../ext/MitsubaLoader/IElement.h"
#include "irrlicht.h"

namespace irr { namespace ext { namespace MitsubaLoader {

class CElementMatrix : public IElement
{
public:
	enum class Type
	{
		ARBITRARY,
		TRANSLATION,
		ROTATION,
		SCALE
	};

public:
	CElementMatrix(CElementMatrix::Type _type)
		:type(_type) {};


	virtual bool processAttributes(const char** _args) override;
	virtual bool onEndTag(asset::IAssetManager& _assetManager, IElement* _parent) override;
	virtual IElement::Type getType() const override { return IElement::Type::MATRIX;  };
	virtual std::string getLogName() const override { return "matrix"; };

	const core::matrix4SIMD getMatrix() const { return matrix; }

private:
	core::matrix4SIMD matrix;
	const CElementMatrix::Type type;

private:
	static std::pair<bool, core::matrix4SIMD> getMatrixFromString(std::string _str);
};

}
}
}

#endif