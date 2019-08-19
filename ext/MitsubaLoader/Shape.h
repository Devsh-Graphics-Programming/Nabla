#ifndef __C_SHAPE_H_INCLUDED__
#define __C_SHAPE_H_INCLUDED__

#include "../../ext/MitsubaLoader/IElement.h"
#include "irrlicht.h"

namespace irr { namespace ext { namespace MitsubaLoader {

class CShape : public IElement
{
public:
	CShape()
		: mesh(nullptr) {};

	inline asset::ICPUMesh* getMesh() { return mesh; }
	inline core::matrix4SIMD& getTransformMatrix() { return transform; }
	inline const core::matrix4SIMD& getTransformMatrix() const { return transform; }

	virtual bool processAttributes(const char** _atts) override;
	virtual bool processChildData(IElement* _child) override;
	virtual bool onEndTag(asset::IAssetManager& _assetManager) override;

	virtual IElement::Type getType() const override { return IElement::Type::SHAPE; };
	virtual std::string getLogName() const { return "shape"; };

	virtual ~CShape() = default;

protected:
	void flipNormals(asset::IAssetManager& _assetManager)
	{
		for (int i = 0; i < mesh->getMeshBufferCount(); i++)
			_assetManager.getMeshManipulator()->flipSurfaces(mesh->getMeshBuffer(i));
	}

protected:
	std::string type;
	asset::ICPUMesh* mesh;
	core::matrix4SIMD transform;

};


}
}
}

#endif