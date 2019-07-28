#ifndef __C_MITSUBA_SCENE_H_INCLUDED__
#define __C_MITSUBA_SCENE_H_INCLUDED__

#include "IElement.h"
#include "irr/asset/SCPUMesh.h"

#include <memory>

namespace irr { namespace ext { namespace MitsubaLoader {


class CMitsubaScene : public IElement
{
public:
	//! Constructor
	CMitsubaScene()
		:mesh(new asset::SCPUMesh) {}

	virtual bool processAttributes(const char** _arguments) override;
	virtual bool onEndTag(asset::IAssetManager& assetManager, IElement* parent) override;
	virtual IElement::Type getType() const override { return IElement::Type::SCENE; };
	virtual std::string getName() const override { return "scene"; };
	virtual bool processChildData(IElement* child) override;

	asset::SCPUMesh* releaseMesh() { return mesh; mesh = nullptr; }
	//void appendMesh(asset::ICPUMesh* mesh);

private:
	asset::SCPUMesh* mesh;

};

}
}
}
#endif