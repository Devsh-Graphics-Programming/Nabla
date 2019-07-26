#ifndef __C_MITSUBA_SCENE_H_INCLUDED__
#define __C_MITSUBA_SCENE_H_INCLUDED__

#include "IElement.h"
#include "irr/asset/SCPUMesh.h"

namespace irr { namespace ext { namespace MitsubaLoader {

//TODO: functions like processAttributes should have bool as return type (true if all arguemnts are valid)
//TODO: change every std::cout << .. to log

//root element
class CMitsubaScene : public IElement
{
public:
	//! Constructor
	CMitsubaScene()
		//acceptable?
		:mesh(new asset::SCPUMesh) {}

	virtual bool processAttributes(const char** _arguments) override;
	virtual void onEndTag(asset::IAssetManager& assetManager, IElement* parent) override;
	virtual IElement::Type getType() const override { return IElement::Type::SCENE; };
	virtual void processChildData(IElement* child) override;

	inline asset::SCPUMesh* releaseMesh() { return mesh; mesh = nullptr; }
	//void appendMesh(asset::ICPUMesh* mesh);

private:
	asset::SCPUMesh* mesh;
};

}
}
}
#endif