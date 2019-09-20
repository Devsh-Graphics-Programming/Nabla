#ifndef __C_MITSUBA_SCENE_H_INCLUDED__
#define __C_MITSUBA_SCENE_H_INCLUDED__

#include "../../ext/MitsubaLoader/IElement.h"
#include "irr/asset/ICPUMesh.h"

#include <memory>

namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

class CMitsubaScene : public IElement
{
public:
	virtual bool processAttributes(const char** _arguments) override;
	virtual bool onEndTag(asset::IAssetLoader::IAssetLoaderOverride* _override) override;
	virtual IElement::Type getType() const override { return IElement::Type::SCENE; };
	virtual std::string getLogName() const override { return "scene"; };
	virtual bool processChildData(IElement* child) override;

	inline asset::SAssetBundle releaseMeshes()
	{
		return asset::SAssetBundle(std::move(meshes));
	}

private:
	core::vector<core::smart_refctd_ptr<asset::ICPUMesh>> meshes;
};

}
}
}
#endif