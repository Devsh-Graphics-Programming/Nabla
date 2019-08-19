#ifndef __C_SHAPE_CREATOR_H_INCLUDED__
#define __C_SHAPE_CREATOR_H_INCLUDED__

#include "../../ext/MitsubaLoader/PropertyElement.h"

#include "irrlicht.h"

namespace irr { namespace ext { namespace MitsubaLoader {

	
class CShapeCreator
{
public:
	static asset::ICPUMesh* createCube(asset::IAssetManager& _assetManager, const core::vector<SPropertyElementData>& properties);
	static asset::ICPUMesh* createSphere(asset::IAssetManager& _assetManager, const core::vector<SPropertyElementData>& properties, core::matrix4SIMD& transform);
	static asset::ICPUMesh* createCylinder(asset::IAssetManager& _assetManager, const core::vector<SPropertyElementData>& properties, core::matrix4SIMD& transform);
	static asset::ICPUMesh* createRectangle(asset::IAssetManager& _assetManager, const core::vector<SPropertyElementData>& properties);
	static asset::ICPUMesh* createDisk(asset::IAssetManager& _assetManager, const core::vector<SPropertyElementData>& properties);
	static asset::ICPUMesh* createOBJ(asset::IAssetManager& _assetManager, const core::vector<SPropertyElementData>& properties);
	static asset::ICPUMesh* createPLY(asset::IAssetManager& _assetManager, const core::vector<SPropertyElementData>& properties);

};

}
}
}

#endif