#ifndef __C_SHAPE_CREATOR_H_INCLUDED__
#define __C_SHAPE_CREATOR_H_INCLUDED__

#include "../../ext/MitsubaLoader/PropertyElement.h"

#include "irrlicht.h"

namespace irr { namespace ext { namespace MitsubaLoader {

	
class CShapeCreator
{
public:
	static core::smart_refctd_ptr<asset::ICPUMesh> createCube(asset::IAssetManager* _assetManager, const core::vector<SPropertyElementData>& properties);
	static core::smart_refctd_ptr<asset::ICPUMesh> createSphere(asset::IAssetManager* _assetManager, const core::vector<SPropertyElementData>& properties, core::matrix4SIMD& transform);
	static core::smart_refctd_ptr<asset::ICPUMesh> createCylinder(asset::IAssetManager* _assetManager, const core::vector<SPropertyElementData>& properties, core::matrix4SIMD& transform);
	static core::smart_refctd_ptr<asset::ICPUMesh> createRectangle(asset::IAssetManager* _assetManager, const core::vector<SPropertyElementData>& properties);
	static core::smart_refctd_ptr<asset::ICPUMesh> createDisk(asset::IAssetManager* _assetManager, const core::vector<SPropertyElementData>& properties);
	static core::smart_refctd_ptr<asset::ICPUMesh> createOBJ(asset::IAssetManager* _assetManager, const core::vector<SPropertyElementData>& properties);
	static core::smart_refctd_ptr<asset::ICPUMesh> createPLY(asset::IAssetManager* _assetManager, const core::vector<SPropertyElementData>& properties);

};

}
}
}

#endif