#ifndef __I_SHAPE_MATRIX_H_INCLUDED__
#define __I_SHAPE_MATRIX_H_INCLUDED__

#include "../../ext/MitsubaLoader/IElement.h"
#include "irrlicht.h"

namespace irr { namespace ext { namespace MitsubaLoader {

class IShape
{
public:
	IShape()
		: flipNormalsFlag(false), mesh(nullptr) {};

	inline asset::ICPUMesh* getMesh() { return mesh; }
	inline core::matrix4SIMD getTransformMatrix() { return transform; }

	virtual ~IShape() = default;

protected:
	void flipNormals(asset::IAssetManager& _assetManager)
	{
		for (int i = 0; i < mesh->getMeshBufferCount(); i++)
			_assetManager.getMeshManipulator()->flipSurfaces(mesh->getMeshBuffer(i));
	}

protected:
	asset::ICPUMesh* mesh;
	core::matrix4SIMD transform;
	bool flipNormalsFlag;

};


}
}
}

#endif