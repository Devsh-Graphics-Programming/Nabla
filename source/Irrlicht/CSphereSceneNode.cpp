// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CSphereSceneNode.h"
#include "IVideoDriver.h"
#include "ISceneManager.h"
#include "IrrlichtDevice.h"
#include "irr/asset/IAssetManager.h"
#include "irr/asset/IGeometryCreator.h"

#include "os.h"

namespace irr
{
namespace scene
{


//! constructor
CSphereSceneNode::CSphereSceneNode(float radius, uint32_t polyCountX, uint32_t polyCountY, IDummyTransformationSceneNode* parent, ISceneManager* mgr, int32_t id,
			const core::vector3df& position, const core::vector3df& rotation, const core::vector3df& scale)
: IMeshSceneNode(parent, mgr, id, position, rotation, scale), Mesh(0),
	Radius(radius), PolyCountX(polyCountX), PolyCountY(polyCountY)
{
	#ifdef _IRR_DEBUG
	setDebugName("CSphereSceneNode");
	#endif

    asset::ICPUMesh* cpumesh = SceneManager->getDevice()->getAssetManager().getGeometryCreator()->createSphereMesh(radius, polyCountX, polyCountY);
	auto res = SceneManager->getVideoDriver()->getGPUObjectsFromAssets(&cpumesh, (&cpumesh)+1);
    Mesh = res.size() ? res.front() : nullptr;
    assert(Mesh);
}



//! destructor
CSphereSceneNode::~CSphereSceneNode()
{
	if (Mesh)
		Mesh->drop();
}


//! renders the node.
void CSphereSceneNode::render()
{
	video::IVideoDriver* driver = SceneManager->getVideoDriver();

	if (Mesh && driver)
	{
		driver->setMaterial(Mesh->getMeshBuffer(0)->getMaterial());
		driver->setTransform(video::E4X3TS_WORLD, AbsoluteTransformation);

		driver->drawMeshBuffer(Mesh->getMeshBuffer(0));
	}
}



//! returns the axis aligned bounding box of this node
const core::aabbox3d<float>& CSphereSceneNode::getBoundingBox()
{
	return Mesh ? Mesh->getBoundingBox() : Box;
}


void CSphereSceneNode::OnRegisterSceneNode()
{
	if (IsVisible)
		SceneManager->registerNodeForRendering(this);

	ISceneNode::OnRegisterSceneNode();
}


//! returns the material based on the zero based index i. To get the amount
//! of materials used by this scene node, use getMaterialCount().
//! This function is needed for inserting the node into the scene hirachy on a
//! optimal position for minimizing renderstate changes, but can also be used
//! to directly modify the material of a scene node.
video::SGPUMaterial& CSphereSceneNode::getMaterial(uint32_t i)
{
	if (i>0 || !Mesh)
		return ISceneNode::getMaterial(i);
	else
		return Mesh->getMeshBuffer(i)->getMaterial();
}


//! returns amount of materials used by this scene node.
uint32_t CSphereSceneNode::getMaterialCount() const
{
	return 1;
}

//! Creates a clone of this scene node and its children.
ISceneNode* CSphereSceneNode::clone(IDummyTransformationSceneNode* newParent, ISceneManager* newManager)
{
	if (!newParent)
		newParent = Parent;
	if (!newManager)
		newManager = SceneManager;

	CSphereSceneNode* nb = new CSphereSceneNode(Radius, PolyCountX, PolyCountY, newParent,
		newManager, ID, RelativeTranslation);

	nb->cloneMembers(this, newManager);
	nb->getMaterial(0) = Mesh->getMeshBuffer(0)->getMaterial();

	if ( newParent )
		nb->drop();
	return nb;
}

} // end namespace scene
} // end namespace irr

