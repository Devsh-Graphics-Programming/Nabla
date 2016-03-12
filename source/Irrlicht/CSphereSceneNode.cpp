// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CSphereSceneNode.h"
#include "IVideoDriver.h"
#include "ISceneManager.h"

#include "os.h"

namespace irr
{
namespace scene
{


//! constructor
CSphereSceneNode::CSphereSceneNode(f32 radius, u32 polyCountX, u32 polyCountY, ISceneNode* parent, ISceneManager* mgr, s32 id,
			const core::vector3df& position, const core::vector3df& rotation, const core::vector3df& scale)
: IMeshSceneNode(parent, mgr, id, position, rotation, scale), Mesh(0),
	Radius(radius), PolyCountX(polyCountX), PolyCountY(polyCountY)
{
	#ifdef _DEBUG
	setDebugName("CSphereSceneNode");
	#endif

	Mesh = SceneManager->getGeometryCreator()->createSphereMeshGPU(SceneManager->getVideoDriver(),radius, polyCountX, polyCountY);
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
		driver->setTransform(video::ETS_WORLD, AbsoluteTransformation);

		driver->drawMeshBuffer(Mesh->getMeshBuffer(0), (AutomaticCullingState & scene::EAC_COND_RENDER) ? query:NULL);

		if ( DebugDataVisible & scene::EDS_BBOX )
		{
			video::SMaterial m;
			driver->setMaterial(m);
			driver->draw3DBox(Mesh->getMeshBuffer(0)->getBoundingBox(), video::SColor(255,255,255,255));
		}
	}
}


//! Removes a child from this scene node.
//! Implemented here, to be able to remove the shadow properly, if there is one,
//! or to remove attached childs.
bool CSphereSceneNode::removeChild(ISceneNode* child)
{
	return ISceneNode::removeChild(child);
}



//! returns the axis aligned bounding box of this node
const core::aabbox3d<f32>& CSphereSceneNode::getBoundingBox() const
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
video::SMaterial& CSphereSceneNode::getMaterial(u32 i)
{
	if (i>0 || !Mesh)
		return ISceneNode::getMaterial(i);
	else
		return Mesh->getMeshBuffer(i)->getMaterial();
}


//! returns amount of materials used by this scene node.
u32 CSphereSceneNode::getMaterialCount() const
{
	return 1;
}

//! Creates a clone of this scene node and its children.
ISceneNode* CSphereSceneNode::clone(ISceneNode* newParent, ISceneManager* newManager)
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

