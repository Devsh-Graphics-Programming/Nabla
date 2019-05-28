// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CCubeSceneNode.h"
#include "IVideoDriver.h"
#include "ISceneManager.h"
#include "os.h"
#include "SMaterial.h"
#include "IrrlichtDevice.h"
#include "irr/asset/IAssetManager.h"
#include "irr/asset/IGeometryCreator.h"

namespace irr
{
namespace scene
{


	/*
        011         111
          /6,8------/5        y
         /  |      / |        ^  z
        /   |     /  |        | /
    010 3,9-------2  |        |/
        |   7- - -10,4 101     *---->x
        |  /      |  /
        |/        | /
        0------11,1/
       000       100
	*/

//! constructor
CCubeSceneNode::CCubeSceneNode(float size, IDummyTransformationSceneNode* parent, ISceneManager* mgr,
		int32_t id, const core::vector3df& position,
		const core::vector3df& rotation, const core::vector3df& scale)
	: IMeshSceneNode(parent, mgr, id, position, rotation, scale),
	Mesh(0), Size(size)
{
	#ifdef _IRR_DEBUG
	setDebugName("CCubeSceneNode");
	#endif

	setSize();
}


CCubeSceneNode::~CCubeSceneNode()
{
	if (Mesh)
		Mesh->drop();
}


void CCubeSceneNode::setSize()
{
	if (Mesh)
		Mesh->drop();

    asset::ICPUMesh* cpumesh = SceneManager->getDevice()->getAssetManager().getGeometryCreator()->createCubeMesh(core::vector3df(Size));
    auto res = SceneManager->getVideoDriver()->getGPUObjectsFromAssets(&cpumesh, (&cpumesh)+1);
    Mesh = res.size() ? res.front() : nullptr;
    assert(Mesh);
}


//! renders the node.
void CCubeSceneNode::render()
{
	video::IVideoDriver* driver = SceneManager->getVideoDriver();

    if (canProceedPastFence())
    {
        driver->setTransform(video::E4X3TS_WORLD, AbsoluteTransformation);


        // for debug purposes only:
        video::SGPUMaterial mat = Mesh->getMeshBuffer(0)->getMaterial();

        driver->setMaterial(mat);
        driver->drawMeshBuffer(Mesh->getMeshBuffer(0));
    }
    else if (DebugDataVisible)
        driver->setTransform(video::E4X3TS_WORLD, AbsoluteTransformation);

	// for debug purposes only:
	if (DebugDataVisible)
	{
		video::SGPUMaterial m;
		driver->setMaterial(m);

		// show mesh
		if (DebugDataVisible & scene::EDS_MESH_WIRE_OVERLAY)
		{
			m.Wireframe = true;
			driver->setMaterial(m);

			driver->drawMeshBuffer(Mesh->getMeshBuffer(0));
		}
	}
}


//! returns the axis aligned bounding box of this node
const core::aabbox3d<float>& CCubeSceneNode::getBoundingBox()
{
	return Mesh->getMeshBuffer(0)->getBoundingBox();
}



void CCubeSceneNode::OnRegisterSceneNode()
{
	if (IsVisible)
		SceneManager->registerNodeForRendering(this);
	ISceneNode::OnRegisterSceneNode();
}


//! returns the material based on the zero based index i.
video::SGPUMaterial& CCubeSceneNode::getMaterial(uint32_t i)
{
	return Mesh->getMeshBuffer(0)->getMaterial();
}


//! returns amount of materials used by this scene node.
uint32_t CCubeSceneNode::getMaterialCount() const
{
	return 1;
}


//! Creates a clone of this scene node and its children.
ISceneNode* CCubeSceneNode::clone(IDummyTransformationSceneNode* newParent, ISceneManager* newManager)
{
	if (!newParent)
		newParent = Parent;
	if (!newManager)
		newManager = SceneManager;

	CCubeSceneNode* nb = new CCubeSceneNode(Size, newParent,
		newManager, ID, RelativeTranslation);

	nb->cloneMembers(this, newManager);
	nb->getMaterial(0) = getMaterial(0);

	if ( newParent )
		nb->drop();
	return nb;
}

} // end namespace scene
} // end namespace irr

