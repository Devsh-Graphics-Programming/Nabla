// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CCubeSceneNode.h"
#include "IVideoDriver.h"
#include "ISceneManager.h"
#include "os.h"

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
CCubeSceneNode::CCubeSceneNode(f32 size, IDummyTransformationSceneNode* parent, ISceneManager* mgr,
		s32 id, const core::vector3df& position,
		const core::vector3df& rotation, const core::vector3df& scale)
	: IMeshSceneNode(parent, mgr, id, position, rotation, scale),
	Mesh(0), Size(size)
{
	#ifdef _DEBUG
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
	Mesh = SceneManager->getGeometryCreator()->createCubeMeshGPU(SceneManager->getVideoDriver(),core::vector3df(Size));
}


//! renders the node.
void CCubeSceneNode::render()
{
	video::IVideoDriver* driver = SceneManager->getVideoDriver();
	driver->setTransform(video::E4X3TS_WORLD, AbsoluteTransformation);


	// for debug purposes only:
	video::SMaterial mat = Mesh->getMeshBuffer(0)->getMaterial();

	driver->setMaterial(mat);
	driver->drawMeshBuffer(Mesh->getMeshBuffer(0), (AutomaticCullingState & scene::EAC_COND_RENDER) ? query:NULL);

	// for debug purposes only:
	if (DebugDataVisible)
	{
		video::SMaterial m;
		m.AntiAliasing=0;
		driver->setMaterial(m);

		if (DebugDataVisible & scene::EDS_BBOX)
		{
			driver->draw3DBox(Mesh->getMeshBuffer(0)->getBoundingBox(), video::SColor(255,255,255,255));
		}
		if (DebugDataVisible & scene::EDS_BBOX_BUFFERS)
		{
			driver->draw3DBox(Mesh->getMeshBuffer(0)->getBoundingBox(),
					video::SColor(255,190,128,128));
		}

		// show mesh
		if (DebugDataVisible & scene::EDS_MESH_WIRE_OVERLAY)
		{
			m.Wireframe = true;
			driver->setMaterial(m);

			driver->drawMeshBuffer(Mesh->getMeshBuffer(0), (AutomaticCullingState & scene::EAC_COND_RENDER) ? query:NULL);
		}
	}
}


//! returns the axis aligned bounding box of this node
const core::aabbox3d<f32>& CCubeSceneNode::getBoundingBox()
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
video::SMaterial& CCubeSceneNode::getMaterial(u32 i)
{
	return Mesh->getMeshBuffer(0)->getMaterial();
}


//! returns amount of materials used by this scene node.
u32 CCubeSceneNode::getMaterialCount() const
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

