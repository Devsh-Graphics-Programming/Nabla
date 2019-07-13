// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CMeshSceneNode.h"
#include "IVideoDriver.h"
#include "ISceneManager.h"
#include "ICameraSceneNode.h"
#include "IAnimatedMesh.h"
#include "IMaterialRenderer.h"
#include "IFileSystem.h"

namespace irr
{
namespace scene
{



//! constructor
CMeshSceneNode::CMeshSceneNode(video::IGPUMesh* mesh, IDummyTransformationSceneNode* parent, ISceneManager* mgr, int32_t id,
			const core::vector3df& position, const core::vector3df& rotation,
			const core::vector3df& scale)
: IMeshSceneNode(parent, mgr, id, position, rotation, scale), Mesh(0),
	PassCount(0), ReferencingMeshMaterials(true)
{
	#ifdef _IRR_DEBUG
	setDebugName("CMeshSceneNode");
	#endif

	setMesh(mesh);
}


//! destructor
CMeshSceneNode::~CMeshSceneNode()
{
	if (Mesh)
		Mesh->drop();
}


//! frame
void CMeshSceneNode::OnRegisterSceneNode()
{
	if (IsVisible)
	{
		// because this node supports rendering of mixed mode meshes consisting of
		// transparent and solid material at the same time, we need to go through all
		// materials, check of what type they are and register this node for the right
		// render pass according to that.

		video::IVideoDriver* driver = SceneManager->getVideoDriver();

		PassCount = 0;
		int transparentCount = 0;
		int solidCount = 0;

		// count transparent and solid materials in this scene node
		if (ReferencingMeshMaterials && Mesh)
		{
			// count mesh materials

			for (uint32_t i=0; i<Mesh->getMeshBufferCount(); ++i)
			{
                video::IGPUMeshBuffer* mb = Mesh->getMeshBuffer(i);
				if (!mb||(mb->getIndexCount()<1 && !mb->isIndexCountGivenByXFormFeedback()))
                    continue;

				video::IMaterialRenderer* rnd = driver->getMaterialRenderer(mb->getMaterial().MaterialType);

				if (rnd && rnd->isTransparent())
					++transparentCount;
				else
					++solidCount;

				if (solidCount && transparentCount)
					break;
			}
		}
		else
		{
			// count copied materials

			for (uint32_t i=0; i<Materials.size(); ++i)
			{
                video::IGPUMeshBuffer* mb = Mesh->getMeshBuffer(i);
				if (!mb||(mb->getIndexCount()<1 && !mb->isIndexCountGivenByXFormFeedback()))
                    continue;

				video::IMaterialRenderer* rnd =
					driver->getMaterialRenderer(Materials[i].MaterialType);

				if (rnd && rnd->isTransparent())
					++transparentCount;
				else
					++solidCount;

				if (solidCount && transparentCount)
					break;
			}
		}

		// register according to material types counted

		if (solidCount)
			SceneManager->registerNodeForRendering(this, scene::ESNRP_SOLID);

		if (transparentCount)
			SceneManager->registerNodeForRendering(this, scene::ESNRP_TRANSPARENT);

		ISceneNode::OnRegisterSceneNode();
	}
}


//! renders the node.
void CMeshSceneNode::render()
{
	video::IVideoDriver* driver = SceneManager->getVideoDriver();

	if (!Mesh || !driver)
		return;

	bool isTransparentPass =
		SceneManager->getSceneNodeRenderPass() == scene::ESNRP_TRANSPARENT;

	++PassCount;

	if (canProceedPastFence())
    {
        driver->setTransform(video::E4X3TS_WORLD, AbsoluteTransformation);

        for (uint32_t i=0; i<Mesh->getMeshBufferCount(); ++i)
        {
            video::IGPUMeshBuffer* mb = Mesh->getMeshBuffer(i);
            if (mb)
            {
                const video::SGPUMaterial& material = ReferencingMeshMaterials ? mb->getMaterial() : Materials[i];

                video::IMaterialRenderer* rnd = driver->getMaterialRenderer(material.MaterialType);
                bool transparent = (rnd && rnd->isTransparent());

                // only render transparent buffer if this is the transparent render pass
                // and solid only in solid pass
                if (transparent == isTransparentPass)
                {
                    driver->setMaterial(material);
                    driver->drawMeshBuffer(mb);
                }
            }
        }
    }

	// for debug purposes only:
	if (DebugDataVisible && PassCount==1)
	{
        driver->setTransform(video::E4X3TS_WORLD, AbsoluteTransformation);

		video::SGPUMaterial m;
		driver->setMaterial(m);

		// show mesh
		if (DebugDataVisible & scene::EDS_MESH_WIRE_OVERLAY)
		{
			m.Wireframe = true;
			driver->setMaterial(m);

			for (uint32_t g=0; g<Mesh->getMeshBufferCount(); ++g)
			{
				driver->drawMeshBuffer(Mesh->getMeshBuffer(g));
			}
		}
	}
}



//! returns the axis aligned bounding box of this node
const core::aabbox3d<float>& CMeshSceneNode::getBoundingBox()
{
	return Mesh ? Mesh->getBoundingBox() : Box;
}


//! returns the material based on the zero based index i. To get the amount
//! of materials used by this scene node, use getMaterialCount().
//! This function is needed for inserting the node into the scene hierarchy on a
//! optimal position for minimizing renderstate changes, but can also be used
//! to directly modify the material of a scene node.
video::SGPUMaterial& CMeshSceneNode::getMaterial(uint32_t i)
{
	if (Mesh && ReferencingMeshMaterials && i<Mesh->getMeshBufferCount())
		return Mesh->getMeshBuffer(i)->getMaterial();

	if (i >= Materials.size())
		return ISceneNode::getMaterial(i);

	return Materials[i];
}


//! returns amount of materials used by this scene node.
uint32_t CMeshSceneNode::getMaterialCount() const
{
	if (Mesh && ReferencingMeshMaterials)
		return Mesh->getMeshBufferCount();

	return Materials.size();
}


//! Sets a new mesh
void CMeshSceneNode::setMesh(video::IGPUMesh* mesh)
{
	if (mesh)
	{
		mesh->grab();
		if (Mesh)
			Mesh->drop();

		Mesh = mesh;
		copyMaterials();
	}
}



void CMeshSceneNode::copyMaterials()
{
	Materials.clear();

	if (Mesh)
	{
		video::SGPUMaterial mat;

		for (uint32_t i=0; i<Mesh->getMeshBufferCount(); ++i)
		{
            video::IGPUMeshBuffer* mb = Mesh->getMeshBuffer(i);
			if (mb)
				mat = mb->getMaterial();

			Materials.push_back(mat);
		}
	}
}



//! Sets if the scene node should not copy the materials of the mesh but use them in a read only style.
/* In this way it is possible to change the materials a mesh causing all mesh scene nodes
referencing this mesh to change too. */
void CMeshSceneNode::setReferencingMeshMaterials(const bool &referencing)
{
	ReferencingMeshMaterials = referencing;
}


//! Returns if the scene node should not copy the materials of the mesh but use them in a read only style
bool CMeshSceneNode::isReferencingeMeshMaterials() const
{
	return ReferencingMeshMaterials;
}


//! Creates a clone of this scene node and its children.
ISceneNode* CMeshSceneNode::clone(IDummyTransformationSceneNode* newParent, ISceneManager* newManager)
{
	if (!newParent)
		newParent = Parent;
	if (!newManager)
		newManager = SceneManager;

	CMeshSceneNode* nb = new CMeshSceneNode(Mesh, newParent,
		newManager, ID, RelativeTranslation, RelativeRotation, RelativeScale);

	nb->cloneMembers(this, newManager);
	nb->ReferencingMeshMaterials = ReferencingMeshMaterials;
	nb->Materials = Materials;

	if (newParent)
		nb->drop();
	return nb;
}


} // end namespace scene
} // end namespace irr

