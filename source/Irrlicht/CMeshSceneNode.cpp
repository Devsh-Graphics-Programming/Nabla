// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CMeshSceneNode.h"
#include "IVideoDriver.h"
#include "ISceneManager.h"
#include "ICameraSceneNode.h"
#include "IAnimatedMesh.h"
#include "IFileSystem.h"

namespace irr
{
namespace scene
{



//! constructor
CMeshSceneNode::CMeshSceneNode(core::smart_refctd_ptr<video::IGPUMesh>&& mesh, IDummyTransformationSceneNode* parent, ISceneManager* mgr, int32_t id,
			const core::vector3df& position, const core::vector3df& rotation,
			const core::vector3df& scale)
: IMeshSceneNode(parent, mgr, id, position, rotation, scale), Mesh(),
	PassCount(0)
{
	#ifdef _IRR_DEBUG
	setDebugName("CMeshSceneNode");
	#endif

	setMesh(std::move(mesh));
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
		if (Mesh)
		{
			// count mesh materials

			for (uint32_t i=0; i<Mesh->getMeshBufferCount(); ++i)
			{
				video::IGPUMeshBuffer* mb = Mesh->getMeshBuffer(i);
				if (!mb || mb->getIndexCount()<1)
                    continue;

#ifndef NEW_SHADERS
				video::IMaterialRenderer* rnd = driver->getMaterialRenderer(0);
				if (rnd && rnd->isTransparent())
					++transparentCount;
				else
#endif
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

#ifndef NEW_SHADERS
	if (canProceedPastFence())
    {
        driver->setTransform(video::E4X3TS_WORLD, core::matrix3x4SIMD().set(AbsoluteTransformation));

        for (uint32_t i=0; i<Mesh->getMeshBufferCount(); ++i)
        {
            video::IGPUMeshBuffer* mb = Mesh->getMeshBuffer(i);
            if (mb)
            {
                const video::SGPUMaterial& material = mb->getMaterial();

                video::IMaterialRenderer* rnd = driver->getMaterialRenderer(0);
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
#endif
}



//! returns the axis aligned bounding box of this node
const core::aabbox3d<float>& CMeshSceneNode::getBoundingBox()
{
	return Mesh ? Mesh->getBoundingBox() : Box;
}


//! Sets a new mesh
void CMeshSceneNode::setMesh(core::smart_refctd_ptr<video::IGPUMesh>&& mesh)
{
	if (!mesh)
		return;
	
	Mesh = mesh;
}


} // end namespace scene
} // end namespace irr

