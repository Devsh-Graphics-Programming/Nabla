// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CSkinnedMeshSceneNode.h"
#include "CSkinnedMesh.h"
#include "IMaterialRenderer.h"
#include "IMesh.h"
#include "ISceneManager.h"

namespace irr
{
namespace scene
{



//! Get CurrentFrameNr and update transiting settings
void CSkinnedMeshSceneNode::buildFrameNr(const uint32_t& deltaTimeMs)
{
	if ((StartFrame==EndFrame))
	{
		CurrentFrameNr = StartFrame; //Support for non animated meshes
	}
	else if (Looping)
	{
		// play animation looped
		CurrentFrameNr += float(deltaTimeMs) * FramesPerSecond;

		// We have no interpolation between EndFrame and StartFrame,
		// the last frame must be identical to first one with our current solution.
		if (FramesPerSecond > 0.f) //forwards...
		{
			if (CurrentFrameNr > EndFrame)
				CurrentFrameNr = StartFrame + fmod(CurrentFrameNr - StartFrame, EndFrame-StartFrame);
		}
		else //backwards...
		{
			if (CurrentFrameNr < StartFrame)
				CurrentFrameNr = EndFrame - fmod(EndFrame - CurrentFrameNr, EndFrame-StartFrame);
		}
	}
	else
	{
		// play animation non looped

		CurrentFrameNr += float(deltaTimeMs) * FramesPerSecond;
		if (FramesPerSecond > 0.f) //forwards...
		{
			if (CurrentFrameNr > EndFrame)
			{
				CurrentFrameNr = EndFrame;
				if (LoopCallBack)
					LoopCallBack->OnAnimationEnd(this);
			}
		}
		else //backwards...
		{
			if (CurrentFrameNr < StartFrame)
			{
				CurrentFrameNr = StartFrame;
				if (LoopCallBack)
					LoopCallBack->OnAnimationEnd(this);
			}
		}
	}
}


void CSkinnedMeshSceneNode::OnRegisterSceneNode()
{
	if (IsVisible)
	{
	    if (!mesh)
            return;

		// because this node supports rendering of mixed mode meshes consisting of
		// transparent and solid material at the same time, we need to go through all
		// materials, check of what type they are and register this node for the right
		// render pass according to that.

		video::IVideoDriver* driver = SceneManager->getVideoDriver();

		PassCount = 0;
		int transparentCount = 0;
		int solidCount = 0;

        // count copied materials
        for (uint32_t i=0; i<Materials.size(); ++i)
        {
            scene::IGPUMeshBuffer* mb = mesh->getMeshBuffer(i);
            if (!mb||mb->getIndexCount()<1)
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

		// register according to material types counted

		if (solidCount)
			SceneManager->registerNodeForRendering(this, scene::ESNRP_SOLID);

		if (transparentCount)
			SceneManager->registerNodeForRendering(this, scene::ESNRP_TRANSPARENT);

		ISceneNode::OnRegisterSceneNode();
	}
}



//! OnAnimate() is called just before rendering the whole scene.
void CSkinnedMeshSceneNode::OnAnimate(uint32_t timeMs)
{
	if (LastTimeMs==0)	// first frame
	{
		LastTimeMs = timeMs;
        buildFrameNr(0);
	}
	else
    {
        // set CurrentFrameNr
        uint32_t deltaTime = timeMs-LastTimeMs;
        if (deltaTime>=uint32_t(desiredUpdateFrequency))
        {
            buildFrameNr(deltaTime);
            LastTimeMs = timeMs;
        }
    }

	updateAbsolutePosition();

	if (boneStateManager->getBoneUpdateMode()==ISkinningStateManager::EBUM_CONTROL)
    {
        ISceneNode::OnAnimate(timeMs);
        boneStateManager->performBoning();
    }
    else
    {
        boneStateManager->setFrame(getFrameNr(),0);
        boneStateManager->performBoning();
        ISceneNode::OnAnimate(timeMs);
    }
}


//! renders the node.
void CSkinnedMeshSceneNode::render()
{
	video::IVideoDriver* driver = SceneManager->getVideoDriver();

	if (!mesh || !driver)
		return;


	bool isTransparentPass =
		SceneManager->getSceneNodeRenderPass() == scene::ESNRP_TRANSPARENT;

	++PassCount;



    if (canProceedPastFence())
    {
        driver->setTransform(video::E4X3TS_WORLD, AbsoluteTransformation);

        // render original meshes
        for (uint32_t i=0; i<mesh->getMeshBufferCount(); ++i)
        {
            scene::IGPUMeshBuffer* mb = mesh->getMeshBuffer(i);
            if (mb)
            {
                const video::SMaterial& material = Materials[i];

                video::IMaterialRenderer* rnd = driver->getMaterialRenderer(material.MaterialType);
                bool transparent = (rnd && rnd->isTransparent());

                // only render transparent buffer if this is the transparent render pass
                // and solid only in solid pass
                if (transparent == isTransparentPass)
                {
                    driver->setMaterial(material);
                    driver->drawMeshBuffer(mb,(AutomaticCullingState & scene::EAC_COND_RENDER) ? query:NULL);
                }
            }
        }
    }

	// for debug purposes only:
	if (DebugDataVisible && PassCount==1)
	{
        driver->setTransform(video::E4X3TS_WORLD, AbsoluteTransformation);

		video::SMaterial debug_mat;
        debug_mat.Thickness = 3.f;
		driver->setMaterial(debug_mat);

		if (DebugDataVisible & scene::EDS_BBOX)
			driver->draw3DBox(Box, video::SColor(255,255,255,255));

		// show bounding box
		if (DebugDataVisible & scene::EDS_BBOX_BUFFERS)
		{
			for (uint32_t g=0; g< mesh->getMeshBufferCount(); ++g)
			{
				const IGPUMeshBuffer* mb = mesh->getMeshBuffer(g);

				driver->draw3DBox(mb->getBoundingBox(), video::SColor(255,190,128,128));
			}
		}
/**
		// show skeleton
		if (DebugDataVisible & scene::EDS_SKELETON)
		{
			if (mesh->getMeshType() == EMT_ANIMATED_SKINNED)
			{
				// draw skeleton

				for (uint32_t g=0; g < static_cast<ICPUSkinnedMesh*>(mesh)->getAllJoints().size(); ++g)
				{
					ICPUSkinnedMesh::SJoint *joint = static_cast<ICPUSkinnedMesh*>(Mesh)->getAllJoints()[g];

					driver->setTransform(video::E4X3TS_WORLD, concatenateBFollowedByA(AbsoluteTransformation,concatenateBFollowedByA(joint->GlobalAnimatedMatrix,joint->GlobalInversedMatrix)));
                    driver->draw3DBox(joint->bbox, video::SColor(255,51,66,255));
				}
			}
		}

		// show mesh
		if (DebugDataVisible & scene::EDS_MESH_WIRE_OVERLAY)
		{
			debug_mat.Wireframe = true;
			debug_mat.ZBuffer = video::ECFN_NEVER;
			driver->setMaterial(debug_mat);

			for (uint32_t g=0; g<mesh->getMeshBufferCount(); ++g)
			{
				IGPUMeshBuffer* mb = mesh->getMeshBuffer(g);
				driver->setTransform(video::E4X3TS_WORLD, AbsoluteTransformation);
				driver->drawMeshBuffer(mb,(AutomaticCullingState & scene::EAC_COND_RENDER) ? query:NULL);
			}
		}
**/
	}
}


//! Sets a new mesh
void CSkinnedMeshSceneNode::setMesh(IGPUSkinnedMesh* inMesh, const ISkinningStateManager::E_BONE_UPDATE_MODE& boneControl)
{
    if (mesh)
        mesh->drop();
    if (boneStateManager)
        boneStateManager->drop();

    if (!inMesh || !inMesh->getBoneReferenceHierarchy())
    {
        mesh = NULL;
        boneStateManager = NULL;
        return;
    }

    inMesh->grab();
    mesh = inMesh;
    boneStateManager = new CSkinningStateManager(boneControl, SceneManager->getVideoDriver(), mesh->getBoneReferenceHierarchy());

    uint32_t ID = boneStateManager->addInstance(this);
    assert(ID==0);// not instanced so this will always be true!

    setFrameLoop(mesh->getBoneReferenceHierarchy()->getKeys()[0], mesh->getBoneReferenceHierarchy()->getKeys()[mesh->getBoneReferenceHierarchy()->getKeyFrameCount()-1]);
    boneStateManager->performBoning();


    Materials.clear();
    Materials.resize(mesh->getMeshBufferCount());

    for (uint32_t i=0; i<mesh->getMeshBufferCount(); ++i)
    {
        IGPUMeshBuffer* mb = mesh->getMeshBuffer(i);
        if (mb)
            Materials[i] = mb->getMaterial();
        else
            Materials[i] = video::SMaterial();
    }
}


//! sets the frames between the animation is looped.
//! the default is 0 - MaximalFrameCount of the mesh.
bool CSkinnedMeshSceneNode::setFrameLoop(const float& begin, const float& end)
{
	const float maxFrameCount = mesh->getLastFrame();
	if (end < begin)
	{
		StartFrame = core::s32_clamp(end, mesh->getFirstFrame(), maxFrameCount);
		EndFrame = core::s32_clamp(begin, StartFrame, maxFrameCount);
	}
	else
	{
		StartFrame = core::s32_clamp(begin, mesh->getFirstFrame(), maxFrameCount);
		EndFrame = core::s32_clamp(end, StartFrame, maxFrameCount);
	}
	if (FramesPerSecond < 0)
		setCurrentFrame((float)EndFrame);
	else
		setCurrentFrame((float)StartFrame);

	return true;
}



void CSkinnedMeshSceneNode::setAnimationEndCallback(IAnimationEndCallBack<ISkinnedMeshSceneNode>* callback)
{
	if (callback == LoopCallBack)
		return;

	if (LoopCallBack)
		LoopCallBack->drop();

	LoopCallBack = callback;

	if (LoopCallBack)
		LoopCallBack->grab();
}






} // end namespace scene
} // end namespace irr

