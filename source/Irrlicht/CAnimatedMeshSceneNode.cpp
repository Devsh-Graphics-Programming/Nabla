// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CAnimatedMeshSceneNode.h"
#include "IVideoDriver.h"
#include "ISceneManager.h"
#include "os.h"
#include "CSkinnedMesh.h"
#include "IDummyTransformationSceneNode.h"
#include "IBoneSceneNode.h"
#include "IMaterialRenderer.h"
#include "IMesh.h"
#include "IMeshCache.h"
#include "IAnimatedMesh.h"
#include "quaternion.h"


namespace irr
{
namespace scene
{


//! constructor
CAnimatedMeshSceneNode::CAnimatedMeshSceneNode(ICPUAnimatedMesh* mesh,
		ISceneNode* parent, ISceneManager* mgr, s32 id,
		const core::vector3df& position,
		const core::vector3df& rotation,
		const core::vector3df& scale)
: IAnimatedMeshSceneNode(parent, mgr, id, position, rotation, scale), Mesh(0),
	StartFrame(0), EndFrame(0), FramesPerSecond(0.025f),
	CurrentFrameNr(0.f), LastTimeMs(0),
	TransitionTime(0), Transiting(0.f), TransitingBlend(0.f),
	JointMode(EJUOR_NONE), JointsUsed(false),
	Looping(true), RenderFromIdentity(false),
	LoopCallBack(0), PassCount(0)
{
	#ifdef _DEBUG
	setDebugName("CAnimatedMeshSceneNode");
	#endif

	setMesh(mesh);
}


//! destructor
CAnimatedMeshSceneNode::~CAnimatedMeshSceneNode()
{
	if (Mesh)
		Mesh->drop();

	if (LoopCallBack)
		LoopCallBack->drop();
}


//! Sets the current frame. From now on the animation is played from this frame.
void CAnimatedMeshSceneNode::setCurrentFrame(f32 frame)
{
	// if you pass an out of range value, we just clamp it
	CurrentFrameNr = core::clamp ( frame, (f32)StartFrame, (f32)EndFrame );

	beginTransition(); //transit to this frame if enabled
}


//! Returns the currently displayed frame number.
f32 CAnimatedMeshSceneNode::getFrameNr() const
{
	return CurrentFrameNr;
}


//! Get CurrentFrameNr and update transiting settings
void CAnimatedMeshSceneNode::buildFrameNr(u32 timeMs)
{
	if (Transiting!=0.f)
	{
		TransitingBlend += (f32)(timeMs) * Transiting;
		if (TransitingBlend > 1.f)
		{
			Transiting=0.f;
			TransitingBlend=0.f;
		}
	}

	if ((StartFrame==EndFrame))
	{
		CurrentFrameNr = (f32)StartFrame; //Support for non animated meshes
	}
	else if (Looping)
	{
		// play animation looped
		CurrentFrameNr += timeMs * FramesPerSecond;

		// We have no interpolation between EndFrame and StartFrame,
		// the last frame must be identical to first one with our current solution.
		if (FramesPerSecond > 0.f) //forwards...
		{
			if (CurrentFrameNr > EndFrame)
				CurrentFrameNr = StartFrame + fmod(CurrentFrameNr - StartFrame, (f32)(EndFrame-StartFrame));
		}
		else //backwards...
		{
			if (CurrentFrameNr < StartFrame)
				CurrentFrameNr = EndFrame - fmod(EndFrame - CurrentFrameNr, (f32)(EndFrame-StartFrame));
		}
	}
	else
	{
		// play animation non looped

		CurrentFrameNr += timeMs * FramesPerSecond;
		if (FramesPerSecond > 0.f) //forwards...
		{
			if (CurrentFrameNr > (f32)EndFrame)
			{
				CurrentFrameNr = (f32)EndFrame;
				if (LoopCallBack)
					LoopCallBack->OnAnimationEnd(this);
			}
		}
		else //backwards...
		{
			if (CurrentFrameNr < (f32)StartFrame)
			{
				CurrentFrameNr = (f32)StartFrame;
				if (LoopCallBack)
					LoopCallBack->OnAnimationEnd(this);
			}
		}
	}
}


void CAnimatedMeshSceneNode::OnRegisterSceneNode()
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
		ICPUMesh* meshForCurrentFrame = getMeshForCurrentFrame();/*
		if (ReferencingMeshMaterials && meshForCurrentFrame)
		{
			// count mesh materials

			for (u32 i=0; i<meshForCurrentFrame->getMeshBufferCount(); ++i)
			{
				scene::ICPUMeshBuffer* mb = meshForCurrentFrame->getMeshBuffer(i);
				if (!mb||mb->getIndexCount()<1)
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
		else */if (meshForCurrentFrame)
		{
			// count copied materials

			for (u32 i=0; i<Materials.size(); ++i)
			{
				scene::ICPUMeshBuffer* mb = Mesh->getMeshBuffer(i);
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
		}

		// register according to material types counted

		if (solidCount)
			SceneManager->registerNodeForRendering(this, scene::ESNRP_SOLID);

		if (transparentCount)
			SceneManager->registerNodeForRendering(this, scene::ESNRP_TRANSPARENT);

		ISceneNode::OnRegisterSceneNode();
	}
}

ICPUMesh * CAnimatedMeshSceneNode::getMeshForCurrentFrame()
{
    if (!Mesh)
        return NULL;

	if(Mesh->getMeshType() != EMT_ANIMATED_SKINNED)
	{
		s32 frameNr = (s32) getFrameNr();
		s32 frameBlend = (s32) (core::fract ( getFrameNr() ) * 1000.f);
		return Mesh->getMesh(frameNr, frameBlend);
	}
	else
	{
#ifndef _IRR_COMPILE_WITH_SKINNED_MESH_SUPPORT_
		return 0;
#else

		// As multiple scene nodes may be sharing the same skinned mesh, we have to
		// re-animate it every frame to ensure that this node gets the mesh that it needs.

		CCPUSkinnedMesh* skinnedMesh = reinterpret_cast<CCPUSkinnedMesh*>(Mesh);

		if (JointMode == EJUOR_CONTROL)//write to mesh
			skinnedMesh->transferJointsToMesh(JointChildSceneNodes);
		else
			skinnedMesh->animateMesh(getFrameNr(), 1.0f);

		// Update the skinned mesh for the current joint transforms.
		skinnedMesh->skinMesh();

		if (JointMode == EJUOR_READ)//read from mesh
		{
			skinnedMesh->recoverJointsFromMesh(JointChildSceneNodes);

			//---slow---
			for (u32 n=0;n<JointChildSceneNodes.size();++n)
				if (JointChildSceneNodes[n]->getParent()==this)
				{
					JointChildSceneNodes[n]->updateAbsolutePositionOfAllChildren(); //temp, should be an option
				}
		}

		if(JointMode == EJUOR_CONTROL)
		{
			// For meshes other than EJUOR_CONTROL, this is done by calling animateMesh()
			skinnedMesh->updateBoundingBox();
		}

		return skinnedMesh;
#endif
	}
}


//! OnAnimate() is called just before rendering the whole scene.
void CAnimatedMeshSceneNode::OnAnimate(u32 timeMs)
{
	if (LastTimeMs==0)	// first frame
	{
		LastTimeMs = timeMs;
	}

	// set CurrentFrameNr
	buildFrameNr(timeMs-LastTimeMs);

	// update bbox
	if (Mesh)
	{
		scene::ICPUMesh * mesh = getMeshForCurrentFrame();

		if (mesh)
			Box = mesh->getBoundingBox();
	}
	LastTimeMs = timeMs;

	IAnimatedMeshSceneNode::OnAnimate(timeMs);
}


//! renders the node.
void CAnimatedMeshSceneNode::render()
{
	video::IVideoDriver* driver = SceneManager->getVideoDriver();

	if (!Mesh || !driver)
		return;


	bool isTransparentPass =
		SceneManager->getSceneNodeRenderPass() == scene::ESNRP_TRANSPARENT;

	++PassCount;

	scene::ICPUMesh* m = getMeshForCurrentFrame();

	if(m)
	{
		Box = m->getBoundingBox();
	}
	else
	{
		#ifdef _DEBUG
			os::Printer::log("Animated Mesh returned no mesh to render.", Mesh->getDebugName(), ELL_WARNING);
		#endif
		return;
	}

	driver->setTransform(video::ETS_WORLD, AbsoluteTransformation);

	// for debug purposes only:

	bool tformChanged = false;
	// render original meshes
    for (u32 i=0; i<m->getMeshBufferCount(); ++i)
    {
        scene::ICPUMeshBuffer* mb = m->getMeshBuffer(i);
        if (mb)
        {
            const video::SMaterial& material = /*ReferencingMeshMaterials ? mb->getMaterial() : */Materials[i];

            video::IMaterialRenderer* rnd = driver->getMaterialRenderer(material.MaterialType);
            bool transparent = (rnd && rnd->isTransparent());

            // only render transparent buffer if this is the transparent render pass
            // and solid only in solid pass
            if (transparent == isTransparentPass)
            {
                if (RenderFromIdentity)
                {
                    tformChanged = true;
                    driver->setTransform(video::ETS_WORLD, core::IdentityMatrix );
                }
                else if (Mesh->getMeshType() == EMT_ANIMATED_SKINNED)
                {
                    tformChanged = true;
                    driver->setTransform(video::ETS_WORLD, AbsoluteTransformation * static_cast<SCPUSkinMeshBuffer*>(mb)->Transformation);
                }

                driver->setMaterial(material);
                driver->drawMeshBuffer(mb,(AutomaticCullingState & scene::EAC_COND_RENDER) ? query:NULL);
            }
        }
    }

    if (tformChanged)
        driver->setTransform(video::ETS_WORLD, AbsoluteTransformation);

	// for debug purposes only:
	if (DebugDataVisible && PassCount==1)
	{
		video::SMaterial debug_mat;
		debug_mat.AntiAliasing=0;

		debug_mat.ZBuffer = video::ECFN_NEVER;
		driver->setMaterial(debug_mat);

		if (DebugDataVisible & scene::EDS_BBOX)
			driver->draw3DBox(Box, video::SColor(255,255,255,255));

		// show bounding box
		if (DebugDataVisible & scene::EDS_BBOX_BUFFERS)
		{
			for (u32 g=0; g< m->getMeshBufferCount(); ++g)
			{
				const ICPUMeshBuffer* mb = m->getMeshBuffer(g);

				if (Mesh->getMeshType() == EMT_ANIMATED_SKINNED)
					driver->setTransform(video::ETS_WORLD, AbsoluteTransformation * static_cast<const SCPUSkinMeshBuffer*>(mb)->Transformation);
				driver->draw3DBox(mb->getBoundingBox(), video::SColor(255,190,128,128));
			}
		}

		// show skeleton
		if (DebugDataVisible & scene::EDS_SKELETON)
		{
			if (Mesh->getMeshType() == EMT_ANIMATED_SKINNED)
			{
				// draw skeleton

				for (u32 g=0; g < static_cast<ICPUSkinnedMesh*>(Mesh)->getAllJoints().size(); ++g)
				{
					ICPUSkinnedMesh::SJoint *joint = static_cast<ICPUSkinnedMesh*>(Mesh)->getAllJoints()[g];

					for (u32 n=0;n<joint->Children.size();++n)
					{
					    //! This might be fucked up by state pollution in transformations
						driver->draw3DLine(joint->GlobalAnimatedMatrix.getTranslation(),
								joint->Children[n]->GlobalAnimatedMatrix.getTranslation(),
								video::SColor(255,51,66,255));
					}
				}
			}
		}

		// show mesh
		if (DebugDataVisible & scene::EDS_MESH_WIRE_OVERLAY)
		{
			debug_mat.Wireframe = true;
			debug_mat.ZBuffer = video::ECFN_NEVER;
			driver->setMaterial(debug_mat);

			for (u32 g=0; g<m->getMeshBufferCount(); ++g)
			{
				ICPUMeshBuffer* mb = m->getMeshBuffer(g);
				if (RenderFromIdentity)
					driver->setTransform(video::ETS_WORLD, core::IdentityMatrix );
				else if (Mesh->getMeshType() == EMT_ANIMATED_SKINNED)
					driver->setTransform(video::ETS_WORLD, AbsoluteTransformation * static_cast<SCPUSkinMeshBuffer*>(mb)->Transformation);
				driver->drawMeshBuffer(mb,(AutomaticCullingState & scene::EAC_COND_RENDER) ? query:NULL);
			}
		}
	}
}


//! Returns the current start frame number.
s32 CAnimatedMeshSceneNode::getStartFrame() const
{
	return StartFrame;
}


//! Returns the current start frame number.
s32 CAnimatedMeshSceneNode::getEndFrame() const
{
	return EndFrame;
}


//! sets the frames between the animation is looped.
//! the default is 0 - MaximalFrameCount of the mesh.
bool CAnimatedMeshSceneNode::setFrameLoop(s32 begin, s32 end)
{
	const s32 maxFrameCount = Mesh->getFrameCount() - 1;
	if (end < begin)
	{
		StartFrame = core::s32_clamp(end, 0, maxFrameCount);
		EndFrame = core::s32_clamp(begin, StartFrame, maxFrameCount);
	}
	else
	{
		StartFrame = core::s32_clamp(begin, 0, maxFrameCount);
		EndFrame = core::s32_clamp(end, StartFrame, maxFrameCount);
	}
	if (FramesPerSecond < 0)
		setCurrentFrame((f32)EndFrame);
	else
		setCurrentFrame((f32)StartFrame);

	return true;
}


//! sets the speed with witch the animation is played
void CAnimatedMeshSceneNode::setAnimationSpeed(f32 framesPerSecond)
{
	FramesPerSecond = framesPerSecond * 0.001f;
}


f32 CAnimatedMeshSceneNode::getAnimationSpeed() const
{
	return FramesPerSecond * 1000.f;
}


//! returns the axis aligned bounding box of this node
const core::aabbox3d<f32>& CAnimatedMeshSceneNode::getBoundingBox() const
{
	return Box;
}


//! returns the material based on the zero based index i. To get the amount
//! of materials used by this scene node, use getMaterialCount().
//! This function is needed for inserting the node into the scene hirachy on a
//! optimal position for minimizing renderstate changes, but can also be used
//! to directly modify the material of a scene node.
video::SMaterial& CAnimatedMeshSceneNode::getMaterial(u32 i)
{/*
    ICPUMesh* meshForCurrentFrame = NULL;
    if (ReferencingMeshMaterials)
        meshForCurrentFrame = getMeshForCurrentFrame();

	if (meshForCurrentFrame && i<meshForCurrentFrame->getMeshBufferCount())
		return meshForCurrentFrame->getMeshBuffer(i)->getMaterial();
*/
	if (i >= Materials.size())
		return ISceneNode::getMaterial(i);

	return Materials[i];
}



//! returns amount of materials used by this scene node.
u32 CAnimatedMeshSceneNode::getMaterialCount() const
{/*
    ICPUMesh* meshForCurrentFrame = NULL;
    if (ReferencingMeshMaterials)
        meshForCurrentFrame = getMeshForCurrentFrame();

    if (meshForCurrentFrame)
        return meshForCurrentFrame->getMeshBufferCount();
    else*/
        return Materials.size();
}


//! Returns a pointer to a child node, which has the same transformation as
//! the corresponding joint, if the mesh in this scene node is a skinned mesh.
IBoneSceneNode* CAnimatedMeshSceneNode::getJointNode(const c8* jointName)
{
#ifndef _IRR_COMPILE_WITH_SKINNED_MESH_SUPPORT_
	os::Printer::log("Compiled without _IRR_COMPILE_WITH_SKINNED_MESH_SUPPORT_", ELL_WARNING);
	return 0;
#else

	if (!Mesh || Mesh->getMeshType() != EMT_ANIMATED_SKINNED)
	{
		os::Printer::log("No mesh, or mesh not of skinned mesh type", ELL_WARNING);
		return 0;
	}

	checkJoints();

	ICPUSkinnedMesh *skinnedMesh = static_cast<ICPUSkinnedMesh*>(Mesh);

	const s32 number = skinnedMesh->getJointNumber(jointName);

	if (number == -1)
	{
		os::Printer::log("Joint with specified name not found in skinned mesh", jointName, ELL_DEBUG);
		return 0;
	}

	if ((s32)JointChildSceneNodes.size() <= number)
	{
		os::Printer::log("Joint was found in mesh, but is not loaded into node", jointName, ELL_WARNING);
		return 0;
	}

	return JointChildSceneNodes[number];
#endif
}



//! Returns a pointer to a child node, which has the same transformation as
//! the corresponding joint, if the mesh in this scene node is a skinned mesh.
IBoneSceneNode* CAnimatedMeshSceneNode::getJointNode(u32 jointID)
{
#ifndef _IRR_COMPILE_WITH_SKINNED_MESH_SUPPORT_
	os::Printer::log("Compiled without _IRR_COMPILE_WITH_SKINNED_MESH_SUPPORT_", ELL_WARNING);
	return 0;
#else

	if (!Mesh || Mesh->getMeshType() != EMT_ANIMATED_SKINNED)
	{
		os::Printer::log("No mesh, or mesh not of skinned mesh type", ELL_WARNING);
		return 0;
	}

	checkJoints();

	if (JointChildSceneNodes.size() <= jointID)
	{
		os::Printer::log("Joint not loaded into node", ELL_WARNING);
		return 0;
	}

	return JointChildSceneNodes[jointID];
#endif
}

//! Gets joint count.
u32 CAnimatedMeshSceneNode::getJointCount() const
{
#ifndef _IRR_COMPILE_WITH_SKINNED_MESH_SUPPORT_
	return 0;
#else

	if (!Mesh || Mesh->getMeshType() != EMT_ANIMATED_SKINNED)
		return 0;

	ICPUSkinnedMesh *skinnedMesh = static_cast<ICPUSkinnedMesh*>(Mesh);

	return skinnedMesh->getJointCount();
#endif
}

//! Removes a child from this scene node.
//! Implemented here, to be able to remove the shadow properly, if there is one,
//! or to remove attached childs.
bool CAnimatedMeshSceneNode::removeChild(ISceneNode* child)
{
	if (ISceneNode::removeChild(child))
	{
		if (JointsUsed) //stop weird bugs caused while changing parents as the joints are being created
		{
			for (u32 i=0; i<JointChildSceneNodes.size(); ++i)
			{
				if (JointChildSceneNodes[i] == child)
				{
					JointChildSceneNodes[i] = 0; //remove link to child
					break;
				}
			}
		}
		return true;
	}

	return false;
}


//! Sets looping mode which is on by default. If set to false,
//! animations will not be looped.
void CAnimatedMeshSceneNode::setLoopMode(bool playAnimationLooped)
{
	Looping = playAnimationLooped;
}

//! returns the current loop mode
bool CAnimatedMeshSceneNode::getLoopMode() const
{
	return Looping;
}


//! Sets a callback interface which will be called if an animation
//! playback has ended. Set this to 0 to disable the callback again.
void CAnimatedMeshSceneNode::setAnimationEndCallback(IAnimationEndCallBack* callback)
{
	if (callback == LoopCallBack)
		return;

	if (LoopCallBack)
		LoopCallBack->drop();

	LoopCallBack = callback;

	if (LoopCallBack)
		LoopCallBack->grab();
}


//! Sets a new mesh
void CAnimatedMeshSceneNode::setMesh(ICPUAnimatedMesh* mesh)
{
	if (!mesh)
		return; // won't set null mesh

	if (Mesh != mesh)
	{
		if (Mesh)
			Mesh->drop();

		Mesh = mesh;

		// grab the mesh (it's non-null!)
		Mesh->grab();
	}

	// get materials and bounding box
	Box = Mesh->getBoundingBox();

	ICPUMesh* m = Mesh->getMesh(0,0);
	if (m)
	{
		Materials.clear();
		Materials.reallocate(m->getMeshBufferCount());

		for (u32 i=0; i<m->getMeshBufferCount(); ++i)
		{
			ICPUMeshBuffer* mb = m->getMeshBuffer(i);
			if (mb)
				Materials.push_back(mb->getMaterial());
			else
				Materials.push_back(video::SMaterial());
		}
	}

	// clean up joint nodes
	if (JointsUsed)
	{
		JointsUsed=false;
		checkJoints();
	}

	// get start and begin time
//	setAnimationSpeed(Mesh->getAnimationSpeed());
	setFrameLoop(0, Mesh->getFrameCount());
}


//! Set the joint update mode (0-unused, 1-get joints only, 2-set joints only, 3-move and set)
void CAnimatedMeshSceneNode::setJointMode(E_JOINT_UPDATE_ON_RENDER mode)
{
	checkJoints();
	JointMode=mode;
}

//! Sets the transition time in seconds (note: This needs to enable joints, and setJointmode maybe set to 2)
//! you must call animateJoints(), or the mesh will not animate
void CAnimatedMeshSceneNode::setTransitionTime(f32 time)
{
	const u32 ttime = (u32)core::floor32(time*1000.0f);
	if (TransitionTime==ttime)
		return;
	TransitionTime = ttime;
	if (ttime != 0)
		setJointMode(EJUOR_CONTROL);
	else
		setJointMode(EJUOR_NONE);
}


//! render mesh ignoring its transformation. Used with ragdolls. (culling is unaffected)
void CAnimatedMeshSceneNode::setRenderFromIdentity(bool enable)
{
	RenderFromIdentity=enable;
}


//! updates the joint positions of this mesh
void CAnimatedMeshSceneNode::animateJoints(bool CalculateAbsolutePositions)
{
#ifndef _IRR_COMPILE_WITH_SKINNED_MESH_SUPPORT_
	return;
#else
	if (Mesh && Mesh->getMeshType() == EMT_ANIMATED_SKINNED )
	{
		checkJoints();
		const f32 frame = getFrameNr(); //old?

		CCPUSkinnedMesh* skinnedMesh = reinterpret_cast<CCPUSkinnedMesh*>(Mesh);

		skinnedMesh->transferOnlyJointsHintsToMesh( JointChildSceneNodes );
		skinnedMesh->animateMesh(frame, 1.0f);
		skinnedMesh->recoverJointsFromMesh( JointChildSceneNodes);

		//-----------------------------------------
		//		Transition
		//-----------------------------------------

		if (Transiting != 0.f)
		{
			// Init additional matrices
			if (PretransitingSave.size()<JointChildSceneNodes.size())
			{
				for(u32 n=PretransitingSave.size(); n<JointChildSceneNodes.size(); ++n)
					PretransitingSave.push_back(core::matrix4());
			}

			for (u32 n=0; n<JointChildSceneNodes.size(); ++n)
			{
				//------Position------

				JointChildSceneNodes[n]->setPosition(
						core::lerp(
							PretransitingSave[n].getTranslation(),
							JointChildSceneNodes[n]->getPosition(),
							TransitingBlend));

				//------Rotation------

				//Code is slow, needs to be fixed up

				const core::quaternion RotationStart(PretransitingSave[n].getRotationDegrees()*core::DEGTORAD);
				const core::quaternion RotationEnd(JointChildSceneNodes[n]->getRotation()*core::DEGTORAD);

				core::quaternion QRotation;
				QRotation.slerp(RotationStart, RotationEnd, TransitingBlend);

				core::vector3df tmpVector;
				QRotation.toEuler(tmpVector);
				tmpVector*=core::RADTODEG; //convert from radians back to degrees
				JointChildSceneNodes[n]->setRotation( tmpVector );

				//------Scale------

				//JointChildSceneNodes[n]->setScale(
				//		core::lerp(
				//			PretransitingSave[n].getScale(),
				//			JointChildSceneNodes[n]->getScale(),
				//			TransitingBlend));
			}
		}

		if (CalculateAbsolutePositions)
		{
			//---slow---
			for (u32 n=0;n<JointChildSceneNodes.size();++n)
			{
				if (JointChildSceneNodes[n]->getParent()==this)
				{
					JointChildSceneNodes[n]->updateAbsolutePositionOfAllChildren(); //temp, should be an option
				}
			}
		}
	}
#endif
}

/*!
*/
void CAnimatedMeshSceneNode::checkJoints()
{
#ifndef _IRR_COMPILE_WITH_SKINNED_MESH_SUPPORT_
	return;
#else

	if (!Mesh || Mesh->getMeshType() != EMT_ANIMATED_SKINNED)
		return;

	if (!JointsUsed)
	{
		for (u32 i=0; i<JointChildSceneNodes.size(); ++i)
			removeChild(JointChildSceneNodes[i]);
		JointChildSceneNodes.clear();

		//Create joints for SkinnedMesh
		((CCPUSkinnedMesh*)Mesh)->addJoints(JointChildSceneNodes, this, SceneManager);
		((CCPUSkinnedMesh*)Mesh)->recoverJointsFromMesh(JointChildSceneNodes);

		JointsUsed=true;
		JointMode=EJUOR_READ;
	}
#endif
}

/*!
*/
void CAnimatedMeshSceneNode::beginTransition()
{
	if (!JointsUsed)
		return;

	if (TransitionTime != 0)
	{
		//Check the array is big enough
		if (PretransitingSave.size()<JointChildSceneNodes.size())
		{
			for(u32 n=PretransitingSave.size(); n<JointChildSceneNodes.size(); ++n)
				PretransitingSave.push_back(core::matrix4());
		}

		//Copy the position of joints
		for (u32 n=0;n<JointChildSceneNodes.size();++n)
			PretransitingSave[n]=JointChildSceneNodes[n]->getRelativeTransformation();

		Transiting = core::reciprocal((f32)TransitionTime);
	}
	TransitingBlend = 0.f;
}


/*!
*/
ISceneNode* CAnimatedMeshSceneNode::clone(ISceneNode* newParent, ISceneManager* newManager)
{
	if (!newParent)
		newParent = Parent;
	if (!newManager)
		newManager = SceneManager;

	CAnimatedMeshSceneNode* newNode =
		new CAnimatedMeshSceneNode(Mesh, NULL, newManager, ID, RelativeTranslation,
						 RelativeRotation, RelativeScale);

	if (newParent)
	{
		newNode->setParent(newParent); 	// not in constructor because virtual overload for updateAbsolutePosition won't be called
		newNode->drop();
	}

	newNode->cloneMembers(this, newManager);

	newNode->Materials = Materials;
	newNode->Box = Box;
	newNode->Mesh = Mesh;
	newNode->StartFrame = StartFrame;
	newNode->EndFrame = EndFrame;
	newNode->FramesPerSecond = FramesPerSecond;
	newNode->CurrentFrameNr = CurrentFrameNr;
	newNode->JointMode = JointMode;
	newNode->JointsUsed = JointsUsed;
	newNode->TransitionTime = TransitionTime;
	newNode->Transiting = Transiting;
	newNode->TransitingBlend = TransitingBlend;
	newNode->Looping = Looping;
	//newNode->ReadOnlyMaterials = ReadOnlyMaterials;
	newNode->LoopCallBack = LoopCallBack;
	if (newNode->LoopCallBack)
		newNode->LoopCallBack->grab();
	newNode->PassCount = PassCount;
	newNode->JointChildSceneNodes = JointChildSceneNodes;
	newNode->PretransitingSave = PretransitingSave;
	newNode->RenderFromIdentity = RenderFromIdentity;

	return newNode;
}


} // end namespace scene
} // end namespace irr
