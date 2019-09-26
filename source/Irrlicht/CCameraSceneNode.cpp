// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CCameraSceneNode.h"
#include "ISceneManager.h"
#include "IVideoDriver.h"
#include "os.h"

namespace irr
{
namespace scene
{


//! constructor
CCameraSceneNode::CCameraSceneNode(IDummyTransformationSceneNode* parent, ISceneManager* mgr, int32_t id,
	const core::vector3df& position, const core::vectorSIMDf& lookat)
	: ICameraSceneNode(parent, mgr, id, position),
	Target(lookat), UpVector(0.0f, 1.0f, 0.0f), ZNear(1.0f), ZFar(3000.0f),
	InputReceiverEnabled(true), TargetAndRotationAreBound(false)
{
	#ifdef _IRR_DEBUG
	setDebugName("CCameraSceneNode");
	#endif

	// set default projection
	Fovy = core::PI<float>() / 2.5f;	// Field of view, in radians.

	const video::IVideoDriver* const d = mgr?mgr->getVideoDriver():0;
	if (d)
		Aspect = (float)d->getCurrentRenderTargetSize().Width /
			(float)d->getCurrentRenderTargetSize().Height;
	else
		Aspect = 4.0f / 3.0f;	// Aspect ratio.

	recalculateProjectionMatrix();
	recalculateViewArea();
}


//! Disables or enables the camera to get key or mouse inputs.
void CCameraSceneNode::setInputReceiverEnabled(bool enabled)
{
	InputReceiverEnabled = enabled;
}


//! Returns if the input receiver of the camera is currently enabled.
bool CCameraSceneNode::isInputReceiverEnabled() const
{
	return InputReceiverEnabled;
}


//! Sets the projection matrix of the camera.
void CCameraSceneNode::setProjectionMatrix(const core::matrix4SIMD& projection)
{
	projMatrix = projection;
	concatMatrix = concatenateBFollowedByA(projMatrix,viewMatrix);
}


//! Gets the current projection matrix of the camera
//! \return Returns the current projection matrix of the camera.
const core::matrix4SIMD& CCameraSceneNode::getProjectionMatrix() const
{
	return projMatrix;
}


//! Gets the current view matrix of the camera
//! \return Returns the current view matrix of the camera.
const core::matrix4x3& CCameraSceneNode::getViewMatrix() const
{
	return viewMatrix;
}



//! It is possible to send mouse and key events to the camera. Most cameras
//! may ignore this input, but camera scene nodes which are created for
//! example with scene::ISceneManager::addMayaCameraSceneNode or
//! scene::ISceneManager::addFPSCameraSceneNode, may want to get this input
//! for changing their position, look at target or whatever.
bool CCameraSceneNode::OnEvent(const SEvent& event)
{
	if (!InputReceiverEnabled)
		return false;

	// send events to event receiving animators

	ISceneNodeAnimatorArray::iterator ait = Animators.begin();

	for (; ait != Animators.end(); ++ait)
		if ((*ait)->isEventReceiverEnabled() && (*ait)->OnEvent(event))
			return true;

	// if nobody processed the event, return false
	return false;
}


//! sets the look at target of the camera
//! \param pos: Look at target of the camera.
void CCameraSceneNode::setTarget(const core::vector3df& pos)
{
	Target.set(pos);

	if(TargetAndRotationAreBound)
	{
		const core::vector3df toTarget = Target.getAsVector3df() - getAbsolutePosition();
		ISceneNode::setRotation(toTarget.getHorizontalAngle());
	}
}


//! Sets the rotation of the node.
/** This only modifies the relative rotation of the node.
If the camera's target and rotation are bound ( @see bindTargetAndRotation() )
then calling this will also change the camera's target to match the rotation.
\param rotation New rotation of the node in degrees. */
void CCameraSceneNode::setRotation(const core::vector3df& rotation)
{
	if(TargetAndRotationAreBound)
		Target.set(getAbsolutePosition() + rotation.rotationToDirection());

	ISceneNode::setRotation(rotation);
}


//! Gets the current look at target of the camera
//! \return Returns the current look at target of the camera
const core::vectorSIMDf& CCameraSceneNode::getTarget() const
{
	return Target;
}


//! sets the up vector of the camera
//! \param pos: New upvector of the camera.
void CCameraSceneNode::setUpVector(const core::vector3df& pos)
{
	UpVector.set(pos);
}


//! Gets the up vector of the camera.
//! \return Returns the up vector of the camera.
const core::vectorSIMDf& CCameraSceneNode::getUpVector() const
{
	return UpVector;
}


float CCameraSceneNode::getNearValue() const
{
	return ZNear;
}


float CCameraSceneNode::getFarValue() const
{
	return ZFar;
}


float CCameraSceneNode::getAspectRatio() const
{
	return Aspect;
}


float CCameraSceneNode::getFOV() const
{
	return Fovy;
}


void CCameraSceneNode::setNearValue(float f)
{
	ZNear = f;
	recalculateProjectionMatrix();
}


void CCameraSceneNode::setFarValue(float f)
{
	ZFar = f;
	recalculateProjectionMatrix();
}


void CCameraSceneNode::setAspectRatio(float f)
{
	Aspect = f;
	recalculateProjectionMatrix();
}


void CCameraSceneNode::setFOV(float f)
{
	Fovy = f;
	recalculateProjectionMatrix();
}


void CCameraSceneNode::recalculateProjectionMatrix()
{
	projMatrix = core::matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(Fovy, Aspect, ZNear, ZFar);
	concatMatrix = concatenateBFollowedByA(projMatrix,viewMatrix);
}


//! prerender
void CCameraSceneNode::OnRegisterSceneNode()
{
	if ( SceneManager->getActiveCamera () == this )
		SceneManager->registerNodeForRendering(this, ESNRP_CAMERA);

	ISceneNode::OnRegisterSceneNode();
}


//! render
void CCameraSceneNode::render()
{
	core::vectorSIMDf pos;
	pos.set(getAbsolutePosition());
	core::vectorSIMDf tgtv = core::normalize(Target - pos);

	// if upvector and vector to the target are the same, we have a
	// problem. so solve this problem:
	core::vectorSIMDf up = core::normalize(UpVector);

	core::vectorSIMDf dp = core::dot(tgtv,up);

	if (core::iszero(core::abs(dp)[0]-1.f))
	{
		up.X += 0.5f;
	}

	viewMatrix.buildCameraLookAtMatrixLH(pos.getAsVector3df(), Target.getAsVector3df(), up.getAsVector3df()); // TODO: Change this to static
	concatMatrix = concatenateBFollowedByA(projMatrix,viewMatrix);
	recalculateViewArea();

	video::IVideoDriver* driver = SceneManager->getVideoDriver();
	if ( driver)
	{
		driver->setTransform(video::EPTS_PROJ,projMatrix);
		driver->setTransform(video::E4X3TS_VIEW, viewMatrix );
	}
}


//! returns the axis aligned bounding box of this node
const core::aabbox3d<float>& CCameraSceneNode::getBoundingBox()
{
	return ViewArea.getBoundingBox();
}


//! returns the view frustum. needed sometimes by bsp or lod render nodes.
const SViewFrustum* CCameraSceneNode::getViewFrustum() const
{
	return &ViewArea;
}


void CCameraSceneNode::recalculateViewArea()
{
	ViewArea.setFrom(concatMatrix);
}



//! Set the binding between the camera's rotation adn target.
void CCameraSceneNode::bindTargetAndRotation(bool bound)
{
	TargetAndRotationAreBound = bound;
}


//! Gets the binding between the camera's rotation and target.
bool CCameraSceneNode::getTargetAndRotationBinding(void) const
{
	return TargetAndRotationAreBound;
}


} // end namespace
} // end namespace

