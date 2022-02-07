// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_CAMERA_SCENE_NODE_H_INCLUDED__
#define __NBL_I_CAMERA_SCENE_NODE_H_INCLUDED__

#include "ISceneNode.h"
#include "matrixutil.h"

namespace nbl
{
namespace scene
{
struct SViewFrustum;

//! Scene Node which is a (controlable) camera.
/** The whole scene will be rendered from the cameras point of view.
Because the ICameraScenNode is a SceneNode, it can be attached to any
other scene node, and will follow its parents movement, rotation and so
on.
*/
class ICameraSceneNode : public ISceneNode
{
public:
    //! Constructor
    ICameraSceneNode(IDummyTransformationSceneNode* parent, ISceneManager* mgr, int32_t id,
        const core::vector3df& position = core::vector3df(0, 0, 0),
        const core::vector3df& rotation = core::vector3df(0, 0, 0),
        const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f))
        : ISceneNode(parent, mgr, id, position, rotation, scale),
          Fovy(core::PI<float>() / 2.5f), Aspect(16.f / 9.f),
          ZNear(1.0f), ZFar(3000.0f), leftHanded(true) {}

    //! Sets the projection matrix of the camera.
    /** The matrix class has some methods to build a
		projection matrix. e.g:
		core::matrix4SIMD::buildProjectionMatrixPerspectiveFovLH.
		Note that the matrix will only stay as set by this method until
		one of the following Methods are called: setNearValue,
		setFarValue, setAspectRatio, setFOV.
		The function will figure it out if you've set an orthogonal matrix.
		\param projection The new projection matrix of the camera.
		*/
    virtual void setProjectionMatrix(const core::matrix4SIMD& projection) = 0;

    //! Gets the current projection matrix of the camera.
    /** \return The current projection matrix of the camera. */
    inline const core::matrix4SIMD& getProjectionMatrix() const { return projMatrix; }

    //! Gets the current view matrix of the camera.
    /** \return The current view matrix of the camera. */
    virtual const core::matrix3x4SIMD& getViewMatrix() const = 0;

    virtual const core::matrix4SIMD& getConcatenatedMatrix() const = 0;
#if 0
		//! It is possible to send mouse and key events to the camera.
		/** Most cameras may ignore this input, but camera scene nodes
		which are created for example with
		ISceneManager::addCameraSceneNodeMaya or
		ISceneManager::addCameraSceneNodeFPS, may want to get
		this input for changing their position, look at target or
		whatever. */
		virtual bool OnEvent(const SEvent& event) =0;
#endif
    //! Sets the look at target of the camera
    /** If the camera's target and rotation are bound ( @see
		bindTargetAndRotation() ) then calling this will also change
		the camera's scene node rotation to match the target.
		Note that setTarget uses the current absolute position
		internally, so if you changed setPosition since last rendering you must
		call updateAbsolutePosition before using this function.
		\param pos Look at target of the camera, in world co-ordinates. */
    virtual void setTarget(const core::vector3df& pos) = 0;

    //! Sets the rotation of the node.
    /** This only modifies the relative rotation of the node.
		If the camera's target and rotation are bound ( @see
		bindTargetAndRotation() ) then calling this will also change
		the camera's target to match the rotation.
		\param rotation New rotation of the node in degrees. */
    virtual void setRotation(const core::vector3df& rotation) = 0;

    //! Gets the current look at target of the camera
    /** \return The current look at target of the camera, in world co-ordinates */
    virtual const core::vectorSIMDf& getTarget() const = 0;

    //! Sets the handedness convention for the camera.
    /** \param pos: New upvector of the camera,
		`_leftHanded==true` means Z+ goes into the screen,
		away from the viewer. */
    inline void setLeftHanded(bool _leftHanded = true)
    {
        leftHanded = _leftHanded;
        recomputeProjectionMatrix();
    }

    //! Gets the handedness convention of the camera.
    /** \return Whether the camera is left handed. */
    inline bool getLeftHanded() const { return leftHanded; }

    //! Sets the up vector of the camera.
    /** \param pos: New upvector of the camera. */
    virtual void setUpVector(const core::vectorSIMDf& up) = 0;

    //! Gets the up vector of the camera.
    /** \return The up vector of the camera, in world space. */
    virtual const core::vectorSIMDf& getUpVector() const = 0;

    //! Gets the value of the near plane of the camera.
    /** \return The value of the near plane of the camera. */
    virtual float getNearValue() const { return ZNear; }

    //! Gets the value of the far plane of the camera.
    /** \return The value of the far plane of the camera. */
    virtual float getFarValue() const { return ZFar; }

    //! Gets the aspect ratio of the camera.
    /** \return The aspect ratio of the camera. */
    virtual float getAspectRatio() const { return Aspect; }

    //! Gets the field of view of the camera.
    /** \return The field of view of the camera in radians. */
    inline float getFOV() const { return Fovy; }

    //! Sets the value of the near clipping plane. (default: 1.0f)
    /** \param zn: New z near value. */
    inline void setNearValue(float zn)
    {
        ZNear = zn;
        recomputeProjectionMatrix();
    }

    //! Sets the value of the far clipping plane (default: 2000.0f)
    /** \param zf: New z far value. */
    inline void setFarValue(float zf)
    {
        ZFar = zf;
        recomputeProjectionMatrix();
    }

    //! Sets the aspect ratio (default: 4.0f / 3.0f)
    /** \param aspect: New aspect ratio. */
    inline void setAspectRatio(float aspect)
    {
        Aspect = aspect;
        recomputeProjectionMatrix();
    }

    //! Sets the field of view (Default: PI / 2.5f)
    /** \param fovy: New field of view in radians. */
    inline void setFOV(float fovy)
    {
        Fovy = fovy;
        recomputeProjectionMatrix();
    }

    //! Update the projection matrix from the set FOV, Aspect and Near/Far values
    virtual void recomputeProjectionMatrix() = 0;

    //! Get the view frustum.
    /** Needed sometimes by bspTree or LOD render nodes.
		\return The current view frustum. */
    virtual const SViewFrustum* getViewFrustum() const = 0;

    //! Disables or enables the camera to get key or mouse inputs.
    /** If this is set to true, the camera will respond to key
		inputs otherwise not. */
    virtual void setInputReceiverEnabled(bool enabled) = 0;

    //! Checks if the input receiver of the camera is currently enabled.
    virtual bool isInputReceiverEnabled() const = 0;

    //! Binds the camera scene node's rotation to its target position and vice vera, or unbinds them.
    /** When bound, calling setRotation() will update the camera's
		target position to be along its +Z axis, and likewise calling
		setTarget() will update its rotation so that its +Z axis will
		point at the target point. FPS camera use this binding by
		default; other cameras do not.
		\param bound True to bind the camera's scene node rotation
		and targetting, false to unbind them.
		@see getTargetAndRotationBinding() */
    virtual void bindTargetAndRotation(bool bound) = 0;

    //! Queries if the camera scene node's rotation and its target position are bound together.
    /** @see bindTargetAndRotation() */
    virtual bool getTargetAndRotationBinding(void) const = 0;

protected:
    // cached values
    float Fovy;  // Field of view, in radians.
    float Aspect;  // Aspect ratio.
    float ZNear;  // value of the near view-plane.
    float ZFar;  // Z-value of the far view-plane.

    // actual projection matrix used
    core::matrix4SIMD projMatrix;

    bool leftHanded;
};

}  // end namespace scene
}  // end namespace nbl

#endif
