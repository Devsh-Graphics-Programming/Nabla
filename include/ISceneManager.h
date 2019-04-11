// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_SCENE_MANAGER_H_INCLUDED__
#define __I_SCENE_MANAGER_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/core/Types.h"
#include "irr/core/irrString.h"
#include "path.h"
#include "vector3d.h"
#include "dimension2d.h"
#include "SColor.h"
#include "ESceneNodeTypes.h"
#include "ESceneNodeAnimatorTypes.h"
#include "EMeshWriterEnums.h"
#include "SceneParameters.h"
#include "irr/video/IGPUSkinnedMesh.h"
#include "ISkinnedMeshSceneNode.h"
#include "irr/asset/ICPUMesh.h"

namespace irr
{
	struct SKeyMap;
	struct SEvent;
    class IrrlichtDevice;

namespace io
{
	class IReadFile;
	class IWriteFile;
	class IFileSystem;
} // end namespace io

namespace video
{
	class IVideoDriver;
	class ITexture;
} // end namespace video

namespace scene
{
	//! Enumeration for render passes.
	/** A parameter passed to the registerNodeForRendering() method of the ISceneManager,
	specifying when the node wants to be drawn in relation to the other nodes. */
	enum E_SCENE_NODE_RENDER_PASS
	{
		//! No pass currently active
		ESNRP_NONE =0,

		//! Camera pass. The active view is set up here. The very first pass.
		ESNRP_CAMERA =1,

		//! In this pass, lights are transformed into camera space and added to the driver
		ESNRP_LIGHT =2,

		//! This is used for sky boxes.
		ESNRP_SKY_BOX =4,

		//! All normal objects can use this for registering themselves.
		/** This value will never be returned by
		ISceneManager::getSceneNodeRenderPass(). The scene manager
		will determine by itself if an object is transparent or solid
		and register the object as SNRT_TRANSPARENT or SNRT_SOLD
		automatically if you call registerNodeForRendering with this
		value (which is default). Note that it will register the node
		only as ONE type. If your scene node has both solid and
		transparent material types register it twice (one time as
		SNRT_SOLID, the other time as SNRT_TRANSPARENT) and in the
		render() method call getSceneNodeRenderPass() to find out the
		current render pass and render only the corresponding parts of
		the node. */
		ESNRP_AUTOMATIC =24,

		//! Solid scene nodes or special scene nodes without materials.
		ESNRP_SOLID =8,

		//! Transparent scene nodes, drawn after solid nodes. They are sorted from back to front and drawn in that order.
		ESNRP_TRANSPARENT =16,

		//! Transparent effect scene nodes, drawn after Transparent nodes. They are sorted from back to front and drawn in that order.
		ESNRP_TRANSPARENT_EFFECT =32
	};

	class IAnimatedMeshSceneNode;
	class IBillboardSceneNode;
	class ICameraSceneNode;
	class IDummyTransformationSceneNode;
	class ILightSceneNode;
	class IMeshLoader;
	class IMeshSceneNode;
	class IMeshSceneNodeInstanced;
	class IMeshWriter;
	class ISceneNode;
	class ISceneNodeAnimator;
	class ISceneNodeAnimatorCollisionResponse;

	namespace quake3
	{
		struct IShader;
	} // end namespace quake3

	//! The Scene Manager manages scene nodes, mesh recources, cameras and all the other stuff.
	/** All Scene nodes can be created only here. There is a always growing
	list of scene nodes for lots of purposes: Indoor rendering scene nodes,
	different Camera scene nodes (addCameraSceneNode(), addCameraSceneNodeMaya()),
	Billboards (addBillboardSceneNode()) and so on.
	A scene node is a node in the hierachical scene tree. Every scene node
	may have children, which are other scene nodes. Children move relative
	the their parents position. If the parent of a node is not visible, its
	children won't be visible, too. In this way, it is for example easily
	possible to attach a light to a moving car or to place a walking
	character on a moving platform on a moving ship.
	The SceneManager is also able to load 3d mesh files of different
	formats. Take a look at getMesh() to find out what formats are
	supported. If these formats are not enough, use
	addExternalMeshLoader() to add new formats to the engine.
	*/
	class ISceneManager : public virtual core::IReferenceCounted
	{
	public:
		//! Get the video driver.
		/** \return Pointer to the video Driver.
		This pointer should not be dropped. See IReferenceCounted::drop() for more information. */
		virtual video::IVideoDriver* getVideoDriver() = 0;


		//! Get the active FileSystem
		/** \return Pointer to the FileSystem
		This pointer should not be dropped. See IReferenceCounted::drop() for more information. */
		virtual io::IFileSystem* getFileSystem() = 0;


        virtual IrrlichtDevice* getDevice() = 0;


		//! Adds a cube scene node
		/** \param size: Size of the cube, uniformly in each dimension.
		\param parent: Parent of the scene node. Can be 0 if no parent.
		\param id: Id of the node. This id can be used to identify the scene node.
		\param position: Position of the space relative to its parent
		where the scene node will be placed.
		\param rotation: Initital rotation of the scene node.
		\param scale: Initial scale of the scene node.
		\return Pointer to the created test scene node. This
		pointer should not be dropped. See IReferenceCounted::drop()
		for more information. */
		virtual IMeshSceneNode* addCubeSceneNode(float size=10.0f, IDummyTransformationSceneNode* parent=0, int32_t id=-1,
			const core::vector3df& position = core::vector3df(0,0,0),
			const core::vector3df& rotation = core::vector3df(0,0,0),
			const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f)) = 0;

		//! Adds a sphere scene node of the given radius and detail
		/** \param radius: Radius of the sphere.
		\param polyCount: The number of vertices in horizontal and
		vertical direction. The total polyCount of the sphere is
		polyCount*polyCount. This parameter must be less than 256 to
		stay within the 16-bit limit of the indices of a meshbuffer.
		\param parent: Parent of the scene node. Can be 0 if no parent.
		\param id: Id of the node. This id can be used to identify the scene node.
		\param position: Position of the space relative to its parent
		where the scene node will be placed.
		\param rotation: Initital rotation of the scene node.
		\param scale: Initial scale of the scene node.
		\return Pointer to the created test scene node. This
		pointer should not be dropped. See IReferenceCounted::drop()
		for more information. */
		virtual IMeshSceneNode* addSphereSceneNode(float radius=5.0f, int32_t polyCount=16,
				IDummyTransformationSceneNode* parent=0, int32_t id=-1,
				const core::vector3df& position = core::vector3df(0,0,0),
				const core::vector3df& rotation = core::vector3df(0,0,0),
				const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f)) = 0;

		//! Adds a scene node for rendering an skinned mesh model.
		virtual ISkinnedMeshSceneNode* addSkinnedMeshSceneNode(
                video::IGPUSkinnedMesh* mesh, const ISkinningStateManager::E_BONE_UPDATE_MODE& boneControlMode=ISkinningStateManager::EBUM_NONE,
				IDummyTransformationSceneNode* parent=0, int32_t id=-1,
				const core::vector3df& position = core::vector3df(0,0,0),
				const core::vector3df& rotation = core::vector3df(0,0,0),
				const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f)) = 0;

		//! Adds a scene node for rendering a static mesh.
		/** \param mesh: Pointer to the loaded static mesh to be displayed.
		\param parent: Parent of the scene node. Can be NULL if no parent.
		\param id: Id of the node. This id can be used to identify the scene node.
		\param position: Position of the space relative to its parent where the
		scene node will be placed.
		\param rotation: Initital rotation of the scene node.
		\param scale: Initial scale of the scene node.
		\param alsoAddIfMeshPointerZero: Add the scene node even if a 0 pointer is passed.
		\return Pointer to the created scene node.
		This pointer should not be dropped. See IReferenceCounted::drop() for more information. */
		virtual IMeshSceneNode* addMeshSceneNode(video::IGPUMesh* mesh, IDummyTransformationSceneNode* parent=0, int32_t id=-1,
			const core::vector3df& position = core::vector3df(0,0,0),
			const core::vector3df& rotation = core::vector3df(0,0,0),
			const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f),
			bool alsoAddIfMeshPointerZero=false) = 0;

        virtual IMeshSceneNodeInstanced* addMeshSceneNodeInstanced(IDummyTransformationSceneNode* parent=0, int32_t id=-1,
			const core::vector3df& position = core::vector3df(0,0,0),
			const core::vector3df& rotation = core::vector3df(0,0,0),
			const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f)) = 0;

		//! Adds a camera scene node to the scene tree and sets it as active camera.
		/** This camera does not react on user input like for example the one created with
		addCameraSceneNodeFPS(). If you want to move or animate it, use animators or the
		ISceneNode::setPosition(), ICameraSceneNode::setTarget() etc methods.
		By default, a camera's look at position (set with setTarget()) and its scene node
		rotation (set with setRotation()) are independent. If you want to be able to
		control the direction that the camera looks by using setRotation() then call
		ICameraSceneNode::bindTargetAndRotation(true) on it.
		\param position: Position of the space relative to its parent where the camera will be placed.
		\param lookat: Position where the camera will look at. Also known as target.
		\param parent: Parent scene node of the camera. Can be null. If the parent moves,
		the camera will move too.
		\param id: id of the camera. This id can be used to identify the camera.
		\param makeActive Flag whether this camera should become the active one.
		Make sure you always have one active camera.
		\return Pointer to interface to camera if successful, otherwise 0.
		This pointer should not be dropped. See IReferenceCounted::drop() for more information. */
		virtual ICameraSceneNode* addCameraSceneNode(IDummyTransformationSceneNode* parent = 0,
			const core::vector3df& position = core::vector3df(0,0,0),
			const core::vector3df& lookat = core::vector3df(0,0,100),
			int32_t id=-1, bool makeActive=true) = 0;

		//! Adds a maya style user controlled camera scene node to the scene tree.
		/** This is a standard camera with an animator that provides mouse control similar
		to camera in the 3D Software Maya by Alias Wavefront.
		The camera does not react on setPosition anymore after applying this animator. Instead
		use setTarget, to fix the target the camera the camera hovers around. And setDistance
		to set the current distance from that target, i.e. the radius of the orbit the camera
		hovers on.
		\param parent: Parent scene node of the camera. Can be null.
		\param rotateSpeed: Rotation speed of the camera.
		\param zoomSpeed: Zoom speed of the camera.
		\param translationSpeed: TranslationSpeed of the camera.
		\param id: id of the camera. This id can be used to identify the camera.
		\param distance Initial distance of the camera from the object
		\param makeActive Flag whether this camera should become the active one.
		Make sure you always have one active camera.
		\return Returns a pointer to the interface of the camera if successful, otherwise 0.
		This pointer should not be dropped. See IReferenceCounted::drop() for more information. */
		virtual ICameraSceneNode* addCameraSceneNodeMaya(IDummyTransformationSceneNode* parent=0,
			float rotateSpeed=-1500.f, float zoomSpeed=200.f,
			float translationSpeed=1500.f, int32_t id=-1, float distance=70.f,
			bool makeActive=true) =0;

		//! Adds a camera scene node with an animator which provides mouse and keyboard control appropriate for first person shooters (FPS).
		/** This FPS camera is intended to provide a demonstration of a
		camera that behaves like a typical First Person Shooter. It is
		useful for simple demos and prototyping but is not intended to
		provide a full solution for a production quality game. It binds
		the camera scene node rotation to the look-at target; @see
		ICameraSceneNode::bindTargetAndRotation(). With this camera,
		you look with the mouse, and move with cursor keys. If you want
		to change the key layout, you can specify your own keymap. For
		example to make the camera be controlled by the cursor keys AND
		the keys W,A,S, and D, do something like this:
		\code
		 SKeyMap keyMap[8];
		 keyMap[0].Action = EKA_MOVE_FORWARD;
		 keyMap[0].KeyCode = KEY_UP;
		 keyMap[1].Action = EKA_MOVE_FORWARD;
		 keyMap[1].KeyCode = KEY_KEY_W;

		 keyMap[2].Action = EKA_MOVE_BACKWARD;
		 keyMap[2].KeyCode = KEY_DOWN;
		 keyMap[3].Action = EKA_MOVE_BACKWARD;
		 keyMap[3].KeyCode = KEY_KEY_S;

		 keyMap[4].Action = EKA_STRAFE_LEFT;
		 keyMap[4].KeyCode = KEY_LEFT;
		 keyMap[5].Action = EKA_STRAFE_LEFT;
		 keyMap[5].KeyCode = KEY_KEY_A;

		 keyMap[6].Action = EKA_STRAFE_RIGHT;
		 keyMap[6].KeyCode = KEY_RIGHT;
		 keyMap[7].Action = EKA_STRAFE_RIGHT;
		 keyMap[7].KeyCode = KEY_KEY_D;

		camera = sceneManager->addCameraSceneNodeFPS(0, 100, 500, -1, keyMap, 8);
		\endcode
		\param parent: Parent scene node of the camera. Can be null.
		\param rotateSpeed: Speed in degress with which the camera is
		rotated. This can be done only with the mouse.
		\param moveSpeed: Speed in units per millisecond with which
		the camera is moved. Movement is done with the cursor keys.
		\param id: id of the camera. This id can be used to identify
		the camera.
		\param keyMapArray: Optional pointer to an array of a keymap,
		specifying what keys should be used to move the camera. If this
		is null, the default keymap is used. You can define actions
		more then one time in the array, to bind multiple keys to the
		same action.
		\param keyMapSize: Amount of items in the keymap array.
		\param noVerticalMovement: Setting this to true makes the
		camera only move within a horizontal plane, and disables
		vertical movement as known from most ego shooters. Default is
		'false', with which it is possible to fly around in space, if
		no gravity is there.
		\param jumpSpeed: Speed with which the camera is moved when
		jumping.
		\param invertMouse: Setting this to true makes the camera look
		up when the mouse is moved down and down when the mouse is
		moved up, the default is 'false' which means it will follow the
		movement of the mouse cursor.
		\param makeActive Flag whether this camera should become the active one.
		Make sure you always have one active camera.
		\return Pointer to the interface of the camera if successful,
		otherwise 0. This pointer should not be dropped. See
		IReferenceCounted::drop() for more information. */
		virtual ICameraSceneNode* addCameraSceneNodeFPS(IDummyTransformationSceneNode* parent = 0,
			float rotateSpeed = 100.0f, float moveSpeed = 0.5f, int32_t id=-1,
			SKeyMap* keyMapArray=0, int32_t keyMapSize=0, bool noVerticalMovement=false,
			float jumpSpeed = 0.f, bool invertMouse=false,
			bool makeActive=true) = 0;


		//! Adds a billboard scene node to the scene tree.
		/** A billboard is like a 3d sprite: A 2d element,
		which always looks to the camera. It is usually used for things
		like explosions, fire, lensflares and things like that.
		\param parent Parent scene node of the billboard. Can be null.
		If the parent moves, the billboard will move too.
		\param size Size of the billboard. This size is 2 dimensional
		because a billboard only has width and height.
		\param position Position of the space relative to its parent
		where the billboard will be placed.
		\param id An id of the node. This id can be used to identify
		the node.
		\param colorTop The color of the vertices at the top of the
		billboard (default: white).
		\param colorBottom The color of the vertices at the bottom of
		the billboard (default: white).
		\return Pointer to the billboard if successful, otherwise NULL.
		This pointer should not be dropped. See
		IReferenceCounted::drop() for more information. */
		virtual IBillboardSceneNode* addBillboardSceneNode(IDummyTransformationSceneNode* parent = 0,
			const core::dimension2d<float>& size = core::dimension2d<float>(10.0f, 10.0f),
			const core::vector3df& position = core::vector3df(0,0,0), int32_t id=-1,
			video::SColor colorTop = 0xFFFFFFFF, video::SColor colorBottom = 0xFFFFFFFF) = 0;

		//! Adds a skybox scene node to the scene tree.
		/** A skybox is a big cube with 6 textures on it and
		is drawn around the camera position.
		\param top: Texture for the top plane of the box.
		\param bottom: Texture for the bottom plane of the box.
		\param left: Texture for the left plane of the box.
		\param right: Texture for the right plane of the box.
		\param front: Texture for the front plane of the box.
		\param back: Texture for the back plane of the box.
		\param parent: Parent scene node of the skybox. A skybox usually has no parent,
		so this should be null. Note: If a parent is set to the skybox, the box will not
		change how it is drawn.
		\param id: An id of the node. This id can be used to identify the node.
		\return Pointer to the sky box if successful, otherwise NULL.
		This pointer should not be dropped. See IReferenceCounted::drop() for more information. */
		virtual ISceneNode* addSkyBoxSceneNode(video::ITexture* top, video::ITexture* bottom,
			video::ITexture* left, video::ITexture* right, video::ITexture* front,
			video::ITexture* back, IDummyTransformationSceneNode* parent = 0, int32_t id=-1) = 0;

		//! Adds a skydome scene node to the scene tree.
		/** A skydome is a large (half-) sphere with a panoramic texture
		on the inside and is drawn around the camera position.
		\param texture: Texture for the dome.
		\param horiRes: Number of vertices of a horizontal layer of the sphere.
		\param vertRes: Number of vertices of a vertical layer of the sphere.
		\param texturePercentage: How much of the height of the
		texture is used. Should be between 0 and 1.
		\param spherePercentage: How much of the sphere is drawn.
		Value should be between 0 and 2, where 1 is an exact
		half-sphere and 2 is a full sphere.
		\param radius The Radius of the sphere
		\param parent: Parent scene node of the dome. A dome usually has no parent,
		so this should be null. Note: If a parent is set, the dome will not
		change how it is drawn.
		\param id: An id of the node. This id can be used to identify the node.
		\return Pointer to the sky dome if successful, otherwise NULL.
		This pointer should not be dropped. See IReferenceCounted::drop() for more information. */
		virtual ISceneNode* addSkyDomeSceneNode(video::IVirtualTexture* texture,
			uint32_t horiRes=16, uint32_t vertRes=8,
			float texturePercentage=0.9, float spherePercentage=2.0,float radius = 1000.f,
			IDummyTransformationSceneNode* parent=0, int32_t id=-1) = 0;

		//! Adds a dummy transformation scene node to the scene tree.
		/** This scene node does not render itself, have a bounding box, a render method,
        and is as-if always visible ISceneNode.
		Its actually a base of ISceneNode, and it can be used for doing advanced transformations
		or structuring the scene tree.
		\return Pointer to the created scene node.
		This pointer should not be dropped. See IReferenceCounted::drop() for more information. */
		virtual IDummyTransformationSceneNode* addDummyTransformationSceneNode(
			IDummyTransformationSceneNode* parent=0, int32_t id=-1) = 0;

		//! Gets the root scene node.
		/** This is the scene node which is parent
		of all scene nodes. The root scene node is a special scene node which
		only exists to manage all scene nodes. It will not be rendered and cannot
		be removed from the scene.
		\return Pointer to the root scene node.
		This pointer should not be dropped. See IReferenceCounted::drop() for more information. */
		virtual ISceneNode* getRootSceneNode() = 0;

		//! Get the current active camera.
		/** \return The active camera is returned. Note that this can
		be NULL, if there was no camera created yet.
		This pointer should not be dropped. See IReferenceCounted::drop() for more information. */
		virtual ICameraSceneNode* getActiveCamera() const =0;

		//! Sets the currently active camera.
		/** The previous active camera will be deactivated.
		\param camera: The new camera which should be active. */
		virtual void setActiveCamera(ICameraSceneNode* camera) = 0;

		//! Registers a node for rendering it at a specific time.
		/** This method should only be used by SceneNodes when they get a
		ISceneNode::OnRegisterSceneNode() call.
		\param node: Node to register for drawing. Usually scene nodes would set 'this'
		as parameter here because they want to be drawn.
		\param pass: Specifies when the node wants to be drawn in relation to the other nodes.
		For example, if the node is a shadow, it usually wants to be drawn after all other nodes
		and will use ESNRP_SHADOW for this. See scene::E_SCENE_NODE_RENDER_PASS for details.
		\return scene will be rendered ( passed culling ) */
		virtual uint32_t registerNodeForRendering(ISceneNode* node,
			E_SCENE_NODE_RENDER_PASS pass = ESNRP_AUTOMATIC) = 0;

		//! Draws all the scene nodes.
		/** This can only be invoked between
		IVideoDriver::beginScene() and IVideoDriver::endScene(). Please note that
		the scene is not only drawn when calling this, but also animated
		by existing scene node animators, culling of scene nodes is done, etc. */
		virtual void drawAll() = 0;

		//! Creates a rotation animator, which rotates the attached scene node around itself.
		/** \param rotationSpeed Specifies the speed of the animation in degree per 10 milliseconds.
		\return The animator. Attach it to a scene node with ISceneNode::addAnimator()
		and the animator will animate it.
		If you no longer need the animator, you should call ISceneNodeAnimator::drop().
		See IReferenceCounted::drop() for more information. */
		virtual ISceneNodeAnimator* createRotationAnimator(const core::vector3df& rotationSpeed) = 0;

		//! Creates a fly circle animator, which lets the attached scene node fly around a center.
		/** \param center: Center of the circle.
		\param radius: Radius of the circle.
		\param speed: The orbital speed, in radians per millisecond.
		\param direction: Specifies the upvector used for alignment of the mesh.
		\param startPosition: The position on the circle where the animator will
		begin. Value is in multiples of a circle, i.e. 0.5 is half way around. (phase)
		\param radiusEllipsoid: if radiusEllipsoid != 0 then radius2 froms a ellipsoid
		begin. Value is in multiples of a circle, i.e. 0.5 is half way around. (phase)
		\return The animator. Attach it to a scene node with ISceneNode::addAnimator()
		and the animator will animate it.
		If you no longer need the animator, you should call ISceneNodeAnimator::drop().
		See IReferenceCounted::drop() for more information. */
		virtual ISceneNodeAnimator* createFlyCircleAnimator(
				const core::vector3df& center=core::vector3df(0.f,0.f,0.f),
				float radius=100.f, float speed=0.001f,
				const core::vector3df& direction=core::vector3df(0.f, 1.f, 0.f),
				float startPosition = 0.f,
				float radiusEllipsoid = 0.f) = 0;

		//! Creates a fly straight animator, which lets the attached scene node fly or move along a line between two points.
		/** \param startPoint: Start point of the line.
		\param endPoint: End point of the line.
		\param timeForWay: Time in milli seconds how long the node should need to
		move from the start point to the end point.
		\param loop: If set to false, the node stops when the end point is reached.
		If loop is true, the node begins again at the start.
		\param pingpong Flag to set whether the animator should fly
		back from end to start again.
		\return The animator. Attach it to a scene node with ISceneNode::addAnimator()
		and the animator will animate it.
		If you no longer need the animator, you should call ISceneNodeAnimator::drop().
		See IReferenceCounted::drop() for more information. */
		virtual ISceneNodeAnimator* createFlyStraightAnimator(const core::vector3df& startPoint,
			const core::vector3df& endPoint, uint32_t timeForWay, bool loop=false, bool pingpong = false) = 0;

		//! Creates a texture animator, which switches the textures of the target scene node based on a list of textures.
		/** \param textures: List of textures to use.
		\param timePerFrame: Time in milliseconds, how long any texture in the list
		should be visible.
		\param loop: If set to to false, the last texture remains set, and the animation
		stops. If set to true, the animation restarts with the first texture.
		\return The animator. Attach it to a scene node with ISceneNode::addAnimator()
		and the animator will animate it.
		If you no longer need the animator, you should call ISceneNodeAnimator::drop().
		See IReferenceCounted::drop() for more information. */
		virtual ISceneNodeAnimator* createTextureAnimator(const core::vector<video::ITexture*>& textures,
			int32_t timePerFrame, bool loop=true) = 0;

		//! Creates a scene node animator, which deletes the scene node after some time automatically.
		/** \param timeMs: Time in milliseconds, after when the node will be deleted.
		\return The animator. Attach it to a scene node with ISceneNode::addAnimator()
		and the animator will animate it.
		If you no longer need the animator, you should call ISceneNodeAnimator::drop().
		See IReferenceCounted::drop() for more information. */
		virtual ISceneNodeAnimator* createDeleteAnimator(uint32_t timeMs) = 0;

		//! Creates a follow spline animator.
		/** The animator modifies the position of
		the attached scene node to make it follow a hermite spline.
		It uses a subset of hermite splines: either cardinal splines
		(tightness != 0.5) or catmull-rom-splines (tightness == 0.5).
		The animator moves from one control point to the next in
		1/speed seconds. This code was sent in by Matthias Gall.
		If you no longer need the animator, you should call ISceneNodeAnimator::drop().
		See IReferenceCounted::drop() for more information. */
		virtual ISceneNodeAnimator* createFollowSplineAnimator(int32_t startTime,
			const core::vector< core::vector3df >& points,
			float speed = 1.0f, float tightness = 0.5f, bool loop=true, bool pingpong=false) = 0;

		//! Adds a scene node to the deletion queue.
		/** The scene node is immediatly
		deleted when it's secure. Which means when the scene node does not
		execute animators and things like that. This method is for example
		used for deleting scene nodes by their scene node animators. In
		most other cases, a ISceneNode::remove() call is enough, using this
		deletion queue is not necessary.
		See ISceneManager::createDeleteAnimator() for details.
		\param node: Node to detete. */
		virtual void addToDeletionQueue(IDummyTransformationSceneNode* node) = 0;

		//! Posts an input event to the environment.
		/** Usually you do not have to
		use this method, it is used by the internal engine. */
		virtual bool receiveIfEventReceiverDidNotAbsorb(const SEvent& event) = 0;

		//! Clears the whole scene.
		/** All scene nodes are removed. */
		virtual void clear() = 0;

		//! Get current render pass.
		/** All scene nodes are being rendered in a specific order.
		First lights, cameras, sky boxes, solid geometry, and then transparent
		stuff. During the rendering process, scene nodes may want to know what the scene
		manager is rendering currently, because for example they registered for rendering
		twice, once for transparent geometry and once for solid. When knowing what rendering
		pass currently is active they can render the correct part of their geometry. */
		virtual E_SCENE_NODE_RENDER_PASS getSceneNodeRenderPass() const = 0;

		//! Creates a new scene manager.
		/** This can be used to easily draw and/or store two
		independent scenes at the same time.
		If you no longer need the new scene manager, you should call
		ISceneManager::drop().
		See IReferenceCounted::drop() for more information. */
		virtual ISceneManager* createNewSceneManager(bool cloneContent=false) = 0;

		//! Check if node is culled in current view frustum
		/** Please note that depending on the used culling method this
		check can be rather coarse, or slow. A positive result is
		correct, though, i.e. if this method returns true the node is
		positively not visible. The node might still be invisible even
		if this method returns false.
		\param node The scene node which is checked for culling.
		\return True if node is not visible in the current scene, else
		false. */
		virtual bool isCulled(ISceneNode* node) const =0;
	};


} // end namespace scene
} // end namespace irr

#endif

