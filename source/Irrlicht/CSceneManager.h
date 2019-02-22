// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_SCENE_MANAGER_H_INCLUDED__
#define __C_SCENE_MANAGER_H_INCLUDED__

#include "ISceneManager.h"
#include "ISceneNode.h"
#include "ICursorControl.h"
#include "ISkinningStateManager.h"

#include <map>
#include <string>

namespace irr
{
	class ITimer;
namespace io
{
	class IXMLWriter;
	class IFileSystem;
}
namespace scene
{
	class IAnimatedMeshSceneNode;

	/*!
		The Scene Manager manages scene nodes, mesh recources, cameras and all the other stuff.
	*/
	class CSceneManager : public ISceneManager, public ISceneNode
	{
    protected:
		//! destructor
		virtual ~CSceneManager();

	public:
		//! constructor
		CSceneManager(IrrlichtDevice* device, video::IVideoDriver* driver, irr::ITimer* timer, io::IFileSystem* fs,
			gui::ICursorControl* cursorControl);

		//! returns the video driver
		virtual video::IVideoDriver* getVideoDriver();

		//! return the filesystem
		virtual io::IFileSystem* getFileSystem();

        virtual IrrlichtDevice* getDevice() override;

		//! adds a cube scene node to the scene. It is a simple cube of (1,1,1) size.
		//! the returned pointer must not be dropped.
		virtual IMeshSceneNode* addCubeSceneNode(float size=10.0f, IDummyTransformationSceneNode* parent=0, int32_t id=-1,
			const core::vector3df& position = core::vector3df(0,0,0),	const core::vector3df& rotation = core::vector3df(0,0,0),	const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f));

		//! Adds a sphere scene node to the scene.
		virtual IMeshSceneNode* addSphereSceneNode(float radius=5.0f, int32_t polyCount=16, IDummyTransformationSceneNode* parent=0, int32_t id=-1,
			const core::vector3df& position = core::vector3df(0,0,0),
			const core::vector3df& rotation = core::vector3df(0,0,0),
			const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f));

        //!
        virtual ISkinnedMeshSceneNode* addSkinnedMeshSceneNode(video::IGPUSkinnedMesh* mesh, const ISkinningStateManager::E_BONE_UPDATE_MODE& boneControlMode=ISkinningStateManager::EBUM_NONE,
            IDummyTransformationSceneNode* parent=0, int32_t id=-1,
            const core::vector3df& position = core::vector3df(0,0,0),
            const core::vector3df& rotation = core::vector3df(0,0,0),
            const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f));

		//! adds a scene node for rendering a static mesh
		//! the returned pointer must not be dropped.
		virtual IMeshSceneNode* addMeshSceneNode(video::IGPUMesh* mesh, IDummyTransformationSceneNode* parent=0, int32_t id=-1,
			const core::vector3df& position = core::vector3df(0,0,0),
			const core::vector3df& rotation = core::vector3df(0,0,0),
			const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f),
			bool alsoAddIfMeshPointerZero=false);

        //!
        virtual IMeshSceneNodeInstanced* addMeshSceneNodeInstanced(IDummyTransformationSceneNode* parent=0, int32_t id=-1,
			const core::vector3df& position = core::vector3df(0,0,0),
			const core::vector3df& rotation = core::vector3df(0,0,0),
			const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f));

        //!
		virtual void OnAnimate(uint32_t timeMs);

		//! renders the node.
		virtual void render();

		//! returns the axis aligned bounding box of this node
		virtual const core::aabbox3d<float>& getBoundingBox();

		//! registers a node for rendering it at a specific time.
		virtual uint32_t registerNodeForRendering(ISceneNode* node, E_SCENE_NODE_RENDER_PASS pass = ESNRP_AUTOMATIC);

		//! draws all scene nodes
		virtual void drawAll();

		//! Adds a camera scene node to the tree and sets it as active camera.
		//! \param position: Position of the space relative to its parent where the camera will be placed.
		//! \param lookat: Position where the camera will look at. Also known as target.
		//! \param parent: Parent scene node of the camera. Can be null. If the parent moves,
		//! the camera will move too.
		//! \return Pointer to interface to camera
		virtual ICameraSceneNode* addCameraSceneNode(IDummyTransformationSceneNode* parent = 0,
			const core::vector3df& position = core::vector3df(0,0,0),
			const core::vector3df& lookat = core::vector3df(0,0,100),
			int32_t id=-1, bool makeActive=true);

		//! Adds a camera scene node which is able to be controlle with the mouse similar
		//! like in the 3D Software Maya by Alias Wavefront.
		//! The returned pointer must not be dropped.
		virtual ICameraSceneNode* addCameraSceneNodeMaya(IDummyTransformationSceneNode* parent=0,
			float rotateSpeed=-1500.f, float zoomSpeed=200.f,
			float translationSpeed=1500.f, int32_t id=-1, float distance=70.f,
			bool makeActive=true);

		//! Adds a camera scene node which is able to be controled with the mouse and keys
		//! like in most first person shooters (FPS):
		virtual ICameraSceneNode* addCameraSceneNodeFPS(IDummyTransformationSceneNode* parent = 0,
			float rotateSpeed = 100.0f, float moveSpeed = .5f, int32_t id=-1,
			SKeyMap* keyMapArray=0, int32_t keyMapSize=0,
			bool noVerticalMovement=false, float jumpSpeed = 0.f,
			bool invertMouseY=false, bool makeActive=true);

		//! Adds a billboard scene node to the scene. A billboard is like a 3d sprite: A 2d element,
		//! which always looks to the camera. It is usually used for things like explosions, fire,
		//! lensflares and things like that.
		virtual IBillboardSceneNode* addBillboardSceneNode(IDummyTransformationSceneNode* parent = 0,
			const core::dimension2d<float>& size = core::dimension2d<float>(10.0f, 10.0f),
			const core::vector3df& position = core::vector3df(0,0,0), int32_t id=-1,
			video::SColor shadeTop = 0xFFFFFFFF, video::SColor shadeBottom = 0xFFFFFFFF);

		//! Adds a skybox scene node. A skybox is a big cube with 6 textures on it and
		//! is drawn around the camera position.
		virtual ISceneNode* addSkyBoxSceneNode(video::ITexture* top, video::ITexture* bottom,
			video::ITexture* left, video::ITexture* right, video::ITexture* front,
			video::ITexture* back, IDummyTransformationSceneNode* parent = 0, int32_t id=-1);

		//! Adds a skydome scene node. A skydome is a large (half-) sphere with a
		//! panoramic texture on it and is drawn around the camera position.
		virtual ISceneNode* addSkyDomeSceneNode(video::IVirtualTexture* texture,
			uint32_t horiRes=16, uint32_t vertRes=8,
			float texturePercentage=0.9, float spherePercentage=2.0,float radius = 1000.f,
			IDummyTransformationSceneNode* parent=0, int32_t id=-1);

		//! Adds a dummy transformation scene node to the scene tree.
		virtual IDummyTransformationSceneNode* addDummyTransformationSceneNode(
			IDummyTransformationSceneNode* parent=0, int32_t id=-1);

		//! Returns the root scene node. This is the scene node wich is parent
		//! of all scene nodes. The root scene node is a special scene node which
		//! only exists to manage all scene nodes. It is not rendered and cannot
		//! be removed from the scene.
		//! \return Pointer to the root scene node.
		virtual ISceneNode* getRootSceneNode();

		//! Returns the current active camera.
		//! \return The active camera is returned. Note that this can be NULL, if there
		//! was no camera created yet.
		virtual ICameraSceneNode* getActiveCamera() const;

		//! Sets the active camera. The previous active camera will be deactivated.
		//! \param camera: The new camera which should be active.
		virtual void setActiveCamera(ICameraSceneNode* camera);

		//! creates a rotation animator, which rotates the attached scene node around itself.
		//! \param rotationPerSecond: Specifies the speed of the animation
		//! \return The animator. Attach it to a scene node with ISceneNode::addAnimator()
		//! and the animator will animate it.
		virtual ISceneNodeAnimator* createRotationAnimator(const core::vector3df& rotationPerSecond);

		//! creates a fly circle animator
		/** Lets the attached scene node fly around a center.
		\param center Center relative to node origin
		 \param speed: The orbital speed, in radians per millisecond.
		 \param direction: Specifies the upvector used for alignment of the mesh.
		 \param startPosition: The position on the circle where the animator will
			begin. Value is in multiples  of a circle, i.e. 0.5 is half way around.
		 \return The animator. Attach it to a scene node with ISceneNode::addAnimator()
		 */
		virtual ISceneNodeAnimator* createFlyCircleAnimator(
				const core::vector3df& center=core::vector3df(0.f, 0.f, 0.f),
				float radius=100.f, float speed=0.001f,
				const core::vector3df& direction=core::vector3df(0.f, 1.f, 0.f),
				float startPosition = 0.f,
				float radiusEllipsoid = 0.f);

		//! Creates a fly straight animator, which lets the attached scene node
		//! fly or move along a line between two points.
		virtual ISceneNodeAnimator* createFlyStraightAnimator(const core::vector3df& startPoint,
			const core::vector3df& endPoint, uint32_t timeForWay, bool loop=false,bool pingpong = false);

		//! Creates a texture animator, which switches the textures of the target scene
		//! node based on a list of textures.
		virtual ISceneNodeAnimator* createTextureAnimator(const core::vector<video::ITexture*>& textures,
			int32_t timePerFrame, bool loop);

		//! Creates a scene node animator, which deletes the scene node after
		//! some time automaticly.
		virtual ISceneNodeAnimator* createDeleteAnimator(uint32_t timeMS);

		//! Creates a follow spline animator.
		virtual ISceneNodeAnimator* createFollowSplineAnimator(int32_t startTime,
			const core::vector< core::vector3df >& points,
			float speed, float tightness, bool loop, bool pingpong);


		//! Adds a scene node to the deletion queue.
		virtual void addToDeletionQueue(IDummyTransformationSceneNode* node);
/*
		//! Returns the first scene node with the specified id.
		virtual ISceneNode* getSceneNodeFromId(int32_t id, IDummyTransformationSceneNode* start=0);

		//! Returns the first scene node with the specified name.
		virtual ISceneNode* getSceneNodeFromName(const char* name, IDummyTransformationSceneNode* start=0);

		//! Returns the first scene node with the specified type.
		virtual ISceneNode* getSceneNodeFromType(scene::ESCENE_NODE_TYPE type, IDummyTransformationSceneNode* start=0);

		//! returns scene nodes by type.
		virtual void getSceneNodesFromType(ESCENE_NODE_TYPE type, core::vector<scene::ISceneNode*>& outNodes, IDummyTransformationSceneNode* start=0);
*/
		//! Posts an input event to the environment. Usually you do not have to
		//! use this method, it is used by the internal engine.
		virtual bool receiveIfEventReceiverDidNotAbsorb(const SEvent& event);

		//! Clears the whole scene. All scene nodes are removed.
		virtual void clear();

		//! Removes all children of this scene node
		virtual void removeAll();

		//! Returns current render pass.
		virtual E_SCENE_NODE_RENDER_PASS getSceneNodeRenderPass() const;

		//! Creates a new scene manager.
		virtual ISceneManager* createNewSceneManager(bool cloneContent);

		//! Returns type of the scene node
		virtual ESCENE_NODE_TYPE getType() const { return ESNT_SCENE_MANAGER; }

		//! Get current render time.
		virtual E_SCENE_NODE_RENDER_PASS getCurrentRendertime() const { return CurrentRendertime; }

		//! Set current render time.
		virtual void setCurrentRendertime(E_SCENE_NODE_RENDER_PASS currentRendertime) { CurrentRendertime = currentRendertime; }

		//! returns if node is culled
		virtual bool isCulled(ISceneNode* node) const;

	protected:

		//! clears the deletion list
		void clearDeletionList();

		struct DefaultNodeEntry
		{
				DefaultNodeEntry(ISceneNode* n) :
					Node(n), renderPriority(0x80000000u), Material(video::EMT_SOLID)
				{
					renderPriority = n->getRenderPriorityScore();
					if (n->getMaterialCount())
						Material = n->getMaterial(0).MaterialType;
				}

				bool operator < (const DefaultNodeEntry& other) const
				{
					return (renderPriority < other.renderPriority)||(renderPriority==other.renderPriority && Material<other.Material);
				}

				ISceneNode* Node;
			private:
				uint32_t renderPriority;
				video::E_MATERIAL_TYPE Material;
		};

		//! sort on distance (center) to camera
		struct TransparentNodeEntry
		{
			TransparentNodeEntry(ISceneNode* n, const core::vector3df& camera)
				: Node(n)
			{
				Distance = Node->getAbsoluteTransformation().getTranslation().getDistanceFromSQ(camera);
			}

			bool operator < (const TransparentNodeEntry& other) const
			{
				return Distance > other.Distance;
			}

			ISceneNode* Node;
			private:
				double Distance;
		};

		//! sort on distance (sphere) to camera
		struct DistanceNodeEntry
		{
			DistanceNodeEntry(ISceneNode* n, const core::vector3df& cameraPos)
				: Node(n)
			{
				setNodeAndDistanceFromPosition(n, cameraPos);
			}

			bool operator < (const DistanceNodeEntry& other) const
			{
				return Distance < other.Distance;
			}

			void setNodeAndDistanceFromPosition(ISceneNode* n, const core::vector3df & fromPosition)
			{
				Node = n;
				Distance = Node->getAbsoluteTransformation().getTranslation().getDistanceFromSQ(fromPosition);
				Distance -= Node->getBoundingBox().getExtent().getLengthSQ() * 0.5;
			}

			ISceneNode* Node;
			private:
			double Distance;
		};

		//! video driver
		video::IVideoDriver* Driver;

		//! timer
		irr::ITimer* Timer;

		//! file system
		io::IFileSystem* FileSystem;

        //! parent device
        IrrlichtDevice* Device;

		//! cursor control
		gui::ICursorControl* CursorControl;

		//! render pass lists
		core::vector<ISceneNode*> CameraList;
		core::vector<ISceneNode*> LightList;
		core::vector<ISceneNode*> SkyBoxList;
		core::vector<DefaultNodeEntry> SolidNodeList;
		core::vector<TransparentNodeEntry> TransparentNodeList;
		core::vector<TransparentNodeEntry> TransparentEffectNodeList;

		core::vector<IDummyTransformationSceneNode*> DeletionList;

		//! current active camera
		ICameraSceneNode* ActiveCamera;

        struct ParamStorage
        {
            uint8_t data[16];
        };
		core::unordered_map<std::string,ParamStorage> Parameters;

		video::IGPUBuffer* redundantMeshDataBuf;

		E_SCENE_NODE_RENDER_PASS CurrentRendertime;

		//! constants for reading and writing XML.
		//! Not made static due to portability problems.
		const core::stringw IRR_XML_FORMAT_SCENE;
		const core::stringw IRR_XML_FORMAT_NODE;
		const core::stringw IRR_XML_FORMAT_NODE_ATTR_TYPE;
	};

} // end namespace video
} // end namespace scene

#endif

