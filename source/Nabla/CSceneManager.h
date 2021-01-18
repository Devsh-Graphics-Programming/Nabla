// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_SCENE_MANAGER_H_INCLUDED__
#define __NBL_C_SCENE_MANAGER_H_INCLUDED__

#include "ISceneManager.h"
#include "ISceneNode.h"
#include "ICursorControl.h"
#include "ISkinningStateManager.h"

#include <map>
#include <string>

namespace nbl
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
		CSceneManager(	IrrlichtDevice* device, video::IVideoDriver* driver,
						nbl::ITimer* timer, io::IFileSystem* fs, gui::ICursorControl* cursorControl);

		//! returns the video driver
		virtual video::IVideoDriver* getVideoDriver();

		//! return the filesystem
		virtual io::IFileSystem* getFileSystem();

        virtual IrrlichtDevice* getDevice() override;

        //!
        virtual ISkinnedMeshSceneNode* addSkinnedMeshSceneNode(
			core::smart_refctd_ptr<video::IGPUSkinnedMesh>&& mesh,
			const ISkinningStateManager::E_BONE_UPDATE_MODE& boneControlMode=ISkinningStateManager::EBUM_NONE,
            IDummyTransformationSceneNode* parent=0, int32_t id=-1,
            const core::vector3df& position = core::vector3df(0,0,0),
            const core::vector3df& rotation = core::vector3df(0,0,0),
            const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f)) override;

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
			const core::vectorSIMDf & lookat = core::vectorSIMDf(0,0,100),
			int32_t id=-1, bool makeActive=true) override;

		//! Adds a camera scene node which is able to be controlle with the mouse similar
		//! like in the 3D Software Maya by Alias Wavefront.
		//! The returned pointer must not be dropped.
		virtual ICameraSceneNode* addCameraSceneNodeMaya(IDummyTransformationSceneNode* parent=0,
			float rotateSpeed=-1500.f, float zoomSpeed=200.f,
			float translationSpeed=1500.f, int32_t id=-1, float distance=70.f,
			bool makeActive=true);

		virtual ICameraSceneNode* addCameraSceneNodeModifiedMaya(IDummyTransformationSceneNode* parent = 0,
			float rotateSpeed = -1500.f, float zoomSpeed = 200.f,
			float translationSpeed = 1500.f, int32_t id = -1, float distance = 70.f,
			float scrlZoomSpeed = 10.0f, bool zoomlWithRMB = false,
			bool makeActive = true) override;

		//! Adds a camera scene node which is able to be controled with the mouse and keys
		//! like in most first person shooters (FPS):
		virtual ICameraSceneNode* addCameraSceneNodeFPS(IDummyTransformationSceneNode* parent = 0,
			float rotateSpeed = 100.0f, float moveSpeed = .5f, int32_t id=-1,
			SKeyMap* keyMapArray=0, int32_t keyMapSize=0,
			bool noVerticalMovement=false, float jumpSpeed = 0.f,
			bool invertMouseY=false, bool makeActive=true);

		//! Adds a skybox scene node. A skybox is a big cube with 6 textures on it and
		//! is drawn around the camera position.
		virtual IMeshSceneNode* addSkyBoxSceneNode(	core::smart_refctd_ptr<video::IGPUImageView>&& cubemap,
												IDummyTransformationSceneNode* parent = 0, int32_t id = -1) override;

		//! Adds a skydome scene node. A skydome is a large (half-) sphere with a
		//! panoramic texture on it and is drawn around the camera position.
		virtual IMeshSceneNode* addSkyDomeSceneNode(core::smart_refctd_ptr<video::IGPUImageView>&& texture,
												uint32_t horiRes = 16, uint32_t vertRes = 8, float texturePercentage = 0.9,
												float spherePercentage = 2.0, float radius = 1000.f,
												IDummyTransformationSceneNode * parent = 0, int32_t id = -1) override;

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
		virtual ISceneNodeAnimator* createRotationAnimator(const core::vector3df& rotationPerSecond) override;

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
				const core::vectorSIMDf& direction=core::vectorSIMDf(0.f, 1.f, 0.f),
				float startPosition = 0.f,
				float radiusEllipsoid = 0.f) override;

		//! Creates a fly straight animator, which lets the attached scene node
		//! fly or move along a line between two points.
		virtual ISceneNodeAnimator* createFlyStraightAnimator(const core::vectorSIMDf& startPoint,
			const core::vectorSIMDf& endPoint, uint32_t timeForWay, bool loop=false,bool pingpong = false) override;

		//! Creates a scene node animator, which deletes the scene node after
		//! some time automaticly.
		virtual ISceneNodeAnimator* createDeleteAnimator(uint32_t timeMS) override;

		//! Creates a follow spline animator.
		virtual ISceneNodeAnimator* createFollowSplineAnimator(int32_t startTime,
			const core::vector< core::vector3df >& points,
			float speed, float tightness, bool loop, bool pingpong) override;


		//! Adds a scene node to the deletion queue.
		virtual void addToDeletionQueue(IDummyTransformationSceneNode* node);

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
					Node(n), renderPriority(0x80000000u)
				{
					renderPriority = n->getRenderPriorityScore();
#ifdef OLD_SHADERS
					if (n->getMaterialCount())
						Material = n->getMaterial(0).MaterialType;
#endif
				}

				bool operator < (const DefaultNodeEntry& other) const
				{
#ifdef OLD_SHADERS
					return (renderPriority < other.renderPriority)||(renderPriority==other.renderPriority && Material<other.Material);
#else
					return renderPriority < other.renderPriority;
#endif
				}

				ISceneNode* Node;
			private:
				uint32_t renderPriority;
#ifdef OLD_SHADERS
				video::E_MATERIAL_TYPE Material;
#endif
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
		nbl::ITimer* Timer;

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

		core::smart_refctd_ptr<video::IGPUBuffer> redundantMeshDataBuf;

		E_SCENE_NODE_RENDER_PASS CurrentRendertime;

		//! constants for reading and writing XML.
		//! Not made static due to portability problems.
		const core::stringw NBL_XML_FORMAT_SCENE;
		const core::stringw NBL_XML_FORMAT_NODE;
		const core::stringw NBL_XML_FORMAT_NODE_ATTR_TYPE;
	};

} // end namespace video
} // end namespace scene

#endif

