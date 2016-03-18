// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_SCENE_MANAGER_H_INCLUDED__
#define __C_SCENE_MANAGER_H_INCLUDED__

#include "ISceneManager.h"
#include "ISceneNode.h"
#include "ICursorControl.h"
#include "irrString.h"
#include "irrArray.h"
#include "IMeshLoader.h"
#include "ILightManager.h"
#include "CMeshManipulator.h"

#include <map>
#include <string>

namespace irr
{
namespace io
{
	class IXMLWriter;
	class IFileSystem;
}
namespace scene
{
	class IGeometryCreator;
	class IAnimatedMeshSceneNode;

	/*!
		The Scene Manager manages scene nodes, mesh recources, cameras and all the other stuff.
	*/
	class CSceneManager : public ISceneManager, public ISceneNode
	{
	public:

		//! constructor
		CSceneManager(video::IVideoDriver* driver, io::IFileSystem* fs,
			gui::ICursorControl* cursorControl);

		//! destructor
		virtual ~CSceneManager();

		//! gets a mesh. loads it if needed. returned pointer must not be dropped.
		virtual ICPUMesh* getMesh(const io::path& filename);

		//! gets a mesh. loads it if needed. returned pointer must not be dropped.
		virtual ICPUMesh* getMesh(io::IReadFile* file);

		//! Returns an interface to the mesh cache which is shared beween all existing scene managers.
		virtual IMeshCache<ICPUMesh>* getMeshCache();

		//! returns the video driver
		virtual video::IVideoDriver* getVideoDriver();

		//! return the filesystem
		virtual io::IFileSystem* getFileSystem();

		//! adds a cube scene node to the scene. It is a simple cube of (1,1,1) size.
		//! the returned pointer must not be dropped.
		virtual IMeshSceneNode* addCubeSceneNode(f32 size=10.0f, ISceneNode* parent=0, s32 id=-1,
			const core::vector3df& position = core::vector3df(0,0,0),	const core::vector3df& rotation = core::vector3df(0,0,0),	const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f));

		//! Adds a sphere scene node to the scene.
		virtual IMeshSceneNode* addSphereSceneNode(f32 radius=5.0f, s32 polyCount=16, ISceneNode* parent=0, s32 id=-1,
			const core::vector3df& position = core::vector3df(0,0,0),
			const core::vector3df& rotation = core::vector3df(0,0,0),
			const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f));

		//! adds a scene node for rendering an animated mesh model
		virtual IAnimatedMeshSceneNode* addAnimatedMeshSceneNode(ICPUAnimatedMesh* mesh, ISceneNode* parent=0, s32 id=-1,
			const core::vector3df& position = core::vector3df(0,0,0),
			const core::vector3df& rotation = core::vector3df(0,0,0),
			const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f),
			bool alsoAddIfMeshPointerZero=false);


		//! adds a scene node for rendering a static mesh
		//! the returned pointer must not be dropped.
		virtual IMeshSceneNode* addMeshSceneNode(IGPUMesh* mesh, ISceneNode* parent=0, s32 id=-1,
			const core::vector3df& position = core::vector3df(0,0,0),
			const core::vector3df& rotation = core::vector3df(0,0,0),
			const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f),
			bool alsoAddIfMeshPointerZero=false);

		//! renders the node.
		virtual void render();

		//! returns the axis aligned bounding box of this node
		virtual const core::aabbox3d<f32>& getBoundingBox() const;

		//! registers a node for rendering it at a specific time.
		virtual u32 registerNodeForRendering(ISceneNode* node, E_SCENE_NODE_RENDER_PASS pass = ESNRP_AUTOMATIC);

		//! draws all scene nodes
		virtual void drawAll();

		//! Adds a camera scene node to the tree and sets it as active camera.
		//! \param position: Position of the space relative to its parent where the camera will be placed.
		//! \param lookat: Position where the camera will look at. Also known as target.
		//! \param parent: Parent scene node of the camera. Can be null. If the parent moves,
		//! the camera will move too.
		//! \return Pointer to interface to camera
		virtual ICameraSceneNode* addCameraSceneNode(ISceneNode* parent = 0,
			const core::vector3df& position = core::vector3df(0,0,0),
			const core::vector3df& lookat = core::vector3df(0,0,100),
			s32 id=-1, bool makeActive=true);

		//! Adds a camera scene node which is able to be controlle with the mouse similar
		//! like in the 3D Software Maya by Alias Wavefront.
		//! The returned pointer must not be dropped.
		virtual ICameraSceneNode* addCameraSceneNodeMaya(ISceneNode* parent=0,
			f32 rotateSpeed=-1500.f, f32 zoomSpeed=200.f,
			f32 translationSpeed=1500.f, s32 id=-1, f32 distance=70.f,
			bool makeActive=true);

		//! Adds a camera scene node which is able to be controled with the mouse and keys
		//! like in most first person shooters (FPS):
		virtual ICameraSceneNode* addCameraSceneNodeFPS(ISceneNode* parent = 0,
			f32 rotateSpeed = 100.0f, f32 moveSpeed = .5f, s32 id=-1,
			SKeyMap* keyMapArray=0, s32 keyMapSize=0,
			bool noVerticalMovement=false, f32 jumpSpeed = 0.f,
			bool invertMouseY=false, bool makeActive=true);

		//! Adds a dynamic light scene node. The light will cast dynamic light on all
		//! other scene nodes in the scene, which have the material flag video::MTF_LIGHTING
		//! turned on. (This is the default setting in most scene nodes).
		virtual ILightSceneNode* addLightSceneNode(ISceneNode* parent = 0,
			const core::vector3df& position = core::vector3df(0,0,0),
			video::SColorf color = video::SColorf(1.0f, 1.0f, 1.0f),
			f32 range=100.0f, s32 id=-1);

		//! Adds a billboard scene node to the scene. A billboard is like a 3d sprite: A 2d element,
		//! which always looks to the camera. It is usually used for things like explosions, fire,
		//! lensflares and things like that.
		virtual IBillboardSceneNode* addBillboardSceneNode(ISceneNode* parent = 0,
			const core::dimension2d<f32>& size = core::dimension2d<f32>(10.0f, 10.0f),
			const core::vector3df& position = core::vector3df(0,0,0), s32 id=-1,
			video::SColor shadeTop = 0xFFFFFFFF, video::SColor shadeBottom = 0xFFFFFFFF);

		//! Adds a skybox scene node. A skybox is a big cube with 6 textures on it and
		//! is drawn around the camera position.
		virtual ISceneNode* addSkyBoxSceneNode(video::ITexture* top, video::ITexture* bottom,
			video::ITexture* left, video::ITexture* right, video::ITexture* front,
			video::ITexture* back, ISceneNode* parent = 0, s32 id=-1);

		//! Adds a skydome scene node. A skydome is a large (half-) sphere with a
		//! panoramic texture on it and is drawn around the camera position.
		virtual ISceneNode* addSkyDomeSceneNode(video::ITexture* texture,
			u32 horiRes=16, u32 vertRes=8,
			f32 texturePercentage=0.9, f32 spherePercentage=2.0,f32 radius = 1000.f,
			ISceneNode* parent=0, s32 id=-1);

		//! Adds a dummy transformation scene node to the scene graph.
		virtual IDummyTransformationSceneNode* addDummyTransformationSceneNode(
			ISceneNode* parent=0, s32 id=-1);

		//! Adds an empty scene node.
		virtual ISceneNode* addEmptySceneNode(ISceneNode* parent, s32 id=-1);

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
				f32 radius=100.f, f32 speed=0.001f,
				const core::vector3df& direction=core::vector3df(0.f, 1.f, 0.f),
				f32 startPosition = 0.f,
				f32 radiusEllipsoid = 0.f);

		//! Creates a fly straight animator, which lets the attached scene node
		//! fly or move along a line between two points.
		virtual ISceneNodeAnimator* createFlyStraightAnimator(const core::vector3df& startPoint,
			const core::vector3df& endPoint, u32 timeForWay, bool loop=false,bool pingpong = false);

		//! Creates a texture animator, which switches the textures of the target scene
		//! node based on a list of textures.
		virtual ISceneNodeAnimator* createTextureAnimator(const core::array<video::ITexture*>& textures,
			s32 timePerFrame, bool loop);

		//! Creates a scene node animator, which deletes the scene node after
		//! some time automaticly.
		virtual ISceneNodeAnimator* createDeleteAnimator(u32 timeMS);


		//! Creates a special scene node animator for doing automatic collision detection
		//! and response.
		virtual ISceneNodeAnimatorCollisionResponse* createCollisionResponseAnimator(
			ITriangleSelector* world, ISceneNode* sceneNode,
			const core::vector3df& ellipsoidRadius = core::vector3df(30,60,30),
			const core::vector3df& gravityPerSecond = core::vector3df(0,-1.0f,0),
			const core::vector3df& ellipsoidTranslation = core::vector3df(0,0,0),
			f32 slidingValue = 0.0005f);

		//! Creates a follow spline animator.
		virtual ISceneNodeAnimator* createFollowSplineAnimator(s32 startTime,
			const core::array< core::vector3df >& points,
			f32 speed, f32 tightness, bool loop, bool pingpong);


		//! Creates a simple ITriangleSelector, based on a mesh.
		virtual ITriangleSelector* createTriangleSelector(ICPUMesh* mesh, ISceneNode* node);

		//! Creates a simple ITriangleSelector, based on a mesh.
		virtual ITriangleSelector* createOctreeTriangleSelector(ICPUMesh* mesh,
			ISceneNode* node, s32 minimalPolysPerNode);

		//! Creates a simple dynamic ITriangleSelector, based on a axis aligned bounding box.
		virtual ITriangleSelector* createTriangleSelectorFromBoundingBox(
			ISceneNode* node);

		//! Creates a meta triangle selector.
		virtual IMetaTriangleSelector* createMetaTriangleSelector();

		//! Adds an external mesh loader.
		virtual void addExternalMeshLoader(IMeshLoader* externalLoader);

		//! Returns the number of mesh loaders supported by Irrlicht at this time
		virtual u32 getMeshLoaderCount() const;

		//! Retrieve the given mesh loader
		virtual IMeshLoader* getMeshLoader(u32 index) const;

		//! Returns a pointer to the scene collision manager.
		virtual ISceneCollisionManager* getSceneCollisionManager();

		//! Returns a pointer to the mesh manipulator.
		virtual IMeshManipulator* getMeshManipulator();

		//! Adds a scene node to the deletion queue.
		virtual void addToDeletionQueue(ISceneNode* node);

		//! Returns the first scene node with the specified id.
		virtual ISceneNode* getSceneNodeFromId(s32 id, ISceneNode* start=0);

		//! Returns the first scene node with the specified name.
		virtual ISceneNode* getSceneNodeFromName(const c8* name, ISceneNode* start=0);

		//! Returns the first scene node with the specified type.
		virtual ISceneNode* getSceneNodeFromType(scene::ESCENE_NODE_TYPE type, ISceneNode* start=0);

		//! returns scene nodes by type.
		virtual void getSceneNodesFromType(ESCENE_NODE_TYPE type, core::array<scene::ISceneNode*>& outNodes, ISceneNode* start=0);

		//! Posts an input event to the environment. Usually you do not have to
		//! use this method, it is used by the internal engine.
		virtual bool postEventFromUser(const SEvent& event);

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

		//! Returns the default scene node factory which can create all built in scene nodes
		virtual ISceneNodeFactory* getDefaultSceneNodeFactory();

		//! Adds a scene node factory to the scene manager.
		/** Use this to extend the scene manager with new scene node types which it should be
		able to create automaticly, for example when loading data from xml files. */
		virtual void registerSceneNodeFactory(ISceneNodeFactory* factoryToAdd);

		//! Returns amount of registered scene node factories.
		virtual u32 getRegisteredSceneNodeFactoryCount() const;

		//! Returns a scene node factory by index
		virtual ISceneNodeFactory* getSceneNodeFactory(u32 index);

		//! Returns a typename from a scene node type or null if not found
		virtual const c8* getSceneNodeTypeName(ESCENE_NODE_TYPE type);

		//! Returns a typename from a scene node animator type or null if not found
		virtual const c8* getAnimatorTypeName(ESCENE_NODE_ANIMATOR_TYPE type);

		//! Adds a scene node to the scene by name
		virtual ISceneNode* addSceneNode(const char* sceneNodeTypeName, ISceneNode* parent=0);

		//! creates a scene node animator based on its type name
		virtual ISceneNodeAnimator* createSceneNodeAnimator(const char* typeName, ISceneNode* target=0);

		//! Returns the default scene node animator factory which can create all built-in scene node animators
		virtual ISceneNodeAnimatorFactory* getDefaultSceneNodeAnimatorFactory();

		//! Adds a scene node animator factory to the scene manager.
		virtual void registerSceneNodeAnimatorFactory(ISceneNodeAnimatorFactory* factoryToAdd);

		//! Returns amount of registered scene node animator factories.
		virtual u32 getRegisteredSceneNodeAnimatorFactoryCount() const;

		//! Returns a scene node animator factory by index
		virtual ISceneNodeAnimatorFactory* getSceneNodeAnimatorFactory(u32 index);

		//! Returns a mesh writer implementation if available
		virtual IMeshWriter* createMeshWriter(EMESH_WRITER_TYPE type);

		//! Sets ambient color of the scene
		virtual void setAmbientLight(const video::SColorf &ambientColor);

		//! Returns ambient color of the scene
		virtual const video::SColorf& getAmbientLight() const;

		//! Register a custom callbacks manager which gets callbacks during scene rendering.
		virtual void setLightManager(ILightManager* lightManager);

		//! Get current render time.
		virtual E_SCENE_NODE_RENDER_PASS getCurrentRendertime() const { return CurrentRendertime; }

		//! Set current render time.
		virtual void setCurrentRendertime(E_SCENE_NODE_RENDER_PASS currentRendertime) { CurrentRendertime = currentRendertime; }

		//! Get an instance of a geometry creator.
		virtual const IGeometryCreator* getGeometryCreator(void) const { return GeometryCreator; }

		//! returns if node is culled
		virtual bool isCulled(const ISceneNode* node) const;

	protected:

		//! clears the deletion list
		void clearDeletionList();

		//! writes a scene node
		void writeSceneNode(io::IXMLWriter* writer, ISceneNode* node, ISceneUserDataSerializer* userDataSerializer, const fschar_t* currentPath=0, bool init=false);

		struct DefaultNodeEntry
		{
			DefaultNodeEntry(ISceneNode* n) :
				Node(n), TextureValue(0)
			{
				if (n->getMaterialCount())
					TextureValue = (n->getMaterial(0).getTexture(0));
			}

			bool operator < (const DefaultNodeEntry& other) const
			{
				return (TextureValue < other.TextureValue);
			}

			ISceneNode* Node;
			private:
			void* TextureValue;
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
				f64 Distance;
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
			f64 Distance;
		};

		//! video driver
		video::IVideoDriver* Driver;

		//! file system
		io::IFileSystem* FileSystem;

		//! cursor control
		gui::ICursorControl* CursorControl;

		//! collision manager
		ISceneCollisionManager* CollisionManager;

		//! render pass lists
		core::array<ISceneNode*> CameraList;
		core::array<ISceneNode*> LightList;
		core::array<ISceneNode*> SkyBoxList;
		core::array<DefaultNodeEntry> SolidNodeList;
		core::array<TransparentNodeEntry> TransparentNodeList;
		core::array<TransparentNodeEntry> TransparentEffectNodeList;

		core::array<IMeshLoader*> MeshLoaderList;
		core::array<ISceneNode*> DeletionList;
		core::array<ISceneNodeFactory*> SceneNodeFactoryList;
		core::array<ISceneNodeAnimatorFactory*> SceneNodeAnimatorFactoryList;

		//! current active camera
		ICameraSceneNode* ActiveCamera;
		core::vector3df camWorldPos; // Position of camera for transparent nodes.

		video::SColorf AmbientLight;

        struct ParamStorage
        {
            uint8_t data[16];
        };
		std::map<std::string,ParamStorage> Parameters;

		//! Mesh cache
		IMeshCache<ICPUMesh>* MeshCache;
		video::IGPUBuffer* redundantMeshDataBuf;

		E_SCENE_NODE_RENDER_PASS CurrentRendertime;

		//! An optional callbacks manager to allow the user app finer control
		//! over the scene lighting and rendering.
		ILightManager* LightManager;

		//! constants for reading and writing XML.
		//! Not made static due to portability problems.
		const core::stringw IRR_XML_FORMAT_SCENE;
		const core::stringw IRR_XML_FORMAT_NODE;
		const core::stringw IRR_XML_FORMAT_NODE_ATTR_TYPE;

		IGeometryCreator* GeometryCreator;
		CMeshManipulator* MeshManipulator;
	};

} // end namespace video
} // end namespace scene

#endif

