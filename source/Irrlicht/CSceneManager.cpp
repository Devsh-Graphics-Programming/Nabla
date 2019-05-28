// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"
#include "CSceneManager.h"
#include "IVideoDriver.h"
#include "IFileSystem.h"
#include "IMaterialRenderer.h"
#include "IReadFile.h"
#include "IWriteFile.h"

#include "os.h"

// We need this include for the case of skinned mesh support without
// any such loader
#include "CSkinnedMeshSceneNode.h"
#include "irr/video/CGPUSkinnedMesh.h"

#include "CBillboardSceneNode.h"
#include "CCubeSceneNode.h"
#include "CSphereSceneNode.h"
#include "CCameraSceneNode.h"
#include "CMeshSceneNode.h"
#include "CMeshSceneNodeInstanced.h"
#include "CSkyBoxSceneNode.h"
#include "CSkyDomeSceneNode.h"

#include "CSceneNodeAnimatorRotation.h"
#include "CSceneNodeAnimatorFlyCircle.h"
#include "CSceneNodeAnimatorFlyStraight.h"
#include "CSceneNodeAnimatorTexture.h"
#include "CSceneNodeAnimatorDelete.h"
#include "CSceneNodeAnimatorFollowSpline.h"
#include "CSceneNodeAnimatorCameraFPS.h"
#include "CSceneNodeAnimatorCameraMaya.h"

#include "irr/asset/CGeometryCreator.h"

namespace irr
{
namespace scene
{

//! constructor
CSceneManager::CSceneManager(IrrlichtDevice* device, video::IVideoDriver* driver, irr::ITimer* timer, io::IFileSystem* fs,
		gui::ICursorControl* cursorControl)
: ISceneNode(0, 0), Driver(driver), Timer(timer), FileSystem(fs), Device(device),
	CursorControl(cursorControl),
	ActiveCamera(0), CurrentRendertime(ESNRP_NONE),
	IRR_XML_FORMAT_SCENE(L"irr_scene"), IRR_XML_FORMAT_NODE(L"node"), IRR_XML_FORMAT_NODE_ATTR_TYPE(L"type")
{
	#ifdef _IRR_DEBUG
	ISceneManager::setDebugName("CSceneManager ISceneManager");
	ISceneNode::setDebugName("CSceneManager ISceneNode");
	#endif

	// root node's scene manager
	SceneManager = this;

	if (Driver)
		Driver->grab();

	if (FileSystem)
		FileSystem->grab();

	if (CursorControl)
		CursorControl->grab();

	{
        size_t redundantMeshDataBufSize = sizeof(char)*24*3+ //data for the skybox positions
                                        0;
        void* tmpMem = _IRR_ALIGNED_MALLOC(redundantMeshDataBufSize,_IRR_SIMD_ALIGNMENT);
        {
            char* skyBoxesVxPositions = (char*)tmpMem;
            skyBoxesVxPositions[0*3+0] = -1;
            skyBoxesVxPositions[0*3+1] = -1;
            skyBoxesVxPositions[0*3+2] = -1;

            skyBoxesVxPositions[1*3+0] = 1;
            skyBoxesVxPositions[1*3+1] =-1;
            skyBoxesVxPositions[1*3+2] =-1;

            skyBoxesVxPositions[2*3+0] = 1;
            skyBoxesVxPositions[2*3+1] = 1;
            skyBoxesVxPositions[2*3+2] =-1;

            skyBoxesVxPositions[3*3+0] =-1;
            skyBoxesVxPositions[3*3+1] = 1;
            skyBoxesVxPositions[3*3+2] =-1;

            // create left side
            skyBoxesVxPositions[4*3+0] = 1;
            skyBoxesVxPositions[4*3+1] =-1;
            skyBoxesVxPositions[4*3+2] =-1;

            skyBoxesVxPositions[5*3+0] = 1;
            skyBoxesVxPositions[5*3+1] =-1;
            skyBoxesVxPositions[5*3+2] = 1;

            skyBoxesVxPositions[6*3+0] = 1;
            skyBoxesVxPositions[6*3+1] = 1;
            skyBoxesVxPositions[6*3+2] = 1;

            skyBoxesVxPositions[7*3+0] = 1;
            skyBoxesVxPositions[7*3+1] = 1;
            skyBoxesVxPositions[7*3+2] =-1;

            // create back side
            skyBoxesVxPositions[8*3+0] = 1;
            skyBoxesVxPositions[8*3+1] =-1;
            skyBoxesVxPositions[8*3+2] = 1;

            skyBoxesVxPositions[9*3+0] =-1;
            skyBoxesVxPositions[9*3+1] =-1;
            skyBoxesVxPositions[9*3+2] = 1;

            skyBoxesVxPositions[10*3+0] =-1;
            skyBoxesVxPositions[10*3+1] = 1;
            skyBoxesVxPositions[10*3+2] = 1;

            skyBoxesVxPositions[11*3+0] = 1;
            skyBoxesVxPositions[11*3+1] = 1;
            skyBoxesVxPositions[11*3+2] = 1;

            // create right side
            skyBoxesVxPositions[12*3+0] =-1;
            skyBoxesVxPositions[12*3+1] =-1;
            skyBoxesVxPositions[12*3+2] = 1;

            skyBoxesVxPositions[13*3+0] =-1;
            skyBoxesVxPositions[13*3+1] =-1;
            skyBoxesVxPositions[13*3+2] =-1;

            skyBoxesVxPositions[14*3+0] =-1;
            skyBoxesVxPositions[14*3+1] = 1;
            skyBoxesVxPositions[14*3+2] =-1;

            skyBoxesVxPositions[15*3+0] =-1;
            skyBoxesVxPositions[15*3+1] = 1;
            skyBoxesVxPositions[15*3+2] = 1;

            // create top side
            skyBoxesVxPositions[16*3+0] = 1;
            skyBoxesVxPositions[16*3+1] = 1;
            skyBoxesVxPositions[16*3+2] =-1;

            skyBoxesVxPositions[17*3+0] = 1;
            skyBoxesVxPositions[17*3+1] = 1;
            skyBoxesVxPositions[17*3+2] = 1;

            skyBoxesVxPositions[18*3+0] =-1;
            skyBoxesVxPositions[18*3+1] = 1;
            skyBoxesVxPositions[18*3+2] = 1;

            skyBoxesVxPositions[19*3+0] =-1;
            skyBoxesVxPositions[19*3+1] = 1;
            skyBoxesVxPositions[19*3+2] =-1;

            // create bottom side
            skyBoxesVxPositions[20*3+0] = 1;
            skyBoxesVxPositions[20*3+1] =-1;
            skyBoxesVxPositions[20*3+2] = 1;

            skyBoxesVxPositions[21*3+0] = 1;
            skyBoxesVxPositions[21*3+1] =-1;
            skyBoxesVxPositions[21*3+2] =-1;

            skyBoxesVxPositions[22*3+0] =-1;
            skyBoxesVxPositions[22*3+1] =-1;
            skyBoxesVxPositions[22*3+2] =-1;

            skyBoxesVxPositions[23*3+0] =-1;
            skyBoxesVxPositions[23*3+1] =-1;
            skyBoxesVxPositions[23*3+2] = 1;
        }
        video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
        reqs.vulkanReqs.size = redundantMeshDataBufSize;
        reqs.vulkanReqs.alignment = 4;
        reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
        reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
        reqs.mappingCapability = video::IDriverMemoryAllocation::EMCAF_NO_MAPPING_ACCESS;
        reqs.prefersDedicatedAllocation = true;
        reqs.requiresDedicatedAllocation = true;
        redundantMeshDataBuf = SceneManager->getVideoDriver()->createGPUBufferOnDedMem(reqs,true);
        if (redundantMeshDataBuf)
            redundantMeshDataBuf->updateSubRange(video::IDriverMemoryAllocation::MemoryRange(0,reqs.vulkanReqs.size),tmpMem);
        _IRR_ALIGNED_FREE(tmpMem);
	}
}


//! destructor
CSceneManager::~CSceneManager()
{
	clearDeletionList();

	//! force to remove hardwareTextures from the driver
	//! because Scenes may hold internally data bounded to sceneNodes
	//! which may be destroyed twice
    if (redundantMeshDataBuf)
        redundantMeshDataBuf->drop();
	if (FileSystem)
		FileSystem->drop();

	if (CursorControl)
		CursorControl->drop();

	if (ActiveCamera)
		ActiveCamera->drop();
	ActiveCamera = 0;

	// remove all nodes and animators before dropping the driver
	// as render targets may be destroyed twice

	removeAll();
	removeAnimators();

	if (Driver)
		Driver->drop();
}


//! returns the video driver
video::IVideoDriver* CSceneManager::getVideoDriver()
{
	return Driver;
}

//! Get the active FileSystem
/** \return Pointer to the FileSystem
This pointer should not be dropped. See IReferenceCounted::drop() for more information. */
io::IFileSystem* CSceneManager::getFileSystem()
{
	return FileSystem;
}

IrrlichtDevice * CSceneManager::getDevice()
{
    return Device;
}

//! adds a test scene node for test purposes to the scene. It is a simple cube of (1,1,1) size.
//! the returned pointer must not be dropped.
IMeshSceneNode* CSceneManager::addCubeSceneNode(float size, IDummyTransformationSceneNode* parent,
		int32_t id, const core::vector3df& position,
		const core::vector3df& rotation, const core::vector3df& scale)
{
	if (!parent)
		parent = this;

	IMeshSceneNode* node = new CCubeSceneNode(size, parent, this, id, position, rotation, scale);
	node->drop();

	return node;
}


//! Adds a sphere scene node for test purposes to the scene.
IMeshSceneNode* CSceneManager::addSphereSceneNode(float radius, int32_t polyCount,
		IDummyTransformationSceneNode* parent, int32_t id, const core::vector3df& position,
		const core::vector3df& rotation, const core::vector3df& scale)
{
	if (!parent)
		parent = this;

	IMeshSceneNode* node = new CSphereSceneNode(radius, polyCount, polyCount, parent, this, id, position, rotation, scale);
	node->drop();

	return node;
}

//! adds a scene node for rendering a static mesh
//! the returned pointer must not be dropped.
IMeshSceneNode* CSceneManager::addMeshSceneNode(video::IGPUMesh* mesh, IDummyTransformationSceneNode* parent, int32_t id,
	const core::vector3df& position, const core::vector3df& rotation,
	const core::vector3df& scale, bool alsoAddIfMeshPointerZero)
{
	if (!alsoAddIfMeshPointerZero && !mesh)
		return 0;

	if (!parent)
		parent = this;

	IMeshSceneNode* node = new CMeshSceneNode(mesh, parent, this, id, position, rotation, scale);
	node->drop();

	return node;
}

IMeshSceneNodeInstanced* CSceneManager::addMeshSceneNodeInstanced(IDummyTransformationSceneNode* parent, int32_t id,
    const core::vector3df& position, const core::vector3df& rotation, const core::vector3df& scale)
{
	if (!parent)
		parent = this;

	CMeshSceneNodeInstanced* node = new CMeshSceneNodeInstanced(parent, this, id, position, rotation, scale);
	node->drop();

	return node;
}

//! adds a scene node for rendering an animated mesh model
ISkinnedMeshSceneNode* CSceneManager::addSkinnedMeshSceneNode(
    video::IGPUSkinnedMesh* mesh, const ISkinningStateManager::E_BONE_UPDATE_MODE& boneControlMode,
    IDummyTransformationSceneNode* parent, int32_t id,
    const core::vector3df& position, const core::vector3df& rotation, const core::vector3df& scale)
{
	if (!mesh)
		return 0;

	if (!parent)
		parent = this;

	ISkinnedMeshSceneNode* node =
		new CSkinnedMeshSceneNode(mesh, boneControlMode, parent, this, id, position, rotation, scale);
	node->drop();

	return node;
}

//! Adds a camera scene node to the tree and sets it as active camera.
//! \param position: Position of the space relative to its parent where the camera will be placed.
//! \param lookat: Position where the camera will look at. Also known as target.
//! \param parent: Parent scene node of the camera. Can be null. If the parent moves,
//! the camera will move too.
//! \return Returns pointer to interface to camera
ICameraSceneNode* CSceneManager::addCameraSceneNode(IDummyTransformationSceneNode* parent,
	const core::vector3df& position, const core::vector3df& lookat, int32_t id,
	bool makeActive)
{
	if (!parent)
		parent = this;

	ICameraSceneNode* node = new CCameraSceneNode(parent, this, id, position, lookat);

	if (makeActive)
		setActiveCamera(node);
	node->drop();

	return node;
}


//! Adds a camera scene node which is able to be controlled with the mouse similar
//! to in the 3D Software Maya by Alias Wavefront.
//! The returned pointer must not be dropped.
ICameraSceneNode* CSceneManager::addCameraSceneNodeMaya(IDummyTransformationSceneNode* parent,
	float rotateSpeed, float zoomSpeed, float translationSpeed, int32_t id, float distance,
	bool makeActive)
{
	ICameraSceneNode* node = addCameraSceneNode(parent, core::vector3df(),
			core::vector3df(0,0,100), id, makeActive);
	if (node)
	{
		ISceneNodeAnimator* anm = new CSceneNodeAnimatorCameraMaya(CursorControl,
			rotateSpeed, zoomSpeed, translationSpeed, distance);

		node->addAnimator(anm);
		anm->drop();
	}

	return node;
}


//! Adds a camera scene node which is able to be controlled with the mouse and keys
//! like in most first person shooters (FPS):
ICameraSceneNode* CSceneManager::addCameraSceneNodeFPS(IDummyTransformationSceneNode* parent,
	float rotateSpeed, float moveSpeed, int32_t id, SKeyMap* keyMapArray,
	int32_t keyMapSize, bool noVerticalMovement, float jumpSpeed,
	bool invertMouseY, bool makeActive)
{
	ICameraSceneNode* node = addCameraSceneNode(parent, core::vector3df(),
			core::vector3df(0,0,100), id, makeActive);
	if (node)
	{
		ISceneNodeAnimator* anm = new CSceneNodeAnimatorCameraFPS(CursorControl,
				rotateSpeed, moveSpeed, jumpSpeed,
				keyMapArray, keyMapSize, noVerticalMovement, invertMouseY);

		// Bind the node's rotation to its target. This is consistent with 1.4.2 and below.
		node->bindTargetAndRotation(true);
		node->addAnimator(anm);
		anm->drop();
	}

	return node;
}




//! Adds a billboard scene node to the scene. A billboard is like a 3d sprite: A 2d element,
//! which always looks to the camera. It is usually used for things like explosions, fire,
//! lensflares and things like that.
IBillboardSceneNode* CSceneManager::addBillboardSceneNode(IDummyTransformationSceneNode* parent,
	const core::dimension2d<float>& size, const core::vector3df& position, int32_t id,
	video::SColor colorTop, video::SColor colorBottom)
{
	if (!parent)
		parent = this;

	IBillboardSceneNode* node = new CBillboardSceneNode(parent, this, id, position, size,
		colorTop, colorBottom);
	node->drop();

	return node;
}

//! Adds a skybox scene node. A skybox is a big cube with 6 textures on it and
//! is drawn around the camera position.
ISceneNode* CSceneManager::addSkyBoxSceneNode(video::ITexture* top, video::ITexture* bottom,
	video::ITexture* left, video::ITexture* right, video::ITexture* front,
	video::ITexture* back, IDummyTransformationSceneNode* parent, int32_t id)
{
	if (!parent)
		parent = this;

	ISceneNode* node = new CSkyBoxSceneNode(top, bottom, left, right,
			front, back, redundantMeshDataBuf,0, parent, this, id);

	node->drop();
	return node;
}


//! Adds a skydome scene node. A skydome is a large (half-) sphere with a
//! panoramic texture on it and is drawn around the camera position.
ISceneNode* CSceneManager::addSkyDomeSceneNode(video::IVirtualTexture* texture,
	uint32_t horiRes, uint32_t vertRes, float texturePercentage,float spherePercentage, float radius,
	IDummyTransformationSceneNode* parent, int32_t id)
{
	if (!parent)
		parent = this;

	ISceneNode* node = new CSkyDomeSceneNode(texture, horiRes, vertRes,
		texturePercentage, spherePercentage, radius, parent, this, id);

	node->drop();
	return node;
}

//! Adds a dummy transformation scene node to the scene tree.
IDummyTransformationSceneNode* CSceneManager::addDummyTransformationSceneNode(
	IDummyTransformationSceneNode* parent, int32_t id)
{
	if (!parent)
		parent = this;

	IDummyTransformationSceneNode* node = new IDummyTransformationSceneNode(parent);
	node->drop();

	return node;
}

//! Returns the root scene node. This is the scene node wich is parent
//! of all scene nodes. The root scene node is a special scene node which
//! only exists to manage all scene nodes. It is not rendered and cannot
//! be removed from the scene.
//! \return Returns a pointer to the root scene node.
ISceneNode* CSceneManager::getRootSceneNode()
{
	return this;
}


//! Returns the current active camera.
//! \return The active camera is returned. Note that this can be NULL, if there
//! was no camera created yet.
ICameraSceneNode* CSceneManager::getActiveCamera() const
{
	return ActiveCamera;
}


//! Sets the active camera. The previous active camera will be deactivated.
//! \param camera: The new camera which should be active.
void CSceneManager::setActiveCamera(ICameraSceneNode* camera)
{
	if (camera)
		camera->grab();
	if (ActiveCamera)
		ActiveCamera->drop();

	ActiveCamera = camera;
}


//! renders the node.
void CSceneManager::render()
{
}


//! returns the axis aligned bounding box of this node
const core::aabbox3d<float>& CSceneManager::getBoundingBox()
{
	_IRR_DEBUG_BREAK_IF(true) // Bounding Box of Scene Manager wanted.

	// should never be used.
	return *((core::aabbox3d<float>*)0);
}


//! returns if node is culled
bool CSceneManager::isCulled(ISceneNode* node) const
{
	const ICameraSceneNode* cam = getActiveCamera();
	if (!cam)
	{
		return false;
	}

    core::aabbox3d<float> tbox = node->getBoundingBox();
    if (tbox.MinEdge==tbox.MaxEdge)
        return true;

    auto cullMode = node->getAutomaticCulling();
    if (cullMode & (scene::EAC_BOX|scene::EAC_FRUSTUM_BOX))
    {
		node->getAbsoluteTransformation().transformBoxEx(tbox);
        // can be seen by a bounding box ?
        if ((cullMode & scene::EAC_BOX) && !tbox.intersectsWithBox(cam->getViewFrustum()->getBoundingBox()))
            return true;
        // can be seen by cam pyramid planes ?
        if ((cullMode & scene::EAC_FRUSTUM_BOX) && !cam->getViewFrustum()->intersectsAABB(tbox))
            return true;
	}

	return false;
}


//! registers a node for rendering it at a specific time.
uint32_t CSceneManager::registerNodeForRendering(ISceneNode* node, E_SCENE_NODE_RENDER_PASS pass)
{
	uint32_t taken = 0;

	switch(pass)
	{
		// take camera if it is not already registered
	case ESNRP_CAMERA:
		{
			taken = 1;
			for (uint32_t i = 0; i != CameraList.size(); ++i)
			{
				if (CameraList[i] == node)
				{
					taken = 0;
					break;
				}
			}
			if (taken)
			{
				CameraList.push_back(node);
			}
		}
		break;

	case ESNRP_SKY_BOX:
		SkyBoxList.push_back(node);
		taken = 1;
		break;
	case ESNRP_SOLID:
		if (!isCulled(node))
		{
			SolidNodeList.push_back(node);
			taken = 1;
		}
		break;
	case ESNRP_TRANSPARENT:
		if (!isCulled(node))
		{
			TransparentNodeList.push_back(TransparentNodeEntry(node, ActiveCamera->getAbsolutePosition()));
			taken = 1;
		}
		break;
	case ESNRP_TRANSPARENT_EFFECT:
		if (!isCulled(node))
		{
			TransparentEffectNodeList.push_back(TransparentNodeEntry(node, ActiveCamera->getAbsolutePosition()));
			taken = 1;
		}
		break;
	case ESNRP_AUTOMATIC:
		if (!isCulled(node))
		{
			const uint32_t count = node->getMaterialCount();

			taken = 0;
			for (uint32_t i=0; i<count; ++i)
			{
				video::IMaterialRenderer* rnd =
					Driver->getMaterialRenderer(node->getMaterial(i).MaterialType);
				if (rnd && rnd->isTransparent())
				{
					// register as transparent node
					TransparentNodeEntry e(node, ActiveCamera->getAbsolutePosition());
					TransparentNodeList.push_back(e);
					taken = 1;
					break;
				}
			}

			// not transparent, register as solid
			if (!taken)
			{
				SolidNodeList.push_back(node);
				taken = 1;
			}
		}
		break;

	default: // ignore this one
		break;
	}

#ifdef _IRR_SCENEMANAGER_DEBUG
	int32_t index = Parameters.findAttribute ( "calls" );
	Parameters.setAttribute ( index, Parameters.getAttributeAsInt ( index ) + 1 );

	if (!taken)
	{
		index = Parameters.findAttribute ( "culled" );
		Parameters.setAttribute ( index, Parameters.getAttributeAsInt ( index ) + 1 );
	}
#endif

	return taken;
}

//!
void CSceneManager::OnAnimate(uint32_t timeMs)
{
    size_t prevSize = Children.size();
    for (size_t i=0; i<prevSize;)
    {
        IDummyTransformationSceneNode* tmpChild = Children[i];
        if (tmpChild->isISceneNode())
            static_cast<ISceneNode*>(tmpChild)->OnAnimate(timeMs);
        else
            OnAnimate_static(tmpChild,timeMs);

        if (Children[i]>tmpChild)
            prevSize = Children.size();
        else
            i++;
    }
}

//! This method is called just before the rendering process of the whole scene.
//! draws all scene nodes
void CSceneManager::drawAll()
{
	if (!Driver)
		return;

#ifdef _IRR_SCENEMANAGER_DEBUG
	// reset attributes
	Parameters.setAttribute ( "culled", 0 );
	Parameters.setAttribute ( "calls", 0 );
	Parameters.setAttribute ( "drawn_solid", 0 );
	Parameters.setAttribute ( "drawn_transparent", 0 );
	Parameters.setAttribute ( "drawn_transparent_effect", 0 );
#endif

	uint32_t i; // new ISO for scoping problem in some compilers

	// reset all transforms
	Driver->setMaterial(video::SGPUMaterial());
	Driver->setTransform(video::EPTS_PROJ,core::matrix4SIMD());
	Driver->setTransform ( video::E4X3TS_VIEW, core::matrix4x3() );
	Driver->setTransform ( video::E4X3TS_WORLD, core::matrix4x3() );

	// TODO: This should not use an attribute here but a real parameter when necessary (too slow!)
	Driver->setAllowZWriteOnTransparent( *((bool*)&(Parameters[ALLOW_ZWRITE_ON_TRANSPARENT])) );

	// do animations and other stuff.
	OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(Timer->getTime()).count());

	/*!
		First Scene Node for prerendering should be the active camera
		consistent Camera is needed for culling
	*/
	if (ActiveCamera)
	{
		ActiveCamera->render();
	}

	// let all nodes register themselves
	OnRegisterSceneNode();

	//render camera scenes
	{
		CurrentRendertime = ESNRP_CAMERA;

		for (i=0; i<CameraList.size(); ++i)
			CameraList[i]->render();

		CameraList.clear();
	}

	// render skyboxes
	{
		CurrentRendertime = ESNRP_SKY_BOX;

        for (i=0; i<SkyBoxList.size(); ++i)
            SkyBoxList[i]->render();

		SkyBoxList.clear();
	}


	// render default objects
	{
		CurrentRendertime = ESNRP_SOLID;

		std::sort(SolidNodeList.begin(),SolidNodeList.end()); // sort by textures

        for (i=0; i<SolidNodeList.size(); ++i)
            SolidNodeList[i].Node->render();

#ifdef _IRR_SCENEMANAGER_DEBUG
		Parameters.setAttribute("drawn_solid", (int32_t) SolidNodeList.size() );
#endif
		SolidNodeList.clear();
	}

	// render transparent objects.
	{
		CurrentRendertime = ESNRP_TRANSPARENT;

		std::sort(TransparentNodeList.begin(),TransparentNodeList.end()); // sort by distance from camera
        for (i=0; i<TransparentNodeList.size(); ++i)
            TransparentNodeList[i].Node->render();

#ifdef _IRR_SCENEMANAGER_DEBUG
		Parameters.setAttribute ( "drawn_transparent", (int32_t) TransparentNodeList.size() );
#endif
		TransparentNodeList.clear();
	}

	// render transparent effect objects.
	{
		CurrentRendertime = ESNRP_TRANSPARENT_EFFECT;

		std::sort(TransparentEffectNodeList.begin(),TransparentEffectNodeList.end()); // sort by distance from camera
        for (i=0; i<TransparentEffectNodeList.size(); ++i)
            TransparentEffectNodeList[i].Node->render();
#ifdef _IRR_SCENEMANAGER_DEBUG
		Parameters.setAttribute ( "drawn_transparent_effect", (int32_t) TransparentEffectNodeList.size() );
#endif
		TransparentEffectNodeList.clear();
	}

	LightList.clear();
	clearDeletionList();

	CurrentRendertime = ESNRP_NONE;
}

//! creates a rotation animator, which rotates the attached scene node around itself.
ISceneNodeAnimator* CSceneManager::createRotationAnimator(const core::vector3df& rotationPerSecond)
{
	ISceneNodeAnimator* anim = new CSceneNodeAnimatorRotation(std::chrono::duration_cast<std::chrono::milliseconds>(Timer->getTime()).count(),
		rotationPerSecond);

	return anim;
}


//! creates a fly circle animator, which lets the attached scene node fly around a center.
ISceneNodeAnimator* CSceneManager::createFlyCircleAnimator(
		const core::vector3df& center, float radius, float speed,
		const core::vector3df& direction,
		float startPosition,
		float radiusEllipsoid)
{
	const float orbitDurationMs = (core::DEGTORAD * 360.f) / speed;
	const uint32_t effectiveTime = std::chrono::duration_cast<std::chrono::milliseconds>(Timer->getTime()).count() + (uint32_t)(orbitDurationMs * startPosition);

	ISceneNodeAnimator* anim = new CSceneNodeAnimatorFlyCircle(
			effectiveTime, center,
			radius, speed, direction,radiusEllipsoid);
	return anim;
}


//! Creates a fly straight animator, which lets the attached scene node
//! fly or move along a line between two points.
ISceneNodeAnimator* CSceneManager::createFlyStraightAnimator(const core::vector3df& startPoint,
					const core::vector3df& endPoint, uint32_t timeForWay, bool loop,bool pingpong)
{
	ISceneNodeAnimator* anim = new CSceneNodeAnimatorFlyStraight(startPoint,
		endPoint, timeForWay, loop, std::chrono::duration_cast<std::chrono::milliseconds>(Timer->getTime()).count(), pingpong);

	return anim;
}


//! Creates a texture animator, which switches the textures of the target scene
//! node based on a list of textures.
ISceneNodeAnimator* CSceneManager::createTextureAnimator(const core::vector<video::ITexture*>& textures,
	int32_t timePerFrame, bool loop)
{
	ISceneNodeAnimator* anim = new CSceneNodeAnimatorTexture(textures,
		timePerFrame, loop, std::chrono::duration_cast<std::chrono::milliseconds>(Timer->getTime()).count());

	return anim;
}


//! Creates a scene node animator, which deletes the scene node after
//! some time automaticly.
ISceneNodeAnimator* CSceneManager::createDeleteAnimator(uint32_t when)
{
	return new CSceneNodeAnimatorDelete(this, std::chrono::duration_cast<std::chrono::milliseconds>(Timer->getTime()).count() + when);
}


//! Creates a follow spline animator.
ISceneNodeAnimator* CSceneManager::createFollowSplineAnimator(int32_t startTime,
	const core::vector< core::vector3df >& points,
	float speed, float tightness, bool loop, bool pingpong)
{
	ISceneNodeAnimator* a = new CSceneNodeAnimatorFollowSpline(startTime, points,
		speed, tightness, loop, pingpong);
	return a;
}


//! Adds a scene node to the deletion queue.
void CSceneManager::addToDeletionQueue(IDummyTransformationSceneNode* node)
{
	if (!node)
		return;

	node->grab();
	DeletionList.push_back(node);
}


//! clears the deletion list
void CSceneManager::clearDeletionList()
{
	if (DeletionList.empty())
		return;

	for (uint32_t i=0; i<DeletionList.size(); ++i)
	{
		DeletionList[i]->remove();
		DeletionList[i]->drop();
	}

	DeletionList.clear();
}

/*
//! Returns the first scene node with the specified name.
ISceneNode* CSceneManager::getSceneNodeFromName(const char* name, IDummyTransformationSceneNode* start)
{
	if (start == 0)
		start = getRootSceneNode();

	if (!strcmp(start->getName(),name))
		return start;

	IDummyTransformationSceneNode* node = 0;

	const IDummyTransformationSceneNodeArray& list = start->getChildren();
	IDummyTransformationSceneNodeArray::ConstIterator it = list.begin();
	for (; it!=list.end(); ++it)
	{
		node = getSceneNodeFromName(name, *it);
		if (node)
			return node;
	}

	return 0;
}


//! Returns the first scene node with the specified id.
ISceneNode* CSceneManager::getSceneNodeFromId(int32_t id, IDummyTransformationSceneNode* start)
{
	if (start == 0)
		start = getRootSceneNode();

	if (start->getID() == id)
		return start;

	ISceneNode* node = 0;

	const IDummyTransformationSceneNodeArray& list = start->getChildren();
	IDummyTransformationSceneNodeArray::ConstIterator it = list.begin();
	for (; it!=list.end(); ++it)
	{
		node = getSceneNodeFromId(id, *it);
		if (node)
			return node;
	}

	return 0;
}


//! Returns the first scene node with the specified type.
ISceneNode* CSceneManager::getSceneNodeFromType(scene::ESCENE_NODE_TYPE type, IDummyTransformationSceneNode* start)
{
	if (start == 0)
		start = getRootSceneNode();

	if (start->getType() == type || ESNT_ANY == type)
		return start;

	ISceneNode* node = 0;

	const IDummyTransformationSceneNodeArray& list = start->getChildren();
	IDummyTransformationSceneNodeArray::ConstIterator it = list.begin();
	for (; it!=list.end(); ++it)
	{
		node = getSceneNodeFromType(type, *it);
		if (node)
			return node;
	}

	return 0;
}


//! returns scene nodes by type.
void CSceneManager::getSceneNodesFromType(ESCENE_NODE_TYPE type, core::vector<scene::ISceneNode*>& outNodes, IDummyTransformationSceneNode* start)
{
	if (start == 0)
		start = getRootSceneNode();

	if (start->getType() == type || ESNT_ANY == type)
		outNodes.push_back(start);

	const IDummyTransformationSceneNodeArray& list = start->getChildren();
	IDummyTransformationSceneNodeArray::ConstIterator it = list.begin();

	for (; it!=list.end(); ++it)
	{
		getSceneNodesFromType(type, outNodes, *it);
	}
}
*/

//! Posts an input event to the environment. Usually you do not have to
//! use this method, it is used by the internal engine.
bool CSceneManager::receiveIfEventReceiverDidNotAbsorb(const SEvent& event)
{
	bool ret = false;
	ICameraSceneNode* cam = getActiveCamera();
	if (cam)
		ret = cam->OnEvent(event);

	return ret;
}


//! Removes all children of this scene node
void CSceneManager::removeAll()
{
	ISceneNode::removeAll();
	setActiveCamera(0);
	// Make sure the driver is reset, might need a more complex method at some point
	if (Driver)
		Driver->setMaterial(video::SGPUMaterial());
}


//! Clears the whole scene. All scene nodes are removed.
void CSceneManager::clear()
{
	removeAll();
}


//! Returns current render pass.
E_SCENE_NODE_RENDER_PASS CSceneManager::getSceneNodeRenderPass() const
{
	return CurrentRendertime;
}

//! Creates a new scene manager.
ISceneManager* CSceneManager::createNewSceneManager(bool cloneContent)
{
    CSceneManager* manager = new CSceneManager(Device, Driver, Timer, FileSystem, CursorControl);

    if (cloneContent)
        manager->cloneMembers(this, manager);

    return manager;
}


} // end namespace scene
} // end namespace irr

