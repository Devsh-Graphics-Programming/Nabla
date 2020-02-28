// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"
#include "CSceneManager.h"
#include "IVideoDriver.h"
#include "IFileSystem.h"
#include "IReadFile.h"
#include "IWriteFile.h"
#include "IrrlichtDevice.h"

#include "os.h"

// We need this include for the case of skinned mesh support without
// any such loader
#include "CSkinnedMeshSceneNode.h"
#include "irr/video/CGPUSkinnedMesh.h"

#include "CCameraSceneNode.h"
#include "CMeshSceneNode.h"

#include "CSceneNodeAnimatorRotation.h"
#include "CSceneNodeAnimatorFlyCircle.h"
#include "CSceneNodeAnimatorFlyStraight.h"
#include "CSceneNodeAnimatorDelete.h"
#include "CSceneNodeAnimatorFollowSpline.h"
#include "CSceneNodeAnimatorCameraFPS.h"
#include "CSceneNodeAnimatorCameraMaya.h"
#include "CSceneNodeAnimatorCameraModifiedMaya.h"

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

//! adds a scene node for rendering a static mesh
//! the returned pointer must not be dropped.
IMeshSceneNode* CSceneManager::addMeshSceneNode(core::smart_refctd_ptr<video::IGPUMesh>&& mesh, IDummyTransformationSceneNode* parent, int32_t id,
	const core::vector3df& position, const core::vector3df& rotation,
	const core::vector3df& scale, bool alsoAddIfMeshPointerZero)
{
	if (!alsoAddIfMeshPointerZero && !mesh)
		return 0;

	if (!parent)
		parent = this;

	IMeshSceneNode* node = new CMeshSceneNode(std::move(mesh), parent, this, id, position, rotation, scale);
	node->drop();

	return node;
}

IMeshSceneNodeInstanced* CSceneManager::addMeshSceneNodeInstanced(IDummyTransformationSceneNode* parent, int32_t id,
    const core::vector3df& position, const core::vector3df& rotation, const core::vector3df& scale)
{
	if (!parent)
		parent = this;
#ifdef NEW_SHADERS
	return nullptr;
#else
	CMeshSceneNodeInstanced* node = new CMeshSceneNodeInstanced(parent, this, id, position, rotation, scale);
	node->drop();

	return node;
#endif
}

//! adds a scene node for rendering an animated mesh model
ISkinnedMeshSceneNode* CSceneManager::addSkinnedMeshSceneNode(
    core::smart_refctd_ptr<video::IGPUSkinnedMesh>&& mesh, const ISkinningStateManager::E_BONE_UPDATE_MODE& boneControlMode,
    IDummyTransformationSceneNode* parent, int32_t id,
    const core::vector3df& position, const core::vector3df& rotation, const core::vector3df& scale)
{
	if (!mesh)
		return 0;

	if (!parent)
		parent = this;

	auto node = new CSkinnedMeshSceneNode(std::move(mesh), boneControlMode, parent, this, id, position, rotation, scale);
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
	const core::vector3df& position, const core::vectorSIMDf& lookat, int32_t id,
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
			core::vectorSIMDf(0,0,100), id, makeActive);
	if (node)
	{
		ISceneNodeAnimator* anm = new CSceneNodeAnimatorCameraMaya(CursorControl,
			rotateSpeed, zoomSpeed, translationSpeed, distance);

		node->addAnimator(anm);
		anm->drop();
	}

	return node;
}

ICameraSceneNode* CSceneManager::addCameraSceneNodeModifiedMaya(IDummyTransformationSceneNode* parent,
	float rotateSpeed, float zoomSpeed,
	float translationSpeed, int32_t id, float distance,
	float scrlZoomSpeed, bool zoomWithRMB,
	bool makeActive)
{
	ICameraSceneNode* node = addCameraSceneNode(parent, core::vector3df(),
		core::vectorSIMDf(0, 0, 100), id, makeActive);
	if (node)
	{
		ISceneNodeAnimator* anm = new CSceneNodeAnimatorCameraModifiedMaya(CursorControl,
			rotateSpeed, zoomSpeed, translationSpeed, distance, scrlZoomSpeed, zoomWithRMB);

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
			core::vectorSIMDf(0,0,100), id, makeActive);
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



//! Adds a skybox scene node. A skybox is a big cube with 6 textures on it and
//! is drawn around the camera position.
IMeshSceneNode* CSceneManager::addSkyBoxSceneNode(core::smart_refctd_ptr<video::IGPUImageView>&& cubemap, IDummyTransformationSceneNode* parent, int32_t id)
{
	if (!parent)
		parent = this;
#ifdef NEW_SHADERS
	return nullptr;
#else
	ISceneNode* node = new CSkyBoxSceneNode(std::move(top), std::move(bottom), std::move(left), std::move(right),
											std::move(front), std::move(back), core::smart_refctd_ptr(redundantMeshDataBuf), 0, parent, this, id);

	node->drop();
	return node;
#endif
}


//! Adds a skydome scene node. A skydome is a large (half-) sphere with a
//! panoramic texture on it and is drawn around the camera position.
IMeshSceneNode* CSceneManager::addSkyDomeSceneNode(	core::smart_refctd_ptr<video::IGPUImageView>&& texture, uint32_t horiRes,
													uint32_t vertRes, float texturePercentage, float spherePercentage, float radius,
													IDummyTransformationSceneNode* parent, int32_t id)
{
	if (!parent)
		parent = this;
#ifdef NEW_SHADERS
	return nullptr;
#else
	ISceneNode* node = new CSkyDomeSceneNode(std::move(texture), horiRes, vertRes,
		texturePercentage, spherePercentage, radius, parent, this, id);

	node->drop();
	return node;
#endif
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

    if (node->getAutomaticCulling())
    {
		node->getAbsoluteTransformation().transformBoxEx(tbox);
        // can be seen by cam pyramid planes ?
        if (cam->getViewFrustum()->intersectsAABB(tbox))
            return true;
	}

	return false;
}


//! registers a node for rendering it at a specific time.
uint32_t CSceneManager::registerNodeForRendering(ISceneNode* node, E_SCENE_NODE_RENDER_PASS pass)
{
	assert(false);
	return 0;
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

	uint32_t i; // new ISO for scoping problem in some compilers

#ifndef NEW_SHADERS
	// reset all transforms
	Driver->setMaterial(video::SGPUMaterial());
	Driver->setTransform(video::EPTS_PROJ,core::matrix4SIMD());
	Driver->setTransform(video::E4X3TS_VIEW,core::matrix3x4SIMD());
	Driver->setTransform(video::E4X3TS_WORLD,core::matrix3x4SIMD());
#endif
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

		std::stable_sort(SolidNodeList.begin(),SolidNodeList.end()); // sort by textures

        for (i=0; i<SolidNodeList.size(); ++i)
            SolidNodeList[i].Node->render();

		SolidNodeList.clear();
	}

	// render transparent objects.
	{
		CurrentRendertime = ESNRP_TRANSPARENT;

		std::stable_sort(TransparentNodeList.begin(),TransparentNodeList.end()); // sort by distance from camera
        for (i=0; i<TransparentNodeList.size(); ++i)
            TransparentNodeList[i].Node->render();

		TransparentNodeList.clear();
	}

	// render transparent effect objects.
	{
		CurrentRendertime = ESNRP_TRANSPARENT_EFFECT;

		std::stable_sort(TransparentEffectNodeList.begin(),TransparentEffectNodeList.end()); // sort by distance from camera
        for (i=0; i<TransparentEffectNodeList.size(); ++i)
            TransparentEffectNodeList[i].Node->render();

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
		const core::vectorSIMDf& direction,
		float startPosition,
		float radiusEllipsoid)
{
	const float orbitDurationMs = core::radians(360.f) / speed;
	const uint32_t effectiveTime = std::chrono::duration_cast<std::chrono::milliseconds>(Timer->getTime()).count() + (uint32_t)(orbitDurationMs * startPosition);

	ISceneNodeAnimator* anim = new CSceneNodeAnimatorFlyCircle(
			effectiveTime, center,
			radius, speed, direction,radiusEllipsoid);
	return anim;
}


//! Creates a fly straight animator, which lets the attached scene node
//! fly or move along a line between two points.
ISceneNodeAnimator* CSceneManager::createFlyStraightAnimator(const core::vectorSIMDf& startPoint,
					const core::vectorSIMDf& endPoint, uint32_t timeForWay, bool loop,bool pingpong)
{
	ISceneNodeAnimator* anim = new CSceneNodeAnimatorFlyStraight(startPoint,
		endPoint, timeForWay, loop, std::chrono::duration_cast<std::chrono::milliseconds>(Timer->getTime()).count(), pingpong);

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
#ifndef NEW_SHADERS
	if (Driver)
		Driver->setMaterial(video::SGPUMaterial());
#endif
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

