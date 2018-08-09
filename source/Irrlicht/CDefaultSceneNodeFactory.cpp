// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CDefaultSceneNodeFactory.h"
#include "ISceneManager.h"
#include "IDummyTransformationSceneNode.h"
#include "ICameraSceneNode.h"
#include "IBillboardSceneNode.h"
#include "IAnimatedMeshSceneNode.h"
#include "IMeshSceneNode.h"

namespace irr
{
namespace scene
{


CDefaultSceneNodeFactory::CDefaultSceneNodeFactory(ISceneManager* mgr)
: Manager(mgr)
{

	#ifdef _DEBUG
	setDebugName("CDefaultSceneNodeFactory");
	#endif

	// don't grab the scene manager here to prevent cyclic references

	SupportedSceneNodeTypes.push_back(SSceneNodeTypePair(ESNT_CUBE, "cube"));
	SupportedSceneNodeTypes.push_back(SSceneNodeTypePair(ESNT_SPHERE, "sphere"));
	SupportedSceneNodeTypes.push_back(SSceneNodeTypePair(ESNT_SKY_BOX, "skyBox"));
	SupportedSceneNodeTypes.push_back(SSceneNodeTypePair(ESNT_SKY_DOME, "skyDome"));
	SupportedSceneNodeTypes.push_back(SSceneNodeTypePair(ESNT_SHADOW_VOLUME, "shadowVolume"));
	// Legacy support;
	SupportedSceneNodeTypes.push_back(SSceneNodeTypePair(ESNT_MESH, "mesh"));
	SupportedSceneNodeTypes.push_back(SSceneNodeTypePair(ESNT_EMPTY, "empty"));
	SupportedSceneNodeTypes.push_back(SSceneNodeTypePair(ESNT_DUMMY_TRANSFORMATION, "dummyTransformation"));
	SupportedSceneNodeTypes.push_back(SSceneNodeTypePair(ESNT_CAMERA, "camera"));
	SupportedSceneNodeTypes.push_back(SSceneNodeTypePair(ESNT_BILLBOARD, "billBoard"));
	SupportedSceneNodeTypes.push_back(SSceneNodeTypePair(ESNT_ANIMATED_MESH, "animatedMesh"));
	SupportedSceneNodeTypes.push_back(SSceneNodeTypePair(ESNT_VOLUME_LIGHT, "volumeLight"));

	// legacy, for version <= 1.4.x irr files
	SupportedSceneNodeTypes.push_back(SSceneNodeTypePair(ESNT_CAMERA_MAYA, "cameraMaya"));
	SupportedSceneNodeTypes.push_back(SSceneNodeTypePair(ESNT_CAMERA_FPS, "cameraFPS"));
}


//! adds a scene node to the scene graph based on its type id
ISceneNode* CDefaultSceneNodeFactory::addSceneNode(ESCENE_NODE_TYPE type, IDummyTransformationSceneNode* parent)
{
	switch(type)
	{
	case ESNT_CUBE:
		return Manager->addCubeSceneNode(10, parent);
	case ESNT_SPHERE:
		return Manager->addSphereSceneNode(5, 16, parent);
	case ESNT_SKY_BOX:
		return Manager->addSkyBoxSceneNode(0,0,0,0,0,0, parent);
	case ESNT_SKY_DOME:
		return Manager->addSkyDomeSceneNode(0, 16, 8, 0.9f, 2.0f, 1000.0f, parent);
	case ESNT_SHADOW_VOLUME:
		return 0;
	case ESNT_MESH:
		return Manager->addMeshSceneNode(0, parent, -1, core::vector3df(),
										 core::vector3df(), core::vector3df(1,1,1), true);
	case ESNT_CAMERA:
		return Manager->addCameraSceneNode(parent);
	case ESNT_CAMERA_MAYA:
		return Manager->addCameraSceneNodeMaya(parent);
	case ESNT_CAMERA_FPS:
		return Manager->addCameraSceneNodeFPS(parent);
	case ESNT_BILLBOARD:
		return Manager->addBillboardSceneNode(parent);
#ifdef NEW_MESHES
	case ESNT_ANIMATED_MESH:
		return NULL;
    case ESNT_SKINNED_MESH:
        return NULL;
#endif
	default:
		break;
	}

	return 0;
}


//! adds a scene node to the scene graph based on its type name
ISceneNode* CDefaultSceneNodeFactory::addSceneNode(const char* typeName, IDummyTransformationSceneNode* parent)
{
	return addSceneNode( getTypeFromName(typeName), parent );
}


//! returns amount of scene node types this factory is able to create
uint32_t CDefaultSceneNodeFactory::getCreatableSceneNodeTypeCount() const
{
	return SupportedSceneNodeTypes.size();
}


//! returns type of a createable scene node type
ESCENE_NODE_TYPE CDefaultSceneNodeFactory::getCreateableSceneNodeType(uint32_t idx) const
{
	if (idx<SupportedSceneNodeTypes.size())
		return SupportedSceneNodeTypes[idx].Type;
	else
		return ESNT_UNKNOWN;
}


//! returns type name of a createable scene node type
const char* CDefaultSceneNodeFactory::getCreateableSceneNodeTypeName(uint32_t idx) const
{
	if (idx<SupportedSceneNodeTypes.size())
		return SupportedSceneNodeTypes[idx].TypeName.c_str();
	else
		return 0;
}


//! returns type name of a createable scene node type
const char* CDefaultSceneNodeFactory::getCreateableSceneNodeTypeName(ESCENE_NODE_TYPE type) const
{
	for (uint32_t i=0; i<SupportedSceneNodeTypes.size(); ++i)
		if (SupportedSceneNodeTypes[i].Type == type)
			return SupportedSceneNodeTypes[i].TypeName.c_str();

	return 0;
}


ESCENE_NODE_TYPE CDefaultSceneNodeFactory::getTypeFromName(const char* name) const
{
	for (uint32_t i=0; i<SupportedSceneNodeTypes.size(); ++i)
		if (SupportedSceneNodeTypes[i].TypeName == name)
			return SupportedSceneNodeTypes[i].Type;

	return ESNT_UNKNOWN;
}

} // end namespace scene
} // end namespace irr

