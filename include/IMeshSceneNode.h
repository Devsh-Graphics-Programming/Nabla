// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_MESH_SCENE_NODE_H_INCLUDED__
#define __I_MESH_SCENE_NODE_H_INCLUDED__

#include "ISceneNode.h"
#include "irr/asset/IMesh.h"
#include "irr/video/IGPUMesh.h"

namespace irr
{
namespace scene
{


//! A scene node displaying a static mesh
class IMeshSceneNode : public ISceneNode
{
public:

	//! Constructor
	/** Use setMesh() to set the mesh to display.
	*/
	IMeshSceneNode(IDummyTransformationSceneNode* parent, ISceneManager* mgr, int32_t id,
			const core::vector3df& position = core::vector3df(0,0,0),
			const core::vector3df& rotation = core::vector3df(0,0,0),
			const core::vector3df& scale = core::vector3df(1,1,1))
		: ISceneNode(parent, mgr, id, position, rotation, scale) {}

	//! Sets a new mesh to display
	/** \param mesh Mesh to display. */
	virtual void setMesh(video::IGPUMesh* mesh) = 0;

	//! Get the currently defined mesh for display.
	/** \return Pointer to mesh which is displayed by this node. */
	virtual video::IGPUMesh* getMesh(void) = 0;


	virtual void setReferencingMeshMaterials(const bool &referencing) = 0;

	virtual bool isReferencingeMeshMaterials() const = 0;
};

} // end namespace scene
} // end namespace irr


#endif

