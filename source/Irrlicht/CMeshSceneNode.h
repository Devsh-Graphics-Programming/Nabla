// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_MESH_SCENE_NODE_H_INCLUDED__
#define __C_MESH_SCENE_NODE_H_INCLUDED__

#include "IMeshSceneNode.h"
#include "irr/asset/IMesh.h"

namespace irr
{
namespace scene
{


	class CMeshSceneNode : public IMeshSceneNode
	{
    protected:
		//! destructor
		virtual ~CMeshSceneNode() {}

	public:

		//! constructor
		CMeshSceneNode(core::smart_refctd_ptr<video::IGPUMesh>&& mesh, IDummyTransformationSceneNode* parent,
			ISceneManager* mgr, int32_t id, const core::vector3df& position = core::vector3df(0,0,0),
			const core::vector3df& rotation = core::vector3df(0,0,0), const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f));

		//!
		virtual bool supportsDriverFence() const {return true;}

		//! frame
		virtual void OnRegisterSceneNode();

		//! renders the node.
		virtual void render();

		//! returns the axis aligned bounding box of this node
		virtual const core::aabbox3d<float>& getBoundingBox() override;

		//! Returns type of the scene node
		virtual ESCENE_NODE_TYPE getType() const override { return ESNT_MESH; }

		//! Sets a new mesh
		virtual void setMesh(core::smart_refctd_ptr<video::IGPUMesh>&& mesh) override;

		//! Returns the current mesh
		virtual video::IGPUMesh* getMesh(void) override { return Mesh.get(); }
		virtual const video::IGPUMesh* getMesh(void) const override { return Mesh.get(); }

	protected:
		core::aabbox3d<float> Box;

        core::smart_refctd_ptr<video::IGPUMesh> Mesh;

		int32_t PassCount;
	};

} // end namespace scene
} // end namespace irr

#endif

