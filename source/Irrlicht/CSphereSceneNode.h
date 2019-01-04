// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_SHPERE_SCENE_NODE_H_INCLUDED__
#define __C_SHPERE_SCENE_NODE_H_INCLUDED__

#include "IMeshSceneNode.h"
#include "irr/asset/IMesh.h"

namespace irr
{
namespace scene
{

	class CSphereSceneNode : public IMeshSceneNode
	{
    protected:
		//! destructor
		virtual ~CSphereSceneNode();

	public:
		//! constructor
		CSphereSceneNode(float size, uint32_t polyCountX, uint32_t polyCountY, IDummyTransformationSceneNode* parent, ISceneManager* mgr, int32_t id,
			const core::vector3df& position = core::vector3df(0,0,0),
			const core::vector3df& rotation = core::vector3df(0,0,0),
			const core::vector3df& scale = core::vector3df(1.0f, 1.0f, 1.0f));

		virtual void OnRegisterSceneNode();

		//! renders the node.
		virtual void render();

		//! returns the axis aligned bounding box of this node
		virtual const core::aabbox3d<float>& getBoundingBox();

		//! returns the material based on the zero based index i. To get the amount
		//! of materials used by this scene node, use getMaterialCount().
		//! This function is needed for inserting the node into the scene hirachy on a
		//! optimal position for minimizing renderstate changes, but can also be used
		//! to directly modify the material of a scene node.
		virtual video::SGPUMaterial& getMaterial(uint32_t i);

		//! returns amount of materials used by this scene node.
		virtual uint32_t getMaterialCount() const;

		//! Returns type of the scene node
		virtual ESCENE_NODE_TYPE getType() const { return ESNT_SPHERE; }

		//! Creates a clone of this scene node and its children.
		virtual ISceneNode* clone(IDummyTransformationSceneNode* newParent=0, ISceneManager* newManager=0);

		//! The mesh cannot be changed
		virtual void setMesh(video::IGPUMesh* mesh) {}

		//! Returns the current mesh
		virtual video::IGPUMesh* getMesh() { return Mesh; }

		//! Sets if the scene node should not copy the materials of the mesh but use them in a read only style.
		/* In this way it is possible to change the materials a mesh causing all mesh scene nodes
		referencing this mesh to change too. */
		virtual void setReferencingMeshMaterials(const bool &referencing) {}

		//! Returns if the scene node should not copy the materials of the mesh but use them in a read only style
		virtual bool isReferencingeMeshMaterials() const { return true; }

	private:

        video::IGPUMesh* Mesh;
		core::aabbox3d<float> Box;
		float Radius;
		uint32_t PolyCountX;
		uint32_t PolyCountY;
	};

} // end namespace scene
} // end namespace irr

#endif

