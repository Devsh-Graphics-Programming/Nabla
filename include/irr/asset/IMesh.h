// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_MESH_H_INCLUDED__
#define __I_MESH_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "SMaterial.h"
#include "aabbox3d.h"

namespace irr
{
namespace asset
{
	//! Possible types of (animated) meshes.
	enum E_MESH_TYPE
	{
		//! Unknown animated mesh type.
		EMT_UNKNOWN = 0,

        //!
		EMT_NOT_ANIMATED,

        //!
		EMT_ANIMATED_FRAME_BASED,

		//! generic skinned mesh
		EMT_ANIMATED_SKINNED
	};


	//! Class which holds the geometry of an object.
	/** An IMesh is nothing more than a collection of some mesh buffers
	(IMeshBuffer). SMesh is a simple implementation of an IMesh.
	A mesh is usually added to an IMeshSceneNode in order to be rendered.
	*/
	template <class T>
	class IMesh : public virtual core::IReferenceCounted
	{
	protected:
		virtual ~IMesh() {}
	public:

		//! Get the amount of mesh buffers.
		/** \return Amount of mesh buffers (IMeshBuffer) in this mesh. */
		virtual uint32_t getMeshBufferCount() const = 0;

		//! Get pointer to a mesh buffer.
		/** \param nr: Zero based index of the mesh buffer. The maximum value is
		getMeshBufferCount() - 1;
		\return Pointer to the mesh buffer or 0 if there is no such
		mesh buffer. */
		virtual T* getMeshBuffer(uint32_t nr) const = 0;

		//! Get an axis aligned bounding box of the mesh.
		/** \return Bounding box of this mesh. */
		virtual const core::aabbox3d<float>& getBoundingBox() const = 0;

		//! Set user-defined axis aligned bounding box
		/** \param box New bounding box to use for the mesh. */
		virtual void setBoundingBox( const core::aabbox3df& box) = 0;

		//! recalculates the bounding box
		virtual void recalculateBoundingBox()
		{
		    core::aabbox3df tmpBox;
			if (getMeshBufferCount())
			{
				tmpBox = getMeshBuffer(0)->getBoundingBox();
				for (uint32_t i=1; i<getMeshBufferCount(); ++i)
                {
					tmpBox.addInternalBox(getMeshBuffer(i)->getBoundingBox());
                }
			}
			else
				tmpBox.reset(0.0f, 0.0f, 0.0f);

            setBoundingBox(tmpBox);
		}

		//! Sets a flag of all contained materials to a new value.
		/** \param flag: Flag to set in all materials.
		\param newvalue: New value to set in all materials. */
		virtual void setMaterialFlag(video::E_MATERIAL_FLAG flag, bool newvalue) = 0;

		//! Get mesh type.
		/** @returns Mesh type. */
		virtual E_MESH_TYPE getMeshType() const = 0;
	};

} // end namespace asset
} // end namespace irr

#endif

