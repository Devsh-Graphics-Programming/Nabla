// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_MESH_H_INCLUDED__
#define __NBL_ASSET_I_MESH_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "aabbox3d.h"

namespace nbl
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
			//! The cached bounding box of this mesh
			core::aabbox3d<float> cachedBoundingBox;

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
			virtual T* getMeshBuffer(uint32_t nr) = 0;

			virtual const T* getMeshBuffer(uint32_t nr) const = 0;

			//! Get an axis aligned bounding box of the mesh.
			/** \return Bounding box of this mesh. */
			virtual const core::aabbox3df& getBoundingBox() const { return cachedBoundingBox; }

			//! Set user-defined axis aligned bounding box
			/** \param box New bounding box to use for the mesh. */
			virtual void setBoundingBox(const core::aabbox3df& box) { cachedBoundingBox = box; }

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

			//! Get mesh type.
			/** @returns Mesh type. */
			virtual E_MESH_TYPE getMeshType() const = 0;
	};

} // end namespace asset
} // end namespace nbl

#endif

