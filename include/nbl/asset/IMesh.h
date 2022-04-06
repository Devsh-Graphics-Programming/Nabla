// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_MESH_H_INCLUDED__
#define __NBL_ASSET_I_MESH_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "aabbox3d.h"
#include "obbox3d.h"

namespace nbl
{
namespace asset
{
	//! Class which holds the geometry of an object.
	/** An IMesh is nothing more than a collection of some mesh buffers
	(IMeshBuffer). 
	*/
	template <class T>
	class IMesh : public virtual core::IReferenceCounted
	{
		protected:
			//! The cached bounding box of this mesh
			core::aabbox3d<float> m_cachedBoundingBox;
      core::OBB m_orientedBoundingBox;

			virtual ~IMesh() {}

		public:
			//! Get iterator range of the attached meshbuffers
			virtual core::SRange<const T* const> getMeshBuffers() const = 0;
			virtual core::SRange<T* const> getMeshBuffers() = 0;

			//! Get an axis aligned bounding box of the mesh.
			/** \return Bounding box of this mesh. */
			inline const core::aabbox3df& getBoundingBox() const { return m_cachedBoundingBox; }

			//!
			inline virtual void setBoundingBox(const core::aabbox3df& newBoundingBox)
			{
				m_cachedBoundingBox = newBoundingBox;
			}

      //! Get an oriented bounding box of the mesh.
      /** \return Oriented Bounding box of this mesh. */
      inline const core::OBB& getOrientedBoundingBox() const { return m_orientedBoundingBox; }

      //!
      inline virtual void setBoundingBox(const core::OBB& newBoundingBox)
      {
        m_orientedBoundingBox = newBoundingBox;
      }
	};

} // end namespace asset
} // end namespace nbl

#endif

