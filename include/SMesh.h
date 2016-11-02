// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __S_MESH_H_INCLUDED__
#define __S_MESH_H_INCLUDED__

#include "IMesh.h"
#include "IMeshBuffer.h"
#include "aabbox3d.h"
#include "irrArray.h"

namespace irr
{
namespace scene
{
	//! Simple implementation of the IMesh interface.
	class SCPUMesh : public ICPUMesh
	{
	    public:
		//! constructor
		SCPUMesh()
		{
			#ifdef _DEBUG
			setDebugName("SCPUMesh");
			#endif
		}

		//! destructor
		virtual ~SCPUMesh()
		{
			// drop buffers
			for (u32 i=0; i<MeshBuffers.size(); ++i)
				MeshBuffers[i]->drop();
		}

		//! clean mesh
		virtual void clear()
		{
			for (u32 i=0; i<MeshBuffers.size(); ++i)
				MeshBuffers[i]->drop();
			MeshBuffers.clear();
			BoundingBox.reset ( 0.f, 0.f, 0.f );
		}


		//! returns amount of mesh buffers.
		virtual u32 getMeshBufferCount() const
		{
			return MeshBuffers.size();
		}

		//! returns pointer to a mesh buffer
		virtual ICPUMeshBuffer* getMeshBuffer(u32 nr) const
		{
		    if (MeshBuffers.size())
                return MeshBuffers[nr];
            else
                return NULL;
		}

		//! returns an axis aligned bounding box
		virtual const core::aabbox3d<f32>& getBoundingBox() const
		{
			return BoundingBox;
		}

		//! set user axis aligned bounding box
		virtual void setBoundingBox( const core::aabbox3df& box)
		{
			BoundingBox = box;
		}

		//! recalculates the bounding box
		void recalculateBoundingBox(const bool recomputeSubBoxes=false)
		{
			if (MeshBuffers.size())
			{
			    if (recomputeSubBoxes)
                    MeshBuffers[0]->recalculateBoundingBox();

				BoundingBox = MeshBuffers[0]->getBoundingBox();
				for (u32 i=1; i<MeshBuffers.size(); ++i)
                {
                    if (recomputeSubBoxes)
                        MeshBuffers[i]->recalculateBoundingBox();

					BoundingBox.addInternalBox(MeshBuffers[i]->getBoundingBox());
                }
			}
			else
				BoundingBox.reset(0.0f, 0.0f, 0.0f);
		}

		//! adds a MeshBuffer
		/** The bounding box is not updated automatically. */
		void addMeshBuffer(ICPUMeshBuffer* buf)
		{
			if (buf)
			{
				buf->grab();
				MeshBuffers.push_back(buf);
			}
		}

		//! sets a flag of all contained materials to a new value
		virtual void setMaterialFlag(video::E_MATERIAL_FLAG flag, bool newvalue)
		{
			for (u32 i=0; i<MeshBuffers.size(); ++i)
				MeshBuffers[i]->getMaterial().setFlag(flag, newvalue);
		}

		virtual E_MESH_TYPE getMeshType() const {return EMT_NOT_ANIMATED;}

		//! The bounding box of this mesh
		core::aabbox3d<f32> BoundingBox;
    //private:
		//! The meshbuffers of this mesh
		core::array<ICPUMeshBuffer*> MeshBuffers;
	};

	//! Simple implementation of the IMesh interface.
	class SGPUMesh : public IGPUMesh
	{
    public:
		//! constructor
		SGPUMesh()
		{
			#ifdef _DEBUG
			setDebugName("SGPUMesh");
			#endif
		}

		//! destructor
		virtual ~SGPUMesh()
		{
			// drop buffers
			for (u32 i=0; i<MeshBuffers.size(); ++i)
				MeshBuffers[i]->drop();
		}

		//! clean mesh
		virtual void clear()
		{
			for (u32 i=0; i<MeshBuffers.size(); ++i)
				MeshBuffers[i]->drop();
		}


		//! returns amount of mesh buffers.
		virtual u32 getMeshBufferCount() const
		{
            return MeshBuffers.size();
		}

		//! returns pointer to a mesh buffer
		virtual IGPUMeshBuffer* getMeshBuffer(u32 nr) const
		{
		    if (MeshBuffers.size())
                return MeshBuffers[nr];
            else
                return NULL;
		}

		//! returns an axis aligned bounding box
		virtual const core::aabbox3d<f32>& getBoundingBox() const
		{
			return BoundingBox;
		}

		//! set user axis aligned bounding box
		virtual void setBoundingBox( const core::aabbox3df& box)
		{
			BoundingBox = box;
		}

		//! adds a MeshBuffer
		/** The bounding box is not updated automatically. */
		void addMeshBuffer(IGPUMeshBuffer* buf)
		{
			if (buf)
			{
				buf->grab();
				MeshBuffers.push_back(buf);
			}
		}

		//! sets a flag of all contained materials to a new value
		virtual void setMaterialFlag(video::E_MATERIAL_FLAG flag, bool newvalue)
		{
			for (u32 i=0; i<MeshBuffers.size(); ++i)
				MeshBuffers[i]->getMaterial().setFlag(flag, newvalue);
		}

		virtual E_MESH_TYPE getMeshType() const {return EMT_NOT_ANIMATED;}

    //private:
		//! The bounding box of this mesh
		core::aabbox3d<f32> BoundingBox;

		//! The meshbuffers of this mesh
		core::array<IGPUMeshBuffer*> MeshBuffers;
	};


} // end namespace scene
} // end namespace irr

#endif

