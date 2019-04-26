// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __S_GPU_MESH_H_INCLUDED__
#define __S_GPU_MESH_H_INCLUDED__

#include "irr/video/IGPUMesh.h"
#include "irr/video/IGPUMeshBuffer.h"
#include "aabbox3d.h"
#include "coreutil.h"

namespace irr
{
namespace video
{
	//! Simple implementation of the IMesh interface.
	class SGPUMesh : public video::IGPUMesh
	{
            core::LeakDebugger* leakDebugger;
        protected:
            //! destructor
            virtual ~SGPUMesh()
            {
                if (leakDebugger)
                    leakDebugger->deregisterObj(this);

                // drop buffers
                for (uint32_t i=0; i<MeshBuffers.size(); ++i)
                    MeshBuffers[i]->drop();
            }
        public:
            //! constructor
            SGPUMesh(core::LeakDebugger* dbgr=NULL) : leakDebugger(dbgr)
            {
                if (leakDebugger)
                    leakDebugger->registerObj(this);

                #ifdef _IRR_DEBUG
                setDebugName("SGPUMesh");
                #endif
            }

            //! clean mesh
            virtual void clear()
            {
                for (uint32_t i=0; i<MeshBuffers.size(); ++i)
                    MeshBuffers[i]->drop();
            }

            //! returns amount of mesh buffers.
            virtual uint32_t getMeshBufferCount() const
            {
                return MeshBuffers.size();
            }

            //! returns pointer to a mesh buffer
            virtual video::IGPUMeshBuffer* getMeshBuffer(uint32_t nr) const
            {
                if (MeshBuffers.size())
                    return MeshBuffers[nr];
                else
                    return NULL;
            }

            //! returns an axis aligned bounding box
            virtual const core::aabbox3d<float>& getBoundingBox() const
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
            void addMeshBuffer(video::IGPUMeshBuffer* buf)
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
                for (uint32_t i=0; i<MeshBuffers.size(); ++i)
                    MeshBuffers[i]->getMaterial().setFlag(flag, newvalue);
            }

            virtual asset::E_MESH_TYPE getMeshType() const override {return asset::EMT_NOT_ANIMATED;}

        //private:
            //! The bounding box of this mesh
            core::aabbox3d<float> BoundingBox;

            //! The meshbuffers of this mesh
            core::vector<video::IGPUMeshBuffer*> MeshBuffers;
	};


} // end namespace video
} // end namespace irr

#endif//__S_GPU_MESH_H_INCLUDED__

