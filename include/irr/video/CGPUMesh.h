// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_GPU_MESH_H_INCLUDED__
#define __IRR_C_GPU_MESH_H_INCLUDED__

#include "irr/video/IGPUMesh.h"

namespace irr
{
namespace video
{
	//! Simple implementation of the IMesh interface.
	class CGPUMesh final : public video::IGPUMesh
	{
            core::CLeakDebugger* leakDebugger;
        protected:
            //! destructor
            virtual ~CGPUMesh()
            {
                if (leakDebugger)
                    leakDebugger->deregisterObj(this);
            }
        public:
            //! constructor
			CGPUMesh(core::CLeakDebugger* dbgr=NULL) : leakDebugger(dbgr)
            {
                if (leakDebugger)
                    leakDebugger->registerObj(this);

                #ifdef _IRR_DEBUG
                setDebugName("SGPUMesh");
                #endif
            }

            //! clean mesh
            inline void clear()
            {
				MeshBuffers.clear();
            }

            //! returns amount of mesh buffers.
            virtual uint32_t getMeshBufferCount() const override
            {
                return MeshBuffers.size();
            }

            //! returns pointer to a mesh buffer
            virtual video::IGPUMeshBuffer* getMeshBuffer(uint32_t nr) const override
            {
                if (MeshBuffers.size())
                    return MeshBuffers[nr].get();
                else
                    return nullptr;
            }

            //! returns an axis aligned bounding box
            virtual const core::aabbox3d<float>& getBoundingBox() const override
            {
                return BoundingBox;
            }

            //! set user axis aligned bounding box
            virtual void setBoundingBox( const core::aabbox3df& box) override
            {
                BoundingBox = box;
            }

            //! adds a MeshBuffer
            /** The bounding box is not updated automatically. */
            void addMeshBuffer(core::smart_refctd_ptr<video::IGPUMeshBuffer>&& buf)
            {
                if (buf)
                    MeshBuffers.push_back(std::move(buf));
            }

            //! sets a flag of all contained materials to a new value
            virtual void setMaterialFlag(video::E_MATERIAL_FLAG flag, bool newvalue) override
            {
                for (uint32_t i=0; i<MeshBuffers.size(); ++i)
                    MeshBuffers[i]->getMaterial().setFlag(flag, newvalue);
            }

            virtual asset::E_MESH_TYPE getMeshType() const override {return asset::EMT_NOT_ANIMATED;}

        //private:
            //! The bounding box of this mesh
            core::aabbox3d<float> BoundingBox;

            //! The meshbuffers of this mesh
            core::vector<core::smart_refctd_ptr<video::IGPUMeshBuffer> > MeshBuffers;
	};


} // end namespace video
} // end namespace irr

#endif//__S_GPU_MESH_H_INCLUDED__

