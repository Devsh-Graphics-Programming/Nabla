// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_GPU_MESH_H_INCLUDED__
#define __NBL_VIDEO_C_GPU_MESH_H_INCLUDED__

#include "nbl/video/IGPUMesh.h"

namespace nbl
{
namespace video
{
	//! Simple implementation of the IMesh interface.
	class CGPUMesh final : public video::IGPUMesh
	{
            core::CLeakDebugger* leakDebugger;
        protected:
			//! The meshbuffers of this mesh
			core::vector<core::smart_refctd_ptr<video::IGPUMeshBuffer> > MeshBuffers;

            //! destructor
            virtual ~CGPUMesh()
            {
                if (leakDebugger)
                    leakDebugger->deregisterObj(this);
            }
        public:
            //! constructor
			CGPUMesh(core::CLeakDebugger* dbgr=nullptr) : leakDebugger(dbgr)
            {
                if (leakDebugger)
                    leakDebugger->registerObj(this);

                #ifdef _NBL_DEBUG
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

            virtual const video::IGPUMeshBuffer* getMeshBuffer(uint32_t nr) const override
            {
                if (nr < MeshBuffers.size())
                    return MeshBuffers[nr].get();
                else
                    return nullptr;
            }

            //! returns pointer to a mesh buffer
            virtual video::IGPUMeshBuffer* getMeshBuffer(uint32_t nr) override
            {
                return const_cast<video::IGPUMeshBuffer*>(const_cast<const CGPUMesh*>(this)->getMeshBuffer(nr));
            }

            //! adds a MeshBuffer
            /** The bounding box is not updated automatically. */
            void addMeshBuffer(core::smart_refctd_ptr<video::IGPUMeshBuffer>&& buf)
            {
                if (buf)
                    MeshBuffers.push_back(std::move(buf));
            }
	};


} // end namespace video
} // end namespace nbl

#endif

