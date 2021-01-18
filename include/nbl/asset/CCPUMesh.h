// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_CPU_MESH_H_INCLUDED__
#define __NBL_ASSET_C_CPU_MESH_H_INCLUDED__

#include "BuildConfigOptions.h"
#include "nbl/asset/ICPUMesh.h"

namespace nbl
{
namespace asset
{

//! Simple implementation of the IMesh interface.
class CCPUMesh final : public ICPUMesh
{
		core::CLeakDebugger* leakDebugger;
	protected:
		//! destructor
		virtual ~CCPUMesh()
		{
			if (leakDebugger)
				leakDebugger->deregisterObj(this);
		}
	public:
		//! constructor
		CCPUMesh(core::CLeakDebugger* dbgr = NULL) : leakDebugger(dbgr)
		{
			if (leakDebugger)
				leakDebugger->registerObj(this);

	#ifdef _NBL_DEBUG
			setDebugName("SCPUMesh");
	#endif
		}

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            auto cp = core::make_smart_refctd_ptr<CCPUMesh>();
            clone_common(cp.get());
            cp->MeshBuffers = core::vector<core::smart_refctd_ptr<ICPUMeshBuffer>>(MeshBuffers.size());
            for (size_t i = 0u; i < MeshBuffers.size(); ++i)
                cp->MeshBuffers[i] = (_depth > 0u && MeshBuffers[i]) ? core::smart_refctd_ptr_static_cast<ICPUMeshBuffer>(MeshBuffers[i]->clone(_depth-1u)) : MeshBuffers[i];

            return cp;
        }

		//! clean mesh
		virtual void clear()
		{
			MeshBuffers.clear();
			recalculateBoundingBox();
		}

		//! returns amount of mesh buffers.
		virtual uint32_t getMeshBufferCount() const override
		{
			return static_cast<uint32_t>(MeshBuffers.size());
		}

		virtual const ICPUMeshBuffer* getMeshBuffer(uint32_t nr) const override
		{
			if (nr < MeshBuffers.size())
				return MeshBuffers[nr].get();
			else
				return nullptr;
		}

		//! returns pointer to a mesh buffer
		virtual ICPUMeshBuffer* getMeshBuffer(uint32_t nr) override
		{
			assert(!isImmutable_debug());

			return const_cast<ICPUMeshBuffer*>(const_cast<const CCPUMesh*>(this)->getMeshBuffer(nr));
		}

		//! Adds a MeshBuffer
		/** The bounding box is not updated automatically. */
		void addMeshBuffer(core::smart_refctd_ptr<ICPUMeshBuffer>&& buf)
		{
			if (buf)
				MeshBuffers.push_back(std::move(buf));
		}

	private:
		//! The meshbuffers of this mesh
		core::vector<core::smart_refctd_ptr<ICPUMeshBuffer> > MeshBuffers;
};

}
}

#endif