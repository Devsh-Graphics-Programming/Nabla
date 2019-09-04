#ifndef __IRR_C_CPU_MESH_H_INCLUDED__
#define __IRR_C_CPU_MESH_H_INCLUDED__

#include "irr/asset/ICPUMesh.h"

namespace irr
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

	#ifdef _IRR_DEBUG
			setDebugName("SCPUMesh");
	#endif
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
			return MeshBuffers.size();
		}

		//! returns pointer to a mesh buffer
		virtual ICPUMeshBuffer* getMeshBuffer(uint32_t nr) const override
		{
			if (MeshBuffers.size())
				return MeshBuffers[nr].get();
			else
				return nullptr;
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