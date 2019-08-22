#ifndef __IRR_C_CPU_SKINNED_MESH_INCLUDED__
#define __IRR_C_CPU_SKINNED_MESH_INCLUDED__

#include "ICPUSkinnedMesh.h"
#include "irr/asset/ICPUSkinnedMeshBuffer.h"
#include "CFinalBoneHierarchy.h"

namespace irr 
{
namespace asset
{

class CCPUSkinnedMesh : public ICPUSkinnedMesh
{
	protected:
		//! destructor
		virtual ~CCPUSkinnedMesh();

	public:
		//! constructor
		CCPUSkinnedMesh();

		//! Clears internal container of meshbuffers and calls drop() on each
		virtual void clearMeshBuffers();

		//! Meant to be used by loaders only
		void setBoneReferenceHierarchy(core::smart_refctd_ptr<CFinalBoneHierarchy>&& fbh) { referenceHierarchy = std::move(fbh); }

		//! returns amount of mesh buffers.
		virtual uint32_t getMeshBufferCount() const override;

		//! returns pointer to a mesh buffer
		virtual ICPUMeshBuffer* getMeshBuffer(uint32_t nr) const override;

		//! Does the mesh have no animation
		virtual bool isStatic() const override { return !HasAnimation; }

		//Interface for the mesh loaders (finalize should lock these functions, and they should have some prefix like loader_
		//these functions will use the needed arrays, set values, etc to help the loaders

		//! alternative method for adding joints
		virtual core::vector<SJoint*> &getAllJoints() override;

		//! alternative method for adding joints
		virtual const core::vector<SJoint*> &getAllJoints() const override;

		//! loaders should call this after populating the mesh
		virtual void finalize() override;

		//! Adds a new meshbuffer to the mesh
		virtual void addMeshBuffer(core::smart_refctd_ptr<ICPUSkinnedMeshBuffer>&& buf) {return LocalBuffers.push_back(std::move(buf)); }

		//! Adds a new joint to the mesh, access it as last one
		virtual SJoint *addJoint(SJoint *parent = 0) override;

	private:
		void checkForAnimation();

		void calculateGlobalMatrices();

		core::vector<core::smart_refctd_ptr<ICPUSkinnedMeshBuffer> > LocalBuffers;

		core::vector<SJoint*> AllJoints;

		bool HasAnimation;
};

}}

#endif //__IRR_C_CPU_SKINNED_MESH_INCLUDED__