#ifndef __IRR_C_CPU_SKINNED_MESH_INCLUDED__
#define __IRR_C_CPU_SKINNED_MESH_INCLUDED__

#include "irr/asset/ICPUSkinnedMesh.h"
#include "irr/asset/ICPUSkinnedMeshBuffer.h"
#include "CFinalBoneHierarchy.h"

namespace irr 
{
namespace asset
{

class CCPUSkinnedMesh : public ICPUSkinnedMesh
{
	protected:
		//!
		core::smart_refctd_ptr<CFinalBoneHierarchy> referenceHierarchy;

		//! destructor
		virtual ~CCPUSkinnedMesh();

	public:
		//! constructor
		CCPUSkinnedMesh() : HasAnimation(false)
		{
			#ifdef _IRR_DEBUG
				setDebugName("CCPUSkinnedMesh");
			#endif
		}

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            auto cp = core::make_smart_refctd_ptr<CCPUSkinnedMesh>();
            clone_common(cp.get());

            cp->LocalBuffers = core::vector<core::smart_refctd_ptr<ICPUSkinnedMeshBuffer>>(LocalBuffers.size());
            for (size_t i = 0u; i < LocalBuffers.size(); ++i)
                cp->LocalBuffers[i] = (_depth > 0u && LocalBuffers[i]) ? core::smart_refctd_ptr_static_cast<ICPUSkinnedMeshBuffer>(LocalBuffers[i]->clone(_depth-1u)) : LocalBuffers[i];
            cp->HasAnimation = HasAnimation;

            for (size_t i = 0u; i < AllJoints.size(); ++i)
                cp->AllJoints.push_back(new SJoint(AllJoints[i][0]));

            for (size_t i = 0u; i < AllJoints.size(); ++i)
            {
                {
                    const size_t ix = std::find(AllJoints.begin(), AllJoints.end(), cp->AllJoints.back()->Parent) - AllJoints.begin();
                    cp->AllJoints.back()->Parent = cp->AllJoints[ix];
                }
                for (auto& child : cp->AllJoints[i]->Children)
                {
                    const size_t ix = std::find(AllJoints.begin(), AllJoints.end(), child) - AllJoints.begin();
                    child = cp->AllJoints[ix];
                }
            }

            return cp;
        }

		//!
		virtual CFinalBoneHierarchy* getBoneReferenceHierarchy() const override { return referenceHierarchy.get(); }

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
		virtual SJoint *addJoint(SJoint *parent = 0) override
        {
            SJoint *joint = new SJoint;

            AllJoints.push_back(joint);
            if (!parent)
            {
                //Add root joints to array in finalize()
                joint->Parent = nullptr;
            }
            else
            {
                //Set parent (Be careful of the mesh loader also setting the parent)
                parent->Children.push_back(joint);
                joint->Parent = parent;
            }

            return joint;
        }

	private:
		void checkForAnimation();

		void calculateGlobalMatrices();

		core::vector<core::smart_refctd_ptr<ICPUSkinnedMeshBuffer> > LocalBuffers;

		core::vector<SJoint*> AllJoints;

		bool HasAnimation;
};

}}

#endif //__IRR_C_CPU_SKINNED_MESH_INCLUDED__