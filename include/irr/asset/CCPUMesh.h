#ifndef __IRR_C_CPU_MESH_H_INCLUDED__
#define __IRR_C_CPU_MESH_H_INCLUDED__

#include "ICPUMesh.h"

namespace irr { namespace asset
{

//! Simple implementation of the IMesh interface.
class CCPUMesh : public ICPUMesh
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
        BoundingBox.reset(0.f, 0.f, 0.f);
    }

    //! Clears internal container of meshbuffers and calls drop() on each
    virtual void clearMeshBuffers()
    {
        MeshBuffers.clear();
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
            return NULL;
    }

    //! returns an axis aligned bounding box
    virtual const core::aabbox3d<float>& getBoundingBox() const override
    {
        return BoundingBox;
    }

    //! set user axis aligned bounding box
    virtual void setBoundingBox(const core::aabbox3df& box) override
    {
        BoundingBox = box;
    }

    //! recalculates the bounding box
    void recalculateBoundingBox(const bool recomputeSubBoxes = false)
    {
        if (MeshBuffers.size())
        {
            if (recomputeSubBoxes)
                MeshBuffers[0]->recalculateBoundingBox();

            BoundingBox = MeshBuffers[0]->getBoundingBox();
            for (uint32_t i = 1; i < MeshBuffers.size(); ++i)
            {
                if (recomputeSubBoxes)
                    MeshBuffers[i]->recalculateBoundingBox();

                BoundingBox.addInternalBox(MeshBuffers[i]->getBoundingBox());
            }
        }
        else
            BoundingBox.reset(0.0f, 0.0f, 0.0f);
    }

    //! Adds a MeshBuffer
    /** The bounding box is not updated automatically. */
    void addMeshBuffer(core::smart_refctd_ptr<ICPUMeshBuffer>&& buf)
    {
        if (buf)
            MeshBuffers.push_back(std::move(buf));
    }

    //! sets a flag of all contained materials to a new value
    virtual void setMaterialFlag(video::E_MATERIAL_FLAG flag, bool newvalue)
    {
        for (uint32_t i = 0; i < MeshBuffers.size(); ++i)
            MeshBuffers[i]->getMaterial().setFlag(flag, newvalue);
    }

    virtual asset::E_MESH_TYPE getMeshType() const override { return asset::EMT_NOT_ANIMATED; }

//private:
    //! The bounding box of this mesh
    core::aabbox3d<float> BoundingBox;

    //! The meshbuffers of this mesh
    core::vector<core::smart_refctd_ptr<ICPUMeshBuffer> > MeshBuffers;
};

}
}

#endif