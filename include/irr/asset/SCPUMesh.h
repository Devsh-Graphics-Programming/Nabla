#ifndef __IRR_SCPUMESH_H_INCLUDED__
#define __IRR_SCPUMESH_H_INCLUDED__

#include "ICPUMesh.h"

namespace irr { namespace asset
{

//! Simple implementation of the IMesh interface.
class SCPUMesh : public ICPUMesh
{
    core::LeakDebugger* leakDebugger;
protected:
    //! destructor
    virtual ~SCPUMesh()
    {
        if (leakDebugger)
            leakDebugger->deregisterObj(this);

        // drop buffers
        for (uint32_t i = 0; i < MeshBuffers.size(); ++i)
            MeshBuffers[i]->drop();
    }
public:
    //! constructor
    SCPUMesh(core::LeakDebugger* dbgr = NULL) : leakDebugger(dbgr)
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
        for (uint32_t i = 0; i < MeshBuffers.size(); ++i)
            MeshBuffers[i]->drop();
        MeshBuffers.clear();
        BoundingBox.reset(0.f, 0.f, 0.f);
    }

    //! Clears internal container of meshbuffers and calls drop() on each
    virtual void clearMeshBuffers()
    {
        for (uint32_t i = 0; i < MeshBuffers.size(); ++i)
            MeshBuffers[i]->drop();
        MeshBuffers.clear();
    }

    //! returns amount of mesh buffers.
    virtual uint32_t getMeshBufferCount() const
    {
        return MeshBuffers.size();
    }

    //! returns pointer to a mesh buffer
    virtual ICPUMeshBuffer* getMeshBuffer(uint32_t nr) const
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
    virtual void setBoundingBox(const core::aabbox3df& box)
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
        for (uint32_t i = 0; i < MeshBuffers.size(); ++i)
            MeshBuffers[i]->getMaterial().setFlag(flag, newvalue);
    }

    virtual asset::E_MESH_TYPE getMeshType() const { return asset::EMT_NOT_ANIMATED; }

    //! The bounding box of this mesh
    core::aabbox3d<float> BoundingBox;

    //private:
        //! The meshbuffers of this mesh
    core::vector<ICPUMeshBuffer*> MeshBuffers;
};

}}

#endif //__IRR_SCPUMESH_H_INCLUDED__