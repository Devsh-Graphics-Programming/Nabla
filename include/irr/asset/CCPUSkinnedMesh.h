#ifndef __IRR_C_CPU_SKINNED_MESH_INCLUDED__
#define __IRR_C_CPU_SKINNED_MESH_INCLUDED__

#include "ICPUSkinnedMesh.h"
#include "irr/asset/ICPUSkinnedMeshBuffer.h"

namespace irr 
{
namespace scene {
    class CFinalBoneHierarchy;
}

namespace asset
{
class ICPUSkinnedMeshBuffer;

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

    virtual scene::CFinalBoneHierarchy* getBoneReferenceHierarchy() const { return referenceHierarchy; }

    //! Meant to be used by loaders only
    void setBoneReferenceHierarchy(scene::CFinalBoneHierarchy* fbh);

    //! returns amount of mesh buffers.
    virtual uint32_t getMeshBufferCount() const;

    //! returns pointer to a mesh buffer
    virtual ICPUMeshBuffer* getMeshBuffer(uint32_t nr) const;

    //! returns an axis aligned bounding box
    virtual const core::aabbox3d<float>& getBoundingBox() const;

    //! set user axis aligned bounding box
    virtual void setBoundingBox(const core::aabbox3df& box);

    //! sets a flag of all contained materials to a new value
    virtual void setMaterialFlag(video::E_MATERIAL_FLAG flag, bool newvalue);

    //! Does the mesh have no animation
    virtual bool isStatic() const;

    //Interface for the mesh loaders (finalize should lock these functions, and they should have some prefix like loader_
    //these functions will use the needed arrays, set values, etc to help the loaders

    //! exposed for loaders to add mesh buffers
    virtual core::vector<ICPUSkinnedMeshBuffer*> &getMeshBuffers();

    //! alternative method for adding joints
    virtual core::vector<SJoint*> &getAllJoints();

    //! alternative method for adding joints
    virtual const core::vector<SJoint*> &getAllJoints() const;

    //! loaders should call this after populating the mesh
    virtual void finalize();

    //! Adds a new meshbuffer to the mesh, access it as last one
    virtual ICPUSkinnedMeshBuffer *addMeshBuffer();

    //! Adds a new meshbuffer to the mesh
    virtual void addMeshBuffer(ICPUSkinnedMeshBuffer* buf);

    //! Adds a new joint to the mesh, access it as last one
    virtual SJoint *addJoint(SJoint *parent = 0);

private:
    void checkForAnimation();

    void calculateGlobalMatrices();

    core::vector<ICPUSkinnedMeshBuffer*> LocalBuffers;

    core::vector<SJoint*> AllJoints;

    scene::CFinalBoneHierarchy* referenceHierarchy;

    core::aabbox3d<float> BoundingBox;

    bool HasAnimation;
};

}}

#endif //__IRR_C_CPU_SKINNED_MESH_INCLUDED__