#ifndef __IRR_I_CPU_MESH_H_INCLUDED__
#define __IRR_I_CPU_MESH_H_INCLUDED__

#include "IMesh.h"
#include "IAsset.h"
#include "ICPUMeshBuffer.h"

namespace irr { namespace asset
{

class ICPUMesh : public asset::IMesh<ICPUMeshBuffer>, public asset::BlobSerializable, public asset::IAsset
{
public:
    //! Serializes mesh to blob for *.baw file format.
    /** @param _stackPtr Optional pointer to stack memory to write blob on. If _stackPtr==NULL, sufficient amount of memory will be allocated.
        @param _stackSize Size of stack memory pointed by _stackPtr.
        @returns Pointer to memory on which blob was written.
    */
    virtual void* serializeToBlob(void* _stackPtr = NULL, const size_t& _stackSize = 0) const override
    {
        return asset::CorrespondingBlobTypeFor<ICPUMesh>::type::createAndTryOnStack(this, _stackPtr, _stackSize);
    }

    virtual void convertToDummyObject() override {}
    virtual asset::IAsset::E_TYPE getAssetType() const override { return asset::IAsset::ET_MESH; }

    virtual size_t conservativeSizeEstimate() const override { return getMeshBufferCount() * sizeof(void*); }
};

}}

#endif //__IRR_I_CPU_MESH_H_INCLUDED__