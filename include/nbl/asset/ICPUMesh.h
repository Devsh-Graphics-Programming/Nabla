// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_CPU_MESH_H_INCLUDED__
#define __NBL_ASSET_I_CPU_MESH_H_INCLUDED__

#include "nbl/asset/IMesh.h"
#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPUMeshBuffer.h"
#include "nbl/asset/bawformat/blobs/MeshBlob.h"

namespace nbl
{
namespace asset
{
class ICPUMesh final : public IMesh<ICPUMeshBuffer>, public BlobSerializable, public IAsset
{
public:
    //! These are not absolute constants, just the most common situation, there may be setups of assets/resources with completely different relationships.
    _NBL_STATIC_INLINE_CONSTEXPR uint32_t MESHBUFFER_HIERARCHYLEVELS_BELOW = 1u;  //mesh->meshbuffer
    _NBL_STATIC_INLINE_CONSTEXPR uint32_t PIPELINE_HIERARCHYLEVELS_BELOW = MESHBUFFER_HIERARCHYLEVELS_BELOW + 1u;  //meshbuffer->pipeline
    _NBL_STATIC_INLINE_CONSTEXPR uint32_t PIPELINE_LAYOUT_HIERARCHYLEVELS_BELOW = PIPELINE_HIERARCHYLEVELS_BELOW + 1u;  //meshbuffer->pipeline->layout
    _NBL_STATIC_INLINE_CONSTEXPR uint32_t DESC_SET_HIERARCHYLEVELS_BELOW = MESHBUFFER_HIERARCHYLEVELS_BELOW + 1u;  //meshbuffer->ds
    _NBL_STATIC_INLINE_CONSTEXPR uint32_t IMAGEVIEW_HIERARCHYLEVELS_BELOW = DESC_SET_HIERARCHYLEVELS_BELOW + 1u;  //ds->imageview
    _NBL_STATIC_INLINE_CONSTEXPR uint32_t IMAGE_HIERARCHYLEVELS_BELOW = IMAGEVIEW_HIERARCHYLEVELS_BELOW + 1u;  //imageview->image

    //!
    inline core::SRange<const ICPUMeshBuffer* const> getMeshBuffers() const override
    {
        auto begin = reinterpret_cast<const ICPUMeshBuffer* const*>(m_meshBuffers.data());
        return core::SRange<const ICPUMeshBuffer* const>(begin, begin + m_meshBuffers.size());
    }
    inline core::SRange<ICPUMeshBuffer* const> getMeshBuffers() override
    {
        assert(!isImmutable_debug());
        auto begin = reinterpret_cast<ICPUMeshBuffer* const*>(m_meshBuffers.data());
        return core::SRange<ICPUMeshBuffer* const>(begin, begin + m_meshBuffers.size());
    }

    //! Mutable access to the vector of meshbuffers
    inline auto& getMeshBufferVector()
    {
        assert(!isImmutable_debug());
        return m_meshBuffers;
    }

    //!
    inline const core::aabbox3df& getBoundingBox() const  // needed so the compiler doesn't freak out
    {
        return IMesh<ICPUMeshBuffer>::getBoundingBox();
    }
    inline void setBoundingBox(const core::aabbox3df& newBoundingBox) override
    {
        assert(!isImmutable_debug());
        return IMesh<ICPUMeshBuffer>::setBoundingBox(newBoundingBox);
    }

    //! Serializes mesh to blob for *.baw file format.
    /** @param _stackPtr Optional pointer to stack memory to write blob on. If _stackPtr==NULL, sufficient amount of memory will be allocated.
			@param _stackSize Size of stack memory pointed by _stackPtr.
			@returns Pointer to memory on which blob was written.
		*/
    inline void* serializeToBlob(void* _stackPtr = NULL, const size_t& _stackSize = 0) const override
    {
        return CorrespondingBlobTypeFor<ICPUMesh>::type::createAndTryOnStack(this, _stackPtr, _stackSize);
    }

    inline void convertToDummyObject(uint32_t referenceLevelsBelowToConvert = 0u) override
    {
        convertToDummyObject_common(referenceLevelsBelowToConvert);

        if(referenceLevelsBelowToConvert)
            for(auto mesh : getMeshBuffers())
                mesh->convertToDummyObject(referenceLevelsBelowToConvert - 1u);
    }

    _NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_MESH;
    inline E_TYPE getAssetType() const override { return AssetType; }

    inline size_t conservativeSizeEstimate() const override { return m_meshBuffers.size() * sizeof(void*); }

    bool canBeRestoredFrom(const IAsset* _other) const override
    {
        auto other = static_cast<const ICPUMesh*>(_other);
        auto myMBs = getMeshBuffers();
        auto otherMBs = other->getMeshBuffers();
        if(myMBs.size() != otherMBs.size())
            return false;
        for(auto myIt = myMBs.end(), theirIt = otherMBs.begin(); myIt != myMBs.end(); myIt++)
            if(!(*myIt)->canBeRestoredFrom(*theirIt))
                return false;

        return true;
    }

    core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
    {
        auto cp = core::make_smart_refctd_ptr<ICPUMesh>();
        clone_common(cp.get());
        cp->m_cachedBoundingBox = m_cachedBoundingBox;
        cp->m_meshBuffers.resize(m_meshBuffers.size());

        auto outIt = cp->m_meshBuffers.begin();
        for(auto inIt = m_meshBuffers.begin(); inIt != m_meshBuffers.end(); inIt++, outIt++)
        {
            if(_depth > 0u && *inIt)
                *outIt = core::smart_refctd_ptr_static_cast<ICPUMeshBuffer>((*inIt)->clone(_depth - 1u));
            else
                *outIt = *inIt;
        }

        return cp;
    }

private:
    void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
    {
        auto* other = static_cast<ICPUMesh*>(_other);

        if(_levelsBelow)
        {
            --_levelsBelow;
            auto myMBs = getMeshBuffers();
            auto otherMBs = other->getMeshBuffers();
            for(auto myIt = myMBs.end(), theirIt = otherMBs.begin(); myIt != myMBs.end(); myIt++)
                restoreFromDummy_impl_call(*myIt, *theirIt, _levelsBelow);
        }
    }

    bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override
    {
        --_levelsBelow;
        auto mbs = getMeshBuffers();
        for(auto mb : mbs)
            if(mb->isAnyDependencyDummy(_levelsBelow))
                return true;
        return false;
    }

    core::vector<core::smart_refctd_ptr<ICPUMeshBuffer>> m_meshBuffers;
};

}
}

#endif
