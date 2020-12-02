// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_SKIN_MESH_BUFFER_H_INCLUDED__
#define __NBL_ASSET_I_SKIN_MESH_BUFFER_H_INCLUDED__

#include "nbl/asset/ICPUMeshBuffer.h"
#include "nbl/asset/bawformat/blobs/SkinnedMeshBufferBlob.h"


namespace irr
{
namespace asset
{


class ICPUSkinnedMeshBuffer final : public ICPUMeshBuffer
{
        uint32_t indexValMin;
        uint32_t indexValMax;
        uint32_t maxVertexBoneInfluences;
    public:
        //! Default constructor
        ICPUSkinnedMeshBuffer() : indexValMin(0), indexValMax(0), maxVertexBoneInfluences(1)
        {
            #ifdef _NBL_DEBUG
            setDebugName("ICPUSkinnedMeshBuffer");
            #endif
        }

        core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
        {
            auto cp = clone_template<ICPUSkinnedMeshBuffer>();
            auto cp_mb = static_cast<ICPUSkinnedMeshBuffer*>(cp.get());
            
            cp_mb->indexValMin = indexValMin;
            cp_mb->indexValMax = indexValMax;
            cp_mb->maxVertexBoneInfluences = maxVertexBoneInfluences;

            return cp;
        }

		inline void* serializeToBlob(void* _stackPtr = NULL, const size_t& _stackSize = 0) const override
		{
			return asset::CorrespondingBlobTypeFor<ICPUSkinnedMeshBuffer>::type::createAndTryOnStack(this, _stackPtr, _stackSize);
		}

        virtual asset::E_MESH_BUFFER_TYPE getMeshBufferType() const override { return asset::EMBT_ANIMATED_SKINNED; }

        inline void setIndexRange(const uint32_t &minBeforeBaseVxAdd, const uint32_t &maxBeforeBaseVxAdd)
        {
            indexValMin = minBeforeBaseVxAdd;
            indexValMax = maxBeforeBaseVxAdd;
        }

        inline const uint32_t& getIndexMinBound() const {return indexValMin;}
        inline const uint32_t& getIndexMaxBound() const {return indexValMax;}


        inline void setMaxVertexBoneInfluences(const uint32_t& val) {maxVertexBoneInfluences = val;}
        inline const uint32_t& getMaxVertexBoneInfluences() const {return maxVertexBoneInfluences;}
};


} // end namespace asset
} // end namespace irr

#endif

