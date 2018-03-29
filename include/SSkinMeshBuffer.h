// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_SKIN_MESH_BUFFER_H_INCLUDED__
#define __I_SKIN_MESH_BUFFER_H_INCLUDED__

#include "IMeshBuffer.h"


namespace irr
{
namespace scene
{


class SCPUSkinMeshBuffer : public ICPUMeshBuffer
{
        uint32_t indexValMin;
        uint32_t indexValMax;
        uint32_t maxVertexBoneInfluences;
    public:
        //! Default constructor
        SCPUSkinMeshBuffer() : indexValMin(0), indexValMax(0), maxVertexBoneInfluences(1)
        {
            #ifdef _DEBUG
            setDebugName("SCPUSkinMeshBuffer");
            #endif
        }

		virtual void* serializeToBlob(void* _stackPtr = NULL, const size_t& _stackSize = 0) const
		{
			return core::CorrespondingBlobTypeFor<SCPUSkinMeshBuffer>::type::createAndTryOnStack(this, _stackPtr, _stackSize);
		}

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


} // end namespace scene
} // end namespace irr

#endif

