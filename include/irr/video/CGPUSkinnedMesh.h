// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

//New skinned mesh

#ifndef __C_GPU_SKINNED_MESH_H_INCLUDED__
#define __C_GPU_SKINNED_MESH_H_INCLUDED__

#include "irr/video/IGPUSkinnedMesh.h"
#include "CFinalBoneHierarchy.h"
#include "irr/asset/ICPUSkinnedMeshBuffer.h"

namespace irr
{
namespace video
{

    class CGPUSkinnedMesh : public IGPUSkinnedMesh
    {
        private:
            struct CGPUMeshBufferMetaData
            {
				CGPUMeshBufferMetaData(core::smart_refctd_ptr<IGPUMeshBuffer>&& _mb, uint32_t _maxVertexWeightInfluences) :
													mb(std::move(_mb)), maxVertexWeightInfluences(_maxVertexWeightInfluences) {}

                core::smart_refctd_ptr<IGPUMeshBuffer> mb;
                uint32_t maxVertexWeightInfluences;
            };
            core::vector<CGPUMeshBufferMetaData> meshbuffers;

        protected:
			virtual ~CGPUSkinnedMesh() {}

        public:
            CGPUSkinnedMesh(core::smart_refctd_ptr<const asset::CFinalBoneHierarchy>&& boneHierarchy) : IGPUSkinnedMesh(std::move(boneHierarchy))
            {
                #ifdef _IRR_DEBUG
                setDebugName("CGPUSkinnedMesh");
                #endif
            }

            //! Get the amount of mesh buffers.
            virtual uint32_t getMeshBufferCount() const override {return meshbuffers.size();}

            virtual const IGPUMeshBuffer* getMeshBuffer(uint32_t nr) const override
            {
                if (nr < meshbuffers.size())
                    return meshbuffers[nr].mb.get();
                else
                    return nullptr;
            }

            //! Returns the IMesh interface for a frame.
            virtual IGPUMeshBuffer* getMeshBuffer(uint32_t nr) override
            {
                return const_cast<IGPUMeshBuffer*>( const_cast<const CGPUSkinnedMesh*>(this)->getMeshBuffer(nr) );
            }

            //! adds a Mesh
            inline void addMeshBuffer(core::smart_refctd_ptr<IGPUMeshBuffer>&& meshbuffer, const size_t& maxBonesPerVx=4)
            {
                if (meshbuffer)
                    meshbuffers.emplace_back(std::move(meshbuffer),maxBonesPerVx);
            }

            //! can use more efficient shaders this way :D
            virtual const uint32_t& getMaxVertexWeights(const size_t& meshbufferIx) const override {return meshbuffers[meshbufferIx].maxVertexWeightInfluences;}

            virtual uint32_t getMaxVertexWeights() const override
            {
                uint32_t maxVal = 0u;
                for (size_t i=0; i<meshbuffers.size(); i++)
                {
                    if (meshbuffers[i].maxVertexWeightInfluences>maxVal)
                        maxVal = meshbuffers[i].maxVertexWeightInfluences;
                }
                return maxVal;
            }
    };

} // end namespace video
} // end namespace irr

#endif

