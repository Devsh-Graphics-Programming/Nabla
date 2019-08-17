// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

//New skinned mesh

#ifndef __C_SKINNED_MESH_H_INCLUDED__
#define __C_SKINNED_MESH_H_INCLUDED__

#include "irr/video/IGPUSkinnedMesh.h"
#include "CFinalBoneHierarchy.h"
#include "irrString.h"
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
				CGPUMeshBufferMetaData(core::smart_refctd_ptr<video::IGPUMeshBuffer>&& _mb, uint32_t _maxVertexWeightInfluences) :
													mb(std::move(_mb)), maxVertexWeightInfluences(_maxVertexWeightInfluences) {}

                core::smart_refctd_ptr<video::IGPUMeshBuffer> mb;
                uint32_t maxVertexWeightInfluences;
            };
            core::vector<CGPUMeshBufferMetaData> meshbuffers;

        protected:
            virtual ~CGPUSkinnedMesh()
            {
                referenceHierarchy->drop();
            }

        public:
            CGPUSkinnedMesh(scene::CFinalBoneHierarchy* boneHierarchy) : IGPUSkinnedMesh(boneHierarchy)
            {
                #ifdef _IRR_DEBUG
                setDebugName("CGPUSkinnedMesh");
                #endif

                referenceHierarchy->grab();
            }

            //! Get the amount of mesh buffers.
            virtual uint32_t getMeshBufferCount() const {return meshbuffers.size();}

            //! Returns the IMesh interface for a frame.
            virtual video::IGPUMeshBuffer* getMeshBuffer(uint32_t nr) const
            {
                if (nr>=meshbuffers.size())
                    return nullptr;

                return meshbuffers[nr].mb.get();
            }

            //! adds a Mesh
            inline void addMeshBuffer(core::smart_refctd_ptr<video::IGPUMeshBuffer>&& meshbuffer, const size_t& maxBonesPerVx=4)
            {
                if (meshbuffer)
                    meshbuffers.emplace_back(std::move(meshbuffer),maxBonesPerVx);
            }

            //! Sets a flag of all contained materials to a new value.
            /** \param flag: Flag to set in all materials.
            \param newvalue: New value to set in all materials. */
            virtual void setMaterialFlag(video::E_MATERIAL_FLAG flag, bool newvalue)
            {
                for (uint32_t i=0; i<meshbuffers.size(); ++i)
                    meshbuffers[i].mb->getMaterial().setFlag(flag, newvalue);
            }


            //! Gets the frame count of the animated mesh.
            virtual uint32_t getFrameCount() const {return referenceHierarchy->getKeyFrameCount();}
            virtual float getFirstFrame() const
            {
                if (referenceHierarchy->getKeyFrameCount())
                    return referenceHierarchy->getKeys()[0];
                else
                    return 0.f;
            }
            virtual float getLastFrame() const
            {
                if (referenceHierarchy->getKeyFrameCount())
                    return referenceHierarchy->getKeys()[referenceHierarchy->getKeyFrameCount()-1];
                else
                    return 0.f;
            }

            virtual asset::E_MESH_TYPE getMeshType() const override
            {
                return asset::EMT_ANIMATED_SKINNED;
            }

            //! can use more efficient shaders this way :D
            virtual const uint32_t& getMaxVertexWeights(const size_t& meshbufferIx) const override {return meshbuffers[meshbufferIx].maxVertexWeightInfluences;}

            virtual uint32_t getMaxVertexWeights() const override
            {
                uint32_t maxVal = 0;
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

