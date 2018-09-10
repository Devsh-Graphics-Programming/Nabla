// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

//New skinned mesh

#ifndef __C_SKINNED_MESH_H_INCLUDED__
#define __C_SKINNED_MESH_H_INCLUDED__

#include "ISkinnedMesh.h"
#include "CFinalBoneHierarchy.h"
#include "irr/core/irrString.h"
#include "SSkinMeshBuffer.h"

namespace irr
{
namespace scene
{

    class CGPUSkinnedMesh : public IGPUSkinnedMesh
    {
        private:
            struct SGPUMeshBufferMetaData
            {
                IGPUMeshBuffer* mb;
                uint32_t maxVertexWeightInfluences;
            };
            core::vector<SGPUMeshBufferMetaData> meshbuffers;

        protected:
            virtual ~CGPUSkinnedMesh()
            {
                for (size_t i=0; i<meshbuffers.size(); i++)
                    meshbuffers[i].mb->drop();

                referenceHierarchy->drop();
            }

        public:
            CGPUSkinnedMesh(CFinalBoneHierarchy* boneHierarchy) : IGPUSkinnedMesh(boneHierarchy)
            {
                #ifdef _DEBUG
                setDebugName("CGPUSkinnedMesh");
                #endif

                referenceHierarchy->grab();
            }

            //! Get the amount of mesh buffers.
            virtual uint32_t getMeshBufferCount() const {return meshbuffers.size();}

            //! Returns the IMesh interface for a frame.
            virtual IGPUMeshBuffer* getMeshBuffer(uint32_t nr) const
            {
                if (nr>=meshbuffers.size())
                    return NULL;

                return meshbuffers[nr].mb;
            }

            //! adds a Mesh
            inline void addMeshBuffer(IGPUMeshBuffer* meshbuffer, const size_t& maxBonesPerVx=4)
            {
                if (meshbuffer)
                {
                    meshbuffer->grab();
                    SGPUMeshBufferMetaData tmp;
                    tmp.mb = meshbuffer;
                    tmp.maxVertexWeightInfluences = maxBonesPerVx;
                    meshbuffers.push_back(tmp);
                }
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

            virtual E_MESH_TYPE getMeshType() const
            {
                return EMT_ANIMATED_SKINNED;
            }

            //! can use more efficient shaders this way :D
            virtual const uint32_t& getMaxVertexWeights(const size_t& meshbufferIx) const {return meshbuffers[meshbufferIx].maxVertexWeightInfluences;}

            virtual uint32_t getMaxVertexWeights() const
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

	class CCPUSkinnedMesh: public ICPUSkinnedMesh
	{
        protected:
            //! destructor
            virtual ~CCPUSkinnedMesh();

        public:
            //! constructor
            CCPUSkinnedMesh();

            //! Clears internal container of meshbuffers and calls drop() on each
			virtual void clearMeshBuffers();

            virtual CFinalBoneHierarchy* getBoneReferenceHierarchy() const {return referenceHierarchy;}

			//! Meant to be used by loaders only
			void setBoneReferenceHierarchy(CFinalBoneHierarchy* fbh);

            //! returns amount of mesh buffers.
            virtual uint32_t getMeshBufferCount() const;

            //! returns pointer to a mesh buffer
            virtual ICPUMeshBuffer* getMeshBuffer(uint32_t nr) const;

            //! returns an axis aligned bounding box
            virtual const core::aabbox3d<float>& getBoundingBox() const;

            //! set user axis aligned bounding box
            virtual void setBoundingBox( const core::aabbox3df& box);

            //! sets a flag of all contained materials to a new value
            virtual void setMaterialFlag(video::E_MATERIAL_FLAG flag, bool newvalue);

            //! Does the mesh have no animation
            virtual bool isStatic();

            //Interface for the mesh loaders (finalize should lock these functions, and they should have some prefix like loader_
            //these functions will use the needed arrays, set values, etc to help the loaders

            //! exposed for loaders to add mesh buffers
            virtual core::vector<SCPUSkinMeshBuffer*> &getMeshBuffers();

            //! alternative method for adding joints
            virtual core::vector<SJoint*> &getAllJoints();

            //! alternative method for adding joints
            virtual const core::vector<SJoint*> &getAllJoints() const;

            //! loaders should call this after populating the mesh
            virtual void finalize();

            //! Adds a new meshbuffer to the mesh, access it as last one
            virtual SCPUSkinMeshBuffer *addMeshBuffer();

			//! Adds a new meshbuffer to the mesh
			virtual void addMeshBuffer(SCPUSkinMeshBuffer* buf);

            //! Adds a new joint to the mesh, access it as last one
            virtual SJoint *addJoint(SJoint *parent=0);

        private:
            void checkForAnimation();

            void calculateGlobalMatrices();

            core::vector<SCPUSkinMeshBuffer*> LocalBuffers;

            core::vector<SJoint*> AllJoints;

            CFinalBoneHierarchy* referenceHierarchy;

            core::aabbox3d<float> BoundingBox;

            bool HasAnimation;
	};

} // end namespace scene
} // end namespace irr

#endif

