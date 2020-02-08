// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_MESH_SCENE_NODE_INSTANCED_H_INCLUDED__
#define __C_MESH_SCENE_NODE_INSTANCED_H_INCLUDED__

#include "ESceneNodeTypes.h"
#include "IMeshSceneNodeInstanced.h"
#include "ITransformFeedback.h"
#include "IQueryObject.h"
#include "ISceneManager.h"


namespace irr
{
namespace video
{
class SGPUMesh;
}
namespace scene
{

class CMeshSceneNodeInstanced;


//! A scene node displaying a static mesh
//! default instance data is interleaved
class CMeshSceneNodeInstanced : public IMeshSceneNodeInstanced
{
    protected:
        virtual ~CMeshSceneNodeInstanced();

    public:
        static uint32_t recullOrder;

        //! Constructor
        /** Use setMesh() to set the mesh to display.
        */
        CMeshSceneNodeInstanced(IDummyTransformationSceneNode* parent, ISceneManager* mgr, int32_t id,
                const core::vector3df& position = core::vector3df(0,0,0),
                const core::vector3df& rotation = core::vector3df(0,0,0),
                const core::vector3df& scale = core::vector3df(1,1,1));

        //!
        virtual bool supportsDriverFence() const {return true;}

        //! Sets a new mesh to display
        /** \param mesh Mesh to display. */
        virtual bool setLoDMeshes(	const core::vector<MeshLoD>& levelsOfDetail, const size_t& dataSizePerInstanceOutput,
									const video::SGPUMaterial& lodSelectionShader, VaoSetupOverrideFunc vaoSetupOverride,
									const size_t shaderLoDsPerPass=1, void* overrideUserData=nullptr,
									const size_t& extraDataSizePerInstanceInput=0) override;

        //! Get the currently defined mesh for display.
        /** \return Pointer to mesh which is displayed by this node. */
        virtual video::CGPUMesh* getLoDMesh(const size_t &lod) override {return LoD[lod].mesh.get();}

        virtual const core::aabbox3df& getLoDInvariantBBox() const {return LoDInvariantBox;}


        virtual size_t getInstanceCount() const { return core::address_allocator_traits<InstanceDataAddressAllocator>::get_allocated_size(instanceDataAllocator->getAddressAllocator())/dataPerInstanceInputSize; }


        virtual uint32_t addInstance(const core::matrix3x4SIMD& relativeTransform, const void* extraData=NULL) override;

        virtual bool addInstances(uint32_t* instanceIDs, const size_t& instanceCount, const core::matrix3x4SIMD* relativeTransforms, const void* extraData) override;

        virtual void setInstanceTransform(const uint32_t& instanceID, const core::matrix3x4SIMD& relativeTransform) override;

        virtual core::matrix3x4SIMD getInstanceTransform(const uint32_t& instanceID) override;

        virtual void setInstanceVisible(const uint32_t& instanceID, const bool& visible) override;

        virtual void setInstanceData(const uint32_t& instanceID, const void* data) override;

        virtual void removeInstance(const uint32_t& instanceID) override;

        virtual void removeInstances(const size_t& instanceCount, const uint32_t* instanceIDs) override;


        //! frame
        virtual void OnRegisterSceneNode();

        //! renders the node.
        virtual void render();

        //! returns the axis aligned bounding box of this node
        virtual const core::aabbox3d<float>& getBoundingBox()
        {
            if (wantBBoxUpdate&&needsBBoxRecompute)
            {
                Box.reset(0,0,0);
                size_t instanceCount = getInstanceCount();
                size_t allocCapacity = getCurrentInstanceCapacity();

                size_t optimCount = 0;
                for (size_t i=0; optimCount<instanceCount&&i<allocCapacity; i++)
                {
                    if (instanceBBoxes[i].MinEdge.X>instanceBBoxes[i].MaxEdge.X)
                        continue;

                    size_t tmp = optimCount++;
                    if (tmp)
                    {
                        for (size_t j=0; j<3; j++)
                        {
                            if (reinterpret_cast<float*>(&instanceBBoxes[i].MinEdge)[j]<reinterpret_cast<float*>(&Box.MinEdge)[j])
                                reinterpret_cast<float*>(&Box.MinEdge)[j] = reinterpret_cast<float*>(&instanceBBoxes[i].MinEdge)[j];
                        }
                        for (size_t j=0; j<3; j++)
                        {
                            if (reinterpret_cast<float*>(&instanceBBoxes[i].MaxEdge)[j]>reinterpret_cast<float*>(&Box.MaxEdge)[j])
                                reinterpret_cast<float*>(&Box.MaxEdge)[j] = reinterpret_cast<float*>(&instanceBBoxes[i].MaxEdge)[j];
                        }
                    }
                    else
                        Box = instanceBBoxes[i];
                }
                needsBBoxRecompute = false;
            }
            return Box;
        }

        //! Returns type of the scene node
        virtual ESCENE_NODE_TYPE getType() const { return ESNT_MESH_INSTANCED; }

        //! Creates a clone of this scene node and its children.
        virtual ISceneNode* clone(IDummyTransformationSceneNode* newParent=0, ISceneManager* newManager=0) { assert(false); return nullptr; }

    protected:
        void RecullInstances();
        core::aabbox3d<float> Box;
        core::aabbox3d<float> LoDInvariantBox;

        struct LoDData
        {
            core::smart_refctd_ptr<video::CGPUMesh> mesh;
            core::smart_refctd_ptr<video::IQueryObject> query;
            float distanceSQ;
            size_t instancesToDraw;
        };
        core::vector<LoDData> LoD;
        core::vector< core::smart_refctd_ptr<video::ITransformFeedback>> xfb;
        size_t gpuLoDsPerPass;

        bool needsBBoxRecompute;
        size_t instanceBBoxesCount;
        core::aabbox3df* instanceBBoxes;
        bool flagQueryForRetrieval;
        core::smart_refctd_ptr<video::IGPUMeshBuffer> lodCullingPointMesh;
        core::smart_refctd_ptr<video::IGPUBuffer> gpuCulledLodInstanceDataBuffer;

        size_t dataPerInstanceOutputSize;
        size_t extraDataInstanceSize;
        size_t dataPerInstanceInputSize;

        int32_t PassCount;

        inline size_t getCurrentInstanceCapacity() const
        {
            return getBlockIDFromAddr(core::address_allocator_traits<InstanceDataAddressAllocator>::get_total_size(instanceDataAllocator->getAddressAllocator()));
        }

        inline size_t getBlockIDFromAddr(uint32_t instanceID) const
        {
            return instanceDataAllocator->getAddressAllocator().addressToBlockID(instanceID);
        }
};


} // end namespace scene
} // end namespace irr


#endif



