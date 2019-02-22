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

        //! returns the material based on the zero based index i. To get the amount
        //! of materials used by this scene node, use getMaterialCount().
        //! This function is needed for inserting the node into the scene hirachy on a
        //! optimal position for minimizing renderstate changes, but can also be used
        //! to directly modify the material of a scene node.
        virtual video::SGPUMaterial& getMaterial(uint32_t i)
        {
            uint32_t cumMaterialCnt = 0;
            for (size_t j=0; j<LoD.size(); j++)
            {
                if (i-cumMaterialCnt<LoD[j].mesh->getMeshBufferCount())
                    return LoD[j].mesh->getMeshBuffer(i-cumMaterialCnt)->getMaterial();
                else
                    cumMaterialCnt += LoD[j].mesh->getMeshBufferCount();
            }
            return ISceneNode::getMaterial(i);
        }
        //! returns amount of materials used by this scene node.
        virtual uint32_t getMaterialCount() const {return cachedMaterialCount;}

        //! Sets a new mesh to display
        /** \param mesh Mesh to display. */
        virtual bool setLoDMeshes(const core::vector<MeshLoD>& levelsOfDetail, const size_t& dataSizePerInstanceOutput, const video::SGPUMaterial& lodSelectionShader, VaoSetupOverrideFunc vaoSetupOverride, const size_t shaderLoDsPerPass=1, void* overrideUserData=NULL, const size_t& extraDataSizePerInstanceInput=0);

        //! Get the currently defined mesh for display.
        /** \return Pointer to mesh which is displayed by this node. */
        virtual video::SGPUMesh* getLoDMesh(const size_t &lod) {return LoD[lod].mesh;}

        virtual const core::aabbox3df& getLoDInvariantBBox() const {return LoDInvariantBox;}


        virtual size_t getInstanceCount() const { return core::address_allocator_traits<InstanceDataAddressAllocator>::get_allocated_size(instanceDataAllocator->getAddressAllocator())/dataPerInstanceInputSize; }


        virtual uint32_t addInstance(const core::matrix4x3& relativeTransform, const void* extraData=NULL);

        virtual bool addInstances(uint32_t* instanceIDs, const size_t& instanceCount, const core::matrix4x3* relativeTransforms, const void* extraData);

        virtual void setInstanceTransform(const uint32_t& instanceID, const core::matrix4x3& relativeTransform);

        virtual core::matrix4x3 getInstanceTransform(const uint32_t& instanceID);

        virtual void setInstanceVisible(const uint32_t& instanceID, const bool& visible);

        virtual void setInstanceData(const uint32_t& instanceID, const void* data);

        virtual void removeInstance(const uint32_t& instanceID);

        virtual void removeInstances(const size_t& instanceCount, const uint32_t* instanceIDs);


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
        virtual ISceneNode* clone(IDummyTransformationSceneNode* newParent=0, ISceneManager* newManager=0);

    protected:
        void RecullInstances();
        core::aabbox3d<float> Box;
        core::aabbox3d<float> LoDInvariantBox;
        uint32_t cachedMaterialCount;

        struct LoDData
        {
            video::SGPUMesh* mesh;
            video::IQueryObject* query;
            float distanceSQ;
            size_t instancesToDraw;
        };
        core::vector<LoDData> LoD;
        core::vector<video::ITransformFeedback*> xfb;
        size_t gpuLoDsPerPass;

        bool needsBBoxRecompute;
        size_t instanceBBoxesCount;
        core::aabbox3df* instanceBBoxes;
        bool flagQueryForRetrieval;
        video::IGPUMeshBuffer* lodCullingPointMesh;
        video::IGPUBuffer* gpuCulledLodInstanceDataBuffer;

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



