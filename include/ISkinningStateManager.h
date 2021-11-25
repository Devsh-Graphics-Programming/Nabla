// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_I_SKINNING_STATE_MANAGER_H_INCLUDED__
#define __NBL_I_SKINNING_STATE_MANAGER_H_INCLUDED__

#include "IDummyTransformationSceneNode.h"
#include "nbl/core/alloc/PoolAddressAllocator.h"
#include "nbl/video/alloc/ResizableBufferingAllocator.h"

#include "IVideoDriver.h"

namespace nbl
{

namespace scene
{
    class ISkinnedMeshSceneNode;
    //! SUPER-CONSTRAINT::::: ALL IAnimatedMeshSceneNodeInstanced LoD Meshes HAVE TO USE THE SAME CFinalBoneHierarchy!!
    /**     Or We need two separate classes, one which makes instances with bones, one that makes them without and these can use different bone hierarchies!
    **/
    class ISkinningStateManager : public virtual core::IReferenceCounted
    {
            typedef core::PoolAddressAllocatorST<uint32_t>                                              InstanceDataAddressAllocator;
        public:
            constexpr static decltype(InstanceDataAddressAllocator::invalid_address) kInvalidInstanceID = InstanceDataAddressAllocator::invalid_address;

            enum E_BONE_UPDATE_MODE
            {
                //! do nothing, no bones get made, GPU_BONING compatibile
                EBUM_NONE = 0,

                //! get joints positions from the mesh (for attached nodes, etc), due to the possible complicated interdependence in the SceneNode hierarchy
                //! its impossible to enable GPU_BONING for this case (think AnimatedNode instance being child of another or another's bone etc.)
                //! The CPU Boning supports implicit boning outside the main boning call
                EBUM_READ,

                //! control joint positions in the mesh (eg. ragdolls), GPU_BONING useless except just to calculate NormalMatrix and BBox for each bone
                //! Hence GPU_BONING not compatibile
                EBUM_CONTROL,

                EBUM_COUNT
            };
            class IBoneSceneNode : public IDummyTransformationSceneNode
            {
                public:
                    enum E_BONE_SKINNING_SPACE
                    {
                        //! local skinning, standard
                        EBSS_LOCAL=0,

                        //! global skinning
                        EBSS_GLOBAL,

                        EBSS_COUNT
                    };

                    IBoneSceneNode(ISkinningStateManager* owner, const uint32_t& instanceIndex, IDummyTransformationSceneNode* parent, const uint32_t& boneIndex, const core::matrix4x3& initialRelativeMatrix=core::matrix4x3())
                        : IDummyTransformationSceneNode(parent), ownerManager(owner), InstanceID(instanceIndex), BoneIndex(boneIndex), SkinningSpace(EBSS_LOCAL), lastTimePulledAbsoluteTFormForBoning(0)
                    {
                        setRelativeTransformationMatrix(initialRelativeMatrix);
                    }

                    //! How the relative transformation of the bone is used
                    //! THIS WILL NOT MAGICALLY CONVERT THE MATRIX VALUES ON_SWITCH
                    //! YOU NEED TO CALCULATE THEM AND SET THEM BEFORE OR AFTER THE SWITCH!!!
                    inline void setSkinningSpace( const E_BONE_SKINNING_SPACE& space )
                    {
                        if (space>EBSS_GLOBAL||space==SkinningSpace)
                            return;

                        //Do Some owner specific shit here!!!
                        SkinningSpace = space;
                    }

                    //! How the relative transformation of the bone is used
                    inline const E_BONE_SKINNING_SPACE& getSkinningSpace() const {return SkinningSpace;}


                    //! OVERRIDDEN FUNCTIONS
                    inline virtual const core::matrix4x3& getRelativeTransformationMatrix()
                    {
                        if (SkinningSpace==EBSS_GLOBAL)
                        {
                            if (relativeTransNeedsUpdate)
                                AbsoluteTransformation = IDummyTransformationSceneNode::getRelativeTransformationMatrix();
                            return RelativeTransformation;
                        }
                        else
                            return IDummyTransformationSceneNode::getRelativeTransformationMatrix();
                    }

                    inline virtual bool needsAbsoluteTransformRecompute() const
                    {
                        if (SkinningSpace==EBSS_GLOBAL)
                            return false;
                        else
                            return IDummyTransformationSceneNode::needsAbsoluteTransformRecompute();
                    }

                    inline virtual size_t needsDeepAbsoluteTransformRecompute() const
                    {
                        if (SkinningSpace==EBSS_GLOBAL)
                            return 0xdeadbeefu;
                        else
                            return IDummyTransformationSceneNode::needsDeepAbsoluteTransformRecompute();
                    }

                    /// if the MODE is READ, have to do implicit boning of an instance :D
                    inline virtual void updateAbsolutePosition()
                    {
                        ownerManager->implicitBone(InstanceID,BoneIndex);
                        bool recompute = relativeTransNeedsUpdate||lastTimeRelativeTransRead[3]<relativeTransChanged;

                        if (Parent&&SkinningSpace!=EBSS_GLOBAL)
                        {
                            uint64_t parentAbsoluteHint = Parent->getAbsoluteTransformLastRecomputeHint();
                            if (lastTimeRelativeTransRead[4] < parentAbsoluteHint)
                            {
                                lastTimeRelativeTransRead[4] = parentAbsoluteHint;
                                recompute = true;
                            }

                            // recompute if local transform has changed
                            if (recompute)
                            {
                                const core::matrix4x3& rel = getRelativeTransformationMatrix();
                                AbsoluteTransformation = concatenateBFollowedByA(Parent->getAbsoluteTransformation(),rel);
                                lastTimeRelativeTransRead[3] = relativeTransChanged;
                            }
                        }
                        else if (recompute)
                        {
                            AbsoluteTransformation = getRelativeTransformationMatrix();
                            lastTimeRelativeTransRead[3] = relativeTransChanged;
                        }
                    }

                    inline bool getTransformChangedBoningHint() const {return lastTimePulledAbsoluteTFormForBoning<lastTimeRelativeTransRead[3];}

                    inline void setTransformChangedBoningHint() {lastTimePulledAbsoluteTFormForBoning = lastTimeRelativeTransRead[3];}

                protected:
                    ISkinningStateManager* ownerManager;
                    uint32_t InstanceID;
                    uint32_t BoneIndex;
                    E_BONE_SKINNING_SPACE SkinningSpace;
                    uint64_t lastTimePulledAbsoluteTFormForBoning;
            };

            //! Constructor
            ISkinningStateManager(const E_BONE_UPDATE_MODE& boneControl, video::IVideoDriver* driver, const asset::CFinalBoneHierarchy* sourceHierarchy)
                    : usingGPUorCPUBoning(-100), boneControlMode(boneControl), referenceHierarchy(sourceHierarchy), instanceData(nullptr), instanceDataSize(0)
            {
                referenceHierarchy->grab();

                instanceFinalBoneDataSize = referenceHierarchy->getBoneCount()*sizeof(FinalBoneData);

                auto bufSize = instanceFinalBoneDataSize*16u; // use more buffer space in the future for GPU scene-tree
                instanceBoneDataAllocator = new std::remove_pointer<decltype(instanceBoneDataAllocator)>::type(driver,core::allocator<uint8_t>(),0u,0u,core::roundDownToPoT(instanceFinalBoneDataSize),bufSize,instanceFinalBoneDataSize);

                actualSizeOfInstanceDataElement = sizeof(BoneHierarchyInstanceData)+referenceHierarchy->getBoneCount()*(sizeof(IBoneSceneNode*)+sizeof(core::matrix4x3)); // TODO: odd bone counts will break alignment badly!
            }

            inline const E_BONE_UPDATE_MODE& getBoneUpdateMode() const {return boneControlMode;}

            virtual video::IGPUBufferView* getBoneDataTBO() const = 0;

            size_t getBoneCount() const { return referenceHierarchy->getBoneCount(); }


            virtual void implicitBone(const size_t& instanceID, const size_t& boneID) = 0;

            virtual void performBoning() = 0;


            //!
            virtual void createBones(const size_t& instanceID) = 0;

            //! always creates bones in EBUM_CONTROL mode
            virtual uint32_t addInstance(ISkinnedMeshSceneNode* attachedNode=nullptr, const bool& createBoneNodes=false) = 0;

            //!
            inline void grabInstance(const uint32_t& ID)
            {
                getBoneHierarchyInstanceFromAddr(ID)->refCount++;
            }

            //! true if deleted
            virtual bool dropInstance(const uint32_t& ID)
            {
                BoneHierarchyInstanceData* instance = getBoneHierarchyInstanceFromAddr(ID);
                if ((instance->refCount--)>1)
                    return false;

                //proceed to delete
                auto instanceBones = getBones(instance);
                for (size_t i=0; i<referenceHierarchy->getBoneLevelRangeEnd(0); i++)
                {
                    instanceBones[i]->remove();
                    instanceBones[i]->drop();
                }
                memset(instance,0,actualSizeOfInstanceDataElement);

                instanceBoneDataAllocator->multi_free_addr(1u,&ID,&instanceFinalBoneDataSize);

                auto instanceCapacity = getDataInstanceCapacity();
                if (instanceDataSize!=instanceCapacity)
                {
                    auto newInstanceDataSize = instanceCapacity*actualSizeOfInstanceDataElement;
                    uint8_t* newInstanceData = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(newInstanceDataSize,_NBL_SIMD_ALIGNMENT));
                    auto oldInstanceDataByteSize = instanceDataSize*actualSizeOfInstanceDataElement;
                    if (newInstanceDataSize<oldInstanceDataByteSize)
                        memcpy(newInstanceData,instanceData,newInstanceDataSize);
                    else
                    {
                        memcpy(newInstanceData,instanceData,oldInstanceDataByteSize);
                        memset(newInstanceData+oldInstanceDataByteSize,0,newInstanceDataSize-oldInstanceDataByteSize);
                    }
                    instanceData = newInstanceData;
                    instanceDataSize = instanceCapacity;
                }
                return true;
            }


            inline void setInterpolation(const bool& isLinear, const uint32_t& ID)
            {
                getBoneHierarchyInstanceFromAddr(ID)->interpolateAnimation = isLinear;
            }

            inline void setFrame(const float& frame, const uint32_t& ID)
            {
                getBoneHierarchyInstanceFromAddr(ID)->frame = frame;
            }

            inline IBoneSceneNode* getBone(const uint32_t& boneID, const uint32_t& ID)
            {
                assert(boneID<referenceHierarchy->getBoneCount());
                return getBones(getBoneHierarchyInstanceFromAddr(ID))[boneID];
            }


            inline size_t getDataInstanceCount() const {return instanceBoneDataAllocator->getAddressAllocator().get_allocated_size()/instanceFinalBoneDataSize;}

        protected:
            virtual ~ISkinningStateManager()
            {
                for (size_t j=instanceBoneDataAllocator->getAddressAllocator().get_combined_offset(); j<instanceBoneDataAllocator->getAddressAllocator().get_total_size(); j+=instanceFinalBoneDataSize)
                {
                    BoneHierarchyInstanceData* currentInstance = getBoneHierarchyInstanceFromAddr(j);
                    if (!currentInstance->refCount)
                        continue;

                    auto instanceBones = getBones(currentInstance);
                    for (size_t i=0; i<referenceHierarchy->getBoneLevelRangeEnd(0); i++)
                    {
                        if (!instanceBones[i])
                            continue;

                        instanceBones[i]->remove();
                        instanceBones[i]->drop();
                    }
                }
                referenceHierarchy->drop();
                instanceBoneDataAllocator->drop();

                if (instanceData)
                    _NBL_ALIGNED_FREE(instanceData);
            }

            int8_t usingGPUorCPUBoning;
            const E_BONE_UPDATE_MODE boneControlMode;
            const asset::CFinalBoneHierarchy* referenceHierarchy;

            size_t actualSizeOfInstanceDataElement;
            class BoneHierarchyInstanceData : public core::AlignedBase<_NBL_SIMD_ALIGNMENT>
            {
                public:
                    BoneHierarchyInstanceData() : refCount(0), frame(0.f), lastAnimatedFrame(-1.f), interpolateAnimation(true), attachedNode(NULL)
                    {
                    }

                    uint32_t refCount;
                    float frame;

                    union
                    {
                        bool needToRecomputeParentBBox;
                        float lastAnimatedFrame;
                    };

                    bool interpolateAnimation;
                    ISkinnedMeshSceneNode* attachedNode; //can be NULL
            };
            inline core::matrix4x3* getGlobalMatrices(BoneHierarchyInstanceData* currentInstance)
            {
                #ifdef _NBL_DEBUG
                size_t addr = reinterpret_cast<size_t>(currentInstance+1u);
                assert((addr&0xfu) == 0u);
                #endif // _NBL_DEBUG
                return reinterpret_cast<core::matrix4x3*>(currentInstance+1u);
            }
            inline IBoneSceneNode** getBones(BoneHierarchyInstanceData* currentInstance)
            {
                #ifdef _NBL_DEBUG
                size_t addr = reinterpret_cast<size_t>(currentInstance+1u);
                assert((addr&0xfu) == 0u);
                #endif // _NBL_DEBUG
                return reinterpret_cast<IBoneSceneNode**>(reinterpret_cast<core::matrix4x3*>(currentInstance+1u)+referenceHierarchy->getBoneCount());
            }
            uint8_t* instanceData;
            uint32_t instanceDataSize;

            uint32_t instanceFinalBoneDataSize;
            #include "nbl/nblpack.h"
            struct FinalBoneData
            {
                core::matrix4x3 SkinningTransform;
                float SkinningNormalMatrix[9];
                float MinBBoxEdge[3];
                float MaxBBoxEdge[3];
                float lastAnimatedFrame; //to pad to 128bit align, maybe parentOffsetRelative?
            } PACK_STRUCT;
            #include "nbl/nblunpack.h"
            video::ResizableBufferingAllocatorST<InstanceDataAddressAllocator,core::allocator<uint8_t>,true>* instanceBoneDataAllocator;

            inline uint32_t getDataInstanceCapacity() const
            {
                const auto& alloc = instanceBoneDataAllocator->getAddressAllocator();
                return alloc.addressToBlockID(alloc.get_total_size());
            }
            inline BoneHierarchyInstanceData* getBoneHierarchyInstanceFromAddr(uint32_t instanceID) const
            {
                const auto& alloc = instanceBoneDataAllocator->getAddressAllocator();
                auto blockID = alloc.addressToBlockID(instanceID);
                return reinterpret_cast<BoneHierarchyInstanceData*>(instanceData+blockID*actualSizeOfInstanceDataElement);
            }
    };

} // end namespace scene
} // end namespace nbl

#endif
