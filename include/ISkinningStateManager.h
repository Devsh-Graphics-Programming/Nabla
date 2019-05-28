#ifndef __I_SKINNING_STATE_MANAGER_H_INCLUDED__
#define __I_SKINNING_STATE_MANAGER_H_INCLUDED__

#include "CFinalBoneHierarchy.h"
#include "IDummyTransformationSceneNode.h"
#include "irr/core/alloc/PoolAddressAllocator.h"
#include "irr/video/ResizableBufferingAllocator.h"

#include "IVideoDriver.h"

namespace irr
{
namespace video
{
    class ITextureBufferObject;
}
namespace scene
{
    class ISkinnedMeshSceneNode;
    /*
    LOCAL SPACE SKINNING:
        Lets think about GPU Boning:
            Need to calculate Hierarchy Positions Unless all Level 1+ bones use Global Skinning Space
            Calculate by pingponging between buffer A and B, K times where K is the depth of the hierarchy of bones!
            For XForm Feedback to work on that, it needs to output all of the joint hierarchy level data out at once for all bones.
            Only Compute Shader/Pixel Shader with GL_LINES would be able to write to non-continous areas of memory.

            INTERMEDIATE REPRESENTATION:
            [{Instance0_J0,Instance1_J0, ... , Instance1_J0},{Instance0_J1,Instance1_J1, ... , InstanceN_J1}, ... ,{Instance0_JM,Instance1_JM, ... , InstanceN_JM}]

            This would make it difficult/impossible to reference parent's data without queries if we were to start culling mid-bone update!!!!
            Only Last Pass could possibly cull!

            +++LAST PASS NEEDS TO REORDER BONE MAJOR ORDER INTO INSTANCE MAJOR!!
                Unless Skinning Shader can take that huge stride into account!!!! (bad for cache anyway)


            1) GPU calculated Final Bone Positions, hence Final BBox for each bone => Final BBox for Node
            2) So either readback or culling performed on GPU => Write to constrained location (compute/imgstore buffer or FBO) or xform feedback compaction
                2a) Would need an additional compaction or index construction step to retrieve instance data (bones and stuff)
                2b) XForm Feedback step compacts the buffer and disallows for using same buffer+pass for different sized aux data (different data per-bone)
                    This also requires the use of atomics to query the offsets and counts of instances for each instance-subtype (per meshes drawn with different materials)
                    ALSO THE INSTANCE-SUBTYPES NEED TO BE IN CONTINUOUS BUFFERS!!! - GRANULE ALLOCATION SCREWED!!!
                    Would effectively require a multi-granule buffer

            This makes the use of one ISkinningStateManager for all instances of crap using sourceHierarchy, very impractical although the GPU instructions remain the same!
            FIRST SOLVE THE ISSUE OF MANY IMeshSceneNodeInstanced(s) USING THE SAME GPU CULLING PROGRAM AND XFMFEEDBACK LAYOUT TAKING MULTIPLE GPU CALLS TO CULL

            So the conclusion is that we need one ISkinningStateManager per IAnimatedSceneNode{Instanced}


            3) To Obtain BBox for Instance
                3a) Min/Max Blending on a 1D Texture Attached to FBO to which we write with GL_POINTS
                3b) Min/Max Blending with atomics after "culling" don't write invisible bones
                3c) Reverse Ping-pong to get child-inclusive bboxes
                3d) During Instance Culling Pass read all N-bones per instance and get BBox that way (bad parallelism, more instances needed for GPU-saturation)


            EVERY SINGLE BONE'S POSITION WOULD HAVE TO BE UPDATED if EJOUR_READ is ENABLED!!! AND READ BACK!!!


            Given the picture we have here, GPU Boning is only a thing we want to consider if we're ever going to have 4096+ instances of the same LoD visible
            (As of AMD Radeon R9 490X)


            Hypothetical GPU-Boning Implementation:
                1a) Fill Buffer LMB with Bone's Local TformMatrices if EJUOR_CONTROL
                1b) Fill Buffer LMB by Compute/XFMFB/FBOPixelShader given a frame
                IF HAS NON-GLOBAL SPACE BONES IN LEVELS DEEPER THAN 1:
                    2) Write to section of Buffer A reading from Buffer LMB for second bone hierarchy level with START shader
                IF HAS NON-GLOBAL SPACE BONES IN LEVELS DEEPER THAN 2:
                    3) Write to section of Buffer B reading from Buffer A for third bone hierarchy level and so on pinponging between A and B with PASS shader
                4) Write to Buffer A with UPDATE shader to calculate BBox and NormalMatrix , as well as join Buffer A and B together
                    (not needed if spec allows same buffer read via TBO range and XFMFB range)
                5) Get Instance-wide BBox Perform Instance Culling!!!

            Since Local Skinning/Global Skinning switch is only available with Bones... GPU Boning always recomputes entire array!!!
    GLOBAL SPACE SKINNING:
        ONLY an OPTION with EBUM_READ or EBUM_CONTROL

        Only Reduces to updating IBoneSceneNode(s) and copying their absolute transformations over to the Output Array
        GPU_BONING useless, calculating the inverses and bboxes less expensive than the GL call unless THOUSANDS of instances
    */
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
            ISkinningStateManager(const E_BONE_UPDATE_MODE& boneControl, video::IVideoDriver* driver, const CFinalBoneHierarchy* sourceHierarchy)
                    : usingGPUorCPUBoning(-100), boneControlMode(boneControl), referenceHierarchy(sourceHierarchy), instanceData(nullptr), instanceDataSize(0)
            {
                referenceHierarchy->grab();

                instanceFinalBoneDataSize = referenceHierarchy->getBoneCount()*sizeof(FinalBoneData);

                auto bufSize = instanceFinalBoneDataSize*16u; // use more buffer space in the future for GPU scene-tree
                instanceBoneDataAllocator = new std::remove_pointer<decltype(instanceBoneDataAllocator)>::type(driver,core::allocator<uint8_t>(),0u,0u,core::roundDownToPoT(instanceFinalBoneDataSize),bufSize,instanceFinalBoneDataSize);

                actualSizeOfInstanceDataElement = sizeof(BoneHierarchyInstanceData)+referenceHierarchy->getBoneCount()*(sizeof(IBoneSceneNode*)+sizeof(core::matrix4x3)); // TODO: odd bone counts will break alignment badly!
            }

            inline const E_BONE_UPDATE_MODE& getBoneUpdateMode() const {return boneControlMode;}

            virtual video::ITextureBufferObject* getBoneDataTBO() const = 0;

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
                    uint8_t* newInstanceData = reinterpret_cast<uint8_t*>(_IRR_ALIGNED_MALLOC(newInstanceDataSize,_IRR_SIMD_ALIGNMENT));
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
                    _IRR_ALIGNED_FREE(instanceData);
            }

            int8_t usingGPUorCPUBoning;
            const E_BONE_UPDATE_MODE boneControlMode;
            const CFinalBoneHierarchy* referenceHierarchy;

            size_t actualSizeOfInstanceDataElement;
            class BoneHierarchyInstanceData : public core::AlignedBase<_IRR_SIMD_ALIGNMENT>
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
                #ifdef _IRR_DEBUG
                size_t addr = reinterpret_cast<size_t>(currentInstance+1u);
                assert((addr&0xfu) == 0u);
                #endif // _IRR_DEBUG
                return reinterpret_cast<core::matrix4x3*>(currentInstance+1u);
            }
            inline IBoneSceneNode** getBones(BoneHierarchyInstanceData* currentInstance)
            {
                #ifdef _IRR_DEBUG
                size_t addr = reinterpret_cast<size_t>(currentInstance+1u);
                assert((addr&0xfu) == 0u);
                #endif // _IRR_DEBUG
                return reinterpret_cast<IBoneSceneNode**>(reinterpret_cast<core::matrix4x3*>(currentInstance+1u)+referenceHierarchy->getBoneCount());
            }
            uint8_t* instanceData;
            uint32_t instanceDataSize;

            uint32_t instanceFinalBoneDataSize;
            #include "irr/irrpack.h"
            struct FinalBoneData
            {
                core::matrix4x3 SkinningTransform;
                float SkinningNormalMatrix[9];
                float MinBBoxEdge[3];
                float MaxBBoxEdge[3];
                float lastAnimatedFrame; //to pad to 128bit align, maybe parentOffsetRelative?
            } PACK_STRUCT;
            #include "irr/irrunpack.h"
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
} // end namespace irr

#endif
