#ifndef __C_SKINNING_STATE_MANAGER_H_INCLUDED__
#define __C_SKINNING_STATE_MANAGER_H_INCLUDED__


#include "ISkinningStateManager.h"
#include "ITextureBufferObject.h"
#include "IVideoDriver.h"

///#define UPDATE_WHOLE_BUFFER

namespace irr
{
namespace scene
{


    class CSkinningStateManager : public ISkinningStateManager
    {
            video::IVideoDriver* Driver;
#ifdef _IRR_COMPILE_WITH_OPENGL_
            video::ITextureBufferObject* TBO;
#endif
        protected:
            virtual ~CSkinningStateManager()
            {
#ifdef _IRR_COMPILE_WITH_OPENGL_
                Driver->removeTextureBufferObject(TBO);
#endif // _IRR_COMPILE_WITH_OPENGL_
            }

        public:
            CSkinningStateManager(const E_BONE_UPDATE_MODE& boneControl, video::IVideoDriver* driver, const asset::CFinalBoneHierarchy* sourceHierarchy)
                                    : ISkinningStateManager(boneControl,driver,sourceHierarchy), Driver(driver)
            {
#ifdef _IRR_COMPILE_WITH_OPENGL_
                TBO = driver->addTextureBufferObject(instanceBoneDataAllocator->getFrontBuffer(),video::ITextureBufferObject::ETBOF_RGBA32F);
#endif // _IRR_COMPILE_WITH_OPENGL_
            }

            //
            const void* getRawBoneData() {return instanceBoneDataAllocator->getBackBufferPointer();}

            //
            virtual video::ITextureBufferObject* getBoneDataTBO() const
            {
#ifdef _IRR_COMPILE_WITH_OPENGL_
                return TBO;
#else
                return NULL;
#endif // _IRR_COMPILE_WITH_OPENGL_
            }

            //
            virtual uint32_t addInstance(ISkinnedMeshSceneNode* attachedNode=NULL, const bool& createBoneNodes=false)
            {
                uint32_t newID = kInvalidInstanceID;

                const uint32_t align = _IRR_SIMD_ALIGNMENT;
                instanceBoneDataAllocator->multi_alloc_addr(1u,&newID,&instanceFinalBoneDataSize,&align);
                if (newID==kInvalidInstanceID)
                    return kInvalidInstanceID;

                //grow instanceData
                auto instanceCapacity = getDataInstanceCapacity();
                if (instanceDataSize!=instanceCapacity)
                {
#ifdef _IRR_COMPILE_WITH_OPENGL_
                    if (TBO->getByteSize()!=instanceBoneDataAllocator->getFrontBuffer()->getSize())
                        TBO->bind(instanceBoneDataAllocator->getFrontBuffer(),video::ITextureBufferObject::ETBOF_RGBA32F); //can't clandestine re-bind because it won't change the length :D
#endif // _IRR_COMPILE_WITH_OPENGL_
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

                BoneHierarchyInstanceData* tmp = getBoneHierarchyInstanceFromAddr(newID);
                tmp->refCount = 1;
                tmp->frame = 0.f;
                tmp->interpolateAnimation = true;
                tmp->attachedNode = attachedNode;
                if (boneControlMode!=EBUM_CONTROL)
                {
                    FinalBoneData* boneData = reinterpret_cast<FinalBoneData*>(reinterpret_cast<uint8_t*>(instanceBoneDataAllocator->getBackBufferPointer())+newID);
                    for (size_t i=0; i<referenceHierarchy->getBoneCount(); i++)
                        boneData[i].lastAnimatedFrame = -1.f;
                }
                switch (boneControlMode)
                {
                    case EBUM_NONE:
                        tmp->lastAnimatedFrame = -1.f;
                        memset(getBones(tmp),0,sizeof(IBoneSceneNode*)*referenceHierarchy->getBoneCount());
                        break;
                    case EBUM_READ:
                        tmp->lastAnimatedFrame = -1.f;
                        if (!createBoneNodes)
                        {
                            memset(getBones(tmp),0,sizeof(IBoneSceneNode*)*referenceHierarchy->getBoneCount());
                            break;
                        }
#if __cplusplus >= 201703L
                        [[fallthrough]];
#endif
                    case EBUM_CONTROL:
                        tmp->needToRecomputeParentBBox = false;
                        for (size_t i=0; i<referenceHierarchy->getBoneCount(); i++)
                        {
                            const asset::CFinalBoneHierarchy::BoneReferenceData& boneData = referenceHierarchy->getBoneData()[i];
                            core::matrix4x3 localMatrix = asset::CFinalBoneHierarchy::getMatrixFromKey(referenceHierarchy->getNonInterpolatedAnimationData(i)[0]).getAsRetardedIrrlichtMatrix();

                            IBoneSceneNode* tmpBone; //! TODO: change to placement new
                            if (boneData.parentOffsetRelative)
                            {
                                tmpBone = new IBoneSceneNode(this,newID,getBones(tmp)[boneData.parentOffsetFromTop],i,localMatrix); //want first frame
                            }
                            else
                                tmpBone = new IBoneSceneNode(this,newID,attachedNode,i,localMatrix);

                            getBones(tmp)[i] = tmpBone;
                        }
                        break;
                    #ifdef _DEBUG
                    default:
                        assert(false);
                        break;
                    #endif // _DEBUG
                }

                instanceBoneDataAllocator->markRangeForPush(newID,newID+instanceFinalBoneDataSize);

                return newID;
            }

            //! true if deleted
            virtual bool dropInstance(const uint32_t& ID)
            {
                if (ISkinningStateManager::dropInstance(ID))
                {
#ifdef _IRR_COMPILE_WITH_OPENGL_
                    if (TBO->getByteSize()!=instanceBoneDataAllocator->getFrontBuffer()->getSize())
                        TBO->bind(instanceBoneDataAllocator->getFrontBuffer(),video::ITextureBufferObject::ETBOF_RGBA32F); //can't clandestine re-bind because it won't change the length :D
#endif // _IRR_COMPILE_WITH_OPENGL_
                    return true;
                }
                else
                    return false;
            }

            virtual void createBones(const size_t& instanceID)
            {
                assert(boneControlMode!=EBUM_NONE);
                if (boneControlMode!=EBUM_READ)
                    return;

                BoneHierarchyInstanceData* tmp = getBoneHierarchyInstanceFromAddr(instanceID);
                for (size_t i=0; i<referenceHierarchy->getBoneCount(); i++)
                {
                    if (getBones(tmp)[i])
                        continue;

                    const asset::CFinalBoneHierarchy::BoneReferenceData& boneData = referenceHierarchy->getBoneData()[i];
					core::matrix3x4SIMD parentInverse;
					core::matrix3x4SIMD().set(getGlobalMatrices(tmp)[boneData.parentOffsetFromTop]).getInverse(parentInverse);
					const core::matrix4x3 localMatrix = core::matrix3x4SIMD::concatenateBFollowedByA(parentInverse, core::matrix3x4SIMD().set(getGlobalMatrices(tmp)[i])).getAsRetardedIrrlichtMatrix();

                    IBoneSceneNode* tmpBone; //! TODO: change to placement new
                    if (boneData.parentOffsetRelative)
                    {
                        tmpBone = new IBoneSceneNode(this,instanceID,getBones(tmp)[boneData.parentOffsetFromTop],i,localMatrix); //want first frame
                    }
                    else
                        tmpBone = new IBoneSceneNode(this,instanceID,tmp->attachedNode,i,localMatrix);

                    getBones(tmp)[i] = tmpBone;
                }
            }



            virtual void implicitBone(const size_t& instanceID, const size_t& boneID)
            {
                assert(boneID<referenceHierarchy->getBoneCount());
                if (boneControlMode!=EBUM_READ)
                    return;

                BoneHierarchyInstanceData* currentInstance = getBoneHierarchyInstanceFromAddr(instanceID);
                if (currentInstance->frame==currentInstance->lastAnimatedFrame) //in other modes, check if also has no bones!!!
                    return;

                FinalBoneData* boneDataForInstance = reinterpret_cast<FinalBoneData*>(reinterpret_cast<uint8_t*>(instanceBoneDataAllocator->getBackBufferPointer())+instanceID);
                if (boneDataForInstance[boneID].lastAnimatedFrame != currentInstance->frame)
                    return;


                core::matrix4x3 attachedNodeTform;
                if (currentInstance->attachedNode)
                    attachedNodeTform = currentInstance->attachedNode->getAbsoluteTransformation();


                size_t boneStack[256];
                boneStack[0] = boneID;
                size_t boneStackSize = 0;
                while (boneDataForInstance[boneStack[boneStackSize]].lastAnimatedFrame!=currentInstance->frame && boneStack[boneStackSize] >= referenceHierarchy->getBoneLevelRangeEnd(0))
                    boneStack[++boneStackSize] = referenceHierarchy->getBoneData()[boneStack[boneStackSize]].parentOffsetFromTop;

                instanceBoneDataAllocator->markRangeForPush(instanceID+boneStack[boneStackSize]*sizeof(FinalBoneData),instanceID+(boneID+1u)*sizeof(FinalBoneData));

                boneStackSize++;


                float interpolationFactor;
                size_t foundKeyIx = referenceHierarchy->getLowerBoundBoneKeyframes(interpolationFactor,currentInstance->frame);
                float interpolantPrecalcTerm2,interpolantPrecalcTerm3;
                core::quaternion::flerp_interpolant_terms(interpolantPrecalcTerm2,interpolantPrecalcTerm3,interpolationFactor);

                while (boneStackSize--)
                {
                    size_t j = boneStack[boneStackSize];
					asset::CFinalBoneHierarchy::AnimationKeyData upperFrame = (currentInstance->interpolateAnimation ? referenceHierarchy->getInterpolatedAnimationData(j):referenceHierarchy->getNonInterpolatedAnimationData(j))[foundKeyIx];

                    core::matrix3x4SIMD interpolatedLocalTform;
                    if (currentInstance->interpolateAnimation&&interpolationFactor<1.f)
                    {
						asset::CFinalBoneHierarchy::AnimationKeyData lowerFrame = (currentInstance->interpolateAnimation ? referenceHierarchy->getInterpolatedAnimationData(j):referenceHierarchy->getNonInterpolatedAnimationData(j))[foundKeyIx-1];
                        interpolatedLocalTform = referenceHierarchy->getMatrixFromKeys(lowerFrame,upperFrame,interpolationFactor,interpolantPrecalcTerm2,interpolantPrecalcTerm3);
                    }
                    else
                        interpolatedLocalTform = referenceHierarchy->getMatrixFromKey(upperFrame);

                    if (j < referenceHierarchy->getBoneLevelRangeEnd(0))
                        getGlobalMatrices(currentInstance)[j] = interpolatedLocalTform.getAsRetardedIrrlichtMatrix();
                    else
                    {
                        const core::matrix4x3& parentTform = getGlobalMatrices(currentInstance)[referenceHierarchy->getBoneData()[j].parentOffsetFromTop];
                        getGlobalMatrices(currentInstance)[j] = core::matrix3x4SIMD::concatenateBFollowedByA(core::matrix3x4SIMD().set(parentTform), interpolatedLocalTform).getAsRetardedIrrlichtMatrix();
                    }
					boneDataForInstance[j].SkinningTransform = core::matrix3x4SIMD::concatenateBFollowedByA(core::matrix3x4SIMD().set(getGlobalMatrices(currentInstance)[j]), referenceHierarchy->getBoneData()[j].PoseBindMatrix).getAsRetardedIrrlichtMatrix();
					if (referenceHierarchy->flipsXOnOutput())
					for (auto n=0; n<4; n++)
						boneDataForInstance[j].SkinningTransform.pointer()[3*n] = -boneDataForInstance[j].SkinningTransform.pointer()[3*n];


                    core::aabbox3df bbox;
                    bbox.MinEdge.X = referenceHierarchy->getBoneData()[j].MinBBoxEdge[0];
                    bbox.MinEdge.Y = referenceHierarchy->getBoneData()[j].MinBBoxEdge[1];
                    bbox.MinEdge.Z = referenceHierarchy->getBoneData()[j].MinBBoxEdge[2];
                    bbox.MaxEdge.X = referenceHierarchy->getBoneData()[j].MaxBBoxEdge[0];
                    bbox.MaxEdge.Y = referenceHierarchy->getBoneData()[j].MaxBBoxEdge[1];
                    bbox.MaxEdge.Z = referenceHierarchy->getBoneData()[j].MaxBBoxEdge[2];
                    //boneDataForInstance[j].SkinningTransform.transformBoxEx(bbox);
					bbox = core::transformBoxEx(bbox, core::matrix3x4SIMD().set(boneDataForInstance[j].SkinningTransform));
                    //
                    IBoneSceneNode* bone = getBones(currentInstance)[j];
                    if (bone)
                    {
                        if (bone->getSkinningSpace()!=IBoneSceneNode::EBSS_LOCAL)
                            bone->setRelativeTransformationMatrix(core::matrix3x4SIMD::concatenateBFollowedByA(core::matrix3x4SIMD().set(attachedNodeTform), core::matrix3x4SIMD().set(getGlobalMatrices(currentInstance)[j])).getAsRetardedIrrlichtMatrix()/*concatenateBFollowedByA(attachedNodeTform,getGlobalMatrices(currentInstance)[j])*/);
                        else
                        {
                            bone->setRelativeTransformationMatrix(interpolatedLocalTform.getAsRetardedIrrlichtMatrix());
                            bone->updateAbsolutePosition();
                        }
                    }

                    boneDataForInstance[j].MinBBoxEdge[0] = bbox.MinEdge.X;
                    boneDataForInstance[j].MinBBoxEdge[1] = bbox.MinEdge.Y;
                    boneDataForInstance[j].MinBBoxEdge[2] = bbox.MinEdge.Z;
                    boneDataForInstance[j].MaxBBoxEdge[0] = bbox.MaxEdge.X;
                    boneDataForInstance[j].MaxBBoxEdge[1] = bbox.MaxEdge.Y;
                    boneDataForInstance[j].MaxBBoxEdge[2] = bbox.MaxEdge.Z;
                    core::matrix3x4SIMD().set(boneDataForInstance[j].SkinningTransform).getSub3x3InverseTransposePacked(boneDataForInstance[j].SkinningNormalMatrix);

                    boneDataForInstance[j].lastAnimatedFrame = currentInstance->frame;
                }
            }

            inline void TrySwapBoneBuffer()
            {
                instanceBoneDataAllocator->pushBuffer(Driver->getDefaultUpStreamingBuffer());
#ifdef _IRR_COMPILE_WITH_OPENGL_
                if (TBO->getByteSize()!=instanceBoneDataAllocator->getFrontBuffer()->getSize())
                    TBO->bind(instanceBoneDataAllocator->getFrontBuffer(),video::ITextureBufferObject::ETBOF_RGBA32F); //can't clandestine re-bind because it won't change the range length :D
#endif // _IRR_COMPILE_WITH_OPENGL_
            }

            virtual void performBoning()
            {
                if (referenceHierarchy->getHierarchyLevels()==0||instanceBoneDataAllocator->getAddressAllocator().get_allocated_size()==0)
                    return;

                if (usingGPUorCPUBoning>=0&&boneControlMode==EBUM_NONE)
                {
#ifdef _DEBUG
//                    os::Printer::log("GPU Boning NOT SUPPORTED YET!",ELL_ERROR);
#endif // _DEBUG
                }
                else
                {
                    switch (boneControlMode)
                    {
                        case EBUM_NONE:
                        case EBUM_READ:
                            {
                                uint8_t* boneData = reinterpret_cast<uint8_t*>(instanceBoneDataAllocator->getBackBufferPointer());
                                bool notModified = true;
                                uint32_t localFirstDirtyInstance,localLastDirtyInstance;
                                for (size_t i=instanceBoneDataAllocator->getAddressAllocator().get_align_offset(); i<instanceBoneDataAllocator->getAddressAllocator().get_total_size(); i+=instanceFinalBoneDataSize)
                                {
                                    BoneHierarchyInstanceData* currentInstance = getBoneHierarchyInstanceFromAddr(i);
                                    if (!currentInstance->refCount || currentInstance->frame==currentInstance->lastAnimatedFrame) //in other modes, check if also has no bones!!!
                                        continue;

                                    core::matrix4x3 attachedNodeTform;
                                    if (currentInstance->attachedNode)
                                        attachedNodeTform = currentInstance->attachedNode->getAbsoluteTransformation();


                                    float interpolationFactor;
                                    size_t foundKeyIx = referenceHierarchy->getLowerBoundBoneKeyframes(interpolationFactor,currentInstance->frame);
                                    float interpolantPrecalcTerm2,interpolantPrecalcTerm3;
                                    core::quaternion::flerp_interpolant_terms(interpolantPrecalcTerm2,interpolantPrecalcTerm3,interpolationFactor);


                                    FinalBoneData* boneDataForInstance = reinterpret_cast<FinalBoneData*>(boneData+i);
                                    for (size_t j=0; j<referenceHierarchy->getBoneCount(); j++)
                                    {
                                        if (boneDataForInstance[j].lastAnimatedFrame==currentInstance->frame)
                                            continue;
                                        if (notModified)
                                        {
                                            localFirstDirtyInstance = i;
                                            notModified = false;
                                        }
                                        localLastDirtyInstance = i;
                                        boneDataForInstance[j].lastAnimatedFrame = currentInstance->frame;

                                        asset::CFinalBoneHierarchy::AnimationKeyData upperFrame = (currentInstance->interpolateAnimation ? referenceHierarchy->getInterpolatedAnimationData(j):referenceHierarchy->getNonInterpolatedAnimationData(j))[foundKeyIx];

                                        core::matrix3x4SIMD interpolatedLocalTform;
                                        if (currentInstance->interpolateAnimation&&interpolationFactor<1.f)
                                        {
											asset::CFinalBoneHierarchy::AnimationKeyData lowerFrame =  (currentInstance->interpolateAnimation ? referenceHierarchy->getInterpolatedAnimationData(j):referenceHierarchy->getNonInterpolatedAnimationData(j))[foundKeyIx-1];
                                            interpolatedLocalTform = referenceHierarchy->getMatrixFromKeys(lowerFrame,upperFrame,interpolationFactor,interpolantPrecalcTerm2,interpolantPrecalcTerm3);
                                        }
                                        else
                                            interpolatedLocalTform = referenceHierarchy->getMatrixFromKey(upperFrame);

                                        if (j < referenceHierarchy->getBoneLevelRangeEnd(0))
                                            getGlobalMatrices(currentInstance)[j] = interpolatedLocalTform.getAsRetardedIrrlichtMatrix();
                                        else
                                        {
                                            const core::matrix4x3& parentTform = getGlobalMatrices(currentInstance)[referenceHierarchy->getBoneData()[j].parentOffsetFromTop];
                                            getGlobalMatrices(currentInstance)[j] = core::matrix3x4SIMD::concatenateBFollowedByA(core::matrix3x4SIMD().set(parentTform), interpolatedLocalTform).getAsRetardedIrrlichtMatrix();
                                        }
                                        boneDataForInstance[j].SkinningTransform = core::matrix3x4SIMD::concatenateBFollowedByA(core::matrix3x4SIMD().set(getGlobalMatrices(currentInstance)[j]), referenceHierarchy->getBoneData()[j].PoseBindMatrix).getAsRetardedIrrlichtMatrix();
										if (referenceHierarchy->flipsXOnOutput())
										for (auto n=0; n<4; n++)
											boneDataForInstance[j].SkinningTransform.pointer()[3*n] = -boneDataForInstance[j].SkinningTransform.pointer()[3*n];

                                        core::aabbox3df bbox;
                                        bbox.MinEdge.X = referenceHierarchy->getBoneData()[j].MinBBoxEdge[0];
                                        bbox.MinEdge.Y = referenceHierarchy->getBoneData()[j].MinBBoxEdge[1];
                                        bbox.MinEdge.Z = referenceHierarchy->getBoneData()[j].MinBBoxEdge[2];
                                        bbox.MaxEdge.X = referenceHierarchy->getBoneData()[j].MaxBBoxEdge[0];
                                        bbox.MaxEdge.Y = referenceHierarchy->getBoneData()[j].MaxBBoxEdge[1];
                                        bbox.MaxEdge.Z = referenceHierarchy->getBoneData()[j].MaxBBoxEdge[2];
                                        //boneDataForInstance[j].SkinningTransform.transformBoxEx(bbox);
										bbox = core::transformBoxEx(bbox, core::matrix3x4SIMD().set(boneDataForInstance[j].SkinningTransform));
                                        //
                                        if (boneControlMode==EBUM_READ)
                                        {
                                            IBoneSceneNode* bone = getBones(currentInstance)[j];
                                            if (bone)
                                            {
												if (bone->getSkinningSpace() != IBoneSceneNode::EBSS_LOCAL)
													bone->setRelativeTransformationMatrix(core::matrix3x4SIMD::concatenateBFollowedByA(core::matrix3x4SIMD().set(attachedNodeTform), core::matrix3x4SIMD().set(getGlobalMatrices(currentInstance)[j])).getAsRetardedIrrlichtMatrix()/*concatenateBFollowedByA(attachedNodeTform,getGlobalMatrices(currentInstance)[j])*/);
                                                else
                                                {
                                                    bone->setRelativeTransformationMatrix(interpolatedLocalTform.getAsRetardedIrrlichtMatrix());
                                                    bone->updateAbsolutePosition();
                                                }
                                            }
                                        }

                                        boneDataForInstance[j].MinBBoxEdge[0] = bbox.MinEdge.X;
                                        boneDataForInstance[j].MinBBoxEdge[1] = bbox.MinEdge.Y;
                                        boneDataForInstance[j].MinBBoxEdge[2] = bbox.MinEdge.Z;
                                        boneDataForInstance[j].MaxBBoxEdge[0] = bbox.MaxEdge.X;
                                        boneDataForInstance[j].MaxBBoxEdge[1] = bbox.MaxEdge.Y;
                                        boneDataForInstance[j].MaxBBoxEdge[2] = bbox.MaxEdge.Z;
										core::matrix3x4SIMD().set(boneDataForInstance[j].SkinningTransform).getSub3x3InverseTransposePacked(boneDataForInstance[j].SkinningNormalMatrix);
                                    }
                                }

                                if (!notModified)
                                    instanceBoneDataAllocator->markRangeForPush(localFirstDirtyInstance,localLastDirtyInstance+instanceFinalBoneDataSize);

                                TrySwapBoneBuffer();

                                if (!notModified)
                                {
                                    for (size_t i=localFirstDirtyInstance; i<=localLastDirtyInstance; i+=instanceFinalBoneDataSize)
                                    {
                                        BoneHierarchyInstanceData* currentInstance = getBoneHierarchyInstanceFromAddr(i);
                                        if (!currentInstance->refCount || currentInstance->frame==currentInstance->lastAnimatedFrame) //in other modes, check if also has no bones!!!
                                            continue;
                                        currentInstance->lastAnimatedFrame = currentInstance->frame;

                                        core::aabbox3df nodeBBox;
                                        FinalBoneData* boneDataForInstance = reinterpret_cast<FinalBoneData*>(boneData+i);
                                        for (size_t j=0; j<referenceHierarchy->getBoneCount(); j++)
                                        {
                                            if (boneControlMode==EBUM_READ)
                                            {
                                                IBoneSceneNode* bone = getBones(currentInstance)[j];
                                                if (bone)
                                                    bone->updateAbsolutePosition();
                                            }

                                            if (!currentInstance->attachedNode)
                                                continue;

                                            core::aabbox3df bbox;
                                            bbox.MinEdge.X = boneDataForInstance[j].MinBBoxEdge[0];
                                            bbox.MinEdge.Y = boneDataForInstance[j].MinBBoxEdge[1];
                                            bbox.MinEdge.Z = boneDataForInstance[j].MinBBoxEdge[2];
                                            bbox.MaxEdge.X = boneDataForInstance[j].MaxBBoxEdge[0];
                                            bbox.MaxEdge.Y = boneDataForInstance[j].MaxBBoxEdge[1];
                                            bbox.MaxEdge.Z = boneDataForInstance[j].MaxBBoxEdge[2];
                                            if (j)
                                                nodeBBox.addInternalBox(bbox);
                                            else
                                                nodeBBox = bbox;
                                        }

                                        if (currentInstance->attachedNode)
                                            currentInstance->attachedNode->setBoundingBox(nodeBBox);
                                    }
                                }
                            }
                            break;
                        case EBUM_CONTROL:
                            {
                                uint8_t* boneData = reinterpret_cast<uint8_t*>(instanceBoneDataAllocator->getBackBufferPointer());
                                bool notModified = true;
                                uint32_t localFirstDirtyInstance,localLastDirtyInstance;
                                for (size_t i=instanceBoneDataAllocator->getAddressAllocator().get_align_offset(); i<instanceBoneDataAllocator->getAddressAllocator().get_total_size(); i+=instanceFinalBoneDataSize)
                                {
                                    BoneHierarchyInstanceData* currentInstance = getBoneHierarchyInstanceFromAddr(i);
                                    if (!currentInstance->refCount)
                                        continue;

                                    FinalBoneData* boneDataForInstance = reinterpret_cast<FinalBoneData*>(boneData+i);

                                    core::matrix3x4SIMD attachedNodeInverse;
                                    if (currentInstance->attachedNode)
                                    {
                                        currentInstance->attachedNode->updateAbsolutePosition();
										core::matrix3x4SIMD().set(currentInstance->attachedNode->getAbsoluteTransformation()).getInverse(attachedNodeInverse);
                                    }

                                    bool localNotModified = true;
                                    for (size_t j=0; j<referenceHierarchy->getBoneCount(); j++)
                                    {
                                        IBoneSceneNode* bone = getBones(currentInstance)[j];
                                        assert(bone);

                                        bone->updateAbsolutePosition();
                                        if (!bone->getTransformChangedBoningHint())
                                            continue;
                                        bone->setTransformChangedBoningHint();


										boneDataForInstance[j].SkinningTransform = core::matrix3x4SIMD::concatenateBFollowedByA(attachedNodeInverse, core::matrix3x4SIMD::concatenateBFollowedByA(core::matrix3x4SIMD().set(bone->getAbsoluteTransformation()), referenceHierarchy->getBoneData()[j].PoseBindMatrix)).getAsRetardedIrrlichtMatrix();

										core::matrix3x4SIMD().set(boneDataForInstance[j].SkinningTransform).getSub3x3InverseTransposePacked(boneDataForInstance[j].SkinningNormalMatrix);

                                        core::aabbox3df bbox;
                                        bbox.MinEdge.X = referenceHierarchy->getBoneData()[j].MinBBoxEdge[0];
                                        bbox.MinEdge.Y = referenceHierarchy->getBoneData()[j].MinBBoxEdge[1];
                                        bbox.MinEdge.Z = referenceHierarchy->getBoneData()[j].MinBBoxEdge[2];
                                        bbox.MaxEdge.X = referenceHierarchy->getBoneData()[j].MaxBBoxEdge[0];
                                        bbox.MaxEdge.Y = referenceHierarchy->getBoneData()[j].MaxBBoxEdge[1];
                                        bbox.MaxEdge.Z = referenceHierarchy->getBoneData()[j].MaxBBoxEdge[2];
                                        //boneDataForInstance[j].SkinningTransform.transformBoxEx(bbox);
										bbox = core::transformBoxEx(bbox, core::matrix3x4SIMD().set(boneDataForInstance[j].SkinningTransform));
                                        boneDataForInstance[j].MinBBoxEdge[0] = bbox.MinEdge.X;
                                        boneDataForInstance[j].MinBBoxEdge[1] = bbox.MinEdge.Y;
                                        boneDataForInstance[j].MinBBoxEdge[2] = bbox.MinEdge.Z;
                                        boneDataForInstance[j].MaxBBoxEdge[0] = bbox.MaxEdge.X;
                                        boneDataForInstance[j].MaxBBoxEdge[1] = bbox.MaxEdge.Y;
                                        boneDataForInstance[j].MaxBBoxEdge[2] = bbox.MaxEdge.Z;

                                        if (localNotModified)
                                        {
                                            if (notModified)
                                            {
                                                localFirstDirtyInstance = i;
                                                notModified = false;
                                            }
                                            localNotModified = false;
                                        }
                                        localLastDirtyInstance = i;
                                    }

                                    currentInstance->needToRecomputeParentBBox = localNotModified;
                                }


                                if (!notModified)
                                    instanceBoneDataAllocator->markRangeForPush(localFirstDirtyInstance,localLastDirtyInstance+instanceFinalBoneDataSize);

                                TrySwapBoneBuffer();

                                if (!notModified)
                                {
                                    for (uint32_t i=localFirstDirtyInstance; i<=localLastDirtyInstance; i+=instanceFinalBoneDataSize)
                                    {
                                        BoneHierarchyInstanceData* currentInstance = getBoneHierarchyInstanceFromAddr(i);
                                        if (!currentInstance->refCount || !currentInstance->attachedNode || currentInstance->needToRecomputeParentBBox)
                                            continue;

                                        core::aabbox3df nodeBBox;
                                        FinalBoneData* boneDataForInstance = reinterpret_cast<FinalBoneData*>(boneData+i);
                                        for (size_t j=0; j<referenceHierarchy->getBoneCount(); j++)
                                        {
                                            core::aabbox3df bbox;
                                            bbox.MinEdge.X = boneDataForInstance[j].MinBBoxEdge[0];
                                            bbox.MinEdge.Y = boneDataForInstance[j].MinBBoxEdge[1];
                                            bbox.MinEdge.Z = boneDataForInstance[j].MinBBoxEdge[2];
                                            bbox.MaxEdge.X = boneDataForInstance[j].MaxBBoxEdge[0];
                                            bbox.MaxEdge.Y = boneDataForInstance[j].MaxBBoxEdge[1];
                                            bbox.MaxEdge.Z = boneDataForInstance[j].MaxBBoxEdge[2];
                                            if (j)
                                                nodeBBox.addInternalBox(bbox);
                                            else
                                                nodeBBox = bbox;
                                        }

                                        currentInstance->attachedNode->setBoundingBox(nodeBBox);
                                    }
                                }
                            }
                            break;
                    #ifdef _DEBUG
                    default:
                        assert(false);
                        break;
                    #endif // _DEBUG
                    }
                }
            }
    };

} // end namespace scene
} // end namespace irr

#endif
