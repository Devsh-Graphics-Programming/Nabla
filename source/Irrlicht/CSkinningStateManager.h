#ifndef __C_SKINNING_STATE_MANAGER_H_INCLUDED__
#define __C_SKINNING_STATE_MANAGER_H_INCLUDED__


#include "ISkinningStateManager.h"
#include "ITextureBufferObject.h"

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
            CSkinningStateManager(const E_BONE_UPDATE_MODE& boneControl, video::IVideoDriver* driver, const CFinalBoneHierarchy* sourceHierarchy)
                                    : ISkinningStateManager(boneControl,driver,sourceHierarchy), Driver(driver)
            {
#ifdef _IRR_COMPILE_WITH_OPENGL_
                TBO = driver->addTextureBufferObject(finalBoneDataInstanceBuffer->getFrontBuffer(),video::ITextureBufferObject::ETBOF_RGBA32F);
#endif // _IRR_COMPILE_WITH_OPENGL_
            }

            const void* getRawBoneData() {return finalBoneDataInstanceBuffer->getBackBufferPointer();}

#ifdef _IRR_COMPILE_WITH_OPENGL_
            virtual video::ITextureBufferObject* getBoneDataTBO() const {return TBO;}
#else
            virtual video::ITextureBufferObject* getBoneDataTBO() const {return NULL;}
#endif // _IRR_COMPILE_WITH_OPENGL_

            virtual uint32_t addInstance(ISkinnedMeshSceneNode* attachedNode=NULL, const bool& createBoneNodes=false)
            {
                uint32_t newID;
                if (!finalBoneDataInstanceBuffer->Alloc(&newID,1))
                    return 0xdeadbeefu;

                //grow instanceData
                if (finalBoneDataInstanceBuffer->getCapacity()!=instanceDataSize)
                {
                    instanceDataSize = finalBoneDataInstanceBuffer->getCapacity();
#ifdef _IRR_COMPILE_WITH_OPENGL_
                    if (TBO->getByteSize()!=finalBoneDataInstanceBuffer->getFrontBuffer()->getSize())
                        TBO->bind(finalBoneDataInstanceBuffer->getFrontBuffer(),video::ITextureBufferObject::ETBOF_RGBA32F); //can't clandestine re-bind because it won't change the length :D
#endif // _IRR_COMPILE_WITH_OPENGL_
                    size_t instanceDataByteSize = instanceDataSize*actualSizeOfInstanceDataElement;
                    if (instanceData)
                        instanceData = (uint8_t*)realloc(instanceData,instanceDataByteSize);
                    else
                        instanceData = (uint8_t*)malloc(instanceDataByteSize);
                }


                size_t redirect = finalBoneDataInstanceBuffer->getRedirectFromID(newID);
                BoneHierarchyInstanceData* tmp = reinterpret_cast<BoneHierarchyInstanceData*>(instanceData+redirect*actualSizeOfInstanceDataElement);
                tmp->refCount = 1;
                tmp->frame = 0.f;
                tmp->interpolateAnimation = true;
                tmp->attachedNode = attachedNode;
                if (boneControlMode!=EBUM_CONTROL)
                {
                    FinalBoneData* boneData = reinterpret_cast<FinalBoneData*>(finalBoneDataInstanceBuffer->getBackBufferPointer());
                    boneData += redirect*referenceHierarchy->getBoneCount();
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
                    case EBUM_CONTROL:
                        tmp->needToRecomputeParentBBox = false;
                        for (size_t i=0; i<referenceHierarchy->getBoneCount(); i++)
                        {
                            const CFinalBoneHierarchy::BoneReferenceData& boneData = referenceHierarchy->getBoneData()[i];
                            core::matrix4x3 localMatrix = CFinalBoneHierarchy::getMatrixFromKey(referenceHierarchy->getNonInterpolatedAnimationData(i)[0]).getAsRetardedIrrlichtMatrix();

                            IBoneSceneNode* tmpBone;
                            if (boneData.parentOffsetRelative)
                            {
                                tmpBone = new IBoneSceneNode(this,newID,getBones(tmp)[boneData.parentOffsetFromTop],i,localMatrix); //want first frame
                            }
                            else
                                tmpBone = new IBoneSceneNode(this,newID,attachedNode,i,localMatrix);

                            getBones(tmp)[i] = tmpBone;
                        }
                        break;
                }


                if (redirect<=firstDirtyInstance)
                {
                    firstDirtyInstance = redirect;
                    firstDirtyBone = 0;
                }
                if (redirect>=lastDirtyInstance)
                {
                    lastDirtyInstance = redirect;
                    lastDirtyBone = referenceHierarchy->getBoneCount()-1;
                }

                return newID;
            }

            //! true if deleted
            virtual bool dropInstance(const uint32_t& ID)
            {
                size_t oldInstanceDataSize = instanceDataSize;
                if (ISkinningStateManager::dropInstance(ID))
                {
#ifdef _IRR_COMPILE_WITH_OPENGL_
                    if (oldInstanceDataSize!=instanceDataSize && TBO->getByteSize()!=finalBoneDataInstanceBuffer->getFrontBuffer()->getSize())
                        TBO->bind(finalBoneDataInstanceBuffer->getFrontBuffer(),video::ITextureBufferObject::ETBOF_RGBA32F); //can't clandestine re-bind because it won't change the length :D
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

                size_t redirect = finalBoneDataInstanceBuffer->getRedirectFromID(instanceID);
                BoneHierarchyInstanceData* tmp = reinterpret_cast<BoneHierarchyInstanceData*>(instanceData+redirect*actualSizeOfInstanceDataElement);
                for (size_t i=0; i<referenceHierarchy->getBoneCount(); i++)
                {
                    if (getBones(tmp)[i])
                        continue;

                    const CFinalBoneHierarchy::BoneReferenceData& boneData = referenceHierarchy->getBoneData()[i];
                    //core::matrix4x3 parentInverse;
                    //getGlobalMatrices(tmp)[boneData.parentOffsetFromTop].getInverse(parentInverse); // todo maybe inversion simd?
					core::matrix3x4SIMD parentInverse;
					core::matrix3x4SIMD().set(getGlobalMatrices(tmp)[boneData.parentOffsetFromTop]).getInverse(parentInverse);
					const core::matrix4x3 localMatrix = core::matrix3x4SIMD::concatenateBFollowedByA(parentInverse, core::matrix3x4SIMD().set(getGlobalMatrices(tmp)[i])).getAsRetardedIrrlichtMatrix();
					//concatenateBFollowedByA(parentInverse,getGlobalMatrices(tmp)[i]);

                    IBoneSceneNode* tmpBone;
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
                size_t redirect = finalBoneDataInstanceBuffer->getRedirectFromID(instanceID);
                assert(redirect<getDataInstanceCount());

                if (boneControlMode!=EBUM_READ)
                    return;

                BoneHierarchyInstanceData* currentInstance = reinterpret_cast<BoneHierarchyInstanceData*>(instanceData+redirect*actualSizeOfInstanceDataElement);
                if (currentInstance->frame==currentInstance->lastAnimatedFrame) //in other modes, check if also has no bones!!!
                    return;

                FinalBoneData* boneDataForInstance = reinterpret_cast<FinalBoneData*>(reinterpret_cast<uint8_t*>(finalBoneDataInstanceBuffer->getBackBufferPointer())+referenceHierarchy->getBoneCount()*redirect);
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

                if (redirect<firstDirtyInstance)
                {
                    firstDirtyInstance = redirect;
                    firstDirtyBone = boneStack[boneStackSize];
                }
                else if (redirect==firstDirtyInstance&&boneStack[boneStackSize]<firstDirtyBone)
                    firstDirtyBone = boneStack[boneStackSize];
                if (redirect>lastDirtyInstance)
                {
                    lastDirtyInstance = redirect;
                    lastDirtyBone = boneID;
                }
                else if (redirect==lastDirtyInstance&&boneID>lastDirtyBone)
                    lastDirtyBone = boneID;
                boneStackSize++;

                float interpolationFactor;
                size_t foundKeyIx = referenceHierarchy->getLowerBoundBoneKeyframes(interpolationFactor,currentInstance->frame);
                float interpolantPrecalcTerm2,interpolantPrecalcTerm3;
                core::quaternion::flerp_interpolant_terms(interpolantPrecalcTerm2,interpolantPrecalcTerm3,interpolationFactor);

                while (boneStackSize--)
                {
                    size_t j = boneStack[boneStackSize];
                    CFinalBoneHierarchy::AnimationKeyData upperFrame = (currentInstance->interpolateAnimation ? referenceHierarchy->getInterpolatedAnimationData(j):referenceHierarchy->getNonInterpolatedAnimationData(j))[foundKeyIx];

                    //core::matrix4x3 interpolatedLocalTform;
                    core::matrix3x4SIMD interpolatedLocalTform;
                    if (currentInstance->interpolateAnimation&&interpolationFactor<1.f)
                    {
                        CFinalBoneHierarchy::AnimationKeyData lowerFrame = (currentInstance->interpolateAnimation ? referenceHierarchy->getInterpolatedAnimationData(j):referenceHierarchy->getNonInterpolatedAnimationData(j))[foundKeyIx-1];
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
						//concatenateBFollowedByA(parentTform,interpolatedLocalTform);
                    }
					boneDataForInstance[j].SkinningTransform = core::matrix3x4SIMD::concatenateBFollowedByA(core::matrix3x4SIMD().set(getGlobalMatrices(currentInstance)[j]), core::matrix3x4SIMD().set(referenceHierarchy->getBoneData()[j].PoseBindMatrix)).getAsRetardedIrrlichtMatrix();
					//concatenateBFollowedByA(getGlobalMatrices(currentInstance)[j],referenceHierarchy->getBoneData()[j].PoseBindMatrix);


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
                    boneDataForInstance[j].SkinningTransform.getSub3x3InverseTranspose(boneDataForInstance[j].SkinningNormalMatrix);

                    boneDataForInstance[j].lastAnimatedFrame = currentInstance->frame;
                }
            }

            inline void TrySwapBoneBuffer()
            {
                if (firstDirtyInstance<=lastDirtyInstance)
                {
                    finalBoneDataInstanceBuffer->SwapBuffers();
#ifdef _IRR_COMPILE_WITH_OPENGL_
                    if (TBO->getByteSize()!=finalBoneDataInstanceBuffer->getFrontBuffer()->getSize())
                        TBO->bind(finalBoneDataInstanceBuffer->getFrontBuffer(),video::ITextureBufferObject::ETBOF_RGBA32F); //can't clandestine re-bind because it won't change the length :D
#endif // _IRR_COMPILE_WITH_OPENGL_
                    firstDirtyInstance = 0xdeadbeefu;
                    lastDirtyInstance = 0;
                    firstDirtyBone = 0xdeadbeefu;
                    lastDirtyBone = 0;
                }
            }

            virtual void performBoning()
            {
                if (referenceHierarchy->getHierarchyLevels()==0||getDataInstanceCount()==0)
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
                                FinalBoneData* boneData = reinterpret_cast<FinalBoneData*>(finalBoneDataInstanceBuffer->getBackBufferPointer());
                                bool notModified = true;
                                uint32_t localFirstDirtyInstance,localLastDirtyInstance,firstBone,lastBone;
                                for (size_t i=0; i<getDataInstanceCount(); i++)
                                {
                                    BoneHierarchyInstanceData* currentInstance = reinterpret_cast<BoneHierarchyInstanceData*>(instanceData+i*actualSizeOfInstanceDataElement);
                                    if (currentInstance->frame==currentInstance->lastAnimatedFrame) //in other modes, check if also has no bones!!!
                                        continue;

                                    core::matrix4x3 attachedNodeTform;
                                    if (currentInstance->attachedNode)
                                        attachedNodeTform = currentInstance->attachedNode->getAbsoluteTransformation();


                                    float interpolationFactor;
                                    size_t foundBoneIx = referenceHierarchy->getLowerBoundBoneKeyframes(interpolationFactor,currentInstance->frame);
                                    float interpolantPrecalcTerm2,interpolantPrecalcTerm3;
                                    core::quaternion::flerp_interpolant_terms(interpolantPrecalcTerm2,interpolantPrecalcTerm3,interpolationFactor);


                                    FinalBoneData* boneDataForInstance = boneData+referenceHierarchy->getBoneCount()*i;
                                    for (size_t j=0; j<referenceHierarchy->getBoneCount(); j++)
                                    {
                                        if (boneDataForInstance[j].lastAnimatedFrame==currentInstance->frame)
                                            continue;
                                        if (notModified)
                                        {
                                            localFirstDirtyInstance = i;
                                            firstBone = j;
                                            notModified = false;
                                        }
                                        localLastDirtyInstance = i;
                                        lastBone = j;
                                        boneDataForInstance[j].lastAnimatedFrame = currentInstance->frame;

                                        CFinalBoneHierarchy::AnimationKeyData upperFrame = (currentInstance->interpolateAnimation ? referenceHierarchy->getInterpolatedAnimationData(j):referenceHierarchy->getNonInterpolatedAnimationData(j))[foundBoneIx];

                                        core::matrix4x3 interpolatedLocalTform;
                                        if (currentInstance->interpolateAnimation&&interpolationFactor<1.f)
                                        {
                                            CFinalBoneHierarchy::AnimationKeyData lowerFrame =  (currentInstance->interpolateAnimation ? referenceHierarchy->getInterpolatedAnimationData(j):referenceHierarchy->getNonInterpolatedAnimationData(j))[foundBoneIx-1];
                                            interpolatedLocalTform = referenceHierarchy->getMatrixFromKeys(lowerFrame,upperFrame,interpolationFactor,interpolantPrecalcTerm2,interpolantPrecalcTerm3).getAsRetardedIrrlichtMatrix();
                                        }
                                        else
                                            interpolatedLocalTform = referenceHierarchy->getMatrixFromKey(upperFrame).getAsRetardedIrrlichtMatrix();

                                        if (j < referenceHierarchy->getBoneLevelRangeEnd(0))
                                            getGlobalMatrices(currentInstance)[j] = interpolatedLocalTform;
                                        else
                                        {
                                            const core::matrix4x3& parentTform = getGlobalMatrices(currentInstance)[referenceHierarchy->getBoneData()[j].parentOffsetFromTop];
                                            getGlobalMatrices(currentInstance)[j] = core::matrix3x4SIMD::concatenateBFollowedByA(core::matrix3x4SIMD().set(parentTform), core::matrix3x4SIMD().set(interpolatedLocalTform)).getAsRetardedIrrlichtMatrix();
											//concatenateBFollowedByA(parentTform,interpolatedLocalTform);
                                        }
                                        boneDataForInstance[j].SkinningTransform = core::matrix3x4SIMD::concatenateBFollowedByA(core::matrix3x4SIMD().set(getGlobalMatrices(currentInstance)[j]), core::matrix3x4SIMD().set(referenceHierarchy->getBoneData()[j].PoseBindMatrix)).getAsRetardedIrrlichtMatrix();
										//concatenateBFollowedByA(getGlobalMatrices(currentInstance)[j],referenceHierarchy->getBoneData()[j].PoseBindMatrix);


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
                                                    bone->setRelativeTransformationMatrix(interpolatedLocalTform);
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
                                        boneDataForInstance[j].SkinningTransform.getSub3x3InverseTranspose(boneDataForInstance[j].SkinningNormalMatrix);
                                    }
                                }

                                if (!notModified)
                                {
                                    if (localFirstDirtyInstance<firstDirtyInstance)
                                    {
                                        firstDirtyInstance = localFirstDirtyInstance;
                                        firstDirtyBone = firstBone;
                                    }
                                    else if (localFirstDirtyInstance==firstDirtyInstance&&firstBone<firstDirtyBone)
                                        firstDirtyBone = firstBone;
                                    if (localLastDirtyInstance>lastDirtyInstance)
                                    {
                                        lastDirtyInstance = localLastDirtyInstance;
                                        lastDirtyBone = lastBone;
                                    }
                                    else if (localLastDirtyInstance==lastDirtyInstance&&lastBone>lastDirtyBone)
                                        lastDirtyBone = lastBone;


                                    TrySwapBoneBuffer();

                                    for (size_t i=localFirstDirtyInstance; i<=localLastDirtyInstance; i++)
                                    {
                                        BoneHierarchyInstanceData* currentInstance = reinterpret_cast<BoneHierarchyInstanceData*>(instanceData+i*actualSizeOfInstanceDataElement);
                                        if (currentInstance->frame==currentInstance->lastAnimatedFrame) //in other modes, check if also has no bones!!!
                                            continue;
                                        currentInstance->lastAnimatedFrame = currentInstance->frame;

                                        core::aabbox3df nodeBBox;
                                        FinalBoneData* boneDataForInstance = boneData+referenceHierarchy->getBoneCount()*i;
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

                                            if (boneControlMode==EBUM_READ)
                                            {
                                                IBoneSceneNode* bone = getBones(currentInstance)[j];
                                                if (!bone)
                                                    continue;

                                                bone->updateAbsolutePosition();
                                            }
                                        }

                                        if (currentInstance->attachedNode)
                                            currentInstance->attachedNode->setBoundingBox(nodeBBox);
                                    }
                                }
                                else
                                    TrySwapBoneBuffer();
                            }
                            break;
                        case EBUM_CONTROL:
                            {
                                FinalBoneData* boneData = reinterpret_cast<FinalBoneData*>(finalBoneDataInstanceBuffer->getBackBufferPointer());
                                bool notModified = true;
                                uint32_t localFirstDirtyInstance,localLastDirtyInstance,firstBone,lastBone;
                                for (size_t i=0; i<getDataInstanceCount(); i++)
                                {
                                    BoneHierarchyInstanceData* currentInstance = reinterpret_cast<BoneHierarchyInstanceData*>(instanceData+i*actualSizeOfInstanceDataElement);
                                    FinalBoneData* boneDataForInstance = boneData+referenceHierarchy->getBoneCount()*i;

                                    core::matrix3x4SIMD attachedNodeInverse;
									//core::matrix4x3 attachedNodeInverse;
                                    if (currentInstance->attachedNode)
                                    {
                                        currentInstance->attachedNode->updateAbsolutePosition();
                                        //currentInstance->attachedNode->getAbsoluteTransformation().getInverse(attachedNodeInverse); // todo maybe simd inversion?
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


										boneDataForInstance[j].SkinningTransform = core::matrix3x4SIMD::concatenateBFollowedByA(attachedNodeInverse, core::matrix3x4SIMD::concatenateBFollowedByA(core::matrix3x4SIMD().set(bone->getAbsoluteTransformation()), core::matrix3x4SIMD().set(referenceHierarchy->getBoneData()[j].PoseBindMatrix))).getAsRetardedIrrlichtMatrix();
											//concatenateBFollowedByA(attachedNodeInverse,concatenateBFollowedByA(bone->getAbsoluteTransformation(),referenceHierarchy->getBoneData()[j].PoseBindMatrix)); //!may not be FP precise enough :(

                                        boneDataForInstance[j].SkinningTransform.getSub3x3InverseTranspose(boneDataForInstance[j].SkinningNormalMatrix);

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
                                                firstBone = j;
                                                notModified = false;
                                            }
                                            localNotModified = false;
                                        }
                                        localLastDirtyInstance = i;
                                        lastBone = j;
                                    }

                                    currentInstance->needToRecomputeParentBBox = localNotModified;
                                }

                                if (!notModified)
                                {
                                    if (localFirstDirtyInstance<firstDirtyInstance)
                                    {
                                        firstDirtyInstance = localFirstDirtyInstance;
                                        firstDirtyBone = firstBone;
                                    }
                                    else if (localFirstDirtyInstance==firstDirtyInstance&&firstBone<firstDirtyBone)
                                        firstDirtyBone = firstBone;
                                    if (localLastDirtyInstance>lastDirtyInstance)
                                    {
                                        lastDirtyInstance = localLastDirtyInstance;
                                        lastDirtyBone = lastBone;
                                    }
                                    else if (localLastDirtyInstance==lastDirtyInstance&&lastBone>lastDirtyBone)
                                        lastDirtyBone = lastBone;


                                    TrySwapBoneBuffer();

                                    for (size_t i=localFirstDirtyInstance; i<=localLastDirtyInstance; i++)
                                    {
                                        BoneHierarchyInstanceData* currentInstance = reinterpret_cast<BoneHierarchyInstanceData*>(instanceData+i*actualSizeOfInstanceDataElement);
                                        if (!currentInstance->attachedNode || currentInstance->needToRecomputeParentBBox)
                                            continue;

                                        core::aabbox3df nodeBBox;
                                        FinalBoneData* boneDataForInstance = boneData+referenceHierarchy->getBoneCount()*i;
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
                                else
                                    TrySwapBoneBuffer();
                            }
                            break;
                    }
                }
            }
    };

} // end namespace scene
} // end namespace irr

#endif
