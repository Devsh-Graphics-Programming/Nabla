#ifndef __C_FINAL_BONE_HIERARCHY_H_INCLUDED__
#define __C_FINAL_BONE_HIERARCHY_H_INCLUDED__

#include "assert.h"
#include <vector>
#include <set>
#include <algorithm>
#include "ISkinnedMesh.h"
#include "IGPUMappedBuffer.h"
#include "quaternion.h"
#include "irrString.h"

namespace irr
{
namespace scene
{

    //! If it has no animation, make 1 frame of animation with LocalMatrix
    class CFinalBoneHierarchy : public IReferenceCounted
    {
        protected:
            virtual ~CFinalBoneHierarchy()
            {
                if (boneNames)
                    delete [] boneNames;
                if (boneFlatArray)
                    free(boneFlatArray);
                if (boneTreeLevelEnd)
                    free(boneTreeLevelEnd);
/**
                if (boundBuffer)
                    boundBuffer->drop();
**/
                if (keyframes)
                    free(keyframes);
                if (interpolatedAnimations)
                    free(interpolatedAnimations);
                if (nonInterpolatedAnimations)
                    free(nonInterpolatedAnimations);
            }
        public:
            #include "irrpack.h"
            struct BoneReferenceData
            {
                core::matrix4x3 PoseBindMatrix;
                float MinBBoxEdge[3];
                float MaxBBoxEdge[3];
                uint32_t parentOffsetRelative;
                uint32_t parentOffsetFromTop;
            } PACK_STRUCT;
            struct AnimationKeyData
            {
                float Rotation[4];
                float Position[3];
                float Scale[3];
                float Padding[2];
            } PACK_STRUCT;
            #include "irrunpack.h"


            CFinalBoneHierarchy(const std::vector<ICPUSkinnedMesh::SJoint*>& inLevelFixedJoints, const std::vector<size_t>& inJointsLevelEnd)
                    : boneCount(inLevelFixedJoints.size()), NumLevelsInHierarchy(inJointsLevelEnd.size()),
                    ///boundBuffer(NULL),
                    keyframeCount(0), keyframes(NULL), interpolatedAnimations(NULL), nonInterpolatedAnimations(NULL)
            {
                boneFlatArray = (BoneReferenceData*)malloc(sizeof(BoneReferenceData)*boneCount);
                boneNames = new core::stringc[boneCount];
                for (size_t i=0; i<boneCount; i++)
                {
                    ICPUSkinnedMesh::SJoint* joint = inLevelFixedJoints[i];
                    boneFlatArray[i].PoseBindMatrix = joint->GlobalInversedMatrix;
                    boneFlatArray[i].MinBBoxEdge[0] = joint->bbox.MinEdge.X;
                    boneFlatArray[i].MinBBoxEdge[1] = joint->bbox.MinEdge.Y;
                    boneFlatArray[i].MinBBoxEdge[2] = joint->bbox.MinEdge.Z;
                    boneFlatArray[i].MaxBBoxEdge[0] = joint->bbox.MaxEdge.X;
                    boneFlatArray[i].MaxBBoxEdge[1] = joint->bbox.MaxEdge.Y;
                    boneFlatArray[i].MaxBBoxEdge[2] = joint->bbox.MaxEdge.Z;
                    if (joint->Parent)
                    {
                        size_t n=0;
                        for (; n<i; n++)
                        {
                            if (joint->Parent==inLevelFixedJoints[n])
                                break;
                        }
                        assert(n<i);
                        boneFlatArray[i].parentOffsetRelative = i-n;
                        boneFlatArray[i].parentOffsetFromTop = n;
                    }
                    else
                    {
                        boneFlatArray[i].parentOffsetRelative = 0;
                        boneFlatArray[i].parentOffsetFromTop = i;
                    }
                }

                boneTreeLevelEnd = (size_t*)malloc(sizeof(size_t)*NumLevelsInHierarchy);
                memcpy(boneTreeLevelEnd,inJointsLevelEnd.data(),sizeof(size_t)*NumLevelsInHierarchy);

                createAnimationKeys(inLevelFixedJoints);
            }

            inline const size_t& getBoneCount() const {return boneCount;}

            inline const BoneReferenceData* getBoneData() const
            {
                return boneFlatArray;
            }

            inline const core::stringc& getBoneName(const size_t& boneID) const
            {
                return boneNames[boneID];
            }

            inline size_t getBoneIDFromName(const char* name) const
            {
                for (size_t i=0; i<boneCount; i++)
                {
                    if (boneNames[i]==name)
                        return i;
                }

                return 0xdeadbeefu;
            }

            inline const size_t& getHierarchyLevels() const {return NumLevelsInHierarchy;}


            inline const size_t& getBoneLevelRangeStart(const size_t& level) const
            {
                if (level)
                    return boneTreeLevelEnd[level-1];
                else
                    return 0;
            }
            inline const size_t& getBoneLevelRangeLength(const size_t& level) const
            {
                if (level)
                    return boneTreeLevelEnd[level]-boneTreeLevelEnd[level-1];
                else
                    return boneTreeLevelEnd[0];
            }
            inline const size_t& getBoneLevelRangeEnd(const size_t& level) const
            {
                return boneTreeLevelEnd[level];
            }

            inline const size_t& getKeyFrameCount() const {return keyframeCount;}

            inline const float* getKeys() const {return keyframes;}
/**
            //! ready but untested
            inline void putInGPUBuffer(video::IGPUBuffer* buffer, const size_t& byteOffset=0)
            {
                size_t boneDataSize = sizeof(BoneReferenceData)*boneCount;
                if (!buffer||byteOffset+boneDataSize>buffer->getSize())
                    return;
                else if (buffer->isMappedBuffer())
                {
                    if (!dynamic_cast<video::IGPUMappedBuffer*>(buffer)->getPointer())
                        return;
                }
                else if (!buffer->canUpdateSubRange())
                    return;

                buffer->grab();
                if (boundBuffer)
                    boundBuffer->drop();
                boundBuffer = buffer;

                if (buffer->isMappedBuffer())
                    memcpy(reinterpret_cast<uint8_t*>(dynamic_cast<video::IGPUMappedBuffer*>(buffer)->getPointer())+byteOffset,boneFlatArray,boneDataSize);
                else
                    buffer->updateSubRange(byteOffset,boneDataSize,boneFlatArray);
            }
**/


            inline size_t getLowerBoundBoneKeyframes(float& interpolationFactor, const float& frame) const
            {

                size_t foundFrameIx = std::lower_bound(keyframes,keyframes+keyframeCount,frame)-keyframes;

                if (foundFrameIx)
                {
                    if (foundFrameIx==keyframeCount)
                    {
                        interpolationFactor = 1.f;
                        foundFrameIx--;
                    }
                    else//interpolationFactor will always be >0.f
                        interpolationFactor = (frame-keyframes[foundFrameIx-1])/(keyframes[foundFrameIx]-keyframes[foundFrameIx-1]); //! never a divide by zero, asserts make sure!!!
                }
                else
                {
                    interpolationFactor = 1.f;
                }

                return foundFrameIx;
            }

            inline size_t getLowerBoundBoneKeyframes(const float& frame) const
            {
                float tmpDummy;
                return getLowerBoundBoneKeyframes(tmpDummy,frame);
            }

            inline const AnimationKeyData* getInterpolatedAnimationData(const size_t& boneID=0) const {return interpolatedAnimations+keyframeCount*boneID;}

            inline const AnimationKeyData* getNonInterpolatedAnimationData(const size_t& boneID=0) const {return nonInterpolatedAnimations+keyframeCount*boneID;}


            //interpolant of 1 means full B
            static inline core::matrix4x3 getMatrixFromKeys(const AnimationKeyData& keyframeA, const AnimationKeyData& keyframeB, const float& interpolant, const float& interpolantPrecalcTerm2, const float& interpolantPrecalcTerm3)
            {
                core::matrix4x3 outMatrix;

                core::vectorSIMDf tmpPosA(keyframeA.Position);
                core::vectorSIMDf tmpPosB(keyframeB.Position);
                core::vectorSIMDf tmpPos = (tmpPosB-tmpPosA)*interpolant+tmpPosA;

                core::quaternion tmpRotA(keyframeA.Rotation);
                core::quaternion tmpRotB(keyframeB.Rotation);
                const float angle = tmpRotA.dotProduct(tmpRotB).X;
                core::quaternion tmpRot = core::quaternion::normalize(core::quaternion::lerp(tmpRotA,tmpRotB,core::quaternion::flerp_adjustedinterpolant(fabsf(angle),interpolant,interpolantPrecalcTerm2,interpolantPrecalcTerm3),angle<0.f));
                tmpRot.getMatrix(outMatrix,tmpPos);

                core::vectorSIMDf tmpScaleA(keyframeA.Scale);
                core::vectorSIMDf tmpScaleB(keyframeB.Scale);
                core::vectorSIMDf tmpScale = (tmpScaleB-tmpScaleA)*interpolant+tmpScaleA;
                outMatrix(0,0) *= tmpScale.X;
                outMatrix(1,0) *= tmpScale.X;
                outMatrix(2,0) *= tmpScale.X;
                outMatrix(0,1) *= tmpScale.Y;
                outMatrix(1,1) *= tmpScale.Y;
                outMatrix(2,1) *= tmpScale.Y;
                outMatrix(0,2) *= tmpScale.Z;
                outMatrix(1,2) *= tmpScale.Z;
                outMatrix(2,2) *= tmpScale.Z;

                return outMatrix;
            }
            static inline core::matrix4x3 getMatrixFromKeys(const AnimationKeyData& keyframeA, const AnimationKeyData& keyframeB, const float& interpolant)
            {
                float interpolantPrecalcTerm2,interpolantPrecalcTerm3;
                core::quaternion::flerp_interpolant_terms(interpolantPrecalcTerm2,interpolantPrecalcTerm3,interpolant);
                return getMatrixFromKeys(keyframeA,keyframeB,interpolant,interpolantPrecalcTerm2,interpolantPrecalcTerm3);
            }
            static inline core::matrix4x3 getMatrixFromKey(const AnimationKeyData& keyframe)
            {
                return getMatrixFromKeys(keyframe,keyframe,1.f,0.25f,0.f);
            }

        private:
            inline void createAnimationKeys(const std::vector<ICPUSkinnedMesh::SJoint*>& inLevelFixedJoints)
            {
                std::set<float> sortedFrames;
                for (size_t i=0; i<boneCount; i++)
                {
                    ICPUSkinnedMesh::SJoint* joint = inLevelFixedJoints[i];
                    for (size_t j=0; j<joint->RotationKeys.size(); j++)
                        sortedFrames.insert(joint->RotationKeys[j].frame);

                    for (size_t j=0; j<joint->PositionKeys.size(); j++)
                        sortedFrames.insert(joint->PositionKeys[j].frame);

                    for (size_t j=0; j<joint->ScaleKeys.size(); j++)
                        sortedFrames.insert(joint->ScaleKeys[j].frame);
                }

                if (sortedFrames.size()==0)
                    sortedFrames.insert(0.f);

                keyframeCount = sortedFrames.size();
                keyframes = (float*)malloc(sizeof(float)*keyframeCount);
                std::copy(sortedFrames.begin(),sortedFrames.end(),keyframes);
                std::sort(keyframes,keyframes+keyframeCount);

                interpolatedAnimations = (AnimationKeyData*)malloc(sizeof(AnimationKeyData)*keyframeCount*boneCount);
                nonInterpolatedAnimations = (AnimationKeyData*)malloc(sizeof(AnimationKeyData)*keyframeCount*boneCount);
                for (size_t i=0; i<boneCount; i++)
                {
                    AnimationKeyData* tmpAnimationInterpol = interpolatedAnimations+keyframeCount*i;
                    AnimationKeyData* tmpAnimationNonInterpol = nonInterpolatedAnimations+keyframeCount*i;

                    ICPUSkinnedMesh::SJoint* joint = inLevelFixedJoints[i];
                    bool HasAnyKeys = joint->RotationKeys.size()>0||joint->PositionKeys.size()||joint->ScaleKeys.size();
                    switch (joint->RotationKeys.size())
                    {
                        case 0:
                            {
                                core::vector3df rotationDegs = joint->LocalMatrix.getRotationDegrees();
                                core::quaternion rotationQuat;
                                if (!HasAnyKeys)
                                    rotationQuat = core::quaternion::fromEuler(rotationDegs*core::DEGTORAD);
                                for (size_t m=0; m<keyframeCount; m++)
                                {
                                    memcpy(tmpAnimationNonInterpol[m].Rotation,rotationQuat.getPointer(),16);
                                    memcpy(tmpAnimationInterpol[m].Rotation,rotationQuat.getPointer(),16);
                                }
                            }
                            break;
                        default:
                            {
                                core::quaternion current = joint->RotationKeys[0].rotation;
                                size_t foundIndex = std::lower_bound(keyframes,keyframes+keyframeCount,joint->RotationKeys[0].frame)-keyframes;
                                for (size_t m=0; m<foundIndex; m++)
                                {
                                    memcpy(tmpAnimationNonInterpol[m].Rotation,current.getPointer(),16);
                                    memcpy(tmpAnimationInterpol[m].Rotation,current.getPointer(),16);
                                }

                                for (size_t j=1; j<joint->RotationKeys.size(); j++)
                                {
                                    float currentFrame = joint->RotationKeys[j-1].frame;
                                    float nextFrame = joint->RotationKeys[j].frame;
                                    assert(nextFrame>currentFrame);

                                    core::quaternion next = joint->RotationKeys[j].rotation;

                                    size_t nextIndex = std::lower_bound(keyframes+foundIndex+1,keyframes+keyframeCount,nextFrame)-keyframes;
                                    assert(nextIndex>foundIndex);
                                    assert(nextIndex<keyframeCount);
                                    for (; foundIndex<nextIndex; foundIndex++)
                                    {
                                        memcpy(tmpAnimationNonInterpol[foundIndex].Rotation,current.getPointer(),16);

                                        const float fd1 = keyframes[foundIndex] - currentFrame;
                                        const float dWidth = nextFrame-currentFrame;
                                        core::quaternion rotation = core::quaternion::slerp(current, next, fd1/dWidth);
                                        memcpy(tmpAnimationInterpol[foundIndex].Rotation,rotation.getPointer(),16);
                                    }

                                    current = next;
                                }

                                for (; foundIndex<keyframeCount; foundIndex++)
                                {
                                    memcpy(tmpAnimationNonInterpol[foundIndex].Rotation,current.getPointer(),16);
                                    memcpy(tmpAnimationInterpol[foundIndex].Rotation,current.getPointer(),16);
                                }
                            }
                            break;
                    }
                    switch (joint->PositionKeys.size())
                    {
                        case 0:
                            {
                                core::vector3df translation;
                                if (!HasAnyKeys)
                                    translation = joint->LocalMatrix.getTranslation();
                                for (size_t m=0; m<keyframeCount; m++)
                                {
                                    *reinterpret_cast<core::vector3df*>(tmpAnimationNonInterpol[m].Position) = translation;
                                    *reinterpret_cast<core::vector3df*>(tmpAnimationInterpol[m].Position) = translation;
                                }
                            }
                            break;
                        default:
                            {
                                core::vector3df current = joint->PositionKeys[0].position;
                                size_t foundIndex = std::lower_bound(keyframes,keyframes+keyframeCount,joint->PositionKeys[0].frame)-keyframes;
                                for (size_t m=0; m<foundIndex; m++)
                                {
                                    memcpy(tmpAnimationNonInterpol[m].Position,&current,12);
                                    memcpy(tmpAnimationInterpol[m].Position,&current,12);
                                }

                                for (size_t j=1; j<joint->PositionKeys.size(); j++)
                                {
                                    float currentFrame = joint->PositionKeys[j-1].frame;
                                    float nextFrame = joint->PositionKeys[j].frame;
                                    assert(nextFrame>currentFrame);

                                    core::vector3df next = joint->PositionKeys[j].position;

                                    size_t nextIndex = std::lower_bound(keyframes+foundIndex+1,keyframes+keyframeCount,nextFrame)-keyframes;
                                    assert(nextIndex>foundIndex);
                                    assert(nextIndex<keyframeCount);
                                    for (; foundIndex<nextIndex; foundIndex++)
                                    {
                                        memcpy(tmpAnimationNonInterpol[foundIndex].Position,&current,12);

                                        const float fd1 = keyframes[foundIndex] - currentFrame;
                                        core::vector3df position = (next-current)*fd1/(nextFrame-currentFrame) + current;
                                        memcpy(tmpAnimationInterpol[foundIndex].Position,&position,12);
                                    }

                                    current = next;
                                }

                                for (; foundIndex<keyframeCount; foundIndex++)
                                {
                                    memcpy(tmpAnimationNonInterpol[foundIndex].Position,&current,12);
                                    memcpy(tmpAnimationInterpol[foundIndex].Position,&current,12);
                                }
                            }
                            break;
                    }
                    switch (joint->ScaleKeys.size())
                    {
                        case 0:
                            {
                                core::vector3df scale(1.f);
                                if (!HasAnyKeys)
                                    scale = joint->LocalMatrix.getScale();
                                for (size_t m=0; m<keyframeCount; m++)
                                {
                                    *reinterpret_cast<core::vector3df*>(tmpAnimationNonInterpol[m].Scale) = scale;
                                    *reinterpret_cast<core::vector3df*>(tmpAnimationInterpol[m].Scale) = scale;
                                }
                            }
                            break;
                        default:
                            {
                                core::vector3df current = joint->ScaleKeys[0].scale;
                                size_t foundIndex = std::lower_bound(keyframes,keyframes+keyframeCount,joint->ScaleKeys[0].frame)-keyframes;
                                for (size_t m=0; m<foundIndex; m++)
                                {
                                    memcpy(tmpAnimationNonInterpol[m].Scale,&current,12);
                                    memcpy(tmpAnimationInterpol[m].Scale,&current,12);
                                }

                                for (size_t j=1; j<joint->ScaleKeys.size(); j++)
                                {
                                    float currentFrame = joint->ScaleKeys[j-1].frame;
                                    float nextFrame = joint->ScaleKeys[j].frame;
                                    assert(nextFrame>currentFrame);

                                    core::vector3df next = joint->ScaleKeys[j].scale;

                                    size_t nextIndex = std::lower_bound(keyframes+foundIndex+1,keyframes+keyframeCount,nextFrame)-keyframes;
                                    assert(nextIndex>foundIndex);
                                    assert(nextIndex<keyframeCount);
                                    for (; foundIndex<nextIndex; foundIndex++)
                                    {
                                        memcpy(tmpAnimationNonInterpol[foundIndex].Scale,&current,12);

                                        const float fd1 = keyframes[foundIndex] - currentFrame;
                                        core::vector3df scale = (next-current)*fd1/(nextFrame-currentFrame) + current;
                                        memcpy(tmpAnimationInterpol[foundIndex].Scale,&scale,12);
                                    }

                                    current = next;
                                }

                                for (; foundIndex<keyframeCount; foundIndex++)
                                {
                                    memcpy(tmpAnimationNonInterpol[foundIndex].Scale,&current,12);
                                    memcpy(tmpAnimationInterpol[foundIndex].Scale,&current,12);
                                }
                            }
                            break;
                    }
                    /*
                    //debug
                    if (!HasAnyKeys)
                    {
                        core::matrix4x3 diff = joint->LocalMatrix-getMatrixFromKey(tmpAnimationNonInterpol[0]);
                        printf("PosDiff %f,%f,%f \t %f,%f,%f \t %f,%f,%f\n",diff.getColumn(0).X,diff.getColumn(0).Y,diff.getColumn(0).Z,
                                                                            diff.getColumn(1).X,diff.getColumn(1).Y,diff.getColumn(1).Z,
                                                                            diff.getColumn(2).X,diff.getColumn(2).Y,diff.getColumn(2).Z);

                        core::vector3df scale = joint->LocalMatrix.getScale()-joint->LocalMatrix-getMatrixFromKey(tmpAnimationNonInterpol[0]).getScale();
                    }*/
                }
            }

            const size_t boneCount;
            BoneReferenceData* boneFlatArray;
            core::stringc* boneNames;
            const size_t NumLevelsInHierarchy;
            size_t* boneTreeLevelEnd;

            ///video::IGPUBuffer* boundBuffer;

            size_t keyframeCount;
            float* keyframes;
            AnimationKeyData* interpolatedAnimations;
            AnimationKeyData* nonInterpolatedAnimations;
    };

} // end namespace scene
} // end namespace irr

#endif

