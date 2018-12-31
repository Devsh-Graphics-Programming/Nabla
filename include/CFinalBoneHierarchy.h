#ifndef __C_FINAL_BONE_HIERARCHY_H_INCLUDED__
#define __C_FINAL_BONE_HIERARCHY_H_INCLUDED__

#include "assert.h"
#include <algorithm>
#include <functional>
#include "irr/asset/ICPUSkinnedMesh.h"
#include "IGPUBuffer.h"
#include "quaternion.h"
#include "irr/core/Types.h"
#include "irr/core/irrString.h"
#include "irr/asset/bawformat/CBAWFile.h"
#include "matrix3x4SIMD.h"

namespace irr
{
namespace scene
{
    //! If it has no animation, make 1 frame of animation with LocalMatrix
    class CFinalBoneHierarchy : public core::IReferenceCounted, public asset::BlobSerializable
    {
        protected:
            virtual ~CFinalBoneHierarchy()
            {
                if (boneNames)
                    _IRR_DELETE_ARRAY(boneNames,boneCount);
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
            #include "irr/irrpack.h"
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
            #include "irr/irrunpack.h"


            CFinalBoneHierarchy(const core::vector<asset::ICPUSkinnedMesh::SJoint*>& inLevelFixedJoints, const core::vector<size_t>& inJointsLevelEnd)
                    : boneCount(inLevelFixedJoints.size()), NumLevelsInHierarchy(inJointsLevelEnd.size()),
                    ///boundBuffer(NULL),
                    keyframeCount(0), keyframes(NULL), interpolatedAnimations(NULL), nonInterpolatedAnimations(NULL)
            {
                boneFlatArray = (BoneReferenceData*)malloc(sizeof(BoneReferenceData)*boneCount);
                boneNames = _IRR_NEW_ARRAY(core::stringc,boneCount);
                for (size_t i=0; i<boneCount; i++)
                {
                    asset::ICPUSkinnedMesh::SJoint* joint = inLevelFixedJoints[i];
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

			CFinalBoneHierarchy(const void* _bonesBegin, const void* _bonesEnd,
				core::stringc* _boneNamesBegin, core::stringc* _boneNamesEnd,
				const std::size_t* _levelsBegin, const std::size_t* _levelsEnd,
				const float* _keyframesBegin, const float* _keyframesEnd,
				const void* _interpAnimsBegin, const void* _interpAnimsEnd,
				const void* _nonInterpAnimsBegin, const void* _nonInterpAnimsEnd)
			: boneCount((BoneReferenceData*)_bonesEnd - (BoneReferenceData*)_bonesBegin), NumLevelsInHierarchy(_levelsEnd - _levelsBegin), keyframeCount(_keyframesEnd - _keyframesBegin)
			{
				_IRR_DEBUG_BREAK_IF(_bonesBegin > _bonesEnd ||
					_boneNamesBegin > _boneNamesEnd ||
					_levelsBegin > _levelsEnd ||
					_keyframesBegin > _keyframesEnd ||
					_interpAnimsBegin > _interpAnimsEnd ||
					_nonInterpAnimsBegin > _nonInterpAnimsEnd
				)
				_IRR_DEBUG_BREAK_IF(_boneNamesEnd - _boneNamesBegin != static_cast<std::make_signed<decltype(boneCount)>::type>(boneCount))
				_IRR_DEBUG_BREAK_IF((AnimationKeyData*)_interpAnimsEnd - (AnimationKeyData*)_interpAnimsBegin != static_cast<std::make_signed<decltype(boneCount)>::type>(getAnimationCount()))
				_IRR_DEBUG_BREAK_IF((AnimationKeyData*)_nonInterpAnimsEnd - (AnimationKeyData*)_nonInterpAnimsBegin != static_cast<std::make_signed<decltype(boneCount)>::type>(getAnimationCount()))

				boneNames = _IRR_NEW_ARRAY(core::stringc,boneCount);
				boneFlatArray = (BoneReferenceData*)malloc(sizeof(BoneReferenceData)*boneCount);
				boneTreeLevelEnd = (size_t*)malloc(sizeof(size_t)*NumLevelsInHierarchy);
				keyframes = (float*)malloc(sizeof(float)*keyframeCount);
				interpolatedAnimations = (AnimationKeyData*)malloc(sizeof(AnimationKeyData)*getAnimationCount());
				nonInterpolatedAnimations = (AnimationKeyData*)malloc(sizeof(AnimationKeyData)*getAnimationCount());

				for (size_t i = 0; i < boneCount; ++i)
					boneNames[i] = _boneNamesBegin[i];
				memcpy(boneFlatArray, _bonesBegin, sizeof(BoneReferenceData)*boneCount);
				memcpy(boneTreeLevelEnd, _levelsBegin, sizeof(size_t)*NumLevelsInHierarchy);
				memcpy(keyframes, _keyframesBegin, sizeof(float)*keyframeCount);
				memcpy(interpolatedAnimations, _interpAnimsBegin, sizeof(AnimationKeyData)*getAnimationCount());
				memcpy(nonInterpolatedAnimations, _nonInterpAnimsBegin, sizeof(AnimationKeyData)*getAnimationCount());
			}

			virtual void* serializeToBlob(void* _stackPtr = NULL, const size_t& _stackSize = 0) const
			{
				return asset::CorrespondingBlobTypeFor<CFinalBoneHierarchy>::type::createAndTryOnStack(static_cast<const CFinalBoneHierarchy*>(this), _stackPtr, _stackSize);
			}

			inline size_t getSizeOfAllBoneNames() const
			{
				size_t sum = 0;
				for (size_t i = 0; i < boneCount; ++i)
					sum += boneNames[i].size()+1;
				return sum;
			}
			static inline size_t getSizeOfSingleBone()
			{
				return sizeof(*boneFlatArray);
			}
			static inline size_t getSizeOfSingleAnimationData()
			{
				return sizeof(*interpolatedAnimations);
			}

            inline const size_t& getBoneCount() const {return boneCount;}

            inline const BoneReferenceData* getBoneData() const
            {
                return boneFlatArray;
            }

			const size_t* getBoneTreeLevelEnd() const { return boneTreeLevelEnd; }

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


            inline size_t getBoneLevelRangeStart(const size_t& level) const
            {
                if (level)
                    return boneTreeLevelEnd[level-1];
                else
                    return 0;
            }
            inline size_t getBoneLevelRangeLength(const size_t& level) const
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

			inline size_t getAnimationCount() const { return getKeyFrameCount()*getBoneCount(); }
/**
            //! ready but untested
            inline void putInGPUBuffer(video::IGPUBuffer* buffer, const size_t& byteOffset=0)
            {
                size_t boneDataSize = sizeof(BoneReferenceData)*boneCount;
                if (!buffer||byteOffset+boneDataSize>buffer->getSize())
                    return;
                else if (buffer->isMappedBuffer())
                {
                    if (!static_cast<video::IGPUMappedBuffer*>(buffer)->getPointer())
                        return;
                }
                else if (!buffer->canUpdateSubRange())
                    return;

                buffer->grab();
                if (boundBuffer)
                    boundBuffer->drop();
                boundBuffer = buffer;

                if (buffer->isMappedBuffer())
                    memcpy(reinterpret_cast<uint8_t*>(static_cast<video::IGPUMappedBuffer*>(buffer)->getPointer())+byteOffset,boneFlatArray,boneDataSize);
                else
                    buffer->updateSubRange(byteOffset,boneDataSize,boneFlatArray);
            }
**/


            inline size_t getLowerBoundBoneKeyframes(float& interpolationFactor, const float& frame, const float* found) const
            {
                if (found==keyframes) //first or before start
                {
                    interpolationFactor = 1.f;
                    return 0;
                }
                else if (found==keyframes+keyframeCount) //last or after start
                {
                    interpolationFactor = 1.f;
                    return keyframeCount-1;
                }
                //else

                //interpolationFactor will always be >0.f
                float prevKeyframe = *(found-1);
                interpolationFactor = (frame-prevKeyframe)/(*found-prevKeyframe); //! never a divide by zero, asserts make sure!!!
                return found-keyframes;
            }

            inline size_t getLowerBoundBoneKeyframes(float& interpolationFactor, const float& frame) const
            {
                const float* found = std::lower_bound(keyframes,keyframes+keyframeCount,frame);
                return getLowerBoundBoneKeyframes(interpolationFactor, frame, found);
            }

            inline size_t getLowerBoundBoneKeyframes(const float& frame) const
            {
                float tmpDummy;
                return getLowerBoundBoneKeyframes(tmpDummy,frame);
            }

            inline const AnimationKeyData* getInterpolatedAnimationData(const size_t& boneID=0) const {return interpolatedAnimations+keyframeCount*boneID;}

            inline const AnimationKeyData* getNonInterpolatedAnimationData(const size_t& boneID=0) const {return nonInterpolatedAnimations+keyframeCount*boneID;}


            //interpolant of 1 means full B
            static inline void getMatrixFromKeys(core::vectorSIMDf& outPos, core::quaternion& outQuat, core::vectorSIMDf& outScale,
                                                 const AnimationKeyData& keyframeA, const AnimationKeyData& keyframeB,
                                                 const float& interpolant, const float& interpolantPrecalcTerm2, const float& interpolantPrecalcTerm3)
            {
                core::vectorSIMDf tmpPosA(keyframeA.Position);
                core::vectorSIMDf tmpPosB(keyframeB.Position);
                outPos = (tmpPosB-tmpPosA)*interpolant+tmpPosA;

                core::quaternion tmpRotA(keyframeA.Rotation);
                core::quaternion tmpRotB(keyframeB.Rotation);
                const float angle = tmpRotA.dotProduct(tmpRotB).X;
                outQuat = core::quaternion::normalize(core::quaternion::lerp(tmpRotA,tmpRotB,core::quaternion::flerp_adjustedinterpolant(fabsf(angle),interpolant,interpolantPrecalcTerm2,interpolantPrecalcTerm3),angle<0.f));


                core::vectorSIMDf tmpScaleA(keyframeA.Scale);
                core::vectorSIMDf tmpScaleB(keyframeB.Scale);
                outScale = (tmpScaleB-tmpScaleA)*interpolant+tmpScaleA;
            }
            static inline core::matrix3x4SIMD getMatrixFromKeys(const AnimationKeyData& keyframeA, const AnimationKeyData& keyframeB, const float& interpolant, const float& interpolantPrecalcTerm2, const float& interpolantPrecalcTerm3)
            {
                core::vectorSIMDf   tmpPos;
                core::quaternion    tmpRot;
                core::vectorSIMDf   tmpScale;

                getMatrixFromKeys(tmpPos,tmpRot,tmpScale,keyframeA,keyframeB,interpolant,interpolantPrecalcTerm2,interpolantPrecalcTerm3);

                core::matrix3x4SIMD outMatrix;
                outMatrix.setScaleRotationAndTranslation(tmpScale, tmpRot, tmpPos);

                return outMatrix;
            }
            static inline core::matrix3x4SIMD getMatrixFromKeys(const AnimationKeyData& keyframeA, const AnimationKeyData& keyframeB, const float& interpolant)
            {
                float interpolantPrecalcTerm2,interpolantPrecalcTerm3;
                core::quaternion::flerp_interpolant_terms(interpolantPrecalcTerm2,interpolantPrecalcTerm3,interpolant);
                return getMatrixFromKeys(keyframeA,keyframeB,interpolant,interpolantPrecalcTerm2,interpolantPrecalcTerm3);
            }
            static inline core::matrix3x4SIMD getMatrixFromKey(const AnimationKeyData& keyframe)
            {
                return getMatrixFromKeys(keyframe,keyframe,1.f,0.25f,0.f);
            }

            //effectively downsamples our animation
            inline void deleteKeyframes(const size_t& keyframesToRemoveCount, const float* sortedKeyFramesToRemove)
            {
                const float* keyframesIn = keyframes;
                const float* const keyframesEnd = keyframes+keyframeCount;
                const AnimationKeyData* inAnimationsIn = interpolatedAnimations;
                const AnimationKeyData* noAnimationsIn = nonInterpolatedAnimations;

                float* keyframesOut = keyframes;
                AnimationKeyData* inAnimationsOut = interpolatedAnimations;
                AnimationKeyData* noAnimationsOut = nonInterpolatedAnimations;

                auto copyKeyframesFunc = [&](const float* rangeEnd)
                {
                    if (keyframesOut==keyframesIn)
                    {
                        size_t amountToMove = rangeEnd-keyframesIn;
                        keyframesOut += amountToMove;
                        keyframesIn += amountToMove;
                        inAnimationsOut += boneCount*amountToMove;
                        inAnimationsIn += boneCount*amountToMove;
                        noAnimationsOut += boneCount*amountToMove;
                        noAnimationsIn += boneCount*amountToMove;
                        return;
                    }

                    while (keyframesIn<rangeEnd)
                    {
                        *(keyframesOut++) = *(keyframesIn++);
                        for (size_t i=0; i<boneCount; i++)
                        {
                            *(inAnimationsOut++) = *(inAnimationsIn++);
                            *(noAnimationsOut++) = *(noAnimationsIn++);
                        }
                    }
                };

                for (const float* keyFramesToRemoveTmp=sortedKeyFramesToRemove; keyFramesToRemoveTmp<sortedKeyFramesToRemove+keyframesToRemoveCount; keyFramesToRemoveTmp++)
                {
                    const float* found = std::lower_bound(keyframesIn,keyframesEnd,*keyFramesToRemoveTmp);

                    //copy over the lower items
                    copyKeyframesFunc(found);

                    if (found==keyframesEnd) //smallest keyframe to remove is past the range of the array
                        break; //no point cheking the rest
                    else if (*found!=*keyFramesToRemoveTmp) //keyframe not found so can't be removed
                        continue;

                    //need to remove
                    keyframesIn++;
                    inAnimationsIn += boneCount;
                    noAnimationsIn += boneCount;
                }

                //copy over greater items
                copyKeyframesFunc(keyframesEnd);

                //won't resize data buffers because cba
                keyframeCount = keyframesOut-keyframes;
            }

            //effectively upsamples our animation
            inline void insertKeyframes(const size_t& keyframesToAddCount, const float* sortedKeyFramesToAdd)
            {
                const float* keyframesIn = keyframes;
                const float* const keyframesEnd = keyframes+keyframeCount;
                const AnimationKeyData* inAnimationsIn = interpolatedAnimations;
                const AnimationKeyData* noAnimationsIn = nonInterpolatedAnimations;

                float* newKeyframes = (float*)malloc(keyframeCount+keyframesToAddCount);
                float* newKeyframesOut = newKeyframes;
                AnimationKeyData* newInAnimations = (AnimationKeyData*)malloc((keyframeCount+keyframesToAddCount)*boneCount);
                AnimationKeyData* newInAnimationsOut = newInAnimations;
                AnimationKeyData* newNoAnimations = (AnimationKeyData*)malloc((keyframeCount+keyframesToAddCount)*boneCount);
                AnimationKeyData* newNoAnimationsOut = newNoAnimations;

                auto copyKeyframeFunc = [&]()
                {
                    *(newKeyframesOut++) = *(keyframesIn++);
                    for (size_t i=0; i<boneCount; i++)
                    {
                        *(newInAnimationsOut++) = *(inAnimationsIn++);
                        *(newNoAnimationsOut++) = *(noAnimationsIn++);
                    }
                };

                for (const float* keyFramesToAddTmp=sortedKeyFramesToAdd; keyFramesToAddTmp<sortedKeyFramesToAdd+keyframesToAddCount; keyFramesToAddTmp++)
                {
                    const float* found = std::lower_bound(keyframesIn,keyframesEnd,*keyFramesToAddTmp);
                    //copy over the lower items
                    while (keyframesIn<found)
                        copyKeyframeFunc();
                    //need to insert
                    if (found==keyframesEnd||(*found)!=(*keyFramesToAddTmp))
                    {
                        *(newKeyframesOut++) = *keyFramesToAddTmp;

                        float interpolationFactor;
                        getLowerBoundBoneKeyframes(interpolationFactor,*keyFramesToAddTmp,found);
                        float interpolantPrecalcTerm2,interpolantPrecalcTerm3;
                        core::quaternion::flerp_interpolant_terms(interpolantPrecalcTerm2,interpolantPrecalcTerm3,interpolationFactor);
                        for (size_t i=0; i<boneCount; i++)
                        {
                            if (interpolationFactor<1.f)
                            {
                                core::vectorSIMDf   tmpPos;
                                core::quaternion    tmpRot;
                                core::vectorSIMDf   tmpScale;

                                getMatrixFromKeys(tmpPos,tmpRot,tmpScale,*(inAnimationsIn-1),*inAnimationsIn,interpolationFactor,interpolantPrecalcTerm2,interpolantPrecalcTerm3);

                                newInAnimationsOut->Rotation[0] = tmpRot.getPointer()[0];
                                newInAnimationsOut->Rotation[1] = tmpRot.getPointer()[1];
                                newInAnimationsOut->Rotation[2] = tmpRot.getPointer()[2];
                                newInAnimationsOut->Rotation[3] = tmpRot.getPointer()[3];
                                newInAnimationsOut->Position[0] = tmpPos.x; newInAnimationsOut->Position[1] = tmpPos.y; newInAnimationsOut->Position[2] = tmpPos.z;
                                newInAnimationsOut->Scale[0] = tmpScale.x; newInAnimationsOut->Scale[1] = tmpScale.y; newInAnimationsOut->Scale[2] = tmpScale.z;
                                newInAnimationsOut++;
                            }
                            else
                                *(newInAnimationsOut++) = *inAnimationsIn;

                            *(newNoAnimationsOut++) = *noAnimationsIn;
                        }
                    }
                    else //no need to insert
                        copyKeyframeFunc();
                }

                //copy over greater items
                while (keyframesIn<keyframesEnd)
                    copyKeyframeFunc();

                //swap data buffers
                if (keyframes)
                    free(keyframes);
                if (interpolatedAnimations)
                    free(interpolatedAnimations);
                if (nonInterpolatedAnimations)
                    free(nonInterpolatedAnimations);
                keyframes = newKeyframes;
                interpolatedAnimations = newInAnimations;
                nonInterpolatedAnimations = newNoAnimations;
                keyframeCount = newKeyframesOut-newKeyframes;
            }

            //typedef for an interpolation function when adding interpolated offsets to animation
            //keyframe and bone data to transform, keyframe timestamp, this pointer to the CFBH, and whether the keyframe is interpolated
            typedef std::function<void(AnimationKeyData*,const uint32_t&,const float&,const CFinalBoneHierarchy*,const bool&)> AnimationKeyframeTransformFunc;

            // transform animation by the root node in the inclusive range
            inline void transformAnimation(const float& rangeStart, const float& rangeEnd, AnimationKeyframeTransformFunc transformFunc,
                                           const size_t& keyframesToAddCount=0, const float* keyFramesToAdd=NULL)
            {
                //add keyframes if needed
                if (keyframesToAddCount)
                    insertKeyframes(keyframesToAddCount,keyFramesToAdd);

                // apply the transform the the keyframes in the range
                float* start = std::lower_bound(keyframes,keyframes+keyframeCount,rangeStart);
                float* end = std::upper_bound(start,keyframes+keyframeCount,rangeEnd);

                for (size_t i=0; i<boneCount; i++)
                {
                    AnimationKeyData* it = interpolatedAnimations+keyframeCount*i+(start-keyframes);
                    for (float* found=start; found<end; found++)
                        transformFunc(it++,i,*found,this,true);

                    it = nonInterpolatedAnimations+keyframeCount*i+(start-keyframes);
                    for (float* found=start; found<end; found++)
                        transformFunc(it++,i,*found,this,true);
                }
            }

        private:
            inline void createAnimationKeys(const core::vector<asset::ICPUSkinnedMesh::SJoint*>& inLevelFixedJoints)
            {
                core::unordered_set<float> sortedFrames;
                for (size_t i=0; i<boneCount; i++)
                {
                    asset::ICPUSkinnedMesh::SJoint* joint = inLevelFixedJoints[i];
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

                    asset::ICPUSkinnedMesh::SJoint* joint = inLevelFixedJoints[i];
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

        private:
            // bone hierachy independent from animations
            const size_t boneCount;
            BoneReferenceData* boneFlatArray;
            core::stringc* boneNames;
            const size_t NumLevelsInHierarchy;
            size_t* boneTreeLevelEnd;

            ///video::IGPUBuffer* boundBuffer;

            // animation data, independent of bone hierarchy to a degree
            size_t keyframeCount;
            float* keyframes;
            AnimationKeyData* interpolatedAnimations;
            AnimationKeyData* nonInterpolatedAnimations;
    };

} // end namespace scene
} // end namespace irr

#endif

