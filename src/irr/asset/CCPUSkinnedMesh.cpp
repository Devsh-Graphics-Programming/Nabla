// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"

#include "irr/core/core.h"
#include "irr/asset/CCPUSkinnedMesh.h"
#include "IAnimatedMeshSceneNode.h"
#include "CFinalBoneHierarchy.h"
#include "irr/asset/ICPUSkinnedMeshBuffer.h"

#include "os.h"
#include <sstream>
#include <algorithm>
#include "assert.h"

namespace irr
{
namespace asset
{


//! destructor
CCPUSkinnedMesh::~CCPUSkinnedMesh()
{
	for (uint32_t i=0; i<AllJoints.size(); ++i)
		delete AllJoints[i];
}

void CCPUSkinnedMesh::clearMeshBuffers()
{
	LocalBuffers.clear();
	recalculateBoundingBox();
}


//! returns amount of mesh buffers.
uint32_t CCPUSkinnedMesh::getMeshBufferCount() const
{
	return LocalBuffers.size();
}


//! returns pointer to a mesh buffer
ICPUMeshBuffer* CCPUSkinnedMesh::getMeshBuffer(uint32_t nr) const
{
	if (nr < LocalBuffers.size())
		return LocalBuffers[nr].get();
	else
		return nullptr;
}


core::vector<CCPUSkinnedMesh::SJoint*> &CCPUSkinnedMesh::getAllJoints()
{
	return AllJoints;
}


const core::vector<CCPUSkinnedMesh::SJoint*> &CCPUSkinnedMesh::getAllJoints() const
{
	return AllJoints;
}



void CCPUSkinnedMesh::checkForAnimation()
{
	uint32_t i;
	//Check for animation...
	HasAnimation = false;
	for(i=0;i<AllJoints.size();++i)
	{
        if (AllJoints[i]->PositionKeys.size() ||
            AllJoints[i]->ScaleKeys.size() ||
            AllJoints[i]->RotationKeys.size() )
        {
            HasAnimation = true;
            break;
        }
	}

#ifndef NEW_SHADERS
	//meshes with weights, are still counted as animated for ragdolls, etc
	if (!HasAnimation && AllJoints.size())
	{
		for(auto buff : LocalBuffers)
		{
		    if (!buff)
                continue;

            asset::IMeshDataFormatDesc<asset::ICPUBuffer>* desc = buff->getMeshDataAndFormat();
			if (!desc)
                continue;

            if (desc->getMappedBuffer(asset::EVAI_ATTR5)&&desc->getMappedBuffer(asset::EVAI_ATTR6))
            {
                HasAnimation = true;
                break;
            }
		}
	}
#endif
}

void PrintDebugBoneHierarchy(ICPUSkinnedMesh::SJoint* joint, std::string indent="", ICPUSkinnedMesh::SJoint* parentJoint=NULL)
{
    assert(joint->Parent==parentJoint);

    std::ostringstream debug(indent);
    debug.seekp(0,std::ios_base::end);
    debug << "Bone Name: \"" << joint->Name << "\"           BindMt: ";

    for (size_t i=0; i<3; i++)
    for (size_t j=0; j<4; j++)
        debug << joint->GlobalInversedMatrix[i][j] << (i!=2||j!=3 ? ",":"\n");

    debug << indent << "PoseMt: ";
    for (size_t i=0; i<3; i++)
    for (size_t j=0; j<4; j++)
        debug << joint->LocalMatrix[i][j] << (i!=2||j!=3 ? ",":"");

    os::Printer::log(debug.str(),ELL_INFORMATION);

    indent += "\t";
    for (size_t j=0; j<joint->Children.size(); j++)
        PrintDebugBoneHierarchy(joint->Children[j],indent,joint);
}


//! called by loader after populating with mesh and bone data
void CCPUSkinnedMesh::finalize()
{
    for (auto it=LocalBuffers.begin(); it!=LocalBuffers.end();)
	{
		if (!(*it) || (*it)->getIndexCount() == 0)
			it = LocalBuffers.erase(it);
        else
            it++;
	}

	//calculate bounding box
	bool* firstTouch = nullptr;
	if (AllJoints.size())
	{
        firstTouch = new bool[AllJoints.size()];
        for (size_t i=0; i<AllJoints.size(); i++)
        {
            AllJoints[i]->bbox.reset(core::vector3df(0.f));
            firstTouch[i] = true;
        }
	}

	//
	core::aabbox3df BoundingBox(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX);
#ifndef NEW_SHADERS
	for (auto buff : LocalBuffers)
	{
        asset::IMeshDataFormatDesc<asset::ICPUBuffer>* desc = buff->getMeshDataAndFormat();

        if (!desc->getMappedBuffer(asset::EVAI_ATTR5) || !desc->getMappedBuffer(asset::EVAI_ATTR6))
        {
            buff->recalculateBoundingBox();
            buff->setMaxVertexBoneInfluences(0);
            BoundingBox.addInternalBox(buff->getBoundingBox());
        }
        else
        {
            core::aabbox3df bb;
            bb.reset(core::vector3df(0.f));
            buff->setBoundingBox(bb);

            uint32_t maxVertexInfluences = 1;

            for (size_t j=buff->getIndexMinBound(); j<buff->getIndexMaxBound(); j++)
            {
                core::vectorSIMDf origPos, boneWeights;
				uint32_t boneIDs[4];
                if (!buff->getAttribute(origPos,asset::EVAI_ATTR0,j) || !buff->getAttribute(boneIDs,asset::EVAI_ATTR5,j) || !buff->getAttribute(boneWeights,asset::EVAI_ATTR6,j))
                    continue;

                size_t boneID = size_t(boneIDs[0]);
                SJoint* joint = AllJoints[boneID];
                if (firstTouch[boneID])
                {
                    joint->bbox.reset(origPos.getAsVector3df());
                    firstTouch[boneID] = false;
                }
                else
                    joint->bbox.addInternalPoint(origPos.getAsVector3df());

                size_t boneCount = boneWeights.w+1.5f;
                if (boneCount>maxVertexInfluences)
                    maxVertexInfluences = boneCount;

                for (size_t k=1; k<boneCount; k++)
                {
                    boneID = size_t(boneIDs[k]);
                    joint = AllJoints[boneID];
                    if (firstTouch[boneID])
                    {
                        joint->bbox.reset(origPos.getAsVector3df());
                        firstTouch[boneID] = false;
                    }
                    else
                        joint->bbox.addInternalPoint(origPos.getAsVector3df());
                }
            }

            buff->setMaxVertexBoneInfluences(maxVertexInfluences);
        }
	}
#endif
	if (firstTouch)
        delete [] firstTouch;
	setBoundingBox(BoundingBox);

    core::vector<size_t> JointIxLevelEnd;

	if (AllJoints.size())
	{
	    core::vector<SJoint*> jointsReorderedByLevel;
	    core::vector<uint8_t> reorderIndexRedirect;
	    reorderIndexRedirect.resize(AllJoints.size());

		//fix parents
		for(size_t CheckingIdx=0; CheckingIdx < AllJoints.size(); ++CheckingIdx)
        for(size_t n=0; n < AllJoints[CheckingIdx]->Children.size(); n++)
		{
		    assert(!AllJoints[CheckingIdx]->Children[n]->Parent || AllJoints[CheckingIdx]->Children[n]->Parent==AllJoints[CheckingIdx]);
		    AllJoints[CheckingIdx]->Children[n]->Parent = AllJoints[CheckingIdx];
		}

		for(size_t CheckingIdx=0; CheckingIdx < AllJoints.size(); ++CheckingIdx)
		{
			if (!AllJoints[CheckingIdx]->Parent)
            {
                reorderIndexRedirect[CheckingIdx] = jointsReorderedByLevel.size();
                jointsReorderedByLevel.push_back(AllJoints[CheckingIdx]);
            }
        }
        assert(jointsReorderedByLevel.size());


	    JointIxLevelEnd.push_back(jointsReorderedByLevel.size());
	    for (size_t ix = 0; jointsReorderedByLevel.size()<AllJoints.size();)
        {
            for (; ix<JointIxLevelEnd.back(); ix++)
            for(size_t n=0; n < jointsReorderedByLevel[ix]->Children.size(); n++)
            {
                SJoint* joint = jointsReorderedByLevel[ix]->Children[n];
                assert(joint->Parent==jointsReorderedByLevel[ix]);
                for(size_t CheckingIdx=0; CheckingIdx < AllJoints.size(); ++CheckingIdx)
                {
                    if (AllJoints[CheckingIdx] == joint)
                    {
                        reorderIndexRedirect[CheckingIdx] = jointsReorderedByLevel.size();
                        break;
                    }
                }
                jointsReorderedByLevel.push_back(joint);
            }
            JointIxLevelEnd.push_back(jointsReorderedByLevel.size());
        }

        AllJoints = jointsReorderedByLevel;

        // fix the weights
#ifndef NEW_SHADERS
        for (auto buff : LocalBuffers)
        {
            asset::IMeshDataFormatDesc<asset::ICPUBuffer>* desc = buff->getMeshDataAndFormat();
            if (!desc)
                continue;

            if (!desc->getMappedBuffer(asset::EVAI_ATTR5))
                continue;

            for (size_t j=buff->getIndexMinBound(); j<buff->getIndexMaxBound(); j++)
            {
				uint32_t boneIDs[4];
                if (!buff->getAttribute(boneIDs,asset::EVAI_ATTR5,j))
                    continue;

                uint32_t newBoneIDs[4];
                for (size_t k=0; k<4; k++)
                    newBoneIDs[k] = reorderIndexRedirect[boneIDs[k]];

                buff->setAttribute(newBoneIDs,asset::EVAI_ATTR5,j);
            }
        }
	}
#endif


    //--- optimize and check keyframes ---
    for(size_t i=0;i<AllJoints.size();++i)
    {
        core::vector<SPositionKey> &PositionKeys =AllJoints[i]->PositionKeys;
        core::vector<SScaleKey> &ScaleKeys = AllJoints[i]->ScaleKeys;
        core::vector<SRotationKey> &RotationKeys = AllJoints[i]->RotationKeys;

        std::sort(PositionKeys.begin(),PositionKeys.end());
        if (PositionKeys.size()>2)
        {
            for(uint32_t j=0;j<PositionKeys.size()-2;++j)
            {
                if (PositionKeys[j].position == PositionKeys[j+1].position && PositionKeys[j+1].position == PositionKeys[j+2].position)
                {
                    PositionKeys.erase(PositionKeys.begin()+j+1); //the middle key is unneeded
                    --j;
                }
            }
        }

        std::sort(ScaleKeys.begin(),ScaleKeys.end());
        if (ScaleKeys.size()>2)
        {
            for(uint32_t j=0;j<ScaleKeys.size()-2;++j)
            {
                if (ScaleKeys[j].scale == ScaleKeys[j+1].scale && ScaleKeys[j+1].scale == ScaleKeys[j+2].scale)
                {
                    ScaleKeys.erase(ScaleKeys.begin()+j+1); //the middle key is unneeded
                    --j;
                }
            }
        }

        std::sort(RotationKeys.begin(),RotationKeys.end());
        if (RotationKeys.size()>2)
        {
            for(uint32_t j=0;j<RotationKeys.size()-2;++j)
            {
                if ((RotationKeys[j].rotation == RotationKeys[j+1].rotation && RotationKeys[j+1].rotation == RotationKeys[j+2].rotation).all())
                {
                    RotationKeys.erase(RotationKeys.begin()+j+1); //the middle key is unneeded
                    --j;
                }
            }
        }
    }


    checkForAnimation();


	if (!HasAnimation)
    {
        for (auto buff : LocalBuffers)
            buff->setMaxVertexBoneInfluences(0);
	}
    else
    {
        //Needed for animation and skinning...
        assert(JointIxLevelEnd[0]);

        size_t jointID = 0;
        for (; jointID<JointIxLevelEnd[0]; jointID++)
        {
            SJoint* joint = AllJoints[jointID];

            joint->GlobalMatrix = joint->LocalMatrix;
            if (joint->GlobalInversedMatrix==core::matrix3x4SIMD()) //might be pre calculated
            {
                joint->GlobalInversedMatrix = joint->GlobalMatrix;
                joint->GlobalInversedMatrix.makeInverse();
            }
        }

        for (; jointID<AllJoints.size(); jointID++)
        {
            SJoint* joint = AllJoints[jointID];
            assert(joint->Parent);
            joint->GlobalMatrix = concatenateBFollowedByA(joint->Parent->GlobalMatrix,joint->LocalMatrix);
            if (joint->GlobalInversedMatrix==core::matrix3x4SIMD()) //might be pre calculated
            {
                joint->GlobalInversedMatrix = joint->GlobalMatrix;
                joint->GlobalInversedMatrix.makeInverse();
            }
        }

        referenceHierarchy = core::make_smart_refctd_ptr<CFinalBoneHierarchy>(AllJoints,JointIxLevelEnd);
    }
    }
}

} // end namespace scene
} // end namespace irr

