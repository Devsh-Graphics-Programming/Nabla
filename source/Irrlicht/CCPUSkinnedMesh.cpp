// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"

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


//! constructor
CCPUSkinnedMesh::CCPUSkinnedMesh()
: referenceHierarchy(NULL), HasAnimation(false)
{
	#ifdef _IRR_DEBUG
	setDebugName("CCPUSkinnedMesh");
	#endif
}


//! destructor
CCPUSkinnedMesh::~CCPUSkinnedMesh()
{
	for (uint32_t i=0; i<AllJoints.size(); ++i)
		delete AllJoints[i];

	for (uint32_t j=0; j<LocalBuffers.size(); ++j)
	{
		if (LocalBuffers[j])
			LocalBuffers[j]->drop();
	}

	if (referenceHierarchy)
        referenceHierarchy->drop();
}

void CCPUSkinnedMesh::clearMeshBuffers()
{
	for (auto buff : LocalBuffers)
		buff->drop();
	LocalBuffers.clear();
}

void CCPUSkinnedMesh::setBoneReferenceHierarchy(scene::CFinalBoneHierarchy* fbh)
{
	scene::CFinalBoneHierarchy* referenceHierarchyOld = referenceHierarchy;

	if (fbh)
		fbh->grab();

	referenceHierarchy = fbh;

	if (referenceHierarchyOld)
		referenceHierarchyOld->drop();
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
		return LocalBuffers[nr];
	else
		return 0;
}

//! returns an axis aligned bounding box
const core::aabbox3d<float>& CCPUSkinnedMesh::getBoundingBox() const
{
	return BoundingBox;
}


//! set user axis aligned bounding box
void CCPUSkinnedMesh::setBoundingBox( const core::aabbox3df& box)
{
	BoundingBox = box;
}


//! sets a flag of all contained materials to a new value
void CCPUSkinnedMesh::setMaterialFlag(video::E_MATERIAL_FLAG flag, bool newvalue)
{
	for (auto buff : LocalBuffers)
		buff->getMaterial().setFlag(flag,newvalue);
}



core::vector<asset::ICPUSkinnedMeshBuffer*> &CCPUSkinnedMesh::getMeshBuffers()
{
	return LocalBuffers;
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
}

void PrintDebugBoneHierarchy(ICPUSkinnedMesh::SJoint* joint, std::string indent="", ICPUSkinnedMesh::SJoint* parentJoint=NULL)
{
    assert(joint->Parent==parentJoint);

    std::ostringstream debug(indent);
    debug.seekp(0,std::ios_base::end);
    debug << "Bone Name: \"" << joint->Name << "\"           BindMt: ";

    for (size_t i=0; i<11; i++)
        debug << joint->GlobalInversedMatrix.pointer()[i] << ",";

    debug << joint->GlobalInversedMatrix.pointer()[11] << "\n" << indent << "PoseMt: ";
    for (size_t i=0; i<11; i++)
        debug << joint->LocalMatrix.pointer()[i] << ",";

    debug << joint->LocalMatrix.pointer()[11];
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
	    if (!(*it) || (*it)->getIndexCount()==0)
        {
            (*it)->drop();
            it = LocalBuffers.erase(it);
        }
        else
            it++;
	}

	//calculate bounding box
	bool* firstTouch = NULL;
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
	BoundingBox.reset(0,0,0);

	bool firstStaticMesh = true;
	for (auto buff : LocalBuffers)
	{
        asset::IMeshDataFormatDesc<asset::ICPUBuffer>* desc = buff->getMeshDataAndFormat();

        if (!desc->getMappedBuffer(asset::EVAI_ATTR5) || !desc->getMappedBuffer(asset::EVAI_ATTR6))
        {
            buff->recalculateBoundingBox();
            buff->setMaxVertexBoneInfluences(0);
            if (firstStaticMesh)
            {
                BoundingBox.reset(buff->getBoundingBox());
                firstStaticMesh = false;
            }
            else
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
	if (firstTouch)
        delete [] firstTouch;

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
            if (joint->GlobalInversedMatrix.isIdentity())//might be pre calculated
            {
                joint->GlobalInversedMatrix = joint->GlobalMatrix;
                joint->GlobalInversedMatrix.makeInverse(); // slow
            }
        }

        for (; jointID<AllJoints.size(); jointID++)
        {
            SJoint* joint = AllJoints[jointID];
            assert(joint->Parent);
            joint->GlobalMatrix = concatenateBFollowedByA(joint->Parent->GlobalMatrix,joint->LocalMatrix);
            if (joint->GlobalInversedMatrix.isIdentity())//might be pre calculated
            {
                joint->GlobalInversedMatrix = joint->GlobalMatrix;
                joint->GlobalInversedMatrix.makeInverse(); // slow
            }
        }

        referenceHierarchy = new scene::CFinalBoneHierarchy(AllJoints,JointIxLevelEnd);
    }
}


asset::ICPUSkinnedMeshBuffer *CCPUSkinnedMesh::addMeshBuffer()
{
	ICPUSkinnedMeshBuffer *buffer = new ICPUSkinnedMeshBuffer();
	LocalBuffers.push_back(buffer);
	return buffer;
}


void CCPUSkinnedMesh::addMeshBuffer(ICPUSkinnedMeshBuffer* buf)
{
	if (buf)
	{
		buf->grab();
		LocalBuffers.push_back(buf);
	}
}


CCPUSkinnedMesh::SJoint *CCPUSkinnedMesh::addJoint(SJoint *parent)
{
	SJoint *joint=new SJoint;

	AllJoints.push_back(joint);
	if (!parent)
	{
		//Add root joints to array in finalize()
		joint->Parent = NULL;
	}
	else
	{
		//Set parent (Be careful of the mesh loader also setting the parent)
		parent->Children.push_back(joint);
		joint->Parent = parent;
	}

	return joint;
}

bool CCPUSkinnedMesh::isStatic() const
{
	return !HasAnimation;
}



} // end namespace scene
} // end namespace irr

