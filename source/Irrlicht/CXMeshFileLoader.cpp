// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_X_LOADER_

#include "CXMeshFileLoader.h"
#include "os.h"

#include "coreutil.h"
#include "ISceneManager.h"
#include "IVideoDriver.h"
#include "IFileSystem.h"
#include "IReadFile.h"
#include "SVertexManipulator.h"
#include "assert.h"
#include <chrono>
#include <vector>

#ifdef _DEBUG
#define _XREADER_DEBUG
#endif

namespace irr
{
namespace scene
{

//! Constructor
CXMeshFileLoader::CXMeshFileLoader(scene::ISceneManager* smgr, io::IFileSystem* fs)
: SceneManager(smgr), FileSystem(fs), AllJoints(0), AnimatedMesh(0),
	BinaryNumCount(0),
	CurFrame(0), MajorVersion(0), MinorVersion(0), BinaryFormat(false), FloatSize(0)
{
	#ifdef _DEBUG
	setDebugName("CXMeshFileLoader");
	#endif
}


//! returns true if the file maybe is able to be loaded by this class
//! based on the file extension (e.g. ".bsp")
bool CXMeshFileLoader::isALoadableFileExtension(const io::path& filename) const
{
	return core::hasFileExtension ( filename, "x" );
}


//! creates/loads an animated mesh from the file.
//! \return Pointer to the created mesh. Returns 0 if loading failed.
//! If you no longer need the mesh, you should call IAnimatedMesh::drop().
//! See IReferenceCounted::drop() for more information.
ICPUMesh* CXMeshFileLoader::createMesh(io::IReadFile* f)
{
	if (!f)
		return 0;

//#ifdef _XREADER_DEBUG
	auto time = std::chrono::high_resolution_clock::now();
//#endif

	AnimatedMesh = new CCPUSkinnedMesh();
    ICPUMesh* retVal = NULL;

	if (load(f))
	{
		AnimatedMesh->finalize();
		if (AnimatedMesh->isStatic())
        {
            SCPUMesh* staticMesh = new SCPUMesh();
            for (size_t i=0; i<AnimatedMesh->getMeshBufferCount(); i++)
            {
                ICPUMeshBuffer* meshbuffer = new ICPUMeshBuffer();
                staticMesh->addMeshBuffer(meshbuffer);
                meshbuffer->drop();

                ICPUMeshBuffer* origMeshBuffer = AnimatedMesh->getMeshBuffer(i);
                ICPUMeshDataFormatDesc* desc = static_cast<ICPUMeshDataFormatDesc*>(origMeshBuffer->getMeshDataAndFormat());
                meshbuffer->getMaterial() = origMeshBuffer->getMaterial();
                meshbuffer->setPrimitiveType(origMeshBuffer->getPrimitiveType());

                bool doesntNeedIndices = !desc->getIndexBuffer();
                uint32_t largestVertex = origMeshBuffer->getIndexCount();
                meshbuffer->setIndexCount(largestVertex);
                if (doesntNeedIndices)
                {
                    largestVertex = 0;

                    size_t baseVertex = origMeshBuffer->getIndexType()==EIT_32BIT ? ((uint32_t*)origMeshBuffer->getIndices())[0]:((uint16_t*)origMeshBuffer->getIndices())[0];
                    for (size_t j=1; j<origMeshBuffer->getIndexCount(); j++)
                    {
                        uint32_t nextIx = origMeshBuffer->getIndexType()==EIT_32BIT ? ((uint32_t*)origMeshBuffer->getIndices())[j]:((uint16_t*)origMeshBuffer->getIndices())[j];
                        if (nextIx>largestVertex)
                            largestVertex = nextIx;

                        if (doesntNeedIndices&&(baseVertex+j!=nextIx))
                            doesntNeedIndices = false;
                    }

                    if (doesntNeedIndices)
                    {
                        desc->mapIndexBuffer(NULL);
                        meshbuffer->setBaseVertex(baseVertex);
                    }
                }


                E_INDEX_TYPE indexType;
                if (doesntNeedIndices)
                    indexType = EIT_UNKNOWN;
                else
                {
                    core::ICPUBuffer* indexBuffer;
                    if (largestVertex>=0x10000u)
                    {
                        indexType = EIT_32BIT;
                        indexBuffer = new core::ICPUBuffer(4*origMeshBuffer->getIndexCount());
                        for (size_t j=0; j<origMeshBuffer->getIndexCount(); j++)
                           ((uint32_t*)indexBuffer->getPointer())[j] = origMeshBuffer->getIndexType()==EIT_32BIT ? ((uint32_t*)origMeshBuffer->getIndices())[j]:((uint16_t*)origMeshBuffer->getIndices())[j];
                    }
                    else
                    {
                        indexType = EIT_16BIT;
                        indexBuffer = new core::ICPUBuffer(2*origMeshBuffer->getIndexCount());
                        for (size_t j=0; j<origMeshBuffer->getIndexCount(); j++)
                           ((uint16_t*)indexBuffer->getPointer())[j] = origMeshBuffer->getIndexType()==EIT_32BIT ? ((uint32_t*)origMeshBuffer->getIndices())[j]:((uint16_t*)origMeshBuffer->getIndices())[j];
                    }
                    desc->mapIndexBuffer(indexBuffer);
                }
                meshbuffer->setIndexType(indexType);
                meshbuffer->setMeshDataAndFormat(desc);

                meshbuffer->setPositionAttributeIx(origMeshBuffer->getPositionAttributeIx());
                for (size_t j=0; j<EVAI_COUNT; j++)
                {
                    E_VERTEX_ATTRIBUTE_ID attrId = (E_VERTEX_ATTRIBUTE_ID)j;
                    if (!desc->getMappedBuffer(attrId))
                        continue;

                    if (attrId==EVAI_ATTR3)
                    {
                        const core::ICPUBuffer* normalBuffer = desc->getMappedBuffer(EVAI_ATTR3);
                        core::ICPUBuffer* newNormalBuffer = new core::ICPUBuffer(normalBuffer->getSize()/3);
                        for (size_t k=0; k<newNormalBuffer->getSize()/4; k++)
                        {
                            core::vectorSIMDf simdNormal;
                            simdNormal.set(((core::vector3df*)normalBuffer->getPointer())[k]);
                            ((uint32_t*)newNormalBuffer->getPointer())[k] = quantizeNormal2_10_10_10(simdNormal);
                        }
                        desc->mapVertexAttrBuffer(newNormalBuffer,EVAI_ATTR3,ECPA_FOUR,ECT_INT_2_10_10_10_REV);
                        newNormalBuffer->drop();
                    }
                }

                meshbuffer->recalculateBoundingBox();
            }
            staticMesh->recalculateBoundingBox();

            retVal = staticMesh;
            AnimatedMesh->drop();
            AnimatedMesh = 0;
        }
        else
            retVal = AnimatedMesh;
	}
	else
	{
		AnimatedMesh->drop();
		AnimatedMesh = 0;
	}
//#ifdef _XREADER_DEBUG
	std::ostringstream tmpString("Time to load ");
	tmpString.seekp(0,std::ios_base::end);
	tmpString << (BinaryFormat ? "binary" : "ascii") << " X file: " << (std::chrono::high_resolution_clock::now()-time).count() << "ms";
	os::Printer::log(tmpString.str());
//#endif
	//Clear up

	MajorVersion=0;
	MinorVersion=0;
	BinaryFormat=0;
	BinaryNumCount=0;
	FloatSize=0;
	CurFrame=0;
	TemplateMaterials.clear();

	fileContents.str("");
	fileContents.clear();

	for (uint32_t i=0; i<Meshes.size(); ++i)
		delete Meshes[i];
	Meshes.clear();

	return retVal;
}

class SuperSkinningTMPStruct
{
    public:
		inline bool operator<(const SuperSkinningTMPStruct& other) const { return (tmp >= other.tmp); }

        float tmp;
        uint32_t redir;
};

core::matrix4x3 getGlobalMatrix_evil(ICPUSkinnedMesh::SJoint* joint)
{
    //if (joint->GlobalInversedMatrix.isIdentity())
        //return joint->GlobalInversedMatrix;
    if (joint->Parent)
        return concatenateBFollowedByA(getGlobalMatrix_evil(joint->Parent),joint->LocalMatrix);
    else
        return joint->LocalMatrix;
}


bool CXMeshFileLoader::load(io::IReadFile* file)
{
	if (!readFileIntoMemory(file))
		return false;

	if (!parseFile())
		return false;

	for (uint32_t n=0; n<Meshes.size(); ++n)
	{
		SXMesh *mesh=Meshes[n];

		// default material if nothing loaded
		if (!mesh->Materials.size())
		{
			mesh->Materials.push_back(video::SMaterial());
			mesh->Materials[0].DiffuseColor.set(0xff777777);
			mesh->Materials[0].Shininess=0.f;
			mesh->Materials[0].SpecularColor.set(0xff777777);
			mesh->Materials[0].EmissiveColor.set(0xff000000);
		}

		if (mesh->BoneCount>0x100u)
            os::Printer::log("X loader", "Too many bones in mesh, limit is 256!", ELL_WARNING);

		uint32_t i;

		mesh->Buffers.reserve(mesh->Materials.size());

		for (i=0; i<mesh->Materials.size(); ++i)
		{
			mesh->Buffers.push_back( AnimatedMesh->addMeshBuffer() );
			mesh->Buffers.back()->getMaterial() = mesh->Materials[i];
		}

		if (!mesh->FaceMaterialIndices.size())
		{
			mesh->FaceMaterialIndices.resize(mesh->Indices.size() / 3);
			for (i=0; i<mesh->FaceMaterialIndices.size(); ++i)
				mesh->FaceMaterialIndices[i]=0;
		}


		{
			core::vector< uint32_t > verticesLinkIndex;
			core::vector< int16_t > verticesLinkBuffer;
			verticesLinkBuffer.resize(mesh->Vertices.size());

			for (i=0;i<mesh->Vertices.size();++i)
			{
				// watch out for vertices which are not part of the mesh
				// they will keep the -1 and can lead to out-of-bounds access
				verticesLinkBuffer[i]=-1;
			}

			bool warned = false;
			// store meshbuffer number per vertex
			for (i=0;i<mesh->FaceMaterialIndices.size();++i)
			{
				for (uint32_t id=i*3+0;id<=i*3+2;++id)
				{
					if ((verticesLinkBuffer[mesh->Indices[id]] != -1) && (verticesLinkBuffer[mesh->Indices[id]] != (int16_t)mesh->FaceMaterialIndices[i]))
					{
						if (!warned)
						{
							os::Printer::log("X loader", "Duplicated vertex, animation might be corrupted.", ELL_WARNING);
							warned=true;
						}
						const uint32_t tmp = mesh->Vertices.size();
						mesh->Vertices.push_back(mesh->Vertices[ mesh->Indices[id] ]);
						mesh->Indices[id] = tmp;
                        verticesLinkBuffer.push_back(mesh->FaceMaterialIndices[i]);
					}
					else
                        verticesLinkBuffer[ mesh->Indices[id] ] = mesh->FaceMaterialIndices[i];
				}
			}

			if (mesh->FaceMaterialIndices.size() != 0)
			{
				scene::ICPUMeshDataFormatDesc* desc = new scene::ICPUMeshDataFormatDesc();

				core::ICPUBuffer* vPosBuf = new core::ICPUBuffer(mesh->Vertices.size()*4*3);
				desc->mapVertexAttrBuffer(vPosBuf,EVAI_ATTR0,ECPA_THREE,ECT_FLOAT);
				vPosBuf->drop();
				core::ICPUBuffer* vColorBuf = NULL;
				if (mesh->Colors.size())
                {
                    vColorBuf = new core::ICPUBuffer(mesh->Vertices.size()*4);
                    desc->mapVertexAttrBuffer(vColorBuf,EVAI_ATTR1,ECPA_REVERSED_OR_BGRA,ECT_NORMALIZED_UNSIGNED_BYTE);
                    vColorBuf->drop();
                }
				core::ICPUBuffer* vTCBuf = new core::ICPUBuffer(mesh->Vertices.size()*4*2);
                desc->mapVertexAttrBuffer(vTCBuf,EVAI_ATTR2,ECPA_TWO,ECT_FLOAT);
                vTCBuf->drop();
				core::ICPUBuffer* vNormalBuf = new core::ICPUBuffer(mesh->Vertices.size()*4*3);
				desc->mapVertexAttrBuffer(vNormalBuf,EVAI_ATTR3,ECPA_THREE,ECT_FLOAT);
				vNormalBuf->drop();
				core::ICPUBuffer* vTC2Buf = NULL;
				if (mesh->TCoords2.size())
				{
                    vTC2Buf = new core::ICPUBuffer(mesh->Vertices.size()*4*2);
                    desc->mapVertexAttrBuffer(vTC2Buf,EVAI_ATTR4,ECPA_TWO,ECT_FLOAT);
                    vTC2Buf->drop();
				}
				core::ICPUBuffer* vSkinningDataBuf = NULL;
				if (mesh->VertexSkinWeights.size())
                {
                    vSkinningDataBuf = new core::ICPUBuffer(mesh->Vertices.size()*sizeof(SkinnedVertexFinalData));
                    desc->mapVertexAttrBuffer(vSkinningDataBuf,EVAI_ATTR5,ECPA_FOUR,ECT_INTEGER_UNSIGNED_BYTE,8,0);
                    desc->mapVertexAttrBuffer(vSkinningDataBuf,EVAI_ATTR6,ECPA_FOUR,ECT_NORMALIZED_UNSIGNED_INT_2_10_10_10_REV,8,4);
                    vSkinningDataBuf->drop();
                }
				else if (mesh->AttachedJointID!=-1)
                {
                    vSkinningDataBuf = new core::ICPUBuffer(mesh->Vertices.size()*sizeof(SkinnedVertexFinalData));
                    desc->mapVertexAttrBuffer(vSkinningDataBuf,EVAI_ATTR5,ECPA_FOUR,ECT_INTEGER_UNSIGNED_BYTE,8,0);
                    desc->mapVertexAttrBuffer(vSkinningDataBuf,EVAI_ATTR6,ECPA_FOUR,ECT_NORMALIZED_UNSIGNED_INT_2_10_10_10_REV,8,4);
                    vSkinningDataBuf->drop();

                    bool correctBindMatrix = AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix.isIdentity();
                    core::matrix4x3 globalMat,globalMatInvTransp;
                    if (correctBindMatrix)
                    {
                        globalMat = getGlobalMatrix_evil(AnimatedMesh->getAllJoints()[mesh->AttachedJointID]);
                        //
                        globalMatInvTransp(0,0) = globalMat(0,0);
                        globalMatInvTransp(1,0) = globalMat(0,1);
                        globalMatInvTransp(2,0) = globalMat(0,2);
                        globalMatInvTransp(0,1) = globalMat(1,0);
                        globalMatInvTransp(1,1) = globalMat(1,1);
                        globalMatInvTransp(2,1) = globalMat(1,2);
                        globalMatInvTransp(0,2) = globalMat(2,0);
                        globalMatInvTransp(1,2) = globalMat(2,1);
                        globalMatInvTransp(2,2) = globalMat(2,2);
                        globalMatInvTransp(0,3) = globalMat(3,0);
                        globalMatInvTransp(1,3) = globalMat(3,1);
                        globalMatInvTransp(2,3) = globalMat(3,2);
                        globalMatInvTransp.makeInverse();
                    }
                    else
                    {
                        AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix.getInverse(globalMat);
                        globalMatInvTransp(0,0) = AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(0,0);
                        globalMatInvTransp(1,0) = AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(0,1);
                        globalMatInvTransp(2,0) = AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(0,2);
                        globalMatInvTransp(0,1) = AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(1,0);
                        globalMatInvTransp(1,1) = AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(1,1);
                        globalMatInvTransp(2,1) = AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(1,2);
                        globalMatInvTransp(0,2) = AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(2,0);
                        globalMatInvTransp(1,2) = AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(2,1);
                        globalMatInvTransp(2,2) = AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(2,2);
                        globalMatInvTransp(0,3) = AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(3,0);
                        globalMatInvTransp(1,3) = AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(3,1);
                        globalMatInvTransp(2,3) = AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(3,2);
                    }

                    for (size_t j=0; j<mesh->Vertices.size(); j++)
                    {
                        globalMat.transformVect(&mesh->Vertices[j].Pos.X);
                        globalMatInvTransp.mulSub3x3With3x1(&mesh->Vertices[j].Normal.X);

                        reinterpret_cast<SkinnedVertexFinalData*>(vSkinningDataBuf->getPointer())[j].boneWeights = 0x000003ffu;
                        reinterpret_cast<SkinnedVertexFinalData*>(vSkinningDataBuf->getPointer())[j].boneIDs[0] = mesh->AttachedJointID;
                        reinterpret_cast<SkinnedVertexFinalData*>(vSkinningDataBuf->getPointer())[j].boneIDs[1] = 0;
                        reinterpret_cast<SkinnedVertexFinalData*>(vSkinningDataBuf->getPointer())[j].boneIDs[2] = 0;
                        reinterpret_cast<SkinnedVertexFinalData*>(vSkinningDataBuf->getPointer())[j].boneIDs[3] = 0;
                    }
                    vSkinningDataBuf = NULL;
                }

				// store vertices in buffers and remember relation in verticesLinkIndex
				uint32_t* vCountArray = new uint32_t[mesh->Buffers.size()];
				memset(vCountArray, 0, mesh->Buffers.size()*sizeof(uint32_t));
				// count vertices in each buffer and reallocate
				for (i=0; i<mesh->Vertices.size(); ++i)
				{
					if (verticesLinkBuffer[i] != -1)
						++vCountArray[verticesLinkBuffer[i]];
				}

                //cumulative shit
                uint32_t* cumBaseVertex = new uint32_t[mesh->Buffers.size()];
				memset(cumBaseVertex, 0, mesh->Buffers.size()*sizeof(uint32_t));
                for (i=0; i<mesh->Buffers.size(); ++i)
                {
					scene::SCPUSkinMeshBuffer *buffer = mesh->Buffers[i];

                    buffer->setIndexRange(0,vCountArray[i]);
                    if (vCountArray[i]>0x10000u)
                        buffer->setIndexType(EIT_32BIT);
                    else
                        buffer->setIndexType(EIT_16BIT);

                    buffer->setMeshDataAndFormat(desc);

                    if (i>0)
                    {
                        cumBaseVertex[i] = cumBaseVertex[i-1] + vCountArray[i-1];
                        buffer->setBaseVertex(cumBaseVertex[i]);
                    }
				}
				desc->drop();

				verticesLinkIndex.resize(mesh->Vertices.size());
				memset(vCountArray, 0, mesh->Buffers.size()*sizeof(uint32_t));
				// actually store vertices
				for (i=0; i<mesh->Vertices.size(); ++i)
				{
					// if a vertex is missing for some reason, just skip it
					if (verticesLinkBuffer[i]==-1)
						continue;
					scene::SCPUSkinMeshBuffer *buffer = mesh->Buffers[ verticesLinkBuffer[i] ];

                    uint32_t &Ix = vCountArray[ verticesLinkBuffer[i] ];
                    verticesLinkIndex[i] = Ix;

                    uint32_t properIx = Ix+cumBaseVertex[verticesLinkBuffer[i]];
                    ((core::vector3df*)vPosBuf->getPointer())[properIx] = mesh->Vertices[i].Pos;
                    if (vColorBuf)
                        ((uint32_t*)vColorBuf->getPointer())[properIx] = mesh->Colors[i];
                    if (vTCBuf)
                        ((core::vector2df*)vTCBuf->getPointer())[properIx] = mesh->Vertices[i].TCoords;
                    if (vNormalBuf)
                        ((core::vector3df*)vNormalBuf->getPointer())[properIx] = mesh->Vertices[i].Normal;
                    if (vTC2Buf)
                        ((core::vector2df*)vTC2Buf->getPointer())[properIx] = (i<mesh->TCoords2.size())?mesh->TCoords2[i]:mesh->Vertices[i].TCoords;
                    if (vSkinningDataBuf)
                    {
                        const SkinnedVertexIntermediateData& origWeight = mesh->VertexSkinWeights[i];
                        SkinnedVertexFinalData* actualWeight = reinterpret_cast<SkinnedVertexFinalData*>(vSkinningDataBuf->getPointer())+properIx;
                        reinterpret_cast<uint32_t*>(actualWeight->boneIDs)[0] = reinterpret_cast<const uint32_t*>(origWeight.boneIDs)[0];
                        size_t activeBones = 0;
                        for (; activeBones<4; activeBones++)
                        {
                            if (origWeight.boneWeights[activeBones]<=0.f)
                                break;
                        }
                        #define SWAP(x,y) if (sortStuff[y] < sortStuff[x]) { SuperSkinningTMPStruct tmp = sortStuff[x]; sortStuff[x] = sortStuff[y]; sortStuff[y] = tmp; }
                        switch (activeBones)
                        {
                            case 0:
                                actualWeight->boneWeights = 0;
                                break;
                            case 1:
                                actualWeight->boneWeights = 0x000003ffu;
                                break;
                            case 2:
                                {
                                    float sum = origWeight.boneWeights[0]+origWeight.boneWeights[1];
                                    SuperSkinningTMPStruct sortStuff[2];
                                    uint32_t tmpInt[2];
                                    //normalize weights
                                    sum = 1023.f/sum;
                                    //first pass quantize to integer
                                    for (size_t j=0; j<2; j++)
                                    {
                                        float tmp = origWeight.boneWeights[j]*sum;
                                        tmpInt[j] = tmp;
                                        sortStuff[j].tmp = tmp-float(tmpInt[j]); //fract()

                                        sortStuff[j].redir = j;
                                    }

                                    //sort
                                    SWAP(0, 1);

                                    uint32_t leftOver = 1023-tmpInt[0]-tmpInt[1];
                                    {
                                        assert(leftOver<=2); // <=2,<=3
                                        for (uint32_t j=0; j<leftOver; j++)
                                            tmpInt[sortStuff[j].redir]++;
                                    }
                                    actualWeight->boneWeights = tmpInt[0]|(tmpInt[1]<<10)|(0x1u<<30);
                                }
                                break;
                            case 3:
                                {
                                    float sum = origWeight.boneWeights[0]+origWeight.boneWeights[1]+origWeight.boneWeights[2];
                                    SuperSkinningTMPStruct sortStuff[3];
                                    uint32_t tmpInt[3];
                                    //normalize weights
                                    sum = 1023.f/sum;
                                    //first pass quantize to integer
                                    for (size_t j=0; j<3; j++)
                                    {
                                        float tmp = origWeight.boneWeights[j]*sum;
                                        tmpInt[j] = tmp;
                                        sortStuff[j].tmp = tmp-float(tmpInt[j]); //fract()

                                        sortStuff[j].redir = j;
                                    }

                                    //sort
                                    SWAP(1, 2);
                                    SWAP(0, 2);
                                    SWAP(0, 1);

                                    uint32_t leftOver = 1023-tmpInt[0]-tmpInt[1]-tmpInt[2];
                                    {
                                        assert(leftOver<=3); // <=2,<=3
                                        for (uint32_t j=0; j<leftOver; j++)
                                            tmpInt[sortStuff[j].redir]++;
                                    }
                                    actualWeight->boneWeights = tmpInt[0]|(tmpInt[1]<<10)|(tmpInt[2]<<20)|(0x2u<<30);
                                }
                                break;
                            case 4://very precise quantization
                                {
                                    float sum = origWeight.boneWeights[0]+origWeight.boneWeights[1]+origWeight.boneWeights[2]+origWeight.boneWeights[3];
                                    SuperSkinningTMPStruct sortStuff[4];
                                    uint32_t tmpInt[4];
                                    //normalize weights
                                    sum = 1023.f/sum;
                                    //first pass quantize to integer
                                    for (size_t j=0; j<4; j++)
                                    {
                                        float tmp = origWeight.boneWeights[j]*sum;
                                        tmpInt[j] = tmp;
                                        sortStuff[j].tmp = tmp-float(tmpInt[j]); //fract()

                                        sortStuff[j].redir = j;
                                    }

                                    //sort
                                    SWAP(0, 1);
                                    SWAP(2, 3);
                                    SWAP(0, 2);
                                    SWAP(1, 3);
                                    SWAP(1, 2);



                                    uint32_t gap = 1023-tmpInt[0]-tmpInt[1]-tmpInt[2]-tmpInt[3];
                                    assert(gap<=4);
                                    for (uint32_t j=0; j<gap; j++)
                                        tmpInt[sortStuff[j].redir]++;

                                    actualWeight->boneWeights = tmpInt[0]|(tmpInt[1]<<10)|(tmpInt[2]<<20)|(0x3u<<30);
                                }
                                break;
                        }
                        #undef SWAP
                    }

                    Ix++;
				}

				// count indices per buffer and reallocate
				memset(vCountArray, 0, mesh->Buffers.size()*sizeof(uint32_t));
				for (i=0; i<mesh->FaceMaterialIndices.size(); ++i)
					++vCountArray[ mesh->FaceMaterialIndices[i] ];

                uint32_t indexBufferSz = 0;
				for (i=0; i<mesh->Buffers.size(); ++i)
                {
					scene::SCPUSkinMeshBuffer *buffer = mesh->Buffers[ i ];


					uint32_t subBufferSz = vCountArray[i]*3;
					buffer->setIndexCount(subBufferSz);
                    subBufferSz *= (buffer->getIndexType()==EIT_32BIT) ? 4:2;

                    //now cumulative
                    cumBaseVertex[i] = indexBufferSz;
					buffer->setIndexBufferOffset(indexBufferSz);
                    indexBufferSz += subBufferSz;
                }
                core::ICPUBuffer* ixbuf = new core::ICPUBuffer(indexBufferSz);
				desc->mapIndexBuffer(ixbuf);
				ixbuf->drop();
				// create indices per buffer
				memset(vCountArray, 0, mesh->Buffers.size()*sizeof(uint32_t));
				for (i=0; i<mesh->FaceMaterialIndices.size(); ++i)
				{
					scene::SCPUSkinMeshBuffer *buffer = mesh->Buffers[ mesh->FaceMaterialIndices[i] ];

					void* indexBufAlreadyOffset = ((uint8_t*)ixbuf->getPointer())+cumBaseVertex[mesh->FaceMaterialIndices[i]];

                    if (buffer->getIndexType()==EIT_32BIT)
                    {
                        for (uint32_t id=i*3+0; id!=i*3+3; ++id)
                            ((uint32_t*)indexBufAlreadyOffset)[vCountArray[mesh->FaceMaterialIndices[i]]++] = verticesLinkIndex[ mesh->Indices[id] ];
                    }
                    else
                    {
                        for (uint32_t id=i*3+0; id!=i*3+3; ++id)
                            ((uint16_t*)indexBufAlreadyOffset)[vCountArray[mesh->FaceMaterialIndices[i]]++] = verticesLinkIndex[ mesh->Indices[id] ];
                    }
				}
                delete [] cumBaseVertex;
				delete [] vCountArray;
			}
		}
	}

	return true;
}


//! Reads file into memory
bool CXMeshFileLoader::readFileIntoMemory(io::IReadFile* file)
{
	const long size = file->getSize();
	if (size < 12)
	{
		os::Printer::log("X File is too small.", ELL_WARNING);
		return false;
	}

	std::string Buffer;
	Buffer.resize(size);
	//! read all into memory
	if (file->read(&Buffer[0], size) != size)
	{
		os::Printer::log("Could not read from x file.", ELL_WARNING);
		return false;
	}
	fileContents.str(Buffer);

	//! check header "xof "
	char tmp[4];
	fileContents.read(tmp,4);
	if (strncmp(tmp, "xof ", 4)!=0)
	{
		os::Printer::log("Not an x file, wrong header.", ELL_WARNING);
		return false;
	}

	//! read minor and major version, e.g. 0302 or 0303
	fileContents.read(tmp,2);
	tmp[2] = 0x0;
	sscanf(tmp,"%u",&MajorVersion);

	fileContents.read(tmp,2);
	sscanf(tmp,"%u",&MinorVersion);

	//! read format
	fileContents.read(tmp,4);
	if (strncmp(tmp, "txt ", 4) ==0)
		BinaryFormat = false;
	else if (strncmp(tmp, "bin ", 4) ==0)
		BinaryFormat = true;
	else
	{
		os::Printer::log("Only uncompressed x files currently supported.", ELL_WARNING);
		return false;
	}
	BinaryNumCount=0;

	//! read float size
	fileContents.read(tmp,4);
	if (strncmp(tmp, "0032", 4) ==0)
		FloatSize = 4;
	else if (strncmp(tmp, "0064", 4) ==0)
		FloatSize = 8;
	else
	{
		os::Printer::log("Float size not supported.", ELL_WARNING);
		return false;
	}


	{
        std::string stmp;
        std::getline(fileContents,stmp);
	}
	FilePath = io::IFileSystem::getFileDir(file->getFileName()) + "/";

	return true;
}


//! Parses the file
bool CXMeshFileLoader::parseFile()
{
	while(parseDataObject())
	{
		// loop
	}

	return true;
}


//! Parses the next Data object in the file
bool CXMeshFileLoader::parseDataObject()
{
	std::string objectName = getNextToken();

	if (objectName.size() == 0)
		return false;

	// parse specific object
#ifdef _XREADER_DEBUG
	os::Printer::log("debug DataObject:", objectName, ELL_DEBUG);
#endif

	if (objectName == "template")
		return parseDataObjectTemplate();
	else
	if (objectName == "Frame")
	{
		return parseDataObjectFrame( 0 );
	}
	else
	if (objectName == "Mesh")
	{
		// some meshes have no frames at all
		//CurFrame = AnimatedMesh->addJoint(0);

		SXMesh *mesh=new SXMesh;

		//mesh->Buffer=AnimatedMesh->addMeshBuffer();
		Meshes.push_back(mesh);

		return parseDataObjectMesh(*mesh);
	}
	else
	if (objectName == "AnimationSet")
	{
		return parseDataObjectAnimationSet();
	}
	else
	if (objectName == "Material")
	{
		// template materials now available thanks to joeWright
		TemplateMaterials.push_back(SXTemplateMaterial());
		TemplateMaterials.back().Name = getNextToken();
		return parseDataObjectMaterial(TemplateMaterials.back().Material);
	}
	else
	if (objectName == "}")
	{
		os::Printer::log("} found in dataObject", ELL_WARNING);
		return true;
	}

	os::Printer::log("Unknown data object in animation of .x file", objectName, ELL_WARNING);

	return parseUnknownDataObject();
}


bool CXMeshFileLoader::parseDataObjectTemplate()
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: Reading template", ELL_DEBUG);
#endif

	// parse a template data object. Currently not stored.
	std::string name;

	if (!readHeadOfDataObject(&name))
	{
		os::Printer::log("Left delimiter in template data object missing.",
			name, ELL_WARNING);

		return false;
	}

	// read GUID
	getNextToken();

	// read and ignore data members
	while(true)
	{
		std::string s = getNextToken();

		if (s == "}")
			break;

		if (s.size() == 0)
			return false;
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectFrame(ICPUSkinnedMesh::SJoint *Parent)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: Reading frame", ELL_DEBUG);
#endif

	// A coordinate frame, or "frame of reference." The Frame template
	// is open and can contain any object. The Direct3D extensions (D3DX)
	// mesh-loading functions recognize Mesh, FrameTransformMatrix, and
	// Frame template instances as child objects when loading a Frame
	// instance.

	uint32_t JointID=0;

	std::string name;

	if (!readHeadOfDataObject(&name))
	{
		os::Printer::log("No opening brace in Frame found in x file", ELL_WARNING);

		return false;
	}

	ICPUSkinnedMesh::SJoint *joint=0;

	if (name.size())
	{
		for (uint32_t n=0; n < AnimatedMesh->getAllJoints().size(); ++n)
		{
			if (AnimatedMesh->getAllJoints()[n]->Name==name)
			{
				joint=AnimatedMesh->getAllJoints()[n];
				JointID=n;
				break;
			}
		}
	}

	if (!joint)
	{
#ifdef _XREADER_DEBUG
		os::Printer::log("creating joint ", name, ELL_DEBUG);
#endif
		joint=AnimatedMesh->addJoint(Parent);
		joint->Name=name;
		JointID=AnimatedMesh->getAllJoints().size()-1;
	}
	else
	{
#ifdef _XREADER_DEBUG
		os::Printer::log("using joint ", name, ELL_DEBUG);
#endif
		if (Parent)
			Parent->Children.push_back(joint);
	}

	// Now inside a frame.
	// read tokens until closing brace is reached.

	while(true)
	{
		std::string objectName = getNextToken();

#ifdef _XREADER_DEBUG
		os::Printer::log("debug DataObject in frame:", objectName, ELL_DEBUG);
#endif

		if (objectName.size() == 0)
		{
			os::Printer::log("Unexpected ending found in Frame in x file.", ELL_WARNING);

			return false;
		}
		else
		if (objectName == "}")
		{
			break; // frame finished
		}
		else
		if (objectName == "Frame")
		{

			if (!parseDataObjectFrame(joint))
				return false;
		}
		else
		if (objectName == "FrameTransformMatrix")
		{
			if (!parseDataObjectTransformationMatrix(joint->LocalMatrix))
				return false;

			//joint->LocalAnimatedMatrix
			//joint->LocalAnimatedMatrix.makeInverse();
			//joint->LocalMatrix=tmp*joint->LocalAnimatedMatrix;
		}
		else
		if (objectName == "Mesh")
		{
			/*
			frame.Meshes.push_back(SXMesh());
			if (!parseDataObjectMesh(frame.Meshes.back()))
				return false;
			*/
			SXMesh *mesh=new SXMesh;

			mesh->AttachedJointID=JointID;

			Meshes.push_back(mesh);

			if (!parseDataObjectMesh(*mesh))
				return false;
		}
		else
		{
			os::Printer::log("Unknown data object in frame in x file", objectName, ELL_WARNING);
			if (!parseUnknownDataObject())
				return false;
		}
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectTransformationMatrix(core::matrix4x3 &mat)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: Reading Transformation Matrix", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject())
	{
		os::Printer::log("No opening brace in Transformation Matrix found in x file", ELL_WARNING);

		return false;
	}

	readMatrix(mat);

	if (!checkForOneFollowingSemicolons())
	{
		os::Printer::log("No finishing semicolon in Transformation Matrix found in x file", ELL_WARNING);

	}

	if (!checkForClosingBrace())
	{
		os::Printer::log("No closing brace in Transformation Matrix found in x file", ELL_WARNING);

		return false;
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectMesh(SXMesh &mesh)
{
	std::string name;

	if (!readHeadOfDataObject(&name))
	{
#ifdef _XREADER_DEBUG
		os::Printer::log("CXFileReader: Reading mesh", ELL_DEBUG);
#endif
		os::Printer::log("No opening brace in Mesh found in x file", ELL_WARNING);

		return false;
	}

#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: Reading mesh", name, ELL_DEBUG);
#endif

	// read vertex count
	const uint32_t nVertices = readInt();

	// read vertices
	mesh.Vertices.resize(nVertices);
	for (uint32_t n=0; n<nVertices; ++n)
	{
		readVector3(mesh.Vertices[n].Pos);
	}

	if (!checkForTwoFollowingSemicolons())
	{
		os::Printer::log("No finishing semicolon in Mesh Vertex Array found in x file", ELL_WARNING);

	}

	// read faces
	const uint32_t nFaces = readInt();

	mesh.Indices.resize(nFaces * 3);
	mesh.IndexCountPerFace.resize(nFaces);

	core::vector<uint32_t> polygonfaces;
	uint32_t currentIndex = 0;

	for (uint32_t k=0; k<nFaces; ++k)
	{
		const uint32_t fcnt = readInt();

		if (fcnt != 3)
		{
			if (fcnt < 3)
			{
				os::Printer::log("Invalid face count (<3) found in Mesh x file reader.", ELL_WARNING);

				return false;
			}

			// read face indices
			polygonfaces.resize(fcnt);
			uint32_t triangles = (fcnt-2);
			mesh.Indices.resize(mesh.Indices.size() + ((triangles-1)*3));
			mesh.IndexCountPerFace[k] = (uint16_t)(triangles * 3);

			for (uint32_t f=0; f<fcnt; ++f)
				polygonfaces[f] = readInt();

			for (uint32_t jk=0; jk<triangles; ++jk)
			{
				mesh.Indices[currentIndex++] = polygonfaces[0];
				mesh.Indices[currentIndex++] = polygonfaces[jk+1];
				mesh.Indices[currentIndex++] = polygonfaces[jk+2];
			}

			// TODO: change face indices in material list
		}
		else
		{
			mesh.Indices[currentIndex++] = readInt();
			mesh.Indices[currentIndex++] = readInt();
			mesh.Indices[currentIndex++] = readInt();
			mesh.IndexCountPerFace[k] = 3;
		}
	}

	if (!checkForTwoFollowingSemicolons())
	{
		os::Printer::log("No finishing semicolon in Mesh Face Array found in x file", ELL_WARNING);

	}

	// here, other data objects may follow

	while(true)
	{
		std::string objectName = getNextToken();

		if (objectName.size() == 0)
		{
			os::Printer::log("Unexpected ending found in Mesh in x file.", ELL_WARNING);

			return false;
		}
		else
		if (objectName == "}")
		{
			break; // mesh finished
		}

#ifdef _XREADER_DEBUG
		os::Printer::log("debug DataObject in mesh:", objectName, ELL_DEBUG);
#endif

		if (objectName == "MeshNormals")
		{
			if (!parseDataObjectMeshNormals(mesh))
				return false;
		}
		else
		if (objectName == "MeshTextureCoords")
		{
			if (!parseDataObjectMeshTextureCoords(mesh))
				return false;
		}
		else
		if (objectName == "MeshVertexColors")
		{
			if (!parseDataObjectMeshVertexColors(mesh))
				return false;
		}
		else
		if (objectName == "MeshMaterialList")
		{
			if (!parseDataObjectMeshMaterialList(mesh))
				return false;
		}
		else
		if (objectName == "VertexDuplicationIndices")
		{
			// we'll ignore vertex duplication indices
			// TODO: read them
			if (!parseUnknownDataObject())
				return false;
		}
		else
		if (objectName == "DeclData")
		{
			// arbitrary vertex attributes
			// first comes the number of element definitions
			// then the vertex element type definitions
			// with format type;tesselator;semantics;usageindex
			// we want to support 2;0;6;0 == tangent
			//                    2;0;7;0 == binormal
			//                    2;0;3;0 == normal
			//                  1/2;0;5;0 == 1st uv coord
			// and              1/2;0;5;1 == 2nd uv coord
			// type==2 is 3xf32, type==1 is 2xf32
			uint32_t j;
			const uint32_t dcnt = readInt();
			uint16_t size = 0;
			int16_t normalpos = -1;
			int16_t uvpos = -1;
			int16_t uv2pos = -1;
			int16_t tangentpos = -1;
			int16_t binormalpos = -1;
			int16_t normaltype = -1;
			int16_t uvtype = -1;
			int16_t uv2type = -1;
			int16_t tangenttype = -1;
			int16_t binormaltype = -1;
			for (j=0; j<dcnt; ++j)
			{
				const uint32_t type = readInt();
				//const uint32_t tesselator = readInt();
				readInt();
				const uint32_t semantics = readInt();
				const uint32_t index = readInt();
				switch (semantics)
				{
				case 3:
					normalpos = size;
					normaltype = type;
					break;
				case 5:
					if (index==0)
					{
						uvpos = size;
						uvtype = type;
					}
					else if (index==1)
					{
						uv2pos = size;
						uv2type = type;
					}
					break;
				case 6:
					tangentpos = size;
					tangenttype = type;
					break;
				case 7:
					binormalpos = size;
					binormaltype = type;
					break;
				default:
					break;
				}
				switch (type)
				{
				case 0:
					size += 4;
					break;
				case 1:
					size += 8;
					break;
				case 2:
					size += 12;
					break;
				case 3:
					size += 16;
					break;
				case 4:
				case 5:
				case 6:
					size += 4;
					break;
				case 7:
					size += 8;
					break;
				case 8:
				case 9:
					size += 4;
					break;
				case 10:
					size += 8;
					break;
				case 11:
					size += 4;
					break;
				case 12:
					size += 8;
					break;
				case 13:
					size += 4;
					break;
				case 14:
					size += 4;
					break;
				case 15:
					size += 4;
					break;
				case 16:
					size += 8;
					break;
				}
			}
			const uint32_t datasize = readInt();
			uint32_t* data = new uint32_t[datasize];
			for (j=0; j<datasize; ++j)
				data[j]=readInt();

			if (!checkForOneFollowingSemicolons())
			{
				os::Printer::log("No finishing semicolon in DeclData found.", ELL_WARNING);

			}
			if (!checkForClosingBrace())
			{
				os::Printer::log("No closing brace in DeclData.", ELL_WARNING);

				delete [] data;
				return false;
			}
			uint8_t* dataptr = (uint8_t*) data;
			if ((uv2pos != -1) && (uv2type == 1))
				mesh.TCoords2.reserve(mesh.Vertices.size());
			for (j=0; j<mesh.Vertices.size(); ++j)
			{
				if ((normalpos != -1) && (normaltype == 2))
					mesh.Vertices[j].Normal.set(*((core::vector3df*)(dataptr+normalpos)));
				if ((uvpos != -1) && (uvtype == 1))
					mesh.Vertices[j].TCoords.set(*((core::vector2df*)(dataptr+uvpos)));
				if ((uv2pos != -1) && (uv2type == 1))
					mesh.TCoords2.push_back(*((core::vector2df*)(dataptr+uv2pos)));
				dataptr += size;
			}
			delete [] data;
		}
		else
		if (objectName == "FVFData")
		{
			if (!readHeadOfDataObject())
			{
				os::Printer::log("No starting brace in FVFData found.", ELL_WARNING);

				return false;
			}
			const uint32_t dataformat = readInt();
			const uint32_t datasize = readInt();
			uint32_t* data = new uint32_t[datasize];
			for (uint32_t j=0; j<datasize; ++j)
				data[j]=readInt();
			if (dataformat&0x102) // 2nd uv set
			{
				mesh.TCoords2.reserve(mesh.Vertices.size());
				uint8_t* dataptr = (uint8_t*) data;
				const uint32_t size=((dataformat>>8)&0xf)*sizeof(core::vector2df);
				for (uint32_t j=0; j<mesh.Vertices.size(); ++j)
				{
					mesh.TCoords2.push_back(*((core::vector2df*)(dataptr)));
					dataptr += size;
				}
			}
			delete [] data;
			if (!checkForOneFollowingSemicolons())
			{
				os::Printer::log("No finishing semicolon in FVFData found.", ELL_WARNING);

			}
			if (!checkForClosingBrace())
			{
				os::Printer::log("No closing brace in FVFData found in x file", ELL_WARNING);

				return false;
			}
		}
		else
		if (objectName == "XSkinMeshHeader")
		{
			if (!parseDataObjectSkinMeshHeader(mesh))
				return false;
		}
		else
		if (objectName == "SkinWeights")
		{
			//mesh.SkinWeights.push_back(SXSkinWeight());
			//if (!parseDataObjectSkinWeights(mesh.SkinWeights.back()))
			if (!parseDataObjectSkinWeights(mesh))
				return false;
		}
		else
		{
			os::Printer::log("Unknown data object in mesh in x file", objectName, ELL_WARNING);
			if (!parseUnknownDataObject())
				return false;
		}
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectSkinWeights(SXMesh &mesh)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: Reading mesh skin weights", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject())
	{
		os::Printer::log("No opening brace in Skin Weights found in .x file", ELL_WARNING);

		return false;
	}

	std::string TransformNodeName;

	if (!getNextTokenAsString(TransformNodeName))
	{
		os::Printer::log("Unknown syntax while reading transfrom node name string in .x file", ELL_WARNING);

		return false;
	}

	ICPUSkinnedMesh::SJoint *joint=0;

	size_t jointID;
	for (jointID=0; jointID < AnimatedMesh->getAllJoints().size(); jointID++)
	{
		if (AnimatedMesh->getAllJoints()[jointID]->Name==TransformNodeName)
		{
			joint=AnimatedMesh->getAllJoints()[jointID];
			break;
		}
	}

	if (!joint)
	{
#ifdef _XREADER_DEBUG
		os::Printer::log("creating joint for skinning ", TransformNodeName, ELL_DEBUG);
#endif
		jointID = AnimatedMesh->getAllJoints().size();
		joint=AnimatedMesh->addJoint(0);
		joint->Name=TransformNodeName;
	}

	// read vertex indices
	const uint32_t nWeights = readInt();
	uint32_t* vertexIDs = new uint32_t[nWeights];
	uint32_t maxIx = 0;
	for (size_t i=0; i<nWeights; i++)
    {
		vertexIDs[i] = readInt();
		if (vertexIDs[i]>maxIx)
            maxIx = vertexIDs[i];
    }
    size_t oldUsed = mesh.VertexSkinWeights.size();
    if (maxIx>=oldUsed)
    {
        mesh.VertexSkinWeights.resize(maxIx+1);
        memset(mesh.VertexSkinWeights.data()+oldUsed,0,(maxIx+1-oldUsed)*sizeof(SkinnedVertexIntermediateData));
    }

	// read vertex weights
	for (size_t i=0; i<nWeights; ++i)
	{
	    SkinnedVertexIntermediateData& tmp = mesh.VertexSkinWeights[vertexIDs[i]];
        float tmpWeight = readFloat();
        for (size_t j=0; j<4; j++)
        {
            if (tmpWeight<=tmp.boneWeights[j])
                continue;

            tmp.boneIDs[j] = jointID;
            tmp.boneWeights[j] = tmpWeight;
            break;
        }
	}
	delete [] vertexIDs;


	// read matrix offset

	// transforms the mesh vertices to the space of the bone
	// When concatenated to the bone's transform, this provides the
	// world space coordinates of the mesh as affected by the bone
	readMatrix(joint->GlobalInversedMatrix);

	if (!checkForOneFollowingSemicolons())
	{
		os::Printer::log("No finishing semicolon in Skin Weights found in x file", ELL_WARNING);

	}

	if (!checkForClosingBrace())
	{
		os::Printer::log("No closing brace in Skin Weights found in x file", ELL_WARNING);

		return false;
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectSkinMeshHeader(SXMesh& mesh)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: Reading skin mesh header", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject())
	{
		os::Printer::log("No opening brace in Skin Mesh header found in .x file", ELL_WARNING);

		return false;
	}

	//mesh.MaxSkinWeightsPerVertex = readInt();
	//mesh.MaxSkinWeightsPerFace = readInt();
	readInt();readInt();

	mesh.BoneCount = readInt();

	if (!BinaryFormat)
		getNextToken(); // skip semicolon

	if (!checkForClosingBrace())
	{
		os::Printer::log("No closing brace in skin mesh header in x file", ELL_WARNING);

		return false;
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectMeshNormals(SXMesh &mesh)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: reading mesh normals", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject())
	{
		os::Printer::log("No opening brace in Mesh Normals found in x file", ELL_WARNING);

		return false;
	}

	// read count
	const uint32_t nNormals = readInt();
	core::vector<core::vector3df> normals;
	normals.resize(nNormals);

	// read normals
	for (uint32_t i=0; i<nNormals; ++i)
		readVector3(normals[i]);

	if (!checkForTwoFollowingSemicolons())
	{
		os::Printer::log("No finishing semicolon in Mesh Normals Array found in x file", ELL_WARNING);

	}

	core::vector<uint32_t> normalIndices;
	normalIndices.resize(mesh.Indices.size());

	// read face normal indices
	const uint32_t nFNormals = readInt();

	uint32_t normalidx = 0;
	core::vector<uint32_t> polygonfaces;
	for (uint32_t k=0; k<nFNormals; ++k)
	{
		const uint32_t fcnt = readInt();
		uint32_t triangles = fcnt - 2;
		uint32_t indexcount = triangles * 3;

		if (indexcount != mesh.IndexCountPerFace[k])
		{
			os::Printer::log("Not matching normal and face index count found in x file", ELL_WARNING);

			return false;
		}

		if (indexcount == 3)
		{
			// default, only one triangle in this face
			for (uint32_t h=0; h<3; ++h)
			{
				const uint32_t normalnum = readInt();
				mesh.Vertices[mesh.Indices[normalidx++]].Normal.set(normals[normalnum]);
			}
		}
		else
		{
			polygonfaces.resize(fcnt);
			// multiple triangles in this face
			for (uint32_t h=0; h<fcnt; ++h)
				polygonfaces[h] = readInt();

			for (uint32_t jk=0; jk<triangles; ++jk)
			{
				mesh.Vertices[mesh.Indices[normalidx++]].Normal.set(normals[polygonfaces[0]]);
				mesh.Vertices[mesh.Indices[normalidx++]].Normal.set(normals[polygonfaces[jk+1]]);
				mesh.Vertices[mesh.Indices[normalidx++]].Normal.set(normals[polygonfaces[jk+2]]);
			}
		}
	}

	if (!checkForTwoFollowingSemicolons())
	{
		os::Printer::log("No finishing semicolon in Mesh Face Normals Array found in x file", ELL_WARNING);

	}

	if (!checkForClosingBrace())
	{
		os::Printer::log("No closing brace in Mesh Normals found in x file", ELL_WARNING);

		return false;
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectMeshTextureCoords(SXMesh &mesh)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: reading mesh texture coordinates", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject())
	{
		os::Printer::log("No opening brace in Mesh Texture Coordinates found in x file", ELL_WARNING);

		return false;
	}

	const uint32_t nCoords = readInt();
	for (uint32_t i=0; i<nCoords; ++i)
		readVector2(mesh.Vertices[i].TCoords);

	if (!checkForTwoFollowingSemicolons())
	{
		os::Printer::log("No finishing semicolon in Mesh Texture Coordinates Array found in x file", ELL_WARNING);

	}

	if (!checkForClosingBrace())
	{
		os::Printer::log("No closing brace in Mesh Texture Coordinates Array found in x file", ELL_WARNING);

		return false;
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectMeshVertexColors(SXMesh &mesh)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: reading mesh vertex colors", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject())
	{
		os::Printer::log("No opening brace for Mesh Vertex Colors found in x file", ELL_WARNING);

		return false;
	}

	mesh.HasVertexColors=true;
	const typename decltype(mesh.Colors)::size_type nColors = readInt();
	mesh.Colors.resize(core::max_(mesh.Colors.size(),nColors));
	for (uint32_t i=0; i<nColors; ++i)
	{
		const uint32_t Index=readInt();
		if (Index>=mesh.Vertices.size())
		{
			os::Printer::log("index value in parseDataObjectMeshVertexColors out of bounds", ELL_WARNING);

			return false;
		}
		video::SColor tmpCol;
		readRGBA(tmpCol);
		mesh.Colors[Index] = tmpCol.color;
		checkForOneFollowingSemicolons();
	}

	if (!checkForOneFollowingSemicolons())
	{
		os::Printer::log("No finishing semicolon in Mesh Vertex Colors Array found in x file", ELL_WARNING);

	}

	if (!checkForClosingBrace())
	{
		os::Printer::log("No closing brace in Mesh Texture Coordinates Array found in x file", ELL_WARNING);

		return false;
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectMeshMaterialList(SXMesh &mesh)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: Reading mesh material list", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject())
	{
		os::Printer::log("No opening brace in Mesh Material List found in x file", ELL_WARNING);

		return false;
	}

	// read material count
	mesh.Materials.reserve(readInt());

	// read non triangulated face material index count
	const uint32_t nFaceIndices = readInt();

	// There seems to be a compact representation of "all faces the same material"
	// being represented as 1;1;0;; which means 1 material, 1 face with first material
	// all the other faces have to obey then, so check is disabled
	//if (nFaceIndices != mesh.IndexCountPerFace.size())
	//	os::Printer::log("Index count per face not equal to face material index count in x file.", ELL_WARNING);

	// read non triangulated face indices and create triangulated ones
	mesh.FaceMaterialIndices.resize( mesh.Indices.size() / 3);
	uint32_t triangulatedindex = 0;
	uint32_t ind = 0;
	for (uint32_t tfi=0; tfi<mesh.IndexCountPerFace.size(); ++tfi)
	{
		if (tfi<nFaceIndices)
			ind = readInt();
		const uint32_t fc = mesh.IndexCountPerFace[tfi]/3;
		for (uint32_t k=0; k<fc; ++k)
			mesh.FaceMaterialIndices[triangulatedindex++] = ind;
	}

	// in version 03.02, the face indices end with two semicolons.
	// commented out version check, as version 03.03 exported from blender also has 2 semicolons
	if (!BinaryFormat) // && MajorVersion == 3 && MinorVersion <= 2)
	{
		if (fileContents.peek() == ';')
            fileContents.seekg(1,fileContents.cur);
	}

	// read following data objects

	while(true)
	{
		std::string objectName = getNextToken();

		if (objectName.size() == 0)
		{
			os::Printer::log("Unexpected ending found in Mesh Material list in .x file.", ELL_WARNING);

			return false;
		}
		else
		if (objectName == "}")
		{
			break; // material list finished
		}
		else
		if (objectName == "{")
		{
			// template materials now available thanks to joeWright
			objectName = getNextToken();
			for (uint32_t i=0; i<TemplateMaterials.size(); ++i)
				if (TemplateMaterials[i].Name == objectName)
					mesh.Materials.push_back(TemplateMaterials[i].Material);
			getNextToken(); // skip }
		}
		else
		if (objectName == "Material")
		{
			mesh.Materials.push_back(video::SMaterial());
			if (!parseDataObjectMaterial(mesh.Materials.back()))
				return false;
		}
		else
		if (objectName == ";")
		{
			// ignore
		}
		else
		{
			os::Printer::log("Unknown data object in material list in x file", objectName, ELL_WARNING);
			if (!parseUnknownDataObject())
				return false;
		}
	}
	return true;
}


bool CXMeshFileLoader::parseDataObjectMaterial(video::SMaterial& material)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: Reading mesh material", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject())
	{
		os::Printer::log("No opening brace in Mesh Material found in .x file", ELL_WARNING);

		return false;
	}

	// read RGBA
	readRGBA(material.DiffuseColor); checkForOneFollowingSemicolons();

	// read power
	material.Shininess = readFloat();

	// read specular
	readRGB(material.SpecularColor); checkForOneFollowingSemicolons();

	// read emissive
	readRGB(material.EmissiveColor); checkForOneFollowingSemicolons();

	// read other data objects
	int textureLayer=0;
	while(true)
	{
		core::stringc objectName = getNextToken().c_str();

		if (objectName.size() == 0)
		{
			os::Printer::log("Unexpected ending found in Mesh Material in .x file.", ELL_WARNING);

			return false;
		}
		else
		if (objectName == "}")
		{
			break; // material finished
		}
		else
		if (objectName.equals_ignore_case("TextureFilename"))
		{
			// some exporters write "TextureFileName" instead.
			std::string tmp;
			if (!parseDataObjectTextureFilename(tmp))
				return false;
			core::stringc TextureFileName = tmp.c_str();

			// original name
			if (FileSystem->existFile(TextureFileName))
				material.setTexture(textureLayer, SceneManager->getVideoDriver()->getTexture(TextureFileName));
			// mesh path
			else
			{
				TextureFileName=FilePath + io::IFileSystem::getFileBasename(TextureFileName);
				if (FileSystem->existFile(TextureFileName))
					material.setTexture(textureLayer, SceneManager->getVideoDriver()->getTexture(TextureFileName));
				// working directory
				else
					material.setTexture(textureLayer, SceneManager->getVideoDriver()->getTexture(io::IFileSystem::getFileBasename(TextureFileName)));
			}
			++textureLayer;
		}
		else
		if (objectName.equals_ignore_case("NormalmapFilename"))
		{
			// some exporters write "NormalmapFileName" instead.
			std::string tmp;
			if (!parseDataObjectTextureFilename(tmp))
				return false;
			core::stringc TextureFileName = tmp.c_str();

			// original name
			if (FileSystem->existFile(TextureFileName))
				material.setTexture(1, SceneManager->getVideoDriver()->getTexture(TextureFileName));
			// mesh path
			else
			{
				TextureFileName=FilePath + io::IFileSystem::getFileBasename(TextureFileName);
				if (FileSystem->existFile(TextureFileName))
					material.setTexture(1, SceneManager->getVideoDriver()->getTexture(TextureFileName));
				// working directory
				else
					material.setTexture(1, SceneManager->getVideoDriver()->getTexture(io::IFileSystem::getFileBasename(TextureFileName)));
			}
			if (textureLayer==1)
				++textureLayer;
		}
		else
		{
			os::Printer::log("Unknown data object in material in .x file", objectName.c_str(), ELL_WARNING);
			if (!parseUnknownDataObject())
				return false;
		}
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectAnimationSet()
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: Reading animation set", ELL_DEBUG);
#endif

	std::string AnimationName;

	if (!readHeadOfDataObject(&AnimationName))
	{
		os::Printer::log("No opening brace in Animation Set found in x file", ELL_WARNING);

		return false;
	}
	os::Printer::log("Reading animationset ", AnimationName, ELL_DEBUG);

	while(true)
	{
		std::string objectName = getNextToken();

		if (objectName.size() == 0)
		{
			os::Printer::log("Unexpected ending found in Animation set in x file.", ELL_WARNING);

			return false;
		}
		else
		if (objectName == "}")
		{
			break; // animation set finished
		}
		else
		if (objectName == "Animation")
		{
			if (!parseDataObjectAnimation())
				return false;
		}
		else
		{
			os::Printer::log("Unknown data object in animation set in x file", objectName, ELL_WARNING);
			if (!parseUnknownDataObject())
				return false;
		}
	}
	return true;
}


bool CXMeshFileLoader::parseDataObjectAnimation()
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: reading animation", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject())
	{
		os::Printer::log("No opening brace in Animation found in x file", ELL_WARNING);

		return false;
	}

	//anim.closed = true;
	//anim.linearPositionQuality = true;
	ICPUSkinnedMesh::SJoint animationDump;

	std::string FrameName;

	while(true)
	{
		std::string objectName = getNextToken();

		if (objectName.size() == 0)
		{
			os::Printer::log("Unexpected ending found in Animation in x file.", ELL_WARNING);

			return false;
		}
		else
		if (objectName == "}")
		{
			break; // animation finished
		}
		else
		if (objectName == "AnimationKey")
		{
			if (!parseDataObjectAnimationKey(&animationDump))
				return false;
		}
		else
		if (objectName == "AnimationOptions")
		{
			//TODO: parse options.
			if (!parseUnknownDataObject())
				return false;
		}
		else
		if (objectName == "{")
		{
			// read frame name
			FrameName = getNextToken();

			if (!checkForClosingBrace())
			{
				os::Printer::log("Unexpected ending found in Animation in x file.", ELL_WARNING);

				return false;
			}
		}
		else
		{
			os::Printer::log("Unknown data object in animation in x file", objectName, ELL_WARNING);
			if (!parseUnknownDataObject())
				return false;
		}
	}

	if (FrameName.size() != 0)
	{
#ifdef _XREADER_DEBUG
		os::Printer::log("frame name", FrameName, ELL_DEBUG);
#endif
		ICPUSkinnedMesh::SJoint *joint=0;

		uint32_t n;
		for (n=0; n < AnimatedMesh->getAllJoints().size(); ++n)
		{
			if (AnimatedMesh->getAllJoints()[n]->Name==FrameName)
			{
				joint=AnimatedMesh->getAllJoints()[n];
				break;
			}
		}

		if (!joint)
		{
#ifdef _XREADER_DEBUG
			os::Printer::log("creating joint for animation ", FrameName, ELL_DEBUG);
#endif
			joint=AnimatedMesh->addJoint(0);
			joint->Name=FrameName;
		}

		joint->PositionKeys.reserve(joint->PositionKeys.size()+animationDump.PositionKeys.size());
		for (n=0; n<animationDump.PositionKeys.size(); ++n)
		{
			joint->PositionKeys.push_back(animationDump.PositionKeys[n]);
		}

		joint->ScaleKeys.reserve(joint->ScaleKeys.size()+animationDump.ScaleKeys.size());
		for (n=0; n<animationDump.ScaleKeys.size(); ++n)
		{
			joint->ScaleKeys.push_back(animationDump.ScaleKeys[n]);
		}

		joint->RotationKeys.reserve(joint->RotationKeys.size()+animationDump.RotationKeys.size());
		for (n=0; n<animationDump.RotationKeys.size(); ++n)
		{
			joint->RotationKeys.push_back(animationDump.RotationKeys[n]);
		}
	}
	else
		os::Printer::log("joint name was never given", ELL_WARNING);

	return true;
}


bool CXMeshFileLoader::parseDataObjectAnimationKey(ICPUSkinnedMesh::SJoint *joint)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: reading animation key", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject())
	{
		os::Printer::log("No opening brace in Animation Key found in x file", ELL_WARNING);

		return false;
	}

	// read key type

	const uint32_t keyType = readInt();

	if (keyType > 4)
	{
		os::Printer::log("Unknown key type found in Animation Key in x file", ELL_WARNING);

		return false;
	}

	// read number of keys
	const uint32_t numberOfKeys = readInt();

	// eat the semicolon after the "0".  if there are keys present, readInt()
	// does this for us.  If there aren't, we need to do it explicitly
	if (numberOfKeys == 0)
		checkForOneFollowingSemicolons();

	for (uint32_t i=0; i<numberOfKeys; ++i)
	{
		// read time
		const float time = (float)readInt();

		// read keys
		switch(keyType)
		{
		case 0: //rotation
			{
				//read quaternions

				// read count
				if (readInt() != 4)
				{
					os::Printer::log("Expected 4 numbers in animation key in x file", ELL_WARNING);

					return false;
				}

                core::vectorSIMDf quatern;
				quatern.W = -readFloat();
				quatern.X = readFloat();
				quatern.Y = readFloat();
				quatern.Z = readFloat();

                quatern = normalize(quatern);

				if (!checkForTwoFollowingSemicolons())
				{
					os::Printer::log("No finishing semicolon after quaternion animation key in x file", ELL_WARNING);

				}

				ICPUSkinnedMesh::SRotationKey *key=joint->addRotationKey();
				key->frame=time;
				key->rotation.set(quatern);
			}
			break;
		case 1: //scale
		case 2: //position
			{
				// read vectors

				// read count
				if (readInt() != 3)
				{
					os::Printer::log("Expected 3 numbers in animation key in x file", ELL_WARNING);

					return false;
				}

				core::vector3df vector;
				readVector3(vector);

				if (!checkForTwoFollowingSemicolons())
				{
					os::Printer::log("No finishing semicolon after vector animation key in x file", ELL_WARNING);

				}

				if (keyType==2)
				{
					ICPUSkinnedMesh::SPositionKey *key=joint->addPositionKey();
					key->frame=time;
					key->position=vector;
				}
				else
				{
					ICPUSkinnedMesh::SScaleKey *key=joint->addScaleKey();
					key->frame=time;
					key->scale=vector;
				}
			}
			break;
		case 3:
		case 4:
			{
				// read matrix

				// read count
				if (readInt() != 16)
				{
					os::Printer::log("Expected 16 numbers in animation key in x file", ELL_WARNING);

					return false;
				}

				// read matrix
				core::matrix4x3 mat4x3;
				readMatrix(mat4x3);

				//mat=joint->LocalMatrix*mat;

				if (!checkForOneFollowingSemicolons())
				{
					os::Printer::log("No finishing semicolon after matrix animation key in x file", ELL_WARNING);

				}

				//core::vector3df rotation = mat.getRotationDegrees();

				ICPUSkinnedMesh::SRotationKey *keyR=joint->addRotationKey();
				keyR->frame=time;

				keyR->rotation = core::quaternion(mat4x3);

				ICPUSkinnedMesh::SPositionKey *keyP=joint->addPositionKey();
				keyP->frame=time;
				keyP->position=mat4x3.getTranslation();


				core::vector3df scale=mat4x3.getScale();

				if (scale.X==0)
					scale.X=1;
				if (scale.Y==0)
					scale.Y=1;
				if (scale.Z==0)
					scale.Z=1;
				ICPUSkinnedMesh::SScaleKey *keyS=joint->addScaleKey();
				keyS->frame=time;
				keyS->scale=scale;
			}
			break;
		} // end switch
	}

	if (!checkForOneFollowingSemicolons())
		fileContents.unget();

	if (!checkForClosingBrace())
	{
		os::Printer::log("No closing brace in animation key in x file", ELL_WARNING);

		return false;
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectTextureFilename(std::string& texturename)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: reading texture filename", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject())
	{
		os::Printer::log("No opening brace in Texture filename found in x file", ELL_WARNING);

		return false;
	}

	if (!getNextTokenAsString(texturename))
	{
		os::Printer::log("Unknown syntax while reading texture filename string in x file", ELL_WARNING);

		return false;
	}

	if (!checkForClosingBrace())
	{
		os::Printer::log("No closing brace in Texture filename found in x file", ELL_WARNING);

		return false;
	}

	return true;
}


bool CXMeshFileLoader::parseUnknownDataObject()
{
	// find opening delimiter
	while(true)
	{
		std::string t = getNextToken();

		if (t.size() == 0)
			return false;

		if (t == "{")
			break;
	}

	uint32_t counter = 1;

	// parse until closing delimiter

	while(counter)
	{
		std::string t = getNextToken();

		if (t.size() == 0)
			return false;

		if (t == "{")
			++counter;
		else
		if (t == "}")
			--counter;
	}

	return true;
}


//! checks for closing curly brace, returns false if not there
bool CXMeshFileLoader::checkForClosingBrace()
{
	return (getNextToken() == "}");
}


//! checks for one following semicolon, returns false if not there
bool CXMeshFileLoader::checkForOneFollowingSemicolons()
{
	if (BinaryFormat)
		return true;

	if (getNextToken() == ";")
		return true;
	else
	{
		fileContents.unget();
		return false;
	}
}


//! checks for two following semicolons, returns false if they are not there
bool CXMeshFileLoader::checkForTwoFollowingSemicolons()
{
	if (BinaryFormat)
		return true;

	for (uint32_t k=0; k<2; ++k)
	{
		if (getNextToken() != ";")
		{
			fileContents.unget();
			return false;
		}
	}

	return true;
}


//! reads header of dataobject including the opening brace.
//! returns false if error happened, and writes name of object
//! if there is one
bool CXMeshFileLoader::readHeadOfDataObject(std::string* outname)
{
	std::string nameOrBrace = getNextToken();
	if (nameOrBrace != "{")
	{
		if (outname)
			(*outname) = nameOrBrace;

		if (getNextToken() != "{")
			return false;
	}

	return true;
}


//! returns next parseable token. Returns empty string if no token there
std::string CXMeshFileLoader::getNextToken()
{
	std::string s;

	// process binary-formatted file
	if (BinaryFormat)
	{
		// in binary mode it will only return NAME and STRING token
		// and (correctly) skip over other tokens.

		int16_t tok = readBinWord();
		uint32_t len;

		// standalone tokens
		switch (tok) {
			case 1:
				// name token
				len = readBinDWord();
				s.resize(len);
				fileContents.get(&s[0],len+1);
				return s;
			case 2:
				// string token
				len = readBinDWord();
				s.resize(len);
				fileContents.get(&s[0],len+1);
                fileContents.seekg(2,fileContents.cur);
				return s;
			case 3:
				// integer token
                fileContents.seekg(4,fileContents.cur);
				return "<integer>";
			case 5:
				// GUID token
                fileContents.seekg(16,fileContents.cur);
				return "<guid>";
			case 6:
				len = readBinDWord();
                fileContents.seekg(4*len,fileContents.cur);
				return "<int_list>";
			case 7:
				len = readBinDWord();
                fileContents.seekg(FloatSize*len,fileContents.cur);
				return "<flt_list>";
			case 0x0a:
				return "{";
			case 0x0b:
				return "}";
			case 0x0c:
				return "(";
			case 0x0d:
				return ")";
			case 0x0e:
				return "[";
			case 0x0f:
				return "]";
			case 0x10:
				return "<";
			case 0x11:
				return ">";
			case 0x12:
				return ".";
			case 0x13:
				return ",";
			case 0x14:
				return ";";
			case 0x1f:
				return "template";
			case 0x28:
				return "WORD";
			case 0x29:
				return "DWORD";
			case 0x2a:
				return "FLOAT";
			case 0x2b:
				return "DOUBLE";
			case 0x2c:
				return "CHAR";
			case 0x2d:
				return "UCHAR";
			case 0x2e:
				return "SWORD";
			case 0x2f:
				return "SDWORD";
			case 0x30:
				return "void";
			case 0x31:
				return "string";
			case 0x32:
				return "unicode";
			case 0x33:
				return "cstring";
			case 0x34:
				return "array";
		}
	}
	// process text-formatted file
	else
	{
		findNextNoneWhiteSpace();

		if (fileContents.eof())
			return s;

		while(!fileContents.eof() && !core::isspace(fileContents.peek()))
		{
			// either keep token delimiters when already holding a token, or return if first valid char
			if (fileContents.peek()==';' || fileContents.peek()=='}' || fileContents.peek()=='{' || fileContents.peek()==',')
			{
				if (!s.size())
					s.push_back(fileContents.get());

				break; // stop for delimiter
			}
			s.push_back(fileContents.get());
		}
	}
	return s;
}


//! places pointer to next begin of a token, which must be a number,
// and ignores comments
void CXMeshFileLoader::findNextNoneWhiteSpaceNumber()
{
	if (BinaryFormat)
		return;

	while(!fileContents.eof())
    {
        char p;
	    fileContents >> std::ws >> p;
        if (p == '-' || p == '.' || core::isdigit(p))
        {
            fileContents.unget();
            break;
        }

		// check if this is a comment
		if ((p == '/' && fileContents.peek() == '/') || p == '#')
        {
			std::string stmp;
			std::getline(fileContents,stmp);
        }
	}
}


// places pointer to next begin of a token, and ignores comments
void CXMeshFileLoader::findNextNoneWhiteSpace()
{
	if (BinaryFormat)
		return;

	while(true)
	{
	    fileContents >> std::ws;

		if (fileContents.eof())
			return;

		// check if this is a comment
        char p = fileContents.get();
		if ((p == '/' && fileContents.peek() == '/') || p == '#')
        {
			std::string stmp;
			std::getline(fileContents,stmp);
        }
		else
        {
            fileContents.unget();
			break;
        }
	}
}

//! reads a x file style string
bool CXMeshFileLoader::getNextTokenAsString(std::string& out)
{
	if (BinaryFormat)
	{
		out=getNextToken();
		return true;
	}
	findNextNoneWhiteSpace();

	if (fileContents.eof())
		return false;

	if (fileContents.get() != '"')
    {
        fileContents.unget();
		return false;
    }

	while(!fileContents.eof() && fileContents.peek()!='"')
	{
		out.push_back(fileContents.get());
	}

	char P[2];
	fileContents.read(P,2);
	if ( P[1] != ';' || P[0] != '"')
    {
        fileContents.unget();
        fileContents.unget();
		return false;
    }

	return true;
}


uint16_t CXMeshFileLoader::readBinWord()
{
	if (fileContents.eof())
		return 0;

    char P[2];
    fileContents.read(P,2);

    return *(uint16_t *)P;
}


uint32_t CXMeshFileLoader::readBinDWord()
{
	if (fileContents.eof())
		return 0;

    char P[4];
    fileContents.read(P,4);

	return *(uint32_t *)P;
}


uint32_t CXMeshFileLoader::readInt()
{
	if (BinaryFormat)
	{
		if (!BinaryNumCount)
		{
			const uint16_t tmp = readBinWord(); // 0x06 or 0x03
			if (tmp == 0x06)
				BinaryNumCount = readBinDWord();
			else
				BinaryNumCount = 1; // single int
		}
		--BinaryNumCount;
		return readBinDWord();
	}
	else
	{
		findNextNoneWhiteSpaceNumber();

	    uint32_t retval;
	    fileContents >> retval;
	    return retval;
	}
}


float CXMeshFileLoader::readFloat()
{
	if (BinaryFormat)
	{
		if (!BinaryNumCount)
		{
			const uint16_t tmp = readBinWord(); // 0x07 or 0x42
			if (tmp == 0x07)
				BinaryNumCount = readBinDWord();
			else
				BinaryNumCount = 1; // single int
		}
		--BinaryNumCount;
		if (FloatSize == 8)
		{
		    double tmp;
		    fileContents.read(reinterpret_cast<char*>(&tmp),8);
			return tmp;
		}
		else
		{
		    float tmp;
		    fileContents.read(reinterpret_cast<char*>(&tmp),4);
			return tmp;
		}
	}
	findNextNoneWhiteSpaceNumber();
	float ftmp;
	fileContents >> ftmp;
	return ftmp;
}


// read 2-dimensional vector. Stops at semicolon after second value for text file format
bool CXMeshFileLoader::readVector2(core::vector2df& vec)
{
	vec.X = readFloat();
	vec.Y = readFloat();
	return true;
}


// read 3-dimensional vector. Stops at semicolon after third value for text file format
bool CXMeshFileLoader::readVector3(core::vector3df& vec)
{
	vec.X = readFloat();
	vec.Y = readFloat();
	vec.Z = readFloat();
	return true;
}


// read color without alpha value. Stops after second semicolon after blue value
bool CXMeshFileLoader::readRGB(video::SColor& color)
{
	video::SColorf tmpColor;
	tmpColor.getAsVectorSIMDf().r = readFloat();
	tmpColor.getAsVectorSIMDf().g = readFloat();
	tmpColor.getAsVectorSIMDf().b = readFloat();
	color = tmpColor.toSColor();
	return checkForOneFollowingSemicolons();
}


// read color with alpha value. Stops after second semicolon after blue value
bool CXMeshFileLoader::readRGBA(video::SColor& color)
{
	video::SColorf tmpColor;
	tmpColor.getAsVectorSIMDf().r = readFloat();
	tmpColor.getAsVectorSIMDf().g = readFloat();
	tmpColor.getAsVectorSIMDf().b = readFloat();
	tmpColor.getAsVectorSIMDf().a = readFloat();
	color = tmpColor.toSColor();
	return checkForOneFollowingSemicolons();
}


// read matrix from list of floats
bool CXMeshFileLoader::readMatrix(core::matrix4x3& mat)
{
    for (uint32_t j=0u; j<4u; j++)
    {
        for (uint32_t i=0u; i<3u; i++)
            mat(i,j) = readFloat();
        readFloat();
    }
	return checkForOneFollowingSemicolons();
}


} // end namespace scene
} // end namespace irr

#endif // _IRR_COMPILE_WITH_X_LOADER_

