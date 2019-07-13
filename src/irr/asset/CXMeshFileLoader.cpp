// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_X_LOADER_

#include "CXMeshFileLoader.h"
#include "os.h"

#include "IrrlichtDevice.h"
#include "coreutil.h"
#include "ISceneManager.h"
#include "IVideoDriver.h"
#include "IFileSystem.h"
#include "IReadFile.h"
#include "SVertexManipulator.h"
#include "assert.h"
#include "irr/asset/IAssetManager.h"
#include "irr/asset/SCPUMesh.h"
#include <chrono>
#include <vector>

#ifdef _IRR_DEBUG
#define _XREADER_DEBUG
#endif

namespace irr
{
namespace asset
{

//! Constructor
CXMeshFileLoader::CXMeshFileLoader(IrrlichtDevice* _dev)
: Device(_dev), SceneManager(_dev->getSceneManager()), FileSystem(_dev->getFileSystem())
{
	#ifdef _IRR_DEBUG
	setDebugName("CXMeshFileLoader");
	#endif
}

CXMeshFileLoader::~CXMeshFileLoader()
{
}

bool CXMeshFileLoader::isALoadableFileFormat(io::IReadFile* _file) const
{
    const size_t prevPos = _file->getPos();
    _file->seek(0u);

    char buf[4];
    _file->read(buf, 4);
    if (strncmp(buf, "xof ", 4) != 0)
    {
        _file->seek(prevPos);
        return false;
    }

    _file->seek(4, true);
    _file->read(buf, 4);
    if (strncmp(buf, "txt ", 4) != 0 && strncmp(buf, "bin ", 4) != 0)
    {
        _file->seek(prevPos);
        return false;
    }

    _file->read(buf, 4);
    if (strncmp(buf, "0032", 4) != 0 && strncmp(buf, "0064", 4) != 0)
    {
        _file->seek(prevPos);
        return false;
    }

    _file->seek(prevPos);
    return true;
}

asset::IAsset* CXMeshFileLoader::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
//#ifdef _XREADER_DEBUG
	auto time = std::chrono::high_resolution_clock::now();
//#endif

    SContext ctx(asset::IAssetLoader::SAssetLoadContext{_params, _file}, _override);

	if (!_file)
		return 0;

	ctx.AnimatedMesh = new asset::CCPUSkinnedMesh();
    asset::ICPUMesh* retVal = nullptr;

	if (load(ctx, _file))
	{
        ctx.AnimatedMesh->finalize();
		if (ctx.AnimatedMesh->isStatic())
        {
            asset::SCPUMesh* staticMesh = new asset::SCPUMesh();
            for (size_t i=0; i<ctx.AnimatedMesh->getMeshBufferCount(); i++)
            {
                asset::ICPUMeshBuffer* meshbuffer = new asset::ICPUMeshBuffer();
                staticMesh->addMeshBuffer(meshbuffer);
                meshbuffer->drop();

                asset::ICPUMeshBuffer* origMeshBuffer = ctx.AnimatedMesh->getMeshBuffer(i);
                asset::ICPUMeshDataFormatDesc* desc = static_cast<asset::ICPUMeshDataFormatDesc*>(origMeshBuffer->getMeshDataAndFormat());
                meshbuffer->getMaterial() = origMeshBuffer->getMaterial();
                meshbuffer->setPrimitiveType(origMeshBuffer->getPrimitiveType());

                bool doesntNeedIndices = !desc->getIndexBuffer();
                uint32_t largestVertex = origMeshBuffer->getIndexCount();
                meshbuffer->setIndexCount(largestVertex);
                if (doesntNeedIndices)
                {
                    largestVertex = 0;

                    size_t baseVertex = origMeshBuffer->getIndexType()==asset::EIT_32BIT ? ((uint32_t*)origMeshBuffer->getIndices())[0]:((uint16_t*)origMeshBuffer->getIndices())[0];
                    for (size_t j=1; j<origMeshBuffer->getIndexCount(); j++)
                    {
                        uint32_t nextIx = origMeshBuffer->getIndexType()==asset::EIT_32BIT ? ((uint32_t*)origMeshBuffer->getIndices())[j]:((uint16_t*)origMeshBuffer->getIndices())[j];
                        if (nextIx>largestVertex)
                            largestVertex = nextIx;

                        if (doesntNeedIndices&&(baseVertex+j!=nextIx))
                            doesntNeedIndices = false;
                    }

                    if (doesntNeedIndices)
                    {
                        desc->setIndexBuffer(NULL);
                        meshbuffer->setBaseVertex(baseVertex);
                    }
                }


                asset::E_INDEX_TYPE indexType;
                if (doesntNeedIndices)
                    indexType = asset::EIT_UNKNOWN;
                else
                {
                    asset::ICPUBuffer* indexBuffer;
                    if (largestVertex>=0x10000u)
                    {
                        indexType = asset::EIT_32BIT;
                        indexBuffer = new asset::ICPUBuffer(4*origMeshBuffer->getIndexCount());
                        for (size_t j=0; j<origMeshBuffer->getIndexCount(); j++)
                           ((uint32_t*)indexBuffer->getPointer())[j] = origMeshBuffer->getIndexType()==asset::EIT_32BIT ? ((uint32_t*)origMeshBuffer->getIndices())[j]:((uint16_t*)origMeshBuffer->getIndices())[j];
                    }
                    else
                    {
                        indexType = asset::EIT_16BIT;
                        indexBuffer = new asset::ICPUBuffer(2*origMeshBuffer->getIndexCount());
                        for (size_t j=0; j<origMeshBuffer->getIndexCount(); j++)
                           ((uint16_t*)indexBuffer->getPointer())[j] = origMeshBuffer->getIndexType()==asset::EIT_32BIT ? ((uint32_t*)origMeshBuffer->getIndices())[j]:((uint16_t*)origMeshBuffer->getIndices())[j];
                    }
                    desc->setIndexBuffer(indexBuffer);
                }
                meshbuffer->setIndexType(indexType);
                meshbuffer->setMeshDataAndFormat(desc);

                meshbuffer->setPositionAttributeIx(origMeshBuffer->getPositionAttributeIx());
                for (size_t j=0; j<asset::EVAI_COUNT; j++)
                {
                    asset::E_VERTEX_ATTRIBUTE_ID attrId = (asset::E_VERTEX_ATTRIBUTE_ID)j;
                    if (!desc->getMappedBuffer(attrId))
                        continue;

                    if (attrId==asset::EVAI_ATTR3)
                    {
                        const asset::ICPUBuffer* normalBuffer = desc->getMappedBuffer(asset::EVAI_ATTR3);
                        asset::ICPUBuffer* newNormalBuffer = new asset::ICPUBuffer(normalBuffer->getSize()/3);
                        for (size_t k=0; k<newNormalBuffer->getSize()/4; k++)
                        {
                            core::vectorSIMDf simdNormal;
                            simdNormal.set(((core::vector3df*)normalBuffer->getPointer())[k]);
                            ((uint32_t*)newNormalBuffer->getPointer())[k] = asset::quantizeNormal2_10_10_10(simdNormal);
                        }
                        desc->setVertexAttrBuffer(newNormalBuffer,asset::EVAI_ATTR3,asset::EF_A2B10G10R10_SNORM_PACK32);
                        newNormalBuffer->drop();
                    }
                }

                meshbuffer->recalculateBoundingBox();
            }
            staticMesh->recalculateBoundingBox();

            retVal = staticMesh;
            ctx.AnimatedMesh->drop();
            ctx.AnimatedMesh = 0;
        }
        else
            retVal = ctx.AnimatedMesh;
	}
	else
	{
		ctx.AnimatedMesh->drop();
        ctx.AnimatedMesh = 0;
	}
//#ifdef _XREADER_DEBUG
	std::ostringstream tmpString("Time to load ");
	tmpString.seekp(0,std::ios_base::end);
	tmpString << (ctx.BinaryFormat ? "binary" : "ascii") << " X file: " << (std::chrono::high_resolution_clock::now()-time).count() << "ms";
	os::Printer::log(tmpString.str());
//#endif

	return retVal;
}

class SuperSkinningTMPStruct
{
    public:
		inline bool operator<(const SuperSkinningTMPStruct& other) const { return (tmp >= other.tmp); }

        float tmp;
        uint32_t redir;
};

core::matrix4x3 getGlobalMatrix_evil(asset::ICPUSkinnedMesh::SJoint* joint)
{
    //if (joint->GlobalInversedMatrix.isIdentity())
        //return joint->GlobalInversedMatrix;
    if (joint->Parent)
        return concatenateBFollowedByA(getGlobalMatrix_evil(joint->Parent),joint->LocalMatrix);
    else
        return joint->LocalMatrix;
}


bool CXMeshFileLoader::load(SContext& _ctx, io::IReadFile* file)
{
	if (!readFileIntoMemory(_ctx, file))
		return false;

	if (!parseFile(_ctx))
		return false;

	for (uint32_t n=0; n<_ctx.Meshes.size(); ++n)
	{
		SXMesh *mesh= _ctx.Meshes[n];

		// default material if nothing loaded
		if (!mesh->Materials.size())
		{
			mesh->Materials.push_back(video::SCPUMaterial());
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
			mesh->Buffers.push_back( _ctx.AnimatedMesh->addMeshBuffer() );
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
                asset::ICPUMeshDataFormatDesc* desc = new asset::ICPUMeshDataFormatDesc();

				asset::ICPUBuffer* vPosBuf = new asset::ICPUBuffer(mesh->Vertices.size()*4*3);
				desc->setVertexAttrBuffer(vPosBuf,asset::EVAI_ATTR0,asset::EF_R32G32B32_SFLOAT);
				vPosBuf->drop();
				asset::ICPUBuffer* vColorBuf = NULL;
				if (mesh->Colors.size())
                {
                    vColorBuf = new asset::ICPUBuffer(mesh->Vertices.size()*4);
                    desc->setVertexAttrBuffer(vColorBuf,asset::EVAI_ATTR1,asset::EF_B8G8R8A8_UNORM);
                    vColorBuf->drop();
                }
				asset::ICPUBuffer* vTCBuf = new asset::ICPUBuffer(mesh->Vertices.size()*4*2);
                desc->setVertexAttrBuffer(vTCBuf,asset::EVAI_ATTR2,asset::EF_R32G32_SFLOAT);
                vTCBuf->drop();
				asset::ICPUBuffer* vNormalBuf = new asset::ICPUBuffer(mesh->Vertices.size()*4*3);
				desc->setVertexAttrBuffer(vNormalBuf,asset::EVAI_ATTR3,asset::EF_R32G32B32_SFLOAT);
				vNormalBuf->drop();
				asset::ICPUBuffer* vTC2Buf = NULL;
				if (mesh->TCoords2.size())
				{
                    vTC2Buf = new asset::ICPUBuffer(mesh->Vertices.size()*4*2);
                    desc->setVertexAttrBuffer(vTC2Buf,asset::EVAI_ATTR4,asset::EF_R32G32_SFLOAT);
                    vTC2Buf->drop();
				}
				asset::ICPUBuffer* vSkinningDataBuf = NULL;
				if (mesh->VertexSkinWeights.size())
                {
                    vSkinningDataBuf = new asset::ICPUBuffer(mesh->Vertices.size()*sizeof(SkinnedVertexFinalData));
                    desc->setVertexAttrBuffer(vSkinningDataBuf,asset::EVAI_ATTR5,asset::EF_R8G8B8A8_UINT,8,0);
                    desc->setVertexAttrBuffer(vSkinningDataBuf,asset::EVAI_ATTR6,asset::EF_A2B10G10R10_UNORM_PACK32,8,4);
                    vSkinningDataBuf->drop();
                }
				else if (mesh->AttachedJointID!=-1)
                {
                    vSkinningDataBuf = new asset::ICPUBuffer(mesh->Vertices.size()*sizeof(SkinnedVertexFinalData));
                    desc->setVertexAttrBuffer(vSkinningDataBuf,asset::EVAI_ATTR5,asset::EF_R8G8B8A8_UINT,8,0);
                    desc->setVertexAttrBuffer(vSkinningDataBuf,asset::EVAI_ATTR6,asset::EF_A2B10G10R10_UNORM_PACK32,8,4);
                    vSkinningDataBuf->drop();

                    bool correctBindMatrix = _ctx.AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix.isIdentity();
                    core::matrix4x3 globalMat,globalMatInvTransp;
                    if (correctBindMatrix)
                    {
                        globalMat = getGlobalMatrix_evil(_ctx.AnimatedMesh->getAllJoints()[mesh->AttachedJointID]);
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
                        _ctx.AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix.getInverse(globalMat);
                        globalMatInvTransp(0,0) = _ctx.AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(0,0);
                        globalMatInvTransp(1,0) = _ctx.AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(0,1);
                        globalMatInvTransp(2,0) = _ctx.AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(0,2);
                        globalMatInvTransp(0,1) = _ctx.AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(1,0);
                        globalMatInvTransp(1,1) = _ctx.AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(1,1);
                        globalMatInvTransp(2,1) = _ctx.AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(1,2);
                        globalMatInvTransp(0,2) = _ctx.AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(2,0);
                        globalMatInvTransp(1,2) = _ctx.AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(2,1);
                        globalMatInvTransp(2,2) = _ctx.AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(2,2);
                        globalMatInvTransp(0,3) = _ctx.AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(3,0);
                        globalMatInvTransp(1,3) = _ctx.AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(3,1);
                        globalMatInvTransp(2,3) = _ctx.AnimatedMesh->getAllJoints()[mesh->AttachedJointID]->GlobalInversedMatrix(3,2);
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
					asset::ICPUSkinnedMeshBuffer *buffer = mesh->Buffers[i];

                    buffer->setIndexRange(0,vCountArray[i]);
                    if (vCountArray[i]>0x10000u)
                        buffer->setIndexType(asset::EIT_32BIT);
                    else
                        buffer->setIndexType(asset::EIT_16BIT);

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
                    asset::ICPUSkinnedMeshBuffer *buffer = mesh->Buffers[ verticesLinkBuffer[i] ];

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
                    asset::ICPUSkinnedMeshBuffer *buffer = mesh->Buffers[ i ];


					uint32_t subBufferSz = vCountArray[i]*3;
					buffer->setIndexCount(subBufferSz);
                    subBufferSz *= (buffer->getIndexType()==asset::EIT_32BIT) ? 4:2;

                    //now cumulative
                    cumBaseVertex[i] = indexBufferSz;
					buffer->setIndexBufferOffset(indexBufferSz);
                    indexBufferSz += subBufferSz;
                }
                asset::ICPUBuffer* ixbuf = new asset::ICPUBuffer(indexBufferSz);
				desc->setIndexBuffer(ixbuf);
				ixbuf->drop();
				// create indices per buffer
				memset(vCountArray, 0, mesh->Buffers.size()*sizeof(uint32_t));
				for (i=0; i<mesh->FaceMaterialIndices.size(); ++i)
				{
					asset::ICPUSkinnedMeshBuffer *buffer = mesh->Buffers[ mesh->FaceMaterialIndices[i] ];

					void* indexBufAlreadyOffset = ((uint8_t*)ixbuf->getPointer())+cumBaseVertex[mesh->FaceMaterialIndices[i]];

                    if (buffer->getIndexType()==asset::EIT_32BIT)
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
bool CXMeshFileLoader::readFileIntoMemory(SContext& _ctx, io::IReadFile* file)
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
    _ctx.fileContents.str(Buffer);

	//! check header "xof "
	char tmp[4];
    _ctx.fileContents.read(tmp,4);
	if (strncmp(tmp, "xof ", 4)!=0)
	{
		os::Printer::log("Not an x file, wrong header.", ELL_WARNING);
		return false;
	}

	//! read minor and major version, e.g. 0302 or 0303
    _ctx.fileContents.read(tmp,2);
	tmp[2] = 0x0;
	sscanf(tmp,"%u",&_ctx.MajorVersion);

    _ctx.fileContents.read(tmp,2);
	sscanf(tmp,"%u",&_ctx.MinorVersion);

	//! read format
    _ctx.fileContents.read(tmp,4);
	if (strncmp(tmp, "txt ", 4) ==0)
        _ctx.BinaryFormat = false;
	else if (strncmp(tmp, "bin ", 4) ==0)
        _ctx.BinaryFormat = true;
	else
	{
		os::Printer::log("Only uncompressed x files currently supported.", ELL_WARNING);
		return false;
	}
    _ctx.BinaryNumCount=0;

	//! read float size
    _ctx.fileContents.read(tmp,4);
	if (strncmp(tmp, "0032", 4) ==0)
        _ctx.FloatSize = 4;
	else if (strncmp(tmp, "0064", 4) ==0)
        _ctx.FloatSize = 8;
	else
	{
		os::Printer::log("Float size not supported.", ELL_WARNING);
		return false;
	}


	{
        std::string stmp;
        std::getline(_ctx.fileContents,stmp);
	}
    _ctx.FilePath = io::IFileSystem::getFileDir(file->getFileName()) + "/";

	return true;
}


//! Parses the file
bool CXMeshFileLoader::parseFile(SContext& _ctx)
{
	while(parseDataObject(_ctx))
	{
		// loop
	}

	return true;
}


//! Parses the next Data object in the file
bool CXMeshFileLoader::parseDataObject(SContext& _ctx)
{
	std::string objectName = getNextToken(_ctx);

	if (objectName.size() == 0)
		return false;

	// parse specific object
#ifdef _XREADER_DEBUG
	os::Printer::log("debug DataObject:", objectName, ELL_DEBUG);
#endif

	if (objectName == "template")
		return parseDataObjectTemplate(_ctx);
	else
	if (objectName == "Frame")
	{
		return parseDataObjectFrame( _ctx, 0 );
	}
	else
	if (objectName == "Mesh")
	{
		// some meshes have no frames at all
		//CurFrame = AnimatedMesh->addJoint(0);

		SXMesh *mesh=new SXMesh;

		//mesh->Buffer=AnimatedMesh->addMeshBuffer();
        _ctx.Meshes.push_back(mesh);

		return parseDataObjectMesh(_ctx, *mesh);
	}
	else
	if (objectName == "AnimationSet")
	{
		return parseDataObjectAnimationSet(_ctx);
	}
	else
	if (objectName == "Material")
	{
		// template materials now available thanks to joeWright
        _ctx.TemplateMaterials.push_back(SXTemplateMaterial());
        _ctx.TemplateMaterials.back().Name = getNextToken(_ctx);
		return parseDataObjectMaterial(_ctx, _ctx.TemplateMaterials.back().Material);
	}
	else
	if (objectName == "}")
	{
		os::Printer::log("} found in dataObject", ELL_WARNING);
		return true;
	}

	os::Printer::log("Unknown data object in animation of .x file", objectName, ELL_WARNING);

	return parseUnknownDataObject(_ctx);
}


bool CXMeshFileLoader::parseDataObjectTemplate(SContext& _ctx)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: Reading template", ELL_DEBUG);
#endif

	// parse a template data object. Currently not stored.
	std::string name;

	if (!readHeadOfDataObject(_ctx, &name))
	{
		os::Printer::log("Left delimiter in template data object missing.",
			name, ELL_WARNING);

		return false;
	}

	// read GUID
	getNextToken(_ctx);

	// read and ignore data members
	while(true)
	{
		std::string s = getNextToken(_ctx);

		if (s == "}")
			break;

		if (s.size() == 0)
			return false;
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectFrame(SContext& _ctx, asset::ICPUSkinnedMesh::SJoint *Parent)
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

	if (!readHeadOfDataObject(_ctx, &name))
	{
		os::Printer::log("No opening brace in Frame found in x file", ELL_WARNING);

		return false;
	}

    asset::ICPUSkinnedMesh::SJoint *joint=0;

	if (name.size())
	{
		for (uint32_t n=0; n < _ctx.AnimatedMesh->getAllJoints().size(); ++n)
		{
			if (_ctx.AnimatedMesh->getAllJoints()[n]->Name==name)
			{
				joint= _ctx.AnimatedMesh->getAllJoints()[n];
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
		joint= _ctx.AnimatedMesh->addJoint(Parent);
		joint->Name=name;
		JointID= _ctx.AnimatedMesh->getAllJoints().size()-1;
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
		std::string objectName = getNextToken(_ctx);

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

			if (!parseDataObjectFrame(_ctx, joint))
				return false;
		}
		else
		if (objectName == "FrameTransformMatrix")
		{
			if (!parseDataObjectTransformationMatrix(_ctx, joint->LocalMatrix))
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

            _ctx.Meshes.push_back(mesh);

			if (!parseDataObjectMesh(_ctx, *mesh))
				return false;
		}
		else
		{
			os::Printer::log("Unknown data object in frame in x file", objectName, ELL_WARNING);
			if (!parseUnknownDataObject(_ctx))
				return false;
		}
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectTransformationMatrix(SContext& _ctx, core::matrix4x3 &mat)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: Reading Transformation Matrix", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject(_ctx))
	{
		os::Printer::log("No opening brace in Transformation Matrix found in x file", ELL_WARNING);

		return false;
	}

	readMatrix(_ctx, mat);

	if (!checkForOneFollowingSemicolons(_ctx))
	{
		os::Printer::log("No finishing semicolon in Transformation Matrix found in x file", ELL_WARNING);

	}

	if (!checkForClosingBrace(_ctx))
	{
		os::Printer::log("No closing brace in Transformation Matrix found in x file", ELL_WARNING);

		return false;
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectMesh(SContext& _ctx, SXMesh &mesh)
{
	std::string name;

	if (!readHeadOfDataObject(_ctx, &name))
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
	const uint32_t nVertices = readInt(_ctx);

	// read vertices
	mesh.Vertices.resize(nVertices);
	for (uint32_t n=0; n<nVertices; ++n)
	{
		readVector3(_ctx, mesh.Vertices[n].Pos);
	}

	if (!checkForTwoFollowingSemicolons(_ctx))
	{
		os::Printer::log("No finishing semicolon in Mesh Vertex Array found in x file", ELL_WARNING);

	}

	// read faces
	const uint32_t nFaces = readInt(_ctx);

	mesh.Indices.resize(nFaces * 3);
	mesh.IndexCountPerFace.resize(nFaces);

	core::vector<uint32_t> polygonfaces;
	uint32_t currentIndex = 0;

	for (uint32_t k=0; k<nFaces; ++k)
	{
		const uint32_t fcnt = readInt(_ctx);

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
				polygonfaces[f] = readInt(_ctx);

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
			mesh.Indices[currentIndex++] = readInt(_ctx);
			mesh.Indices[currentIndex++] = readInt(_ctx);
			mesh.Indices[currentIndex++] = readInt(_ctx);
			mesh.IndexCountPerFace[k] = 3;
		}
	}

	if (!checkForTwoFollowingSemicolons(_ctx))
	{
		os::Printer::log("No finishing semicolon in Mesh Face Array found in x file", ELL_WARNING);

	}

	// here, other data objects may follow

	while(true)
	{
		std::string objectName = getNextToken(_ctx);

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
			if (!parseDataObjectMeshNormals(_ctx, mesh))
				return false;
		}
		else
		if (objectName == "MeshTextureCoords")
		{
			if (!parseDataObjectMeshTextureCoords(_ctx, mesh))
				return false;
		}
		else
		if (objectName == "MeshVertexColors")
		{
			if (!parseDataObjectMeshVertexColors(_ctx, mesh))
				return false;
		}
		else
		if (objectName == "MeshMaterialList")
		{
			if (!parseDataObjectMeshMaterialList(_ctx, mesh))
				return false;
		}
		else
		if (objectName == "VertexDuplicationIndices")
		{
			// we'll ignore vertex duplication indices
			// TODO: read them
			if (!parseUnknownDataObject(_ctx))
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
			const uint32_t dcnt = readInt(_ctx);
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
				const uint32_t type = readInt(_ctx);
				//const uint32_t tesselator = readInt();
				readInt(_ctx);
				const uint32_t semantics = readInt(_ctx);
				const uint32_t index = readInt(_ctx);
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
			const uint32_t datasize = readInt(_ctx);
			uint32_t* data = new uint32_t[datasize];
			for (j=0; j<datasize; ++j)
				data[j]=readInt(_ctx);

			if (!checkForOneFollowingSemicolons(_ctx))
			{
				os::Printer::log("No finishing semicolon in DeclData found.", ELL_WARNING);

			}
			if (!checkForClosingBrace(_ctx))
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
			if (!readHeadOfDataObject(_ctx))
			{
				os::Printer::log("No starting brace in FVFData found.", ELL_WARNING);

				return false;
			}
			const uint32_t dataformat = readInt(_ctx);
			const uint32_t datasize = readInt(_ctx);
			uint32_t* data = new uint32_t[datasize];
			for (uint32_t j=0; j<datasize; ++j)
				data[j]=readInt(_ctx);
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
			if (!checkForOneFollowingSemicolons(_ctx))
			{
				os::Printer::log("No finishing semicolon in FVFData found.", ELL_WARNING);

			}
			if (!checkForClosingBrace(_ctx))
			{
				os::Printer::log("No closing brace in FVFData found in x file", ELL_WARNING);

				return false;
			}
		}
		else
		if (objectName == "XSkinMeshHeader")
		{
			if (!parseDataObjectSkinMeshHeader(_ctx, mesh))
				return false;
		}
		else
		if (objectName == "SkinWeights")
		{
			//mesh.SkinWeights.push_back(SXSkinWeight());
			//if (!parseDataObjectSkinWeights(mesh.SkinWeights.back()))
			if (!parseDataObjectSkinWeights(_ctx, mesh))
				return false;
		}
		else
		{
			os::Printer::log("Unknown data object in mesh in x file", objectName, ELL_WARNING);
			if (!parseUnknownDataObject(_ctx))
				return false;
		}
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectSkinWeights(SContext& _ctx, SXMesh &mesh)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: Reading mesh skin weights", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject(_ctx))
	{
		os::Printer::log("No opening brace in Skin Weights found in .x file", ELL_WARNING);

		return false;
	}

	std::string TransformNodeName;

	if (!getNextTokenAsString(_ctx, TransformNodeName))
	{
		os::Printer::log("Unknown syntax while reading transfrom node name string in .x file", ELL_WARNING);

		return false;
	}

    asset::ICPUSkinnedMesh::SJoint *joint=0;

	size_t jointID;
	for (jointID=0; jointID < _ctx.AnimatedMesh->getAllJoints().size(); jointID++)
	{
		if (_ctx.AnimatedMesh->getAllJoints()[jointID]->Name==TransformNodeName)
		{
			joint= _ctx.AnimatedMesh->getAllJoints()[jointID];
			break;
		}
	}

	if (!joint)
	{
#ifdef _XREADER_DEBUG
		os::Printer::log("creating joint for skinning ", TransformNodeName, ELL_DEBUG);
#endif
		jointID =_ctx.AnimatedMesh->getAllJoints().size();
		joint=_ctx.AnimatedMesh->addJoint(0);
		joint->Name=TransformNodeName;
	}

	// read vertex indices
	const uint32_t nWeights = readInt(_ctx);
	uint32_t* vertexIDs = new uint32_t[nWeights];
	uint32_t maxIx = 0;
	for (size_t i=0; i<nWeights; i++)
    {
		vertexIDs[i] = readInt(_ctx);
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
        float tmpWeight = readFloat(_ctx);
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
	readMatrix(_ctx, joint->GlobalInversedMatrix);

	if (!checkForOneFollowingSemicolons(_ctx))
	{
		os::Printer::log("No finishing semicolon in Skin Weights found in x file", ELL_WARNING);

	}

	if (!checkForClosingBrace(_ctx))
	{
		os::Printer::log("No closing brace in Skin Weights found in x file", ELL_WARNING);

		return false;
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectSkinMeshHeader(SContext& _ctx, SXMesh& mesh)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: Reading skin mesh header", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject(_ctx))
	{
		os::Printer::log("No opening brace in Skin Mesh header found in .x file", ELL_WARNING);

		return false;
	}

	//mesh.MaxSkinWeightsPerVertex = readInt();
	//mesh.MaxSkinWeightsPerFace = readInt();
	readInt(_ctx);readInt(_ctx);

	mesh.BoneCount = readInt(_ctx);

	if (!_ctx.BinaryFormat)
		getNextToken(_ctx); // skip semicolon

	if (!checkForClosingBrace(_ctx))
	{
		os::Printer::log("No closing brace in skin mesh header in x file", ELL_WARNING);

		return false;
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectMeshNormals(SContext& _ctx, SXMesh &mesh)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: reading mesh normals", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject(_ctx))
	{
		os::Printer::log("No opening brace in Mesh Normals found in x file", ELL_WARNING);

		return false;
	}

	// read count
	const uint32_t nNormals = readInt(_ctx);
	core::vector<core::vector3df> normals;
	normals.resize(nNormals);

	// read normals
	for (uint32_t i=0; i<nNormals; ++i)
		readVector3(_ctx, normals[i]);

	if (!checkForTwoFollowingSemicolons(_ctx))
	{
		os::Printer::log("No finishing semicolon in Mesh Normals Array found in x file", ELL_WARNING);

	}

	core::vector<uint32_t> normalIndices;
	normalIndices.resize(mesh.Indices.size());

	// read face normal indices
	const uint32_t nFNormals = readInt(_ctx);

	uint32_t normalidx = 0;
	core::vector<uint32_t> polygonfaces;
	for (uint32_t k=0; k<nFNormals; ++k)
	{
		const uint32_t fcnt = readInt(_ctx);
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
				const uint32_t normalnum = readInt(_ctx);
				mesh.Vertices[mesh.Indices[normalidx++]].Normal.set(normals[normalnum]);
			}
		}
		else
		{
			polygonfaces.resize(fcnt);
			// multiple triangles in this face
			for (uint32_t h=0; h<fcnt; ++h)
				polygonfaces[h] = readInt(_ctx);

			for (uint32_t jk=0; jk<triangles; ++jk)
			{
				mesh.Vertices[mesh.Indices[normalidx++]].Normal.set(normals[polygonfaces[0]]);
				mesh.Vertices[mesh.Indices[normalidx++]].Normal.set(normals[polygonfaces[jk+1]]);
				mesh.Vertices[mesh.Indices[normalidx++]].Normal.set(normals[polygonfaces[jk+2]]);
			}
		}
	}

	if (!checkForTwoFollowingSemicolons(_ctx))
	{
		os::Printer::log("No finishing semicolon in Mesh Face Normals Array found in x file", ELL_WARNING);

	}

	if (!checkForClosingBrace(_ctx))
	{
		os::Printer::log("No closing brace in Mesh Normals found in x file", ELL_WARNING);

		return false;
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectMeshTextureCoords(SContext& _ctx, SXMesh &mesh)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: reading mesh texture coordinates", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject(_ctx))
	{
		os::Printer::log("No opening brace in Mesh Texture Coordinates found in x file", ELL_WARNING);

		return false;
	}

	const uint32_t nCoords = readInt(_ctx);
	for (uint32_t i=0; i<nCoords; ++i)
		readVector2(_ctx, mesh.Vertices[i].TCoords);

	if (!checkForTwoFollowingSemicolons(_ctx))
	{
		os::Printer::log("No finishing semicolon in Mesh Texture Coordinates Array found in x file", ELL_WARNING);

	}

	if (!checkForClosingBrace(_ctx))
	{
		os::Printer::log("No closing brace in Mesh Texture Coordinates Array found in x file", ELL_WARNING);

		return false;
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectMeshVertexColors(SContext& _ctx, SXMesh &mesh)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: reading mesh vertex colors", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject(_ctx))
	{
		os::Printer::log("No opening brace for Mesh Vertex Colors found in x file", ELL_WARNING);

		return false;
	}

	mesh.HasVertexColors=true;
	const typename decltype(mesh.Colors)::size_type nColors = readInt(_ctx);
	mesh.Colors.resize(core::max_(mesh.Colors.size(),nColors));
	for (uint32_t i=0; i<nColors; ++i)
	{
		const uint32_t Index=readInt(_ctx);
		if (Index>=mesh.Vertices.size())
		{
			os::Printer::log("index value in parseDataObjectMeshVertexColors out of bounds", ELL_WARNING);

			return false;
		}
		video::SColor tmpCol;
		readRGBA(_ctx, tmpCol);
		mesh.Colors[Index] = tmpCol.color;
		checkForOneFollowingSemicolons(_ctx);
	}

	if (!checkForOneFollowingSemicolons(_ctx))
	{
		os::Printer::log("No finishing semicolon in Mesh Vertex Colors Array found in x file", ELL_WARNING);

	}

	if (!checkForClosingBrace(_ctx))
	{
		os::Printer::log("No closing brace in Mesh Texture Coordinates Array found in x file", ELL_WARNING);

		return false;
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectMeshMaterialList(SContext& _ctx, SXMesh &mesh)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: Reading mesh material list", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject(_ctx))
	{
		os::Printer::log("No opening brace in Mesh Material List found in x file", ELL_WARNING);

		return false;
	}

	// read material count
	mesh.Materials.reserve(readInt(_ctx));

	// read non triangulated face material index count
	const uint32_t nFaceIndices = readInt(_ctx);

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
			ind = readInt(_ctx);
		const uint32_t fc = mesh.IndexCountPerFace[tfi]/3;
		for (uint32_t k=0; k<fc; ++k)
			mesh.FaceMaterialIndices[triangulatedindex++] = ind;
	}

	// in version 03.02, the face indices end with two semicolons.
	// commented out version check, as version 03.03 exported from blender also has 2 semicolons
	if (!_ctx.BinaryFormat) // && MajorVersion == 3 && MinorVersion <= 2)
	{
		if (_ctx.fileContents.peek() == ';')
            _ctx.fileContents.seekg(1, _ctx.fileContents.cur);
	}

	// read following data objects

	while(true)
	{
		std::string objectName = getNextToken(_ctx);

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
			objectName = getNextToken(_ctx);
			for (uint32_t i=0; i< _ctx.TemplateMaterials.size(); ++i)
				if (_ctx.TemplateMaterials[i].Name == objectName)
					mesh.Materials.push_back(_ctx.TemplateMaterials[i].Material);
			getNextToken(_ctx); // skip }
		}
		else
		if (objectName == "Material")
		{
			mesh.Materials.push_back(video::SCPUMaterial());
			if (!parseDataObjectMaterial(_ctx, mesh.Materials.back()))
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
			if (!parseUnknownDataObject(_ctx))
				return false;
		}
	}
	return true;
}


bool CXMeshFileLoader::parseDataObjectMaterial(SContext& _ctx, video::SCPUMaterial& material)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: Reading mesh material", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject(_ctx))
	{
		os::Printer::log("No opening brace in Mesh Material found in .x file", ELL_WARNING);

		return false;
	}

	// read RGBA
	readRGBA(_ctx, material.DiffuseColor); checkForOneFollowingSemicolons(_ctx);

	// read power
	material.Shininess = readFloat(_ctx);

	// read specular
	readRGB(_ctx, material.SpecularColor); checkForOneFollowingSemicolons(_ctx);

	// read emissive
	readRGB(_ctx, material.EmissiveColor); checkForOneFollowingSemicolons(_ctx);

	// read other data objects
	int textureLayer=0;
	while(true)
	{
		core::stringc objectName = getNextToken(_ctx).c_str();

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
			if (!parseDataObjectTextureFilename(_ctx, tmp))
				return false;
			core::stringc TextureFileName = tmp.c_str();

			// original name
            if (FileSystem->existFile(TextureFileName))
            {
                asset::ICPUTexture* texture = static_cast<asset::ICPUTexture*>(
                    interm_getAssetInHierarchy(Device->getAssetManager(), TextureFileName.c_str(), _ctx.Inner.params, 2u, _ctx.loaderOverride)
                );
                material.setTexture(textureLayer, texture);
            }
			// mesh path
			else
			{
				TextureFileName= _ctx.FilePath + io::IFileSystem::getFileBasename(TextureFileName);
                if (FileSystem->existFile(TextureFileName))
                {
                    asset::ICPUTexture* texture = static_cast<asset::ICPUTexture*>(
                        interm_getAssetInHierarchy(Device->getAssetManager(), TextureFileName.c_str(), _ctx.Inner.params, 2u, _ctx.loaderOverride)
                    );
                    material.setTexture(textureLayer, texture);
                }
				// working directory
                else
                {
                    asset::ICPUTexture* texture = static_cast<asset::ICPUTexture*>(
                        interm_getAssetInHierarchy(Device->getAssetManager(), io::IFileSystem::getFileBasename(TextureFileName).c_str(), _ctx.Inner.params, 2u, _ctx.loaderOverride)
                    );
                    material.setTexture(textureLayer, texture);
                }
			}
			++textureLayer;
		}
		else
		if (objectName.equals_ignore_case("NormalmapFilename"))
		{
			// some exporters write "NormalmapFileName" instead.
			std::string tmp;
			if (!parseDataObjectTextureFilename(_ctx, tmp))
				return false;
			core::stringc TextureFileName = tmp.c_str();

			// original name
            if (FileSystem->existFile(TextureFileName))
            {
                asset::ICPUTexture* texture = static_cast<asset::ICPUTexture*>(
                    interm_getAssetInHierarchy(Device->getAssetManager(), TextureFileName.c_str(), _ctx.Inner.params, 2u, _ctx.loaderOverride)
                );
                material.setTexture(1, texture);
            }
			// mesh path
			else
			{
				TextureFileName= _ctx.FilePath + io::IFileSystem::getFileBasename(TextureFileName);
                if (FileSystem->existFile(TextureFileName))
                {
                    asset::ICPUTexture* texture = static_cast<asset::ICPUTexture*>(
                        interm_getAssetInHierarchy(Device->getAssetManager(), TextureFileName.c_str(), _ctx.Inner.params, 2u, _ctx.loaderOverride)
                    );
                    material.setTexture(1, texture);
                }
				// working directory
                else
                {
                    asset::ICPUTexture* texture = static_cast<asset::ICPUTexture*>(
                        interm_getAssetInHierarchy(Device->getAssetManager(), io::IFileSystem::getFileBasename(TextureFileName).c_str(), _ctx.Inner.params, 2u, _ctx.loaderOverride)
                    );
                    material.setTexture(1, texture);
                }
			}
			if (textureLayer==1)
				++textureLayer;
		}
		else
		{
			os::Printer::log("Unknown data object in material in .x file", objectName.c_str(), ELL_WARNING);
			if (!parseUnknownDataObject(_ctx))
				return false;
		}
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectAnimationSet(SContext& _ctx)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: Reading animation set", ELL_DEBUG);
#endif

	std::string AnimationName;

	if (!readHeadOfDataObject(_ctx, &AnimationName))
	{
		os::Printer::log("No opening brace in Animation Set found in x file", ELL_WARNING);

		return false;
	}
	os::Printer::log("Reading animationset ", AnimationName, ELL_DEBUG);

	while(true)
	{
		std::string objectName = getNextToken(_ctx);

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
			if (!parseDataObjectAnimation(_ctx))
				return false;
		}
		else
		{
			os::Printer::log("Unknown data object in animation set in x file", objectName, ELL_WARNING);
			if (!parseUnknownDataObject(_ctx))
				return false;
		}
	}
	return true;
}


bool CXMeshFileLoader::parseDataObjectAnimation(SContext& _ctx)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: reading animation", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject(_ctx))
	{
		os::Printer::log("No opening brace in Animation found in x file", ELL_WARNING);

		return false;
	}

	//anim.closed = true;
	//anim.linearPositionQuality = true;
    asset::ICPUSkinnedMesh::SJoint animationDump;

	std::string FrameName;

	while(true)
	{
		std::string objectName = getNextToken(_ctx);

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
			if (!parseDataObjectAnimationKey(_ctx, &animationDump))
				return false;
		}
		else
		if (objectName == "AnimationOptions")
		{
			//TODO: parse options.
			if (!parseUnknownDataObject(_ctx))
				return false;
		}
		else
		if (objectName == "{")
		{
			// read frame name
			FrameName = getNextToken(_ctx);

			if (!checkForClosingBrace(_ctx))
			{
				os::Printer::log("Unexpected ending found in Animation in x file.", ELL_WARNING);

				return false;
			}
		}
		else
		{
			os::Printer::log("Unknown data object in animation in x file", objectName, ELL_WARNING);
			if (!parseUnknownDataObject(_ctx))
				return false;
		}
	}

	if (FrameName.size() != 0)
	{
#ifdef _XREADER_DEBUG
		os::Printer::log("frame name", FrameName, ELL_DEBUG);
#endif
        asset::ICPUSkinnedMesh::SJoint *joint=0;

		uint32_t n;
		for (n=0; n < _ctx.AnimatedMesh->getAllJoints().size(); ++n)
		{
			if (_ctx.AnimatedMesh->getAllJoints()[n]->Name==FrameName)
			{
				joint= _ctx.AnimatedMesh->getAllJoints()[n];
				break;
			}
		}

		if (!joint)
		{
#ifdef _XREADER_DEBUG
			os::Printer::log("creating joint for animation ", FrameName, ELL_DEBUG);
#endif
			joint= _ctx.AnimatedMesh->addJoint(0);
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


bool CXMeshFileLoader::parseDataObjectAnimationKey(SContext& _ctx, asset::ICPUSkinnedMesh::SJoint *joint)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: reading animation key", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject(_ctx))
	{
		os::Printer::log("No opening brace in Animation Key found in x file", ELL_WARNING);

		return false;
	}

	// read key type

	const uint32_t keyType = readInt(_ctx);

	if (keyType > 4)
	{
		os::Printer::log("Unknown key type found in Animation Key in x file", ELL_WARNING);

		return false;
	}

	// read number of keys
	const uint32_t numberOfKeys = readInt(_ctx);

	// eat the semicolon after the "0".  if there are keys present, readInt()
	// does this for us.  If there aren't, we need to do it explicitly
	if (numberOfKeys == 0)
		checkForOneFollowingSemicolons(_ctx);

	for (uint32_t i=0; i<numberOfKeys; ++i)
	{
		// read time
		const float time = (float)readInt(_ctx);

		// read keys
		switch(keyType)
		{
		case 0: //rotation
			{
				//read quaternions

				// read count
				if (readInt(_ctx) != 4)
				{
					os::Printer::log("Expected 4 numbers in animation key in x file", ELL_WARNING);

					return false;
				}

                core::vectorSIMDf quatern;
				quatern.W = -readFloat(_ctx);
				quatern.X = readFloat(_ctx);
				quatern.Y = readFloat(_ctx);
				quatern.Z = readFloat(_ctx);

                quatern = normalize(quatern);

				if (!checkForTwoFollowingSemicolons(_ctx))
				{
					os::Printer::log("No finishing semicolon after quaternion animation key in x file", ELL_WARNING);

				}

                asset::ICPUSkinnedMesh::SRotationKey *key=joint->addRotationKey();
				key->frame=time;
				key->rotation.set(quatern);
			}
			break;
		case 1: //scale
		case 2: //position
			{
				// read vectors

				// read count
				if (readInt(_ctx) != 3)
				{
					os::Printer::log("Expected 3 numbers in animation key in x file", ELL_WARNING);

					return false;
				}

				core::vector3df vector;
				readVector3(_ctx, vector);

				if (!checkForTwoFollowingSemicolons(_ctx))
				{
					os::Printer::log("No finishing semicolon after vector animation key in x file", ELL_WARNING);

				}

				if (keyType==2)
				{
                    asset::ICPUSkinnedMesh::SPositionKey *key=joint->addPositionKey();
					key->frame=time;
					key->position=vector;
				}
				else
				{
                    asset::ICPUSkinnedMesh::SScaleKey *key=joint->addScaleKey();
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
				if (readInt(_ctx) != 16)
				{
					os::Printer::log("Expected 16 numbers in animation key in x file", ELL_WARNING);

					return false;
				}

				// read matrix
				core::matrix4x3 mat4x3;
				readMatrix(_ctx, mat4x3);
				//mat=joint->LocalMatrix*mat;

				if (!checkForOneFollowingSemicolons(_ctx))
				{
					os::Printer::log("No finishing semicolon after matrix animation key in x file", ELL_WARNING);

				}

				//core::vector3df rotation = mat.getRotationDegrees();

                asset::ICPUSkinnedMesh::SRotationKey *keyR=joint->addRotationKey();
				keyR->frame=time;

				keyR->rotation = core::quaternion(mat4x3);

                asset::ICPUSkinnedMesh::SPositionKey *keyP=joint->addPositionKey();
				keyP->frame=time;
				keyP->position=mat4x3.getTranslation();


				core::vector3df scale=mat4x3.getScale();

				if (scale.X==0)
					scale.X=1;
				if (scale.Y==0)
					scale.Y=1;
				if (scale.Z==0)
					scale.Z=1;
                asset::ICPUSkinnedMesh::SScaleKey *keyS=joint->addScaleKey();
				keyS->frame=time;
				keyS->scale=scale;
			}
			break;
		} // end switch
	}

	if (!checkForOneFollowingSemicolons(_ctx))
		_ctx.fileContents.unget();

	if (!checkForClosingBrace(_ctx))
	{
		os::Printer::log("No closing brace in animation key in x file", ELL_WARNING);

		return false;
	}

	return true;
}


bool CXMeshFileLoader::parseDataObjectTextureFilename(SContext& _ctx, std::string& texturename)
{
#ifdef _XREADER_DEBUG
	os::Printer::log("CXFileReader: reading texture filename", ELL_DEBUG);
#endif

	if (!readHeadOfDataObject(_ctx))
	{
		os::Printer::log("No opening brace in Texture filename found in x file", ELL_WARNING);

		return false;
	}

	if (!getNextTokenAsString(_ctx, texturename))
	{
		os::Printer::log("Unknown syntax while reading texture filename string in x file", ELL_WARNING);

		return false;
	}

	if (!checkForClosingBrace(_ctx))
	{
		os::Printer::log("No closing brace in Texture filename found in x file", ELL_WARNING);

		return false;
	}

	return true;
}


bool CXMeshFileLoader::parseUnknownDataObject(SContext& _ctx)
{
	// find opening delimiter
	while(true)
	{
		std::string t = getNextToken(_ctx);

		if (t.size() == 0)
			return false;

		if (t == "{")
			break;
	}

	uint32_t counter = 1;

	// parse until closing delimiter

	while(counter)
	{
		std::string t = getNextToken(_ctx);

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
bool CXMeshFileLoader::checkForClosingBrace(SContext& _ctx)
{
	return (getNextToken(_ctx) == "}");
}


//! checks for one following semicolon, returns false if not there
bool CXMeshFileLoader::checkForOneFollowingSemicolons(SContext& _ctx)
{
	if (_ctx.BinaryFormat)
		return true;

	if (getNextToken(_ctx) == ";")
		return true;
	else
	{
        _ctx.fileContents.unget();
		return false;
	}
}


//! checks for two following semicolons, returns false if they are not there
bool CXMeshFileLoader::checkForTwoFollowingSemicolons(SContext& _ctx)
{
	if (_ctx.BinaryFormat)
		return true;

	for (uint32_t k=0; k<2; ++k)
	{
		if (getNextToken(_ctx) != ";")
		{
            _ctx.fileContents.unget();
			return false;
		}
	}

	return true;
}


//! reads header of dataobject including the opening brace.
//! returns false if error happened, and writes name of object
//! if there is one
bool CXMeshFileLoader::readHeadOfDataObject(SContext& _ctx, std::string* outname)
{
	std::string nameOrBrace = getNextToken(_ctx);
	if (nameOrBrace != "{")
	{
		if (outname)
			(*outname) = nameOrBrace;

		if (getNextToken(_ctx) != "{")
			return false;
	}

	return true;
}


//! returns next parseable token. Returns empty string if no token there
std::string CXMeshFileLoader::getNextToken(SContext& _ctx)
{
	std::string s;

	// process binary-formatted file
	if (_ctx.BinaryFormat)
	{
		// in binary mode it will only return NAME and STRING token
		// and (correctly) skip over other tokens.

		int16_t tok = readBinWord(_ctx);
		uint32_t len;

		// standalone tokens
		switch (tok) {
			case 1:
				// name token
				len = readBinDWord(_ctx);
				s.resize(len);
                _ctx.fileContents.get(&s[0],len+1);
				return s;
			case 2:
				// string token
				len = readBinDWord(_ctx);
				s.resize(len);
                _ctx.fileContents.get(&s[0],len+1);
                _ctx.fileContents.seekg(2, _ctx.fileContents.cur);
				return s;
			case 3:
				// integer token
                _ctx.fileContents.seekg(4, _ctx.fileContents.cur);
				return "<integer>";
			case 5:
				// GUID token
                _ctx.fileContents.seekg(16, _ctx.fileContents.cur);
				return "<guid>";
			case 6:
				len = readBinDWord(_ctx);
                _ctx.fileContents.seekg(4*len, _ctx.fileContents.cur);
				return "<int_list>";
			case 7:
				len = readBinDWord(_ctx);
                _ctx.fileContents.seekg(_ctx.FloatSize*len, _ctx.fileContents.cur);
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
		findNextNoneWhiteSpace(_ctx);

		if (_ctx.fileContents.eof())
			return s;

		while(!_ctx.fileContents.eof() && !core::isspace(_ctx.fileContents.peek()))
		{
			// either keep token delimiters when already holding a token, or return if first valid char
			if (_ctx.fileContents.peek()==';' || _ctx.fileContents.peek()=='}' || _ctx.fileContents.peek()=='{' || _ctx.fileContents.peek()==',')
			{
				if (!s.size())
					s.push_back(_ctx.fileContents.get());

				break; // stop for delimiter
			}
			s.push_back(_ctx.fileContents.get());
		}
	}
	return s;
}


//! places pointer to next begin of a token, which must be a number,
// and ignores comments
void CXMeshFileLoader::findNextNoneWhiteSpaceNumber(SContext& _ctx)
{
	if (_ctx.BinaryFormat)
		return;

	while(!_ctx.fileContents.eof())
    {
        char p;
	    _ctx.fileContents >> std::ws >> p;
        if (p == '-' || p == '.' || core::isdigit(p))
        {
            _ctx.fileContents.unget();
            break;
        }

		// check if this is a comment
		if ((p == '/' && _ctx.fileContents.peek() == '/') || p == '#')
        {
			std::string stmp;
			std::getline(_ctx.fileContents,stmp);
        }
	}
}


// places pointer to next begin of a token, and ignores comments
void CXMeshFileLoader::findNextNoneWhiteSpace(SContext& _ctx)
{
	if (_ctx.BinaryFormat)
		return;

	while(true)
	{
	    _ctx.fileContents >> std::ws;

		if (_ctx.fileContents.eof())
			return;

		// check if this is a comment
        char p = _ctx.fileContents.get();
		if ((p == '/' && _ctx.fileContents.peek() == '/') || p == '#')
        {
			std::string stmp;
			std::getline(_ctx.fileContents,stmp);
        }
		else
        {
            _ctx.fileContents.unget();
			break;
        }
	}
}

//! reads a x file style string
bool CXMeshFileLoader::getNextTokenAsString(SContext& _ctx, std::string& out)
{
	if (_ctx.BinaryFormat)
	{
		out=getNextToken(_ctx);
		return true;
	}
	findNextNoneWhiteSpace(_ctx);

	if (_ctx.fileContents.eof())
		return false;

	if (_ctx.fileContents.get() != '"')
    {
        _ctx.fileContents.unget();
		return false;
    }

	while(!_ctx.fileContents.eof() && _ctx.fileContents.peek()!='"')
	{
		out.push_back(_ctx.fileContents.get());
	}

	char P[2];
	_ctx.fileContents.read(P,2);
	if ( P[1] != ';' || P[0] != '"')
    {
        _ctx.fileContents.unget();
        _ctx.fileContents.unget();
		return false;
    }

	return true;
}


uint16_t CXMeshFileLoader::readBinWord(SContext& _ctx)
{
	if (_ctx.fileContents.eof())
		return 0;

    char P[2];
    _ctx.fileContents.read(P,2);

    return *(uint16_t *)P;
}


uint32_t CXMeshFileLoader::readBinDWord(SContext& _ctx)
{
	if (_ctx.fileContents.eof())
		return 0;

    char P[4];
    _ctx.fileContents.read(P,4);

	return *(uint32_t *)P;
}


uint32_t CXMeshFileLoader::readInt(SContext& _ctx)
{
	if (_ctx.BinaryFormat)
	{
		if (!_ctx.BinaryNumCount)
		{
			const uint16_t tmp = readBinWord(_ctx); // 0x06 or 0x03
			if (tmp == 0x06)
                _ctx.BinaryNumCount = readBinDWord(_ctx);
			else
                _ctx.BinaryNumCount = 1; // single int
		}
		--_ctx.BinaryNumCount;
		return readBinDWord(_ctx);
	}
	else
	{
		findNextNoneWhiteSpaceNumber(_ctx);

	    uint32_t retval;
	    _ctx.fileContents >> retval;
	    return retval;
	}
}


float CXMeshFileLoader::readFloat(SContext& _ctx)
{
	if (_ctx.BinaryFormat)
	{
		if (!_ctx.BinaryNumCount)
		{
			const uint16_t tmp = readBinWord(_ctx); // 0x07 or 0x42
			if (tmp == 0x07)
                _ctx.BinaryNumCount = readBinDWord(_ctx);
			else
                _ctx.BinaryNumCount = 1; // single int
		}
		--_ctx.BinaryNumCount;
		if (_ctx.FloatSize == 8)
		{
		    double tmp;
		    _ctx.fileContents.read(reinterpret_cast<char*>(&tmp),8);
			return tmp;
		}
		else
		{
		    float tmp;
		    _ctx.fileContents.read(reinterpret_cast<char*>(&tmp),4);
			return tmp;
		}
	}
	findNextNoneWhiteSpaceNumber(_ctx);
	float ftmp;
	_ctx.fileContents >> ftmp;
	return ftmp;
}


// read 2-dimensional vector. Stops at semicolon after second value for text file format
bool CXMeshFileLoader::readVector2(SContext& _ctx, core::vector2df& vec)
{
	vec.X = readFloat(_ctx);
	vec.Y = readFloat(_ctx);
	return true;
}


// read 3-dimensional vector. Stops at semicolon after third value for text file format
bool CXMeshFileLoader::readVector3(SContext& _ctx, core::vector3df& vec)
{
	vec.X = readFloat(_ctx);
	vec.Y = readFloat(_ctx);
	vec.Z = readFloat(_ctx);
	return true;
}


// read color without alpha value. Stops after second semicolon after blue value
bool CXMeshFileLoader::readRGB(SContext& _ctx, video::SColor& color)
{
	video::SColorf tmpColor;
	tmpColor.getAsVectorSIMDf().r = readFloat(_ctx);
	tmpColor.getAsVectorSIMDf().g = readFloat(_ctx);
	tmpColor.getAsVectorSIMDf().b = readFloat(_ctx);
	color = tmpColor.toSColor();
	return checkForOneFollowingSemicolons(_ctx);
}


// read color with alpha value. Stops after second semicolon after blue value
bool CXMeshFileLoader::readRGBA(SContext& _ctx, video::SColor& color)
{
	video::SColorf tmpColor;
	tmpColor.getAsVectorSIMDf().r = readFloat(_ctx);
	tmpColor.getAsVectorSIMDf().g = readFloat(_ctx);
	tmpColor.getAsVectorSIMDf().b = readFloat(_ctx);
	tmpColor.getAsVectorSIMDf().a = readFloat(_ctx);
	color = tmpColor.toSColor();
	return checkForOneFollowingSemicolons(_ctx);
}


// read matrix from list of floats
bool CXMeshFileLoader::readMatrix(SContext& _ctx, core::matrix4x3& mat)
{
    for (uint32_t j=0u; j<4u; j++)
    {
        for (uint32_t i=0u; i<3u; i++)
            mat(i,j) = readFloat(_ctx);
        readFloat(_ctx);
    }
	return checkForOneFollowingSemicolons(_ctx);
}


} // end namespace asset
} // end namespace irr

#endif // _IRR_COMPILE_WITH_X_LOADER_

