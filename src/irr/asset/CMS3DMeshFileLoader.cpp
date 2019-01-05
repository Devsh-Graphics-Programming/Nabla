// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"
#ifdef _IRR_COMPILE_WITH_MS3D_LOADER_

#include "IReadFile.h"
#include "os.h"
#include "CMS3DMeshFileLoader.h"
#include "irr/video/CGPUSkinnedMesh.h"


namespace irr
{
namespace scene
{

#ifdef _DEBUG
#define _IRR_DEBUG_MS3D_LOADER_
#endif

// byte-align structures
#include "irr/irrpack.h"

namespace {
// File header
struct MS3DHeader
{
	char ID[10];
	int Version;
} PACK_STRUCT;

// Vertex information
struct MS3DVertex
{
	uint8_t Flags;
	float Vertex[3];
	char BoneID;
	uint8_t RefCount;
} PACK_STRUCT;

// Triangle information
struct MS3DTriangle
{
	uint16_t Flags;
	uint16_t VertexIndices[3];
	float VertexNormals[3][3];
	float S[3], T[3];
	uint8_t SmoothingGroup;
	uint8_t GroupIndex;
} PACK_STRUCT;

// Material information
struct MS3DMaterial
{
    char Name[32];
    float Ambient[4];
    float Diffuse[4];
    float Specular[4];
    float Emissive[4];
    float Shininess;	// 0.0f - 128.0f
    float Transparency;	// 0.0f - 1.0f
    uint8_t Mode;	// 0, 1, 2 is unused now
    char Texture[128];
    char Alphamap[128];
} PACK_STRUCT;

// Joint information
struct MS3DJoint
{
	uint8_t Flags;
	char Name[32];
	char ParentName[32];
	float Rotation[3];
	float Translation[3];
	uint16_t NumRotationKeyframes;
	uint16_t NumTranslationKeyframes;
} PACK_STRUCT;

// Keyframe data
struct MS3DKeyframe
{
	float Time;
	float Parameter[3];
} PACK_STRUCT;

// vertex weights in 1.8.x
struct MS3DVertexWeights
{
	char boneIds[3];
	uint8_t weights[3];
} PACK_STRUCT;

} // end namespace

// Default alignment
#include "irr/irrunpack.h"

struct SGroup
{
	core::stringc Name;
	core::array<uint16_t> VertexIds;
	uint16_t MaterialIdx;
};

//! Constructor
CMS3DMeshFileLoader::CMS3DMeshFileLoader(video::IVideoDriver *driver)
: Driver(driver), AnimatedMesh(0)
{
	#ifdef _DEBUG
	setDebugName("CMS3DMeshFileLoader");
	#endif
}


//! returns true if the file maybe is able to be loaded by this class
//! based on the file extension (e.g. ".bsp")
bool CMS3DMeshFileLoader::isALoadableFileExtension(const io::path& filename) const
{
	return core::hasFileExtension ( filename, "ms3d" );
}


//! creates/loads an animated mesh from the file.
//! \return Pointer to the created mesh. Returns 0 if loading failed.
//! If you no longer need the mesh, you should call IAnimatedMesh::drop().
//! See IReferenceCounted::drop() for more information.
IAnimatedMesh* CMS3DMeshFileLoader::createMesh(io::IReadFile* file)
{
	if (!file)
		return 0;

	AnimatedMesh = new CSkinnedMesh();

	if ( load(file) )
	{
		AnimatedMesh->finalize();
	}
	else
	{
		AnimatedMesh->drop();
		AnimatedMesh = 0;
	}

	return AnimatedMesh;
}


//! loads a milkshape file
bool CMS3DMeshFileLoader::load(io::IReadFile* file)
{
	if (!file)
		return false;

	// find file size
	const long fileSize = file->getSize();

	// read whole file

	uint8_t* buffer = new uint8_t[fileSize];
	int32_t read = file->read(buffer, fileSize);
	if (read != fileSize)
	{
		delete [] buffer;
		os::Printer::log("Could not read full file. Loading failed", file->getFileName().c_str(), ELL_ERROR);
		return false;
	}

	// read header

	const uint8_t *pPtr = (uint8_t*)((void*)buffer);
	MS3DHeader *pHeader = (MS3DHeader*)pPtr;
	pPtr += sizeof(MS3DHeader);

	if ( strncmp( pHeader->ID, "MS3D000000", 10 ) != 0 )
	{
		delete [] buffer;
		os::Printer::log("Not a valid Milkshape3D Model File. Loading failed", file->getFileName().c_str(), ELL_ERROR);
		return false;
	}

	if ( pHeader->Version < 3 || pHeader->Version > 4 )
	{
		delete [] buffer;
		os::Printer::log("Only Milkshape3D version 3 and 4 (1.3 to 1.8) is supported. Loading failed", file->getFileName().c_str(), ELL_ERROR);
		return false;
	}
#ifdef _IRR_DEBUG_MS3D_LOADER_
	os::Printer::log("Loaded header version", core::stringc(pHeader->Version).c_str());
#endif

	// get pointers to data

	// vertices
	uint16_t numVertices = *(uint16_t*)pPtr;
#ifdef _IRR_DEBUG_MS3D_LOADER_
	os::Printer::log("Load vertices", core::stringc(numVertices).c_str());
#endif
	pPtr += sizeof(uint16_t);
	MS3DVertex *vertices = (MS3DVertex*)pPtr;
	pPtr += sizeof(MS3DVertex) * numVertices;
	if (pPtr > buffer+fileSize)
	{
		delete [] buffer;
		os::Printer::log("Loading failed. Corrupted data found.", file->getFileName().c_str(), ELL_ERROR);
		return false;
	}
	for (uint16_t tmp=0; tmp<numVertices; ++tmp)
	{
		vertices[tmp].Vertex[2] = -vertices[tmp].Vertex[2];
	}

	// triangles
	uint16_t numTriangles = *(uint16_t*)pPtr;
#ifdef _IRR_DEBUG_MS3D_LOADER_
	os::Printer::log("Load Triangles", core::stringc(numTriangles).c_str());
#endif
	pPtr += sizeof(uint16_t);
	MS3DTriangle *triangles = (MS3DTriangle*)pPtr;
	pPtr += sizeof(MS3DTriangle) * numTriangles;
	if (pPtr > buffer+fileSize)
	{
		delete [] buffer;
		os::Printer::log("Loading failed. Corrupted data found.", file->getFileName().c_str(), ELL_ERROR);
		return false;
	}
	for (uint16_t tmp=0; tmp<numTriangles; ++tmp)
	{
		triangles[tmp].VertexNormals[0][2] = -triangles[tmp].VertexNormals[0][2];
		triangles[tmp].VertexNormals[1][2] = -triangles[tmp].VertexNormals[1][2];
		triangles[tmp].VertexNormals[2][2] = -triangles[tmp].VertexNormals[2][2];
	}

	// groups
	uint16_t numGroups = *(uint16_t*)pPtr;
#ifdef _IRR_DEBUG_MS3D_LOADER_
	os::Printer::log("Load Groups", core::stringc(numGroups).c_str());
#endif
	pPtr += sizeof(uint16_t);

	core::array<SGroup> groups;
	groups.reallocate(numGroups);

	//store groups
	uint32_t i;
	for (i=0; i<numGroups; ++i)
	{
		groups.push_back(SGroup());
		SGroup& grp = groups.back();

		// The byte flag is before the name, so add 1
		grp.Name = ((const char*) pPtr) + 1;

		pPtr += 33; // name and 1 byte flags
		uint16_t triangleCount = *(uint16_t*)pPtr;
		pPtr += sizeof(uint16_t);
		grp.VertexIds.reallocate(triangleCount);

		//pPtr += sizeof(uint16_t) * triangleCount; // triangle indices
		for (uint16_t j=0; j<triangleCount; ++j)
		{
			grp.VertexIds.push_back(*(uint16_t*)pPtr);
			pPtr += sizeof (uint16_t);
		}

		grp.MaterialIdx = *(uint8_t*)pPtr;
		if (grp.MaterialIdx == 255)
			grp.MaterialIdx = 0;

		pPtr += sizeof(char); // material index
		if (pPtr > buffer+fileSize)
		{
			delete [] buffer;
			os::Printer::log("Loading failed. Corrupted data found.", file->getFileName().c_str(), ELL_ERROR);
			return false;
		}
	}

	// load materials
	uint16_t numMaterials = *(uint16_t*)pPtr;
#ifdef _IRR_DEBUG_MS3D_LOADER_
	os::Printer::log("Load Materials", core::stringc(numMaterials).c_str());
#endif
	pPtr += sizeof(uint16_t);

	if(numMaterials == 0)
	{
		// if there are no materials, add at least one buffer
		AnimatedMesh->addMeshBuffer();
	}

	for (i=0; i<numMaterials; ++i)
	{
		MS3DMaterial *material = (MS3DMaterial*)pPtr;

		pPtr += sizeof(MS3DMaterial);
		if (pPtr > buffer+fileSize)
		{
			delete [] buffer;
			os::Printer::log("Loading failed. Corrupted data found.", file->getFileName().c_str(), ELL_ERROR);
			return false;
		}

		asset::SSkinMeshBuffer *tmpBuffer = AnimatedMesh->addMeshBuffer();

		tmpBuffer->Material.MaterialType = video::EMT_SOLID;

		tmpBuffer->Material.AmbientColor = video::SColorf(material->Ambient[0], material->Ambient[1], material->Ambient[2], material->Ambient[3]).toSColor ();
		tmpBuffer->Material.DiffuseColor = video::SColorf(material->Diffuse[0], material->Diffuse[1], material->Diffuse[2], material->Diffuse[3]).toSColor ();
		tmpBuffer->Material.EmissiveColor = video::SColorf(material->Emissive[0], material->Emissive[1], material->Emissive[2], material->Emissive[3]).toSColor ();
		tmpBuffer->Material.SpecularColor = video::SColorf(material->Specular[0], material->Specular[1], material->Specular[2], material->Specular[3]).toSColor ();
		tmpBuffer->Material.Shininess = material->Shininess;

		core::stringc TexturePath(material->Texture);
		if (TexturePath.trim()!="")
		{
			TexturePath=stripPathFromString(file->getFileName(),true) + stripPathFromString(TexturePath,false);
			tmpBuffer->Material.setTexture(0, Driver->getTexture(TexturePath));
		}

		core::stringc AlphamapPath=(const char*)material->Alphamap;
		if (AlphamapPath.trim()!="")
		{
			AlphamapPath=stripPathFromString(file->getFileName(),true) + stripPathFromString(AlphamapPath,false);
			tmpBuffer->Material.setTexture(2, Driver->getTexture(AlphamapPath));
		}
	}

	// animation time
	float framesPerSecond = *(float*)pPtr;
#ifdef _IRR_DEBUG_MS3D_LOADER_
	os::Printer::log("FPS", core::stringc(framesPerSecond).c_str());
#endif
	pPtr += sizeof(float) * 2; // fps and current time

	if (framesPerSecond<1.f)
		framesPerSecond=1.f;
	AnimatedMesh->setAnimationSpeed(framesPerSecond);

// ignore, calculated inside SkinnedMesh
//	int32_t frameCount = *(int*)pPtr;
	pPtr += sizeof(int);

	uint16_t jointCount = *(uint16_t*)pPtr;

#ifdef _IRR_DEBUG_MS3D_LOADER_
	os::Printer::log("Joints", core::stringc(jointCount).c_str());
#endif
	pPtr += sizeof(uint16_t);
	if (pPtr > buffer+fileSize)
	{
		delete [] buffer;
		os::Printer::log("Loading failed. Corrupted data found.", file->getFileName().c_str(), ELL_ERROR);
		return false;
	}

	core::array<core::stringc> parentNames;
	parentNames.reallocate(jointCount);

	// load joints
	for (i=0; i<jointCount; ++i)
	{
		uint32_t j;
		MS3DJoint *pJoint = (MS3DJoint*)pPtr;

		pPtr += sizeof(MS3DJoint);
		if (pPtr > buffer+fileSize)
		{
			delete [] buffer;
			os::Printer::log("Loading failed. Corrupted data found.", file->getFileName().c_str(), ELL_ERROR);
			return false;
		}

		ISkinnedMesh::SJoint *jnt = AnimatedMesh->addJoint();

		jnt->Name = pJoint->Name;
#ifdef _IRR_DEBUG_MS3D_LOADER_
		os::Printer::log("Joint", jnt->Name.c_str());
		os::Printer::log("Rotation keyframes", core::stringc(pJoint->NumRotationKeyframes).c_str());
		os::Printer::log("Translation keyframes", core::stringc(pJoint->NumTranslationKeyframes).c_str());
#endif
		jnt->LocalMatrix.makeIdentity();
		jnt->LocalMatrix.setRotationRadians(
			core::vector3df(pJoint->Rotation[0], pJoint->Rotation[1], pJoint->Rotation[2]) );
		// convert right-handed to left-handed
		jnt->LocalMatrix[2]=-jnt->LocalMatrix[2];
		jnt->LocalMatrix[6]=-jnt->LocalMatrix[6];
		jnt->LocalMatrix[8]=-jnt->LocalMatrix[8];
		jnt->LocalMatrix[9]=-jnt->LocalMatrix[9];

		jnt->LocalMatrix.setTranslation(
			core::vector3df(pJoint->Translation[0], pJoint->Translation[1], -pJoint->Translation[2]) );
		jnt->Animatedposition.set(jnt->LocalMatrix.getTranslation());
		jnt->Animatedrotation.set(jnt->LocalMatrix.getRotationDegrees());

		parentNames.push_back( (char*)pJoint->ParentName );

		/*if (pJoint->NumRotationKeyframes ||
			pJoint->NumTranslationKeyframes)
			HasAnimation = true;
		 */

		// get rotation keyframes
		const uint16_t numRotationKeyframes = pJoint->NumRotationKeyframes;
		for (j=0; j < numRotationKeyframes; ++j)
		{
			MS3DKeyframe* kf = (MS3DKeyframe*)pPtr;

			pPtr += sizeof(MS3DKeyframe);
			if (pPtr > buffer+fileSize)
			{
				delete [] buffer;
				os::Printer::log("Loading failed. Corrupted data found.", file->getFileName().c_str(), ELL_ERROR);
				return false;
			}

			ISkinnedMesh::SRotationKey *k=AnimatedMesh->addRotationKey(jnt);
			k->frame = kf->Time * framesPerSecond-1;

			core::matrix4 tmpMatrix;

			tmpMatrix.setRotationRadians(
				core::vector3df(kf->Parameter[0], kf->Parameter[1], kf->Parameter[2]) );
			// convert right-handed to left-handed
			tmpMatrix[2]=-tmpMatrix[2];
			tmpMatrix[6]=-tmpMatrix[6];
			tmpMatrix[8]=-tmpMatrix[8];
			tmpMatrix[9]=-tmpMatrix[9];

			tmpMatrix=jnt->LocalMatrix*tmpMatrix;

			// IRR_TEST_BROKEN_QUATERNION_USE: TODO - switched from tmpMatrix to tmpMatrix.getTransposed() for downward compatibility.
			//								   Not tested so far if this was correct or wrong before quaternion fix!
			k->rotation  = core::quaternion(tmpMatrix.getTransposed());
_
#error "Fix QUATERNIONS FIRST!!!"
		}

		// get translation keyframes
		const uint16_t numTranslationKeyframes = pJoint->NumTranslationKeyframes;
		for (j=0; j<numTranslationKeyframes; ++j)
		{
			MS3DKeyframe* kf = (MS3DKeyframe*)pPtr;
#ifdef __BIG_ENDIAN__
			kf->Time = os::Byteswap::byteswap(kf->Time);
			for (uint32_t l=0; l<3; ++l)
				kf->Parameter[l] = os::Byteswap::byteswap(kf->Parameter[l]);
#endif
			pPtr += sizeof(MS3DKeyframe);
			if (pPtr > buffer+fileSize)
			{
				delete [] buffer;
				os::Printer::log("Loading failed. Corrupted data found.", file->getFileName().c_str(), ELL_ERROR);
				return false;
			}

			ISkinnedMesh::SPositionKey *k=AnimatedMesh->addPositionKey(jnt);
			k->frame = kf->Time * framesPerSecond-1;

			k->position = core::vector3df
				(kf->Parameter[0]+pJoint->Translation[0],
				 kf->Parameter[1]+pJoint->Translation[1],
				 -kf->Parameter[2]-pJoint->Translation[2]);
		}
	}

	core::array<MS3DVertexWeights> vertexWeights;
	float weightFactor=0;

	if (jointCount && (pHeader->Version == 4) && (pPtr < buffer+fileSize))
	{
		int32_t subVersion = *(int32_t*)pPtr; // comment subVersion, always 1

		pPtr += sizeof(int32_t);

		for (uint32_t j=0; j<4; ++j) // four comment groups
		{
#ifdef _IRR_DEBUG_MS3D_LOADER_
			os::Printer::log("Skipping comment group", core::stringc(j+1).c_str());
#endif
			uint32_t numComments = *(uint32_t*)pPtr;

			pPtr += sizeof(uint32_t);
			for (i=0; i<numComments; ++i)
			{
				// according to scorpiomidget this field does
				// not exist for model comments. So avoid to
				// read it
				if (j!=3)
					pPtr += sizeof(int32_t); // index
				int32_t commentLength = *(int32_t*)pPtr;

				pPtr += sizeof(int32_t);
				pPtr += commentLength;
			}

			if (pPtr > buffer+fileSize)
			{
				delete [] buffer;
				os::Printer::log("Loading failed. Corrupted data found.", file->getFileName().c_str(), ELL_ERROR);
				return false;
			}
		}

		if (pPtr < buffer+fileSize)
		{
			subVersion = *(int32_t*)pPtr; // vertex subVersion, 1 or 2

			if (subVersion==1)
				weightFactor=1.f/255.f;
			else
				weightFactor=1.f/100.f;
			pPtr += sizeof(int32_t);

#ifdef _IRR_DEBUG_MS3D_LOADER_
			os::Printer::log("Reading vertex weights");
#endif
			// read vertex weights, ignoring data 'extra' from 1.8.2
			vertexWeights.reallocate(numVertices);
			const char offset = (subVersion==1)?6:10;
			for (i=0; i<numVertices; ++i)
			{
				vertexWeights.push_back(*(MS3DVertexWeights*)pPtr);
				pPtr += offset;
			}

			if (pPtr > buffer+fileSize)
			{
				delete [] buffer;
				os::Printer::log("Loading failed. Corrupted data found.", file->getFileName().c_str(), ELL_ERROR);
				return false;
			}
		}

		if (pPtr < buffer+fileSize)
		{
			subVersion = *(int32_t*)pPtr; // joint subVersion, 1 or 2

			pPtr += sizeof(int32_t);
			// skip joint colors
#ifdef _IRR_DEBUG_MS3D_LOADER_
			os::Printer::log("Skip joint color");
#endif
			pPtr += 3*sizeof(float)*jointCount;

			if (pPtr > buffer+fileSize)
			{
				delete [] buffer;
				os::Printer::log("Loading failed. Corrupted data found", file->getFileName().c_str(), ELL_ERROR);
				return false;
			}
		}

		if (pPtr < buffer+fileSize)
		{
			subVersion = *(int32_t*)pPtr; // model subVersion, 1 or 2

			pPtr += sizeof(int32_t);
#ifdef _IRR_DEBUG_MS3D_LOADER_
			os::Printer::log("Skip model extra information");
#endif
			// now the model extra information would follow
			// we also skip this for now
		}
	}

	//find parent of every joint
	for (uint32_t jointnum=0; jointnum<AnimatedMesh->getAllJoints().size(); ++jointnum)
	{
		for (uint32_t j2=0; j2<AnimatedMesh->getAllJoints().size(); ++j2)
		{
			if (jointnum != j2 && parentNames[jointnum] == AnimatedMesh->getAllJoints()[j2]->Name )
			{
				AnimatedMesh->getAllJoints()[j2]->Children.push_back(AnimatedMesh->getAllJoints()[jointnum]);
				break;
			}
		}
	}

	// create vertices and indices, attach them to the joints.
	video::S3DVertex v;
	core::array<video::S3DVertex> *Vertices;
	core::array<uint16_t> Indices;

	for (i=0; i<numTriangles; ++i)
	{
		uint32_t tmp = groups[triangles[i].GroupIndex].MaterialIdx;
		Vertices = &AnimatedMesh->getMeshBuffers()[tmp]->Vertices_Standard;

		for (int32_t j = 2; j!=-1; --j)
		{
			const uint32_t vertidx = triangles[i].VertexIndices[j];

			v.TCoords.X = triangles[i].S[j];
			v.TCoords.Y = triangles[i].T[j];

			v.Normal.X = triangles[i].VertexNormals[j][0];
			v.Normal.Y = triangles[i].VertexNormals[j][1];
			v.Normal.Z = triangles[i].VertexNormals[j][2];

			if(triangles[i].GroupIndex < groups.size() &&
					groups[triangles[i].GroupIndex].MaterialIdx < AnimatedMesh->getMeshBuffers().size())
				v.Color = AnimatedMesh->getMeshBuffers()[groups[triangles[i].GroupIndex].MaterialIdx]->Material.DiffuseColor;
			else
				v.Color.set(255,255,255,255);

			v.Pos.X = vertices[vertidx].Vertex[0];
			v.Pos.Y = vertices[vertidx].Vertex[1];
			v.Pos.Z = vertices[vertidx].Vertex[2];

			// check if we already have this vertex in our vertex array
			int32_t index = -1;
			for (uint32_t iV = 0; iV < Vertices->size(); ++iV)
			{
				if (v == (*Vertices)[iV])
				{
					index = (int32_t)iV;
					break;
				}
			}

			if (index == -1)
			{
				index = Vertices->size();
				const uint32_t matidx = groups[triangles[i].GroupIndex].MaterialIdx;
				if (vertexWeights.size()==0)
				{
					const int32_t boneid = vertices[vertidx].BoneID;
					if ((uint32_t)boneid < AnimatedMesh->getAllJoints().size())
					{
						ISkinnedMesh::SWeight *w=AnimatedMesh->addWeight(AnimatedMesh->getAllJoints()[boneid]);
						w->buffer_id = matidx;
						w->strength = 1.0f;
						w->vertex_id = index;
					}
				}
				else if (jointCount) // new weights from 1.8.x
				{
					float sum = 1.0f;
					int32_t boneid = vertices[vertidx].BoneID;
					if (((uint32_t)boneid < AnimatedMesh->getAllJoints().size()) && (vertexWeights[vertidx].weights[0] != 0))
					{
						ISkinnedMesh::SWeight *w=AnimatedMesh->addWeight(AnimatedMesh->getAllJoints()[boneid]);
						w->buffer_id = matidx;
						sum -= (w->strength = vertexWeights[vertidx].weights[0]*weightFactor);
						w->vertex_id = index;
					}
					boneid = vertexWeights[vertidx].boneIds[0];
					if (((uint32_t)boneid < AnimatedMesh->getAllJoints().size()) && (vertexWeights[vertidx].weights[1] != 0))
					{
						ISkinnedMesh::SWeight *w=AnimatedMesh->addWeight(AnimatedMesh->getAllJoints()[boneid]);
						w->buffer_id = matidx;
						sum -= (w->strength = vertexWeights[vertidx].weights[1]*weightFactor);
						w->vertex_id = index;
					}
					boneid = vertexWeights[vertidx].boneIds[1];
					if (((uint32_t)boneid < AnimatedMesh->getAllJoints().size()) && (vertexWeights[vertidx].weights[2] != 0))
					{
						ISkinnedMesh::SWeight *w=AnimatedMesh->addWeight(AnimatedMesh->getAllJoints()[boneid]);
						w->buffer_id = matidx;
						sum -= (w->strength = vertexWeights[vertidx].weights[2]*weightFactor);
						w->vertex_id = index;
					}
					boneid = vertexWeights[vertidx].boneIds[2];
					if (((uint32_t)boneid < AnimatedMesh->getAllJoints().size()) && (sum > 0.f))
					{
						ISkinnedMesh::SWeight *w=AnimatedMesh->addWeight(AnimatedMesh->getAllJoints()[boneid]);
						w->buffer_id = matidx;
						w->strength = sum;
						w->vertex_id = index;
					}
					// fallback, if no bone chosen. Seems to be an error in the specs
					boneid = vertices[vertidx].BoneID;
					if ((sum == 1.f) && ((uint32_t)boneid < AnimatedMesh->getAllJoints().size()))
					{
						ISkinnedMesh::SWeight *w=AnimatedMesh->addWeight(AnimatedMesh->getAllJoints()[boneid]);
						w->buffer_id = matidx;
						w->strength = 1.f;
						w->vertex_id = index;
					}
				}

				Vertices->push_back(v);
			}
			Indices.push_back(index);
		}
	}

	//create groups
	int32_t iIndex = -1;
	for (i=0; i<groups.size(); ++i)
	{
		SGroup& grp = groups[i];

		if (grp.MaterialIdx >= AnimatedMesh->getMeshBuffers().size())
			grp.MaterialIdx = 0;

		core::array<uint16_t>& indices = AnimatedMesh->getMeshBuffers()[grp.MaterialIdx]->Indices;

		for (uint32_t k=0; k < grp.VertexIds.size(); ++k)
			for (uint32_t l=0; l<3; ++l)
				indices.push_back(Indices[++iIndex]);
	}

	delete [] buffer;

	return true;
}


core::stringc CMS3DMeshFileLoader::stripPathFromString(const core::stringc& inString, bool returnPath) const
{
	int32_t slashIndex=inString.findLast('/'); // forward slash
	int32_t backSlash=inString.findLast('\\'); // back slash

	if (backSlash>slashIndex) slashIndex=backSlash;

	if (slashIndex==-1)//no slashes found
	{
		if (returnPath)
			return core::stringc(); //no path to return
		else
			return inString;
	}

	if (returnPath)
		return inString.subString(0, slashIndex + 1);
	else
		return inString.subString(slashIndex+1, inString.size() - (slashIndex+1));
}


} // end namespace scene
} // end namespace irr

#endif
