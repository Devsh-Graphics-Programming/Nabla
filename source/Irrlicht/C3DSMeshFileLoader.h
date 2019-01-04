// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_3DS_MESH_FILE_LOADER_H_INCLUDED__
#define __C_3DS_MESH_FILE_LOADER_H_INCLUDED__

#include <string>

#include "IMeshLoader.h"
#include "IFileSystem.h"
#include "ISceneManager.h"
#include "irr/video/SGPUMesh.h"
#include "matrix4.h"

namespace irr
{
namespace scene
{

//! Meshloader capable of loading 3ds meshes.
class C3DSMeshFileLoader : public IMeshLoader
{
protected:
	//! destructor
	virtual ~C3DSMeshFileLoader();

public:
	//! Constructor
	C3DSMeshFileLoader(ISceneManager* smgr, io::IFileSystem* fs);

	//! returns true if the file maybe is able to be loaded by this class
	//! based on the file extension (e.g. ".cob")
	virtual bool isALoadableFileExtension(const io::path& filename) const;

	//! creates/loads an animated mesh from the file.
	//! \return Pointer to the created mesh. Returns 0 if loading failed.
	//! If you no longer need the mesh, you should call IAnimatedMesh::drop().
	//! See IReferenceCounted::drop() for more information.
	virtual IAnimatedMesh* createMesh(io::IReadFile* file);

private:
// byte-align structures
#include "irr/irrpack.h"

	struct ChunkHeader
	{
		uint16_t id;
		int32_t length;
	} PACK_STRUCT;

// Default alignment
#include "irr/irrunpack.h"

	struct ChunkData
	{
		ChunkData() : read(0) {}

		ChunkHeader header;
		int32_t read;
	};

	struct SCurrentMaterial
	{
		void clear() {
			Material=video::SGPUMaterial();
			Name="";
			Filename[0]="";
			Filename[1]="";
			Filename[2]="";
			Filename[3]="";
			Filename[4]="";
			Strength[0]=0.f;
			Strength[1]=0.f;
			Strength[2]=0.f;
			Strength[3]=0.f;
			Strength[4]=0.f;
		}

		video::SGPUMaterial Material;
		std::string Name;
		std::string Filename[5];
		float Strength[5];
	};

	struct SMaterialGroup
	{
		SMaterialGroup() : faceCount(0), faces(0) {};

		SMaterialGroup(const SMaterialGroup& o)
		{
			*this = o;
		}

		~SMaterialGroup()
		{
			clear();
		}

		void clear()
		{
			delete [] faces;
			faces = 0;
			faceCount = 0;
		}

		void operator =(const SMaterialGroup& o)
		{
			MaterialName = o.MaterialName;
			faceCount = o.faceCount;
			faces = new uint16_t[faceCount];
			for (uint16_t i=0; i<faceCount; ++i)
				faces[i] = o.faces[i];
		}

		std::string MaterialName;
		uint16_t faceCount;
		uint16_t* faces;
	};

	bool readChunk(io::IReadFile* file, ChunkData* parent);
	bool readMaterialChunk(io::IReadFile* file, ChunkData* parent);
	bool readFrameChunk(io::IReadFile* file, ChunkData* parent);
	bool readTrackChunk(io::IReadFile* file, ChunkData& data,
				IMeshBuffer* mb, const core::vector3df& pivot);
	bool readObjectChunk(io::IReadFile* file, ChunkData* parent);
	bool readPercentageChunk(io::IReadFile* file, ChunkData* chunk, float& percentage);
	bool readColorChunk(io::IReadFile* file, ChunkData* chunk, video::SColor& out);

	void readChunkData(io::IReadFile* file, ChunkData& data);
	void readString(io::IReadFile* file, ChunkData& data, std::string& out);
	void readVertices(io::IReadFile* file, ChunkData& data);
	void readIndices(io::IReadFile* file, ChunkData& data);
	void readMaterialGroup(io::IReadFile* file, ChunkData& data);
	void readTextureCoords(io::IReadFile* file, ChunkData& data);

	void composeObject(io::IReadFile* file, const std::string& name);
	void loadMaterials(io::IReadFile* file);
	void cleanUp();

	scene::ISceneManager* SceneManager;
	io::IFileSystem* FileSystem;

	float* Vertices;
	uint16_t* Indices;
	uint32_t* SmoothingGroups;
	core::vector<uint16_t> TempIndices;
	float* TCoords;
	uint16_t CountVertices;
	uint16_t CountFaces; // = CountIndices/4
	uint16_t CountTCoords;
	core::vector<SMaterialGroup> MaterialGroups;

	SCurrentMaterial CurrentMaterial;
	core::vector<SCurrentMaterial> Materials;
	core::vector<std::string> MeshBufferNames;
	core::matrix4 TransformationMatrix;

	SMesh* Mesh;
};


} // end namespace scene
} // end namespace irr

#endif

