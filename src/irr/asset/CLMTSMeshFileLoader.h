// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h
//
// I (Nikolaus Gebhardt) did some few changes to Jonas Petersen's original loader:
// - removed setTexturePath() and replaced with the ISceneManager::getStringParameter()-stuff.
// - added EAMT_LMTS enumeration value
// Thanks a lot to Jonas Petersen for his work
// on this and that he gave me his permission to add it into Irrlicht.
/*

CLMTSMeshFileLoader.h

LMTSMeshFileLoader
Written by Jonas Petersen (a.k.a. jox)

Version 1.5 - 15 March 2005

*/

#if !defined(__C_LMTS_MESH_FILE_LOADER_H_INCLUDED__)
#define __C_LMTS_MESH_FILE_LOADER_H_INCLUDED__

#include "IMeshLoader.h"
#include "irr/video/SGPUMesh.h"
#include "IFileSystem.h"
#include "IVideoDriver.h"

namespace irr
{
namespace scene
{

class CLMTSMeshFileLoader : public IMeshLoader
{
protected:
	virtual ~CLMTSMeshFileLoader();
public:
	CLMTSMeshFileLoader(io::IFileSystem* fs, video::IVideoDriver* driver);

	virtual bool isALoadableFileExtension(const io::path& filename) const;

	virtual IAnimatedMesh* createMesh(io::IReadFile* file);

private:
	void constructMesh(SMesh* mesh);
	void loadTextures(SMesh* mesh);
	void cleanup();

// byte-align structures
#include "irr/irrpack.h"

	struct SLMTSHeader
	{
		uint32_t MagicID;
		uint32_t Version;
		uint32_t HeaderSize;
		uint16_t TextureCount;
		uint16_t SubsetCount;
		uint32_t TriangleCount;
		uint16_t SubsetSize;
		uint16_t VertexSize;
	} PACK_STRUCT;

	struct SLMTSTextureInfoEntry
	{
		int8_t Filename[256];
		uint16_t Flags;
	} PACK_STRUCT;

	struct SLMTSSubsetInfoEntry
	{
		uint32_t Offset;
		uint32_t Count;
		uint16_t TextID1;
		uint16_t TextID2;
	} PACK_STRUCT;

	struct SLMTSTriangleDataEntry
	{
		float X;
		float Y;
		float Z;
		float U1;
		float V1;
		float U2;
		float V2;
	} PACK_STRUCT;

// Default alignment
#include "irr/irrunpack.h"

	SLMTSHeader Header;
	SLMTSTextureInfoEntry* Textures;
	SLMTSSubsetInfoEntry* Subsets;
	SLMTSTriangleDataEntry* Triangles;

	video::IVideoDriver* Driver;
	io::IFileSystem* FileSystem;
	bool FlipEndianess;
};

} // end namespace scene
} // end namespace irr

#endif // !defined(__C_LMTS_MESH_FILE_LOADER_H_INCLUDED__)
