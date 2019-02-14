// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h
//
// This file was originally written by ZDimitor.
// I (Nikolaus Gebhardt) did some few changes to this:
// - replaced logging calls to their os:: counterparts
// - removed some logging calls
// - removed setTexture path and replaced it with the directory of the mesh
// - added EAMT_MY3D file type
// - fixed a memory leak when decompressing RLE data.
// - cleaned multi character constant problems with gcc
// - removed octree child scene node generation because irrlicht is now able to draw
//   scene nodes with transparent and sold materials in them at the same time. (see changes.txt)
// Thanks a lot to ZDimitor for his work on this and that he gave me
// his permission to add it into Irrlicht.

//--------------------------------------------------------------------------------
// This tool created by ZDimitor everyone can use it as wants
//--------------------------------------------------------------------------------

#ifndef __CMY3D_MESH_FILE_LOADER_H_INCLUDED__
#define __CMY3D_MESH_FILE_LOADER_H_INCLUDED__


#ifdef _MSC_VER
#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
#endif


#include "IMeshLoader.h"
#include "irr/video/SGPUMesh.h"
#include "IFileSystem.h"
#include "IVideoDriver.h"

#include "ISceneManager.h"

namespace irr
{
namespace scene
{

// byte-align structures
#include "irr/irrpack.h"

struct SMyColor
{   SMyColor () {;}
    SMyColor (int32_t __R, int32_t __G, int32_t __B, int32_t __A)
        : R(__R), G(__G), B(__B), A(__A) {}
    int32_t R, G, B, A;
} PACK_STRUCT;

// material header
struct SMyMaterialHeader
{   int8_t  Name[256];           // material name
    uint32_t Index;
    SMyColor AmbientColor;
    SMyColor DiffuseColor;
    SMyColor EmissiveColor;
    SMyColor SpecularColor;
    float Shininess;
    float Transparency;
    uint32_t TextureCount;        // texture count
} PACK_STRUCT;

// Default alignment
#include "irr/irrunpack.h"

class CMY3DMeshFileLoader : public IMeshLoader
{
protected:
	virtual ~CMY3DMeshFileLoader();
public:
	CMY3DMeshFileLoader(ISceneManager *scmgr, io::IFileSystem* fs);

	virtual bool isALoadableFileExtension(const io::path& filename) const;

	virtual IAnimatedMesh* createMesh(io::IReadFile* file);

	//! getting access to the nodes (with transparent material), creating
	//! while loading .my3d file
	const core::array<ISceneNode*>& getChildNodes() const;

private:

	video::ITexture* readEmbeddedLightmap(io::IReadFile* file, char* namebuf);

	scene::ISceneManager* SceneManager;
	io::IFileSystem* FileSystem;

	struct SMyMaterialEntry
	{
		SMyMaterialEntry ()
		: Texture1FileName("null"), Texture2FileName("null"),
		Texture1(0), Texture2(0), MaterialType(video::EMT_SOLID) {}

		SMyMaterialHeader Header;
		core::stringc Texture1FileName;
		core::stringc Texture2FileName;
		video::ITexture *Texture1;
		video::ITexture *Texture2;
		video::E_MATERIAL_TYPE MaterialType;
	};

	struct SMyMeshBufferEntry
	{
		SMyMeshBufferEntry() : MaterialIndex(-1), MeshBuffer(0) {}
		SMyMeshBufferEntry(int32_t mi, SMeshBufferLightMap* mb)
			: MaterialIndex(mi), MeshBuffer(mb) {}

		int32_t MaterialIndex;
		SMeshBufferLightMap* MeshBuffer;
	};

	SMyMaterialEntry*    getMaterialEntryByIndex     (uint32_t matInd);
	SMeshBufferLightMap* getMeshBufferByMaterialIndex(uint32_t matInd);

	core::array<SMyMaterialEntry>   MaterialEntry;
	core::array<SMyMeshBufferEntry> MeshBufferEntry;

	core::array<ISceneNode*> ChildNodes;
};


} // end namespace scene
} // end namespace irr


#endif // __CMY3D_MESH_FILE_LOADER_H_INCLUDED__
