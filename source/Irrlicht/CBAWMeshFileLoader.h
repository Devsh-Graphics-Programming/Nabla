// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __C_BAW_MESH_FILE_LOADER_H_INCLUDED__
#define __C_BAW_MESH_FILE_LOADER_H_INCLUDED__

#include <map>

#include "IMeshLoader.h"
#include "ISceneManager.h"
#include "IFileSystem.h"
#include "IMesh.h"
#include "CBawFile.h"

namespace irr { namespace scene
{

class CBAWMeshFileLoader : public IMeshLoader
{
private:
	struct SPair
	{
		core::BlobHeaderV1* header;
		void* blob;
	};

	struct SContext
	{
		io::path filePath;
		std::map<uint64_t, SPair> blobs;
	};

protected:
	//! Destructor
	virtual ~CBAWMeshFileLoader();

public:
	//! Constructor
	CBAWMeshFileLoader(scene::ISceneManager* _sm, io::IFileSystem* _fs);

	//! @returns true if the file maybe is able to be loaded by this class
	//! based on the file extension (e.g. ".obj")
	virtual inline bool isALoadableFileExtension(const io::path& filename) const { return core::hasFileExtension(filename, "baw"); }

	//! creates/loads an animated mesh from the file.
	/** @returns Pointer to the created mesh. Returns 0 if loading failed.
	If you no longer need the mesh, you should call IAnimatedMesh::drop().
	See IReferenceCounted::drop() for more information.*/
	virtual ICPUMesh* createMesh(io::IReadFile* file);

private:
	template<typename T>
	T* make(SPair&, SContext&) const;

private:
	scene::ISceneManager* m_sceneMgr;
	io::IFileSystem* m_fileSystem;
};

}} // irr::scene

#endif