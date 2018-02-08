// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __C_BAW_MESH_FILE_LOADER_H_INCLUDED__
#define __C_BAW_MESH_FILE_LOADER_H_INCLUDED__

#include <map>
#include <queue>

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
	struct SBlobData
	{
		const core::BlobHeaderV0* header;
		size_t absOffset; // absolute
		void* createdObj;
		mutable bool validated;

		bool validate(const void* _data) const { 
			validated = true;
			return validated ? true : header->validate(_data); 
		}
		bool isLoaded() const { return createdObj; }
	};

	struct SContext
	{
		void freeLoadedObjects()
		{
			for (std::map<uint64_t, SBlobData>::iterator it = blobs.begin(); it != blobs.end(); ++it)
				if (it->second.createdObj)
					delete it->second.createdObj;
		}

		io::IReadFile* file;
		io::path filePath;
		uint64_t fileVersion;
		std::map<uint64_t, SBlobData> blobs;
		std::deque<uint64_t> queue;
		core::BlobsLoadingManager loadingMgr;
	};

protected:
	//! Destructor
	virtual ~CBAWMeshFileLoader();

public:
	//! Constructor
	CBAWMeshFileLoader(scene::ISceneManager* _sm, io::IFileSystem* _fs);

	//! @returns true if the file maybe is able to be loaded by this class
	//! based on the file extension (e.g. ".baw")
	virtual bool isALoadableFileExtension(const io::path& filename) const { return core::hasFileExtension(filename, "baw"); }

	//! creates/loads an animated mesh from the file.
	/** @returns Pointer to the created mesh. Returns 0 if loading failed.
	If you no longer need the mesh, you should call IAnimatedMesh::drop().
	See IReferenceCounted::drop() for more information.*/
	virtual ICPUMesh* createMesh(io::IReadFile* file);

private:
	//! Verifies whether given file is of appropriate format. Also reads file version and assigns it to passed context object.
	bool verifyFile(SContext& _ctx) const;
	//! Loads and checks correctness of offsets and headers. Also let us know blob count.
	/** @returns true if everythings ok, false otherwise. */
	bool validateHeaders(uint32_t* _blobCnt, uint32_t** _offsets, void** _headers, SContext& _ctx);

	//! Reads `_size` bytes to `_buf` from `_file`, but previously checks whether file is big enough and returns true/false appropriately.
	bool safeRead(io::IReadFile* _file, void* _buf, size_t _size) const;

	//! Loads a blob (i.e. creates object and assigns its address to _data.createdObj) defined by `_data` parameter/
	/** @returns false if loading/creating an object failed or true otherwise.*/
	bool loadBlob(SBlobData& _data, SContext&) const;

	//! Reads blob to memory on stack or allocates sufficient amount on heap if provided stack storage was not big enough.
	/** @returns `_stackPtr` if blob was read to it or pointer to malloc'd memory otherwise.*/
	void* tryReadBlobOnStack(SBlobData& _data, SContext& _ctx, void* _stackPtr, size_t _stackSize) const;

private:
	scene::ISceneManager* m_sceneMgr;
	io::IFileSystem* m_fileSystem;
};

}} // irr::scene

#endif
