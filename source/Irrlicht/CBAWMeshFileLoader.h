// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __C_BAW_MESH_FILE_LOADER_H_INCLUDED__
#define __C_BAW_MESH_FILE_LOADER_H_INCLUDED__

#include <map>
#include <vector>

#include "IMeshLoader.h"
#include "ISceneManager.h"
#include "IFileSystem.h"
#include "IMesh.h"
#include "CBawFile.h"
#include "CBlobsLoadingManager.h"

namespace irr { namespace scene
{

class CBAWMeshFileLoader : public IMeshLoader
{
private:
	struct SBlobData
	{
		const core::BlobHeaderV0* header;
		size_t absOffset; // absolute
		mutable bool validated;

		bool validate(const void* _data) const { 
			validated = true;
			return validated ? true : header->validate(_data); 
		}
	};

	struct SContext
	{
		void releaseLoadedObjects()
		{
			for (size_t i = 0; i < blobs.size(); ++i)
				loadingMgr.releaseObj(blobs[i].header->blobType, createdObjs[blobs[i].header->handle]);
		}

		io::IReadFile* file;
		io::path filePath;
		uint64_t fileVersion;
		std::vector<SBlobData> blobs;
		std::map<uint64_t, void*> createdObjs;
		core::CBlobsLoadingManager loadingMgr;
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
	//! Instantiates new object of appropriate type dependent of blob's type but does not assign dependencies.
	/** @returns Pointer to just created object.
	*/
	void* instantiateObjWithoutDeps(const SBlobData& _data, SContext& _ctx, void* const _stackData, size_t _stackSize);
	//! Finalizes object (i.e. assigns dependcy objects to appropriate fields in the object).
	/** @returns whether everything went ok or not.
	*/
	bool finalizeObj(const SBlobData& _data, SContext& _ctx, void* const _stackData, size_t _stackSize);
	//! Verifies whether given file is of appropriate format. Also reads file version and assigns it to passed context object.
	bool verifyFile(SContext& _ctx) const;
	//! Loads and checks correctness of offsets and headers. Also let us know blob count.
	/** @returns true if everythings ok, false otherwise. */
	bool validateHeaders(uint32_t* _blobCnt, uint32_t** _offsets, void** _headers, SContext& _ctx);

	//! Reads `_size` bytes to `_buf` from `_file`, but previously checks whether file is big enough and returns true/false appropriately.
	bool safeRead(io::IReadFile* _file, void* _buf, size_t _size) const;

	//! Reads blob to memory on stack or allocates sufficient amount on heap if provided stack storage was not big enough.
	/** @returns `_stackPtr` if blob was read to it or pointer to malloc'd memory otherwise.*/
	void* tryReadBlobOnStack(const SBlobData& _data, SContext& _ctx, void* _stackPtr, size_t _stackSize) const;

private:
	scene::ISceneManager* m_sceneMgr;
	io::IFileSystem* m_fileSystem;
};

}} // irr::scene

#endif
