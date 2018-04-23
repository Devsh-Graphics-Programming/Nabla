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
#include "CBAWFile.h"
#include "CBlobsLoadingManager.h"

namespace irr { namespace scene
{

class CBAWMeshFileLoader : public IMeshLoader
{
private:
	struct SBlobData
	{
		core::BlobHeaderV0* header;
		size_t absOffset; // absolute
		void* heapBlob;
		mutable bool validated;

		SBlobData(core::BlobHeaderV0* _hd=NULL, size_t _offset=0xdeadbeefdeadbeef) : header(_hd), absOffset(_offset), heapBlob(NULL), validated(false) {}
		~SBlobData() { free(heapBlob); }
		bool validate() const {
			validated = false;
			return validated ? true : (validated = (heapBlob && header->validate(heapBlob)));
		}
	private:
		// a bit dangerous to leave it copyable but until c++11 I have to to be able to store it in unordered_map
		// SBlobData(const SBlobData&) {}
		SBlobData& operator=(const SBlobData&) {}
	};

	struct SContext
	{
		void releaseLoadedObjects()
		{
			for (std::unordered_map<uint64_t, void*>::iterator it = createdObjs.begin(); it != createdObjs.end(); ++it)
				loadingMgr.releaseObj(blobs[it->first].header->blobType, it->second);
		}
		void releaseAllButThisOne(std::unordered_map<uint64_t, SBlobData>::iterator _thisIt)
		{
			const uint64_t theHandle = _thisIt != blobs.end() ? _thisIt->second.header->handle : 0;
			for (std::unordered_map<uint64_t, void*>::iterator it = createdObjs.begin(); it != createdObjs.end(); ++it)
			{
				if (it->first != theHandle)
					loadingMgr.releaseObj(blobs[it->first].header->blobType, it->second);
			}
		}

		io::IReadFile* file;
		io::path filePath;
		uint64_t fileVersion;
		std::unordered_map<uint64_t, SBlobData> blobs;
		std::unordered_map<uint64_t, void*> createdObjs;
		core::CBlobsLoadingManager loadingMgr;
		unsigned char iv[16];
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
	ICPUMesh* createMesh(io::IReadFile* file, unsigned char pwd[16]);

private:
	//! Verifies whether given file is of appropriate format. Also reads file version and assigns it to passed context object.
	bool verifyFile(SContext& _ctx) const;
	//! Loads and checks correctness of offsets and headers. Also let us know blob count.
	/** @returns true if everythings ok, false otherwise. */
	bool validateHeaders(uint32_t* _blobCnt, uint32_t** _offsets, void** _headers, SContext& _ctx);

	//! Reads `_size` bytes to `_buf` from `_file`, but previously checks whether file is big enough and returns true/false appropriately.
	bool safeRead(io::IReadFile* _file, void* _buf, size_t _size) const;

	//! Reads blob to memory on stack or allocates sufficient amount on heap if provided stack storage was not big enough.
	/** @returns `_stackPtr` if blob was read to it or pointer to malloc'd memory otherwise.*/
	void* tryReadBlobOnStack(const SBlobData& _data, SContext& _ctx, unsigned char pwd[16], void* _stackPtr=NULL, size_t _stackSize=0) const;

	bool decompressLzma(void* _dst, size_t _dstSize, const void* _src, size_t _srcSize) const;
	bool decompressLz4(void* _dst, size_t _dstSize, const void* _src, size_t _srcSize) const;

private:
	scene::ISceneManager* m_sceneMgr;
	io::IFileSystem* m_fileSystem;
};

}} // irr::scene

#endif
