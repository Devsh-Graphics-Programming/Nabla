// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#include "CBAWMeshFileLoader.h"

#include "CFinalBoneHierarchy.h"
#include "SMesh.h"
#include "CSkinnedMesh.h"

namespace irr { namespace scene
{
CBAWMeshFileLoader::~CBAWMeshFileLoader()
{
	if (m_fileSystem)
		m_fileSystem->drop();
}

CBAWMeshFileLoader::CBAWMeshFileLoader(scene::ISceneManager* _sm, io::IFileSystem* _fs) : m_sceneMgr(_sm), m_fileSystem(_fs)
{
#ifdef _DEBUG
	setDebugName("CBAWMeshFileLoader");
#endif
	if (m_fileSystem)
		m_fileSystem->grab();
}

ICPUMesh* CBAWMeshFileLoader::createMesh(io::IReadFile* _file)
{
	SContext ctx{_file};
	if (!verifyFile(ctx))
		return NULL;

	uint32_t blobCnt;
	uint32_t* offsets;
	core::BlobHeaderV0* headers;
	if (!validateHeaders(&blobCnt, &offsets, (void**)&headers, ctx))
		return NULL;

	ctx.filePath = ctx.file->getFileName();
	if (ctx.filePath[ctx.filePath.size() - 1] != '/')
		ctx.filePath += "/";

	const uint32_t BLOBS_FILE_OFFSET = core::BAWFileV0{{}, blobCnt}.calcBlobsOffset();

	uint64_t meshHandle = NULL;

	for (int i = 0; i < blobCnt; ++i)
	{
		SBlobData data = SBlobData{headers+i, BLOBS_FILE_OFFSET + offsets[i], false};
		ctx.blobs.push_back(data);
		ctx.createdObjs.insert(std::make_pair(headers[i].handle, (void*)NULL));
		if (data.header->blobType == core::Blob::EBT_MESH || data.header->blobType == core::Blob::EBT_SKINNED_MESH)
			meshHandle = headers[i].handle;
	}
	free(offsets);
	if (!meshHandle)
	{
		free(headers);
		return NULL;
	}

	uint8_t stack[1u<<14];
	for (size_t i = 0; i < ctx.blobs.size(); ++i)
		if (!(ctx.createdObjs[ctx.blobs[i].header->handle] = instantiateObjWithoutDeps(ctx.blobs[i], ctx, stack, sizeof(stack))))
		{
			free(headers);
			ctx.releaseLoadedObjects();
			return NULL;
		}

	for (size_t i = 0; i < ctx.blobs.size(); ++i)
		finalizeObj(ctx.blobs[i], ctx, stack, sizeof(stack));

	free(headers);
	return reinterpret_cast<ICPUMesh*>(ctx.createdObjs[meshHandle]);
}

void* CBAWMeshFileLoader::instantiateObjWithoutDeps(const SBlobData & _data, SContext & _ctx, void* const _stackData, size_t _stackSize)
{
	const core::BlobHeaderV0* header = _data.header;
	core::BlobLoadingParams params{ m_sceneMgr, m_fileSystem, _ctx.filePath };
	void* blob = tryReadBlobOnStack(_data, _ctx, _stackData, _stackSize);

	if (!_data.validate(blob))
	{
		if (blob != _stackData)
			free(blob);
		return NULL;
	}

	void* retval = _ctx.loadingMgr.instantiateEmpty(header->blobType, blob, header->blobSizeDecompr, params);

	if (blob != _stackData)
		free(blob);

	return retval;
}

bool CBAWMeshFileLoader::finalizeObj(const SBlobData & _data, SContext & _ctx, void * const _stackData, size_t _stackSize)
{
	const core::BlobHeaderV0* header = _data.header;
	core::BlobLoadingParams params{ m_sceneMgr, m_fileSystem, _ctx.filePath };
	void* blob = tryReadBlobOnStack(_data, _ctx, _stackData, _stackSize);

	void* retval = _ctx.loadingMgr.finalize(header->blobType, _ctx.createdObjs[header->handle], blob, header->blobSizeDecompr, _ctx.createdObjs, params);

	if (blob != _stackData)
		free(blob);

	return bool(retval);
}

bool CBAWMeshFileLoader::verifyFile(SContext& _ctx) const
{
	char headerStr[sizeof(core::BAWFileV0::fileHeader)];
	_ctx.file->seek(0);
	if (!safeRead(_ctx.file, headerStr, sizeof(headerStr)))
		return false;

	const char * const headerStrPattern = "IrrlichtBaW BinaryFile";
	if (strcmp(headerStr, headerStrPattern) != 0)
		return false;

	_ctx.fileVersion = ((uint64_t*)headerStr)[3];
	if (_ctx.fileVersion >= 1)
        return false;

	return true;
}

bool CBAWMeshFileLoader::validateHeaders(uint32_t* _blobCnt, uint32_t** _offsets, void** _headers, SContext& _ctx)
{
	if (!_blobCnt)
		return false;

	_ctx.file->seek(sizeof(core::BAWFileV0::fileHeader));
	if (!safeRead(_ctx.file, _blobCnt, sizeof(*_blobCnt)))
		return false;

	uint32_t* const offsets = *_offsets = (uint32_t*)malloc(*_blobCnt * sizeof(uint32_t));
	*_headers = malloc(*_blobCnt * sizeof(core::BlobHeaderV0));
	core::BlobHeaderV0* const headers = (core::BlobHeaderV0*)*_headers;

	bool nope = false;

	if (!safeRead(_ctx.file, offsets, *_blobCnt * sizeof(uint32_t)))
		nope = true;
	if (!safeRead(_ctx.file, headers, *_blobCnt * sizeof(core::BlobHeaderV0)))
		nope = true;
	const uint32_t offsetRelByte = core::BAWFileV0{{}, *_blobCnt}.calcBlobsOffset(); // num of byte to which offsets are relative
	for (uint32_t i = 0; i < *_blobCnt-1; ++i) // whether offsets are in ascending order none of them points past the end of file
		if (offsets[i] >= offsets[i+1] || offsetRelByte + offsets[i] >= _ctx.file->getSize())
			nope = true;

	if (offsetRelByte + offsets[0] >= _ctx.file->getSize()) // in case there's only one blob and so previous loop did not run at all
		nope = true;

	for (uint32_t i = 0; i < *_blobCnt-1; ++i) // whether blobs are tightly packed (do not overlays each other and there's no space bewteen any pair)
		if (offsets[i] + headers[i].blobSizeDecompr != offsets[i+1])
			nope = true;

	if (offsets[*_blobCnt-1] + headers[*_blobCnt-1].blobSizeDecompr >= _ctx.file->getSize()) // whether last blob doesn't "go out of file"
		nope = true;

	if (nope)
	{
		free(offsets);
		free(*_headers);
		return false;
	}
	return true;
}

bool CBAWMeshFileLoader::safeRead(io::IReadFile * _file, void * _buf, size_t _size) const
{
	if (_file->getPos() + _size > _file->getSize())
		return false;
	_file->read(_buf, _size);
	return true;
}

void* CBAWMeshFileLoader::tryReadBlobOnStack(const SBlobData & _data, SContext & _ctx, void * _stackPtr, size_t _stackSize) const
{
	void* dst;
	if (_data.header->blobSizeDecompr <= _stackSize)
		dst = _stackPtr;
	else
		dst = malloc(_data.header->blobSizeDecompr);
	_ctx.file->seek(_data.absOffset);
	_ctx.file->read(dst, _data.header->blobSizeDecompr);
	return dst;
}

}} // irr::scene
