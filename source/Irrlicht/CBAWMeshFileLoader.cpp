// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#include "irr/asset/CBAWMeshFileLoader.h"

#include <stack>

#include "CFinalBoneHierarchy.h"
#include "SMesh.h" // Why is this nowhere to be found?
#include "CSkinnedMesh.h"
#include "os.h"
#include "lzma/LzmaDec.h"
#include "lz4/lz4.h"

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
	unsigned char pwd[16] = "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
	return createMesh(_file, pwd);
}

ICPUMesh* CBAWMeshFileLoader::createMesh(io::IReadFile * _file, unsigned char _pwd[16])
{
#ifdef _DEBUG
	auto time = std::chrono::high_resolution_clock::now();
#endif // _DEBUG

	SContext ctx{ _file };
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

	const uint32_t BLOBS_FILE_OFFSET = core::BAWFileV0{ {}, blobCnt }.calcBlobsOffset();

	core::unordered_map<uint64_t, SBlobData>::iterator meshBlobDataIter;

	for (uint32_t i = 0; i < blobCnt; ++i)
	{
		SBlobData data(headers + i, BLOBS_FILE_OFFSET + offsets[i]);
		const core::unordered_map<uint64_t, SBlobData>::iterator it = ctx.blobs.insert(std::make_pair(headers[i].handle, data)).first;
		if (data.header->blobType == core::Blob::EBT_MESH || data.header->blobType == core::Blob::EBT_SKINNED_MESH)
			meshBlobDataIter = it;
	}
	_IRR_ALIGNED_FREE(offsets);

	const core::BlobLoadingParams params{ m_sceneMgr, m_fileSystem, ctx.filePath };
	core::stack<SBlobData*> toLoad, toFinalize;
	toLoad.push(&meshBlobDataIter->second);
	while (!toLoad.empty())
	{
		SBlobData* data = toLoad.top();
		toLoad.pop();

		const uint64_t handle = data->header->handle;
		const uint32_t size = data->header->blobSizeDecompr;
		const uint32_t blobType = data->header->blobType;
		const void* blob = data->heapBlob = tryReadBlobOnStack(*data, ctx, _pwd);

		if (!blob)
		{
			ctx.releaseLoadedObjects();
			_IRR_ALIGNED_FREE(headers);
			return NULL;
		}

		core::unordered_set<uint64_t> deps = ctx.loadingMgr.getNeededDeps(blobType, blob);
		for (auto it = deps.begin(); it != deps.end(); ++it)
			if (ctx.createdObjs.find(*it) == ctx.createdObjs.end())
				toLoad.push(&ctx.blobs[*it]);

		bool fail = !(ctx.createdObjs[handle] = ctx.loadingMgr.instantiateEmpty(blobType, blob, size, params));

		if (fail)
		{
			ctx.releaseLoadedObjects();
			_IRR_ALIGNED_FREE(headers);
			return NULL;
		}

		if (!deps.size())
		{
			ctx.loadingMgr.finalize(blobType, ctx.createdObjs[handle], blob, size, ctx.createdObjs, params);
			_IRR_ALIGNED_FREE(data->heapBlob);
			blob = data->heapBlob = NULL;
		}
		else
			toFinalize.push(data);
	}

	void* retval = NULL;
	while (!toFinalize.empty())
	{
		SBlobData* data = toFinalize.top();
		toFinalize.pop();

		const void* blob = data->heapBlob;
		const uint64_t handle = data->header->handle;
		const uint32_t size = data->header->blobSizeDecompr;
		const uint32_t blobType = data->header->blobType;

		retval = ctx.loadingMgr.finalize(blobType, ctx.createdObjs[handle], blob, size, ctx.createdObjs, params); // last one will always be mesh
	}

	ctx.releaseAllButThisOne(meshBlobDataIter); // call drop on all loaded objects except mesh
	_IRR_ALIGNED_FREE(headers);

#ifdef _DEBUG
	std::ostringstream tmpString("Time to load ");
	tmpString.seekp(0, std::ios_base::end);
	tmpString << "BAW file: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()-time).count() << "us";
	os::Printer::log(tmpString.str());
#endif // _DEBUG

	return reinterpret_cast<ICPUMesh*>(retval);
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
	if (!safeRead(_ctx.file, _ctx.iv, 16))
		return false;
	uint32_t* const offsets = *_offsets = (uint32_t*)_IRR_ALIGNED_MALLOC(*_blobCnt * sizeof(uint32_t),_IRR_SIMD_ALIGNMENT);
	*_headers = _IRR_ALIGNED_MALLOC(*_blobCnt * sizeof(core::BlobHeaderV0),_IRR_SIMD_ALIGNMENT);
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

	if (offsetRelByte + offsets[*_blobCnt-1] >= _ctx.file->getSize()) // check the last offset
		nope = true;

	for (uint32_t i = 0; i < *_blobCnt-1; ++i) // whether blobs are tightly packed (do not overlays each other and there's no space bewteen any pair)
		if (offsets[i] + headers[i].effectiveSize() != offsets[i+1])
			nope = true;

	if (offsets[*_blobCnt-1] + headers[*_blobCnt-1].effectiveSize() >= _ctx.file->getSize()) // whether last blob doesn't "go out of file"
		nope = true;

	if (nope)
	{
		_IRR_ALIGNED_FREE(offsets);
		_IRR_ALIGNED_FREE(*_headers);
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

void* CBAWMeshFileLoader::tryReadBlobOnStack(const SBlobData & _data, SContext & _ctx, unsigned char _pwd[16], void * _stackPtr, size_t _stackSize) const
{
	void* dst;
	if (_stackPtr && _data.header->blobSizeDecompr <= _stackSize && _data.header->effectiveSize() <= _stackSize)
		dst = _stackPtr;
	else
		dst = _IRR_ALIGNED_MALLOC(core::BlobHeaderV0::calcEncSize(_data.header->blobSizeDecompr),_IRR_SIMD_ALIGNMENT);

	const bool encrypted = (_data.header->compressionType & core::Blob::EBCT_AES128_GCM);
	const bool compressed = (_data.header->compressionType & core::Blob::EBCT_LZ4) || (_data.header->compressionType & core::Blob::EBCT_LZMA);

	void* dstCompressed = dst; // ptr to mem to load possibly compressed data
	if (compressed)
		dstCompressed = _IRR_ALIGNED_MALLOC(_data.header->effectiveSize(),_IRR_SIMD_ALIGNMENT);

	_ctx.file->seek(_data.absOffset);
	_ctx.file->read(dstCompressed, _data.header->effectiveSize());

	if (!_data.header->validate(dstCompressed))
	{
#ifdef _DEBUG
		os::Printer::log("Blob validation failed!", ELL_ERROR);
#endif
		if (compressed)
		{
			_IRR_ALIGNED_FREE(dstCompressed);
			if (dst != _stackPtr)
				_IRR_ALIGNED_FREE(dst);
		}
		else if (dst != _stackPtr)
			_IRR_ALIGNED_FREE(dstCompressed);
		return NULL;
	}

	if (encrypted)
	{
#ifdef _IRR_COMPILE_WITH_OPENSSL_
		const size_t size = _data.header->effectiveSize();
		void* out = _IRR_ALIGNED_MALLOC(size,_IRR_SIMD_ALIGNMENT);
		const bool ok = core::decAes128gcm(dstCompressed, size, out, size, _pwd, _ctx.iv, _data.header->gcmTag);
		if (dstCompressed != _stackPtr)
			_IRR_ALIGNED_FREE(dstCompressed);
		if (!ok)
		{
			if (dst != _stackPtr && dstCompressed != dst)
				_IRR_ALIGNED_FREE(dst);
			_IRR_ALIGNED_FREE(out);
#ifdef _DEBUG
			os::Printer::log("Blob decryption failed!", ELL_ERROR);
#endif
			return NULL;
		}
		dstCompressed = out;
		if (!compressed)
			dst = dstCompressed;
#else
		if (compressed)
		{
			_IRR_ALIGNED_FREE(dstCompressed);
			if (dst != _stackPtr)
				_IRR_ALIGNED_FREE(dst);
		}
		else if (dst != _stackPtr)
			_IRR_ALIGNED_FREE(dstCompressed);
		return NULL;
#endif
	}

	if (compressed)
	{
		const uint8_t comprType = _data.header->compressionType;
		bool res = false;

		if (comprType & core::Blob::EBCT_LZ4)
			res = decompressLz4(dst, _data.header->blobSizeDecompr, dstCompressed, _data.header->blobSize);
		else if (comprType & core::Blob::EBCT_LZMA)
			res = decompressLzma(dst, _data.header->blobSizeDecompr, dstCompressed, _data.header->blobSize);

		_IRR_ALIGNED_FREE(dstCompressed);
		if (!res)
		{
			if (dst != _stackPtr && dst != dstCompressed)
				_IRR_ALIGNED_FREE(dst);
#ifdef _DEBUG
			os::Printer::log("Blob decompression failed!", ELL_ERROR);
#endif
			return NULL;
		}
	}

	return dst;
}

bool CBAWMeshFileLoader::decompressLzma(void* _dst, size_t _dstSize, const void* _src, size_t _srcSize) const
{
	SizeT dstSize = _dstSize;
	SizeT srcSize = _srcSize - LZMA_PROPS_SIZE;
	ELzmaStatus status;
	ISzAlloc alloc{&core::LzmaMemMngmnt::alloc, &core::LzmaMemMngmnt::release};
	const SRes res = LzmaDecode((Byte*)_dst, &dstSize, (const Byte*)(_src)+LZMA_PROPS_SIZE, &srcSize, (const Byte*)_src, LZMA_PROPS_SIZE, LZMA_FINISH_ANY, &status, &alloc);
	if (res != SZ_OK)
		return false;
	return true;
}

bool CBAWMeshFileLoader::decompressLz4(void * _dst, size_t _dstSize, const void * _src, size_t _srcSize) const
{
	int res = LZ4_decompress_safe((const char*)_src, (char*)_dst, _srcSize, _dstSize);
	return res >= 0;
}

}} // irr::scene
