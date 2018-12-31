// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#include "CBAWMeshFileLoader.h"

#include <stack>

#include "CFinalBoneHierarchy.h"
#include "SMesh.h"
#include "CSkinnedMesh.h"
#include "os.h"
#include "lzma/LzmaDec.h"
#include "lz4/lz4.h"
#include "IrrlichtDevice.h"
#include "irr/asset/bawformat/legacy/CBAWLegacy.h"
#include "CMemoryFile.h"

namespace irr { namespace scene
{
CBAWMeshFileLoader::~CBAWMeshFileLoader()
{
    m_device->drop();
}

CBAWMeshFileLoader::CBAWMeshFileLoader(IrrlichtDevice* _dev) : m_device(_dev), m_sceneMgr(_dev->getSceneManager()), m_fileSystem(_dev->getFileSystem())
{
#ifdef _DEBUG
	setDebugName("CBAWMeshFileLoader");
#endif
    m_device->grab();
}

asset::IAsset* CBAWMeshFileLoader::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
#ifdef _DEBUG
	uint32_t time = os::Timer::getRealTime();
#endif // _DEBUG

	SContext ctx{
        asset::IAssetLoader::SAssetLoadContext{
            _params,
            _file
        }
    };
    io::IReadFile* const overridenFile = ctx.inner.mainFile = _override->getLoadFile(ctx.inner.mainFile, ctx.inner.mainFile->getFileName().c_str(), ctx.inner, 0u);

    ctx.inner.mainFile = tryCreateNewestFormatVersionFile(ctx.inner.mainFile, _override);

    auto dropFileIfNeeded = [&] {
        if (ctx.inner.mainFile != overridenFile) // if mainFile is temparary memory file created just to update format to the newest version
            ctx.inner.mainFile->drop();
    };

    if (!verifyFile<asset::BAWFileV1>(ctx, _IRR_BAW_FORMAT_VERSION))
    {
        dropFileIfNeeded();
        return nullptr;
    }

	uint32_t blobCnt;
	uint32_t* offsets;
    asset::BlobHeaderV1* headers;
    if (!validateHeaders<asset::BAWFileV1, asset::BlobHeaderV1>(&blobCnt, &offsets, (void**)&headers, ctx))
    {
        dropFileIfNeeded();
        return nullptr;
    }

	const uint32_t BLOBS_FILE_OFFSET = asset::BAWFileV1{ {}, blobCnt }.calcBlobsOffset();

	core::unordered_map<uint64_t, SBlobData>::iterator meshBlobDataIter;

	for (uint32_t i = 0; i < blobCnt; ++i)
	{
		SBlobData data(headers + i, BLOBS_FILE_OFFSET + offsets[i]);
		const core::unordered_map<uint64_t, SBlobData>::iterator it = ctx.blobs.insert(std::make_pair(headers[i].handle, data)).first;
		if (data.header->blobType == asset::Blob::EBT_MESH || data.header->blobType == asset::Blob::EBT_SKINNED_MESH)
			meshBlobDataIter = it;
	}
	_IRR_ALIGNED_FREE(offsets);

    const std::string rootCacheKey = ctx.inner.mainFile->getFileName().c_str();

	const asset::BlobLoadingParams params{
        m_device,
        m_fileSystem,
        ctx.inner.mainFile->getFileName()[ctx.inner.mainFile->getFileName().size()-1] == '/' ? ctx.inner.mainFile->getFileName() : ctx.inner.mainFile->getFileName()+"/",
        ctx.inner.params,
        _override
    };
	core::stack<SBlobData*> toLoad, toFinalize;
	toLoad.push(&meshBlobDataIter->second);
    toLoad.top()->hierarchyLvl = 0u;
	while (!toLoad.empty())
	{
		SBlobData* data = toLoad.top();
		toLoad.pop();

		const uint64_t handle = data->header->handle;
        const uint32_t size = data->header->blobSizeDecompr;
        const uint32_t blobType = data->header->blobType;
        const std::string thisCacheKey = genSubAssetCacheKey(rootCacheKey, handle);
        const uint32_t hierLvl = data->hierarchyLvl;

        uint8_t decrKey[16];
        size_t decrKeyLen = 16u;
        uint32_t attempt = 0u;
        const void* blob = nullptr;
        // todo: supposedFilename arg is missing (empty string) - what is it? 
        while (_override->getDecryptionKey(decrKey, decrKeyLen, 16u, attempt, ctx.inner.mainFile, "", thisCacheKey, ctx.inner, hierLvl))
        {
            if (!((data->header->compressionType & asset::Blob::EBCT_AES128_GCM) && decrKeyLen != 16u))
                blob = data->heapBlob = tryReadBlobOnStack(*data, ctx, decrKey);
            if (blob)
                break;
            ++attempt;
        }

		if (!blob)
		{
			ctx.releaseLoadedObjects();
			_IRR_ALIGNED_FREE(headers);
            dropFileIfNeeded();
			return nullptr;
		}

		core::unordered_set<uint64_t> deps = ctx.loadingMgr.getNeededDeps(blobType, blob);
        for (auto it = deps.begin(); it != deps.end(); ++it)
        {
            if (ctx.createdObjs.find(*it) == ctx.createdObjs.end())
            {
                toLoad.push(&ctx.blobs[*it]);
                toLoad.top()->hierarchyLvl = hierLvl+1u;
            }
        }

        if (asset::IAsset* found = _override->findCachedAsset(thisCacheKey, nullptr, ctx.inner, hierLvl))
        {
            ctx.createdObjs[handle] = toAddrUsedByBlobsLoadingMgr(found, blobType);
            continue;
        }

		bool fail = !(ctx.createdObjs[handle] = ctx.loadingMgr.instantiateEmpty(blobType, blob, size, params));

		if (fail)
		{
			ctx.releaseLoadedObjects();
			_IRR_ALIGNED_FREE(headers);
            dropFileIfNeeded();
			return nullptr;
		}

		if (!deps.size())
		{
            void* obj = ctx.createdObjs[handle];
			ctx.loadingMgr.finalize(blobType, obj, blob, size, ctx.createdObjs, params);
            _IRR_ALIGNED_FREE(data->heapBlob);
			blob = data->heapBlob = NULL;
            insertAssetIntoCache(ctx, _override, obj, blobType, hierLvl, thisCacheKey);
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
        const uint32_t hierLvl = data->hierarchyLvl;
        const std::string thisCacheKey = genSubAssetCacheKey(rootCacheKey, handle);

		retval = ctx.loadingMgr.finalize(blobType, ctx.createdObjs[handle], blob, size, ctx.createdObjs, params); // last one will always be mesh
        if (!toFinalize.empty()) // don't cache root-asset (mesh) as sub-asset because it'll be cached by asset manager directly (and there's only one IAsset::cacheKey)
            insertAssetIntoCache(ctx, _override, retval, blobType, hierLvl, thisCacheKey);
	}

	ctx.releaseAllButThisOne(meshBlobDataIter); // call drop on all loaded objects except mesh
	_IRR_ALIGNED_FREE(headers);

#ifdef _DEBUG
	time = os::Timer::getRealTime() - time;
	std::ostringstream tmpString("Time to load ");
	tmpString.seekp(0, std::ios_base::end);
	tmpString << "BAW file: " << time << "ms";
	os::Printer::log(tmpString.str());
#endif // _DEBUG

    dropFileIfNeeded();
	return reinterpret_cast<asset::ICPUMesh*>(retval);
}

bool CBAWMeshFileLoader::safeRead(io::IReadFile * _file, void * _buf, size_t _size) const
{
	if (_file->getPos() + _size > _file->getSize())
		return false;
	_file->read(_buf, _size);
	return true;
}

bool CBAWMeshFileLoader::decompressLzma(void* _dst, size_t _dstSize, const void* _src, size_t _srcSize) const
{
	SizeT dstSize = _dstSize;
	SizeT srcSize = _srcSize - LZMA_PROPS_SIZE;
	ELzmaStatus status;
	ISzAlloc alloc{&asset::LzmaMemMngmnt::alloc, &asset::LzmaMemMngmnt::release};
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

io::IReadFile* CBAWMeshFileLoader::createConvertBAW0intoBAW1(io::IReadFile* _baw0file, asset::IAssetLoader::IAssetLoaderOverride* _override)
{
    uint8_t* const baw1mem = new uint8_t[_baw0file->getSize()]; // baw.v1 will be for sure a little smaller than baw.v0

	SContext ctx{
        asset::IAssetLoader::SAssetLoadContext{
            asset::IAssetLoader::SAssetLoadParams{},
            _baw0file
        }
    };

    if (!verifyFile<asset::BAWFileV0>(ctx, 0ull))
    {
        delete[] baw1mem;
        return nullptr;
    }

    uint32_t blobCnt{};
    uint32_t* offsets = nullptr;
    asset::BlobHeaderV0* headers = nullptr;

    if (!validateHeaders<asset::BAWFileV0, asset::BlobHeaderV0>(&blobCnt, &offsets, reinterpret_cast<void**>(&headers), ctx))
    {
        delete[] baw1mem;
        return nullptr;
    }

    const uint32_t baseOffsetv0 = asset::BAWFileV0{{},blobCnt}.calcBlobsOffset();

    std::vector<uint32_t> newoffsets(blobCnt);
    std::vector<asset::MeshDataFormatDescBlobV1> newblobs;
    int32_t offsetDiff = 0;
    for (uint32_t i = 0u; i < blobCnt; ++i)
    {
        asset::BlobHeaderV0& hdr = headers[i];
        const uint32_t offset = offsets[i];
        uint32_t& newoffset = newoffsets[i];

        bool adjustDiff = false;
        uint32_t prevBlobSz{};
        if (hdr.blobType == asset::Blob::EBT_DATA_FORMAT_DESC)
        {
            uint8_t stackmem[1u<<10];
            uint32_t attempt = 0u;
            uint8_t decrKey[16];
            size_t decrKeyLen = 16u;
            void* blob = nullptr;
            while (_override->getDecryptionKey(decrKey, decrKeyLen, 16u, attempt, _baw0file, "", genSubAssetCacheKey(_baw0file->getFileName().c_str(), hdr.handle), ctx.inner, 2u))
            {
                if (!((hdr.compressionType & asset::Blob::EBCT_AES128_GCM) && decrKeyLen != 16u))
                    blob = tryReadBlobOnStack<asset::BlobHeaderV0>(SBlobData_t<asset::BlobHeaderV0>(&hdr, baseOffsetv0+offset), ctx, decrKey, stackmem, sizeof(stackmem));
                if (blob)
                    break;
                ++attempt;
            }
            newblobs.emplace_back(reinterpret_cast<asset::legacy::MeshDataFormatDescBlobV0*>(blob)[0]);

            prevBlobSz = hdr.effectiveSize();
            hdr.compressionType = asset::Blob::EBCT_RAW;
            core::XXHash_256(&newblobs.back(), sizeof(newblobs.back()), hdr.blobHash);
            hdr.blobSizeDecompr = hdr.blobSize = sizeof(newblobs.back());

            adjustDiff = true;
        }
        newoffset = offset + offsetDiff;
        if (adjustDiff)
            offsetDiff += static_cast<int32_t>(sizeof(newblobs.back())) - static_cast<int32_t>(prevBlobSz);
    }
    const char * const headerStr = "IrrlichtBaW BinaryFile";
    uint64_t fileHeader[4] {0u, 0u, 0u, 1u/*baw v1*/};
    memcpy(fileHeader, headerStr, strlen(headerStr));
    uint8_t* baw1mem_tmp = baw1mem;
    memcpy(baw1mem_tmp, fileHeader, sizeof(fileHeader));
    baw1mem_tmp += sizeof(fileHeader);
    memcpy(baw1mem_tmp, &blobCnt, 4);
    baw1mem_tmp += 4;
    memcpy(baw1mem_tmp, ctx.iv, 16);
    baw1mem_tmp += 16;
    memcpy(baw1mem_tmp, newoffsets.data(), newoffsets.size()*4);
    baw1mem_tmp += newoffsets.size()*4;
    memcpy(baw1mem_tmp, headers, blobCnt*sizeof(headers[0])); // blob header in v0 and in v1 is exact same thing, so we can do this

    const uint32_t baseOffsetv1 = asset::BAWFileV1{{}, blobCnt}.calcBlobsOffset();

    uint8_t stackmem[1u<<13]{};
    auto newblobsItr = newblobs.begin();
    size_t newFileSz = 0u;
    for (uint32_t i = 0u; i < blobCnt; ++i)
    {
        uint32_t sz = headers[i].effectiveSize();
        void* blob = nullptr;
        if (headers[i].blobType == asset::Blob::EBT_DATA_FORMAT_DESC)
        {
            blob = &(*(newblobsItr++));
            sz = sizeof(newblobs[0]);
        }
        else
        {
            _baw0file->seek(baseOffsetv0 + offsets[i]);
            if (sz <= sizeof(stackmem))
            {
                blob = stackmem;
            }
            else blob = _IRR_ALIGNED_MALLOC(sz, _IRR_SIMD_ALIGNMENT);

            _baw0file->read(blob, sz);
        }

        baw1mem_tmp = baw1mem + baseOffsetv1 + newoffsets[i];
        memcpy(baw1mem_tmp, blob, sz);

        if (headers[i].blobType != asset::Blob::EBT_DATA_FORMAT_DESC && blob != stackmem)
            _IRR_ALIGNED_FREE(blob);

        newFileSz = baseOffsetv1 + newoffsets[i] + sz;
    }

    _IRR_ALIGNED_FREE(offsets);
    _IRR_ALIGNED_FREE(headers);

    return new io::CMemoryReadFile(baw1mem, newFileSz, _baw0file->getFileName(), true);
}

io::IReadFile* CBAWMeshFileLoader::tryCreateNewestFormatVersionFile(io::IReadFile* _originalFile, asset::IAssetLoader::IAssetLoaderOverride* _override)
{
    using convertFuncT = io::IReadFile*(CBAWMeshFileLoader::*)(io::IReadFile*, asset::IAssetLoader::IAssetLoaderOverride*);
    convertFuncT convertFunc[_IRR_BAW_FORMAT_VERSION]{ &CBAWMeshFileLoader::createConvertBAW0intoBAW1 };

    if (!_originalFile)
        return nullptr;

    _originalFile->grab();

    uint64_t version{};
    _originalFile->seek(24u);
    _originalFile->read(&version, 8u);
    _originalFile->seek(0u);

    io::IReadFile* newestFormatFile = _originalFile;
    while (version != _IRR_BAW_FORMAT_VERSION)
    {
#define CALL_MEMBER_FN(object,ptrToMember)  ((object).*(ptrToMember))
        io::IReadFile* tmp = CALL_MEMBER_FN(*this, convertFunc[version])(newestFormatFile, _override);
        newestFormatFile->drop();
        newestFormatFile = tmp;
        ++version;
        if (!newestFormatFile)
            return nullptr;
    }
    if (newestFormatFile == _originalFile)
        newestFormatFile->drop();

    return newestFormatFile;
#undef CALL_MEMBER_FN
}

}} // irr::scene
