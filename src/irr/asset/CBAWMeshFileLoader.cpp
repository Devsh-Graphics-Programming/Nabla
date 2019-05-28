// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#include "CBAWMeshFileLoader.h"

#include <stack>

#include "CFinalBoneHierarchy.h"
#include "irr/video/SGPUMesh.h"
#include "irr/video/CGPUSkinnedMesh.h"
#include "os.h"
#include "lz4/lib/lz4.h"
#include "IrrlichtDevice.h"
#include "irr/asset/bawformat/legacy/CBAWLegacy.h"
#include "CMemoryFile.h"

#undef Bool
#include "lzma/C/LzmaDec.h"

namespace irr
{
namespace asset
{

struct LzmaMemMngmnt
{
        static void *alloc(ISzAllocPtr, size_t _size) { return _IRR_ALIGNED_MALLOC(_size,_IRR_SIMD_ALIGNMENT); }
        static void release(ISzAllocPtr, void* _addr) { _IRR_ALIGNED_FREE(_addr); }
    private:
        LzmaMemMngmnt() {}
};


CBAWMeshFileLoader::~CBAWMeshFileLoader()
{
}

CBAWMeshFileLoader::CBAWMeshFileLoader(IrrlichtDevice* _dev) : m_device(_dev), m_sceneMgr(_dev->getSceneManager()), m_fileSystem(_dev->getFileSystem())
{
#ifdef _IRR_DEBUG
	setDebugName("CBAWMeshFileLoader");
#endif
}

asset::IAsset* CBAWMeshFileLoader::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
#ifdef _IRR_DEBUG
    auto time = std::chrono::high_resolution_clock::now();
#endif // _IRR_DEBUG

	SContext ctx{
        asset::IAssetLoader::SAssetLoadContext{
            _params,
            _file
        }
    };

    ctx.inner.mainFile = tryCreateNewestFormatVersionFile(ctx.inner.mainFile, _override, std::make_integer_sequence<uint64_t, _IRR_BAW_FORMAT_VERSION>{});

    asset::BlobHeaderV1* headers = nullptr;

    auto exitRoutine = [&] {
        if (ctx.inner.mainFile != _file) // if mainFile is temparary memory file created just to update format to the newest version
            ctx.inner.mainFile->drop();
        ctx.releaseLoadedObjects();
        if (headers)
            _IRR_ALIGNED_FREE(headers);
    };
    auto exiter = core::makeRAIIExiter(exitRoutine);

    if (!verifyFile<asset::BAWFileVn<_IRR_BAW_FORMAT_VERSION>>(ctx))
    {
        return nullptr;
    }

    uint32_t blobCnt{};
	uint32_t* offsets = nullptr;
    if (!validateHeaders<asset::BAWFileVn<_IRR_BAW_FORMAT_VERSION>, asset::BlobHeaderVn<_IRR_BAW_FORMAT_VERSION>>(&blobCnt, &offsets, (void**)&headers, ctx))
    {
        return nullptr;
    }

	const uint32_t BLOBS_FILE_OFFSET = asset::BAWFileVn<_IRR_BAW_FORMAT_VERSION>{ {}, blobCnt }.calcBlobsOffset();

	core::unordered_map<uint64_t, SBlobData>::iterator meshBlobDataIter;

	for (uint32_t i = 0; i < blobCnt; ++i)
	{
		SBlobData data(headers + i, BLOBS_FILE_OFFSET + offsets[i]);
		const core::unordered_map<uint64_t, SBlobData>::iterator it = ctx.blobs.insert(std::make_pair(headers[i].handle, std::move(data))).first;
		if (data.header->blobType == asset::Blob::EBT_MESH || data.header->blobType == asset::Blob::EBT_SKINNED_MESH)
			meshBlobDataIter = it;
	}
	_IRR_ALIGNED_FREE(offsets);

    const std::string rootCacheKey = ctx.inner.mainFile->getFileName().c_str();

	const asset::BlobLoadingParams params{
        this,
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
        while (_override->getDecryptionKey(decrKey, decrKeyLen, attempt, ctx.inner.mainFile, "", thisCacheKey, ctx.inner, hierLvl))
        {
            if (!((data->header->compressionType & asset::Blob::EBCT_AES128_GCM) && decrKeyLen != 16u))
                blob = data->heapBlob = tryReadBlobOnStack(*data, ctx, decrKey);
            if (blob)
                break;
            ++attempt;
        }

		if (!blob)
		{
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

#ifdef _IRR_DEBUG
	std::ostringstream tmpString("Time to load ");
	tmpString.seekp(0, std::ios_base::end);
	tmpString << "BAW file: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()-time).count() << "us";
	os::Printer::log(tmpString.str());
#endif // _IRR_DEBUG

    asset::ICPUMesh* mesh = reinterpret_cast<asset::ICPUMesh*>(retval);
    return mesh;
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

template<>
io::IReadFile* CBAWMeshFileLoader::createConvertIntoVer_spec<1>(SContext& _ctx, io::IReadFile* _baw0file, asset::IAssetLoader::IAssetLoaderOverride* _override, const CommonDataTuple<0>& _common)
{
    uint32_t blobCnt{};
    asset::legacyv0::BlobHeaderV0* headers = nullptr;
    uint32_t* offsets = nullptr;
    uint32_t baseOffsetv0{};
    uint32_t baseOffsetv1{};
    std::tie(blobCnt, headers, offsets, baseOffsetv0, baseOffsetv1) = _common;

    io::CMemoryWriteFile* const baw1mem = new io::CMemoryWriteFile(0u, _baw0file->getFileName());

    std::vector<uint32_t> newoffsets(blobCnt);
    int32_t offsetDiff = 0;
    for (uint32_t i = 0u; i < blobCnt; ++i)
    {
        asset::legacyv0::BlobHeaderV0& hdr = headers[i];
        const uint32_t offset = offsets[i];
        uint32_t& newoffset = newoffsets[i];

        newoffset = offset + offsetDiff;

        bool adjustDiff = false;
        uint32_t prevBlobSz{};
        if (hdr.blobType == asset::Blob::EBT_DATA_FORMAT_DESC)
        {
            uint8_t stackmem[1u<<10];
            uint32_t attempt = 0u;
            uint8_t decrKey[16];
            size_t decrKeyLen = 16u;
            void* blob = nullptr;
            /* to state blob's/asset's hierarchy level we'd have to load (and possibly decrypt and decompress) the blob
            however we don't need to do this here since we know format's version (baw v0) and so we can be sure that hierarchy level for mesh data descriptors is 2
            */
            constexpr uint32_t ICPUMESHDATAFORMATDESC_HIERARCHY_LVL = 2u;
            while (_override->getDecryptionKey(decrKey, decrKeyLen, attempt, _baw0file, "", genSubAssetCacheKey(_baw0file->getFileName().c_str(), hdr.handle), _ctx.inner, ICPUMESHDATAFORMATDESC_HIERARCHY_LVL))
            {
                if (!((hdr.compressionType & asset::Blob::EBCT_AES128_GCM) && decrKeyLen != 16u))
                    blob = tryReadBlobOnStack<asset::legacyv0::BlobHeaderV0>(SBlobData_t<asset::legacyv0::BlobHeaderV0>(&hdr, baseOffsetv0+offset), _ctx, decrKey, stackmem, sizeof(stackmem));
                if (blob)
                    break;
                ++attempt;
            }

            const uint32_t absOffset = baseOffsetv1 + newoffset;
            baw1mem->seek(absOffset);
            baw1mem->write(
                asset::MeshDataFormatDescBlobV1(reinterpret_cast<asset::legacyv0::MeshDataFormatDescBlobV0*>(blob)[0]).getData(),
                sizeof(asset::MeshDataFormatDescBlobV1)
            );

            prevBlobSz = hdr.effectiveSize();
            hdr.compressionType = asset::Blob::EBCT_RAW;
            core::XXHash_256(reinterpret_cast<uint8_t*>(baw1mem->getPointer())+absOffset, sizeof(asset::MeshDataFormatDescBlobV1), hdr.blobHash);
            hdr.blobSizeDecompr = hdr.blobSize = sizeof(asset::MeshDataFormatDescBlobV1);

            adjustDiff = true;
        }
        if (adjustDiff)
            offsetDiff += static_cast<int32_t>(sizeof(asset::MeshDataFormatDescBlobV1)) - static_cast<int32_t>(prevBlobSz);
    }
    uint64_t fileHeader[4] {0u, 0u, 0u, 1u/*baw v1*/};
    memcpy(fileHeader, asset::BAWFileV1::HEADER_STRING, strlen(asset::BAWFileV1::HEADER_STRING));
    baw1mem->seek(0u);
    baw1mem->write(fileHeader, sizeof(fileHeader));
    baw1mem->write(&blobCnt, 4);
    baw1mem->write(_ctx.iv, 16);
    baw1mem->write(newoffsets.data(), newoffsets.size()*4);
    baw1mem->write(headers, blobCnt*sizeof(headers[0])); // blob header in v0 and in v1 is exact same thing, so we can do this

    uint8_t stackmem[1u<<13]{};
    size_t newFileSz = 0u;
    for (uint32_t i = 0u; i < blobCnt; ++i)
    {
        uint32_t sz = headers[i].effectiveSize();
        void* blob = nullptr;
        if (headers[i].blobType == asset::Blob::EBT_DATA_FORMAT_DESC)
        {
            sz = 0u;
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

        baw1mem->seek(baseOffsetv1 + newoffsets[i]);
        baw1mem->write(blob, sz);

        if (headers[i].blobType != asset::Blob::EBT_DATA_FORMAT_DESC && blob != stackmem)
            _IRR_ALIGNED_FREE(blob);

        newFileSz = baseOffsetv1 + newoffsets[i] + sz;
    }

    _IRR_ALIGNED_FREE(offsets);
    _IRR_ALIGNED_FREE(headers);

    auto ret = new io::CMemoryReadFile(baw1mem->getPointer(), baw1mem->getSize(), _baw0file->getFileName());
    baw1mem->drop();
    return ret;
}

}} // irr::scene
