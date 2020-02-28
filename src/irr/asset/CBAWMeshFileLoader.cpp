// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#include "CBAWMeshFileLoader.h"

#include <stack>

#include "os.h"
#include "CMemoryFile.h"
#include "CFinalBoneHierarchy.h"
#include "irr/asset/IAssetManager.h"
#include "irr/asset/bawformat/legacy/CBAWLegacy.h"
#include "irr/video/CGPUMesh.h"
#include "irr/video/CGPUSkinnedMesh.h"

#include "lz4/lib/lz4.h"
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

CBAWMeshFileLoader::CBAWMeshFileLoader(IAssetManager* _manager) : m_manager(_manager), m_fileSystem(_manager->getFileSystem())
{
#ifdef _IRR_DEBUG
	setDebugName("CBAWMeshFileLoader");
#endif
}

SAssetBundle CBAWMeshFileLoader::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
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

    BlobHeaderLatest* headers = nullptr;

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
        return {};
    }

    uint32_t blobCnt{};
	uint32_t* offsets = nullptr;
    if (!validateHeaders<asset::BAWFileVn<_IRR_BAW_FORMAT_VERSION>, asset::BlobHeaderVn<_IRR_BAW_FORMAT_VERSION>>(&blobCnt, &offsets, (void**)&headers, ctx))
    {
        return {};
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
        m_manager,
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
            return {};
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

        auto foundBundle = _override->findCachedAsset(thisCacheKey, nullptr, ctx.inner, hierLvl).getContents();
        if (foundBundle.first!=foundBundle.second)
        {
            ctx.createdObjs[handle] = toAddrUsedByBlobsLoadingMgr(foundBundle.first->get(), blobType);
            continue;
        }

		bool fail = !(ctx.createdObjs[handle] = ctx.loadingMgr.instantiateEmpty(blobType, blob, size, params));

		if (fail)
		{
            return {};
		}

		if (!deps.size())
		{
            void* obj = ctx.createdObjs[handle];
			ctx.loadingMgr.finalize(blobType, obj, blob, size, ctx.createdObjs, params);
            _IRR_ALIGNED_FREE(data->heapBlob);
			blob = data->heapBlob = nullptr;
            insertAssetIntoCache(ctx, _override, obj, blobType, hierLvl, thisCacheKey);
		}
		else
			toFinalize.push(data);
	}

	void* retval = nullptr;
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
    return SAssetBundle({core::smart_refctd_ptr<asset::IAsset>(mesh,core::dont_grab)});
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

//! These should be moved to some legacy baw loader .cpp file
template<>
io::IReadFile* CBAWMeshFileLoader::createConvertIntoVer_spec<2>(SContext& _ctx, io::IReadFile* _baw1file, asset::IAssetLoader::IAssetLoaderOverride* _override, const CommonDataTuple<1>& _common)
{
    uint32_t blobCnt{};
    asset::legacyv1::BlobHeaderV1* headers = nullptr;
    uint32_t* offsets = nullptr;
    uint32_t baseOffsetv1{};
    uint32_t baseOffsetv2{};
    std::tie(blobCnt, headers, offsets, baseOffsetv1, baseOffsetv2) = _common;

    io::CMemoryWriteFile* const baw2mem = new io::CMemoryWriteFile(0u, _baw1file->getFileName());

	std::vector<uint32_t> newoffsets(blobCnt);
	int32_t offsetDiff = 0;
    for (uint32_t i = 0u; i < blobCnt; ++i)
    {
        asset::legacyv1::BlobHeaderV1& hdr = headers[i];
		const uint32_t offset = offsets[i];
		uint32_t& newoffset = newoffsets[i];

		newoffset = offset + offsetDiff;

        if (hdr.blobType == asset::Blob::EBT_FINAL_BONE_HIERARCHY)
        {
			const uint32_t offset = offsets[i];

            uint8_t stackmem[1u<<10];
            uint32_t attempt = 0u;
            uint8_t decrKey[16];
            size_t decrKeyLen = 16u;
            void* blob = nullptr;
            /* to state blob's/asset's hierarchy level we'd have to load (and possibly decrypt and decompress) the blob
            however we don't need to do this here since we know format's version (baw v1) and so we can be sure that hierarchy level for FinalBoneHierarchy is 3 (won't be true for v2 to v3 conversion)
            */
            constexpr uint32_t FINALBONEHIERARCHY_HIERARCHY_LVL = 3u;
            while (_override->getDecryptionKey(decrKey, decrKeyLen, attempt, _baw1file, "", genSubAssetCacheKey(_baw1file->getFileName().c_str(), hdr.handle), _ctx.inner, FINALBONEHIERARCHY_HIERARCHY_LVL))
            {
                if (!((hdr.compressionType & asset::Blob::EBCT_AES128_GCM) && decrKeyLen != 16u))
                    blob = tryReadBlobOnStack<asset::legacyv1::BlobHeaderV1>(SBlobData_t<asset::legacyv1::BlobHeaderV1>(&hdr, baseOffsetv1+offset), _ctx, decrKey, stackmem, sizeof(stackmem));
                if (blob)
                    break;
                ++attempt;
            }
			
			// patch the blob
			if (blob)
			{
				auto* hierarchy = reinterpret_cast<legacyv1::FinalBoneHierarchyBlobV1*>(blob);

				auto bones = hierarchy->getBoneData();
				for (auto j=0u; j<hierarchy->boneCount; j++)
				{
					core::matrix3x4SIMD mtx = bones[j].PoseBindMatrix;
					const auto* in = mtx.rows[0].pointer;
					auto* out = bones[j].PoseBindMatrix.rows[0].pointer;
					out[0] = in[0];
					out[4] = in[1];
					out[8] = in[2];
					out[1] = in[3];
					out[5] = in[4];
					out[9] = in[5];
					out[2] = in[6];
					out[6] = in[7];
					out[10] = in[8];
					out[3] = in[9];
					out[7] = in[10];
					out[11] = in[11];
				}

				const uint32_t absOffset = baseOffsetv2 + newoffset;
				baw2mem->seek(absOffset);
				baw2mem->write(blob, hdr.blobSizeDecompr);
			}
			else
				baw2mem->seek(baseOffsetv2 + newoffset + hdr.blobSizeDecompr);

			offsetDiff += static_cast<int32_t>(hdr.blobSizeDecompr) - static_cast<int32_t>(hdr.effectiveSize());
            hdr.compressionType = asset::Blob::EBCT_RAW;
            core::XXHash_256(blob, hdr.blobSizeDecompr, hdr.blobHash);
			hdr.blobSize = hdr.blobSizeDecompr;

			if (blob)
				_IRR_ALIGNED_FREE(blob);
        }
    }
    uint64_t fileHeader[4] {0u, 0u, 0u, 2u/*baw v2*/};
    memcpy(fileHeader, BAWFileV2::HEADER_STRING, strlen(BAWFileV2::HEADER_STRING));
    baw2mem->seek(0u);
    baw2mem->write(fileHeader, sizeof(fileHeader));
    baw2mem->write(&blobCnt, 4);
    baw2mem->write(_ctx.iv, 16);
    baw2mem->write(newoffsets.data(), newoffsets.size()*4);
    baw2mem->write(headers, blobCnt*sizeof(headers[0])); // blob header in v1 and in v2 is exact same thing, so we can do this

    uint8_t stackmem[1u<<13]{};
    size_t newFileSz = 0u;
    for (uint32_t i = 0u; i < blobCnt; ++i)
    {
        uint32_t sz = headers[i].effectiveSize();
        void* blob = nullptr;
        if (headers[i].blobType == asset::Blob::EBT_FINAL_BONE_HIERARCHY)
        {
            sz = 0u;
        }
        else
        {
            _baw1file->seek(baseOffsetv1 + offsets[i]);
            if (sz <= sizeof(stackmem))
                blob = stackmem;
            else
				blob = _IRR_ALIGNED_MALLOC(sz, _IRR_SIMD_ALIGNMENT);

            _baw1file->read(blob, sz);
        }

        baw2mem->seek(baseOffsetv2 + newoffsets[i]);
        baw2mem->write(blob, sz);

        if (headers[i].blobType != asset::Blob::EBT_DATA_FORMAT_DESC && blob != stackmem)
            _IRR_ALIGNED_FREE(blob);

        newFileSz = baseOffsetv2 + newoffsets[i] + sz;
    }

    _IRR_ALIGNED_FREE(offsets);
    _IRR_ALIGNED_FREE(headers);

    auto ret = new io::CMemoryReadFile(baw2mem->getPointer(), baw2mem->getSize(), _baw1file->getFileName());
    baw2mem->drop();
    return ret;
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
			assert(!blob || blob==stackmem); // its a fixed size blob so should always use stack

            const uint32_t absOffset = baseOffsetv1 + newoffset;
            baw1mem->seek(absOffset);
#ifndef NEW_SHADERS
            baw1mem->write(
                asset::MeshDataFormatDescBlobV1(reinterpret_cast<asset::legacyv0::MeshDataFormatDescBlobV0*>(blob)[0]).getData(),
                sizeof(asset::MeshDataFormatDescBlobV1)
            );

            prevBlobSz = hdr.effectiveSize();
            hdr.compressionType = asset::Blob::EBCT_RAW;
            core::XXHash_256(reinterpret_cast<uint8_t*>(baw1mem->getPointer())+absOffset, sizeof(asset::MeshDataFormatDescBlobV1), hdr.blobHash);
            hdr.blobSizeDecompr = hdr.blobSize = sizeof(asset::MeshDataFormatDescBlobV1);
#endif

            adjustDiff = true;
        }
#ifndef NEW_SHADERS
        if (adjustDiff)
            offsetDiff += static_cast<int32_t>(sizeof(asset::MeshDataFormatDescBlobV1)) - static_cast<int32_t>(prevBlobSz);
#endif
    }
    uint64_t fileHeader[4] {0u, 0u, 0u, 1u/*baw v1*/};
    memcpy(fileHeader, asset::legacyv1::BAWFileV1::HEADER_STRING, strlen(asset::legacyv1::BAWFileV1::HEADER_STRING));
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
				blob = stackmem;
			else
				blob = _IRR_ALIGNED_MALLOC(sz, _IRR_SIMD_ALIGNMENT);

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
