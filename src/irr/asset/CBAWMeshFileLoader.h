// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __C_BAW_MESH_FILE_LOADER_H_INCLUDED__
#define __C_BAW_MESH_FILE_LOADER_H_INCLUDED__


#include "irr/asset/IAssetLoader.h"
#include "ISceneManager.h"
#include "IFileSystem.h"
#include "irr/asset/ICPUMesh.h"
#include "irr/asset/bawformat/CBAWFile.h"
#include "irr/asset/bawformat/CBlobsLoadingManager.h"
#include "irr/asset/ICPUSkinnedMeshBuffer.h"

namespace irr
{
class IrrlichtDevice;
}

namespace irr { namespace asset
{

class CBAWMeshFileLoader : public asset::IAssetLoader
{
private:
    template<typename HeaderT>
	struct SBlobData_t
	{
		HeaderT* header;
		size_t absOffset; // absolute
		void* heapBlob;
		mutable bool validated;
        uint32_t hierarchyLvl;

        SBlobData_t(HeaderT* _hd=NULL, size_t _offset=0xdeadbeefdeadbeef) : header(_hd), absOffset(_offset), heapBlob(NULL), validated(false) {}
		~SBlobData_t() { _IRR_ALIGNED_FREE(heapBlob); }

		bool validate() const {
			validated = false;
			return validated ? true : (validated = (heapBlob && header->validate(heapBlob)));
		}

        SBlobData_t<HeaderT>& operator=(const SBlobData_t<HeaderT>&) = delete;
	};
    using SBlobData = SBlobData_t<asset::BlobHeaderV1>;

	struct SContext
	{
		void releaseLoadedObjects()
		{
			for (auto it = createdObjs.begin(); it != createdObjs.end(); ++it)
				loadingMgr.releaseObj(blobs[it->first].header->blobType, it->second);
            createdObjs.clear();
		}
		void releaseAllButThisOne(core::unordered_map<uint64_t, SBlobData>::iterator _thisIt)
		{
			const uint64_t theHandle = _thisIt != blobs.end() ? _thisIt->second.header->handle : 0;
			for (auto it = createdObjs.begin(); it != createdObjs.end(); ++it)
			{
				if (it->first != theHandle)
					loadingMgr.releaseObj(blobs[it->first].header->blobType, it->second);
			}
            createdObjs.clear();
		}

        asset::IAssetLoader::SAssetLoadContext inner;
		uint64_t fileVersion;
		core::unordered_map<uint64_t, SBlobData> blobs;
		core::unordered_map<uint64_t, void*> createdObjs;
        asset::CBlobsLoadingManager loadingMgr;
		unsigned char iv[16];
	};

protected:
	//! Destructor
	virtual ~CBAWMeshFileLoader();

public:
	//! Constructor
	CBAWMeshFileLoader(IrrlichtDevice* _dev);


    virtual bool isALoadableFileFormat(io::IReadFile* _file) const override
    {
        SContext ctx{
            asset::IAssetLoader::SAssetLoadContext{
                asset::IAssetLoader::SAssetLoadParams{},
                _file
            }
        };
        
        const size_t prevPos = _file->getPos();
        _file->seek(0u);
        bool res = false;
        for (uint32_t i = 0u; i <= _IRR_BAW_FORMAT_VERSION; ++i)
            res |= verifyFile(_IRR_BAW_FORMAT_VERSION-i, ctx);
        _file->seek(prevPos);

        return res;
    }

    virtual const char** getAssociatedFileExtensions() const override
    {
        static const char* ext[]{ "baw", nullptr };
        return ext;
    }

    virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MESH; }

    virtual asset::IAsset* loadAsset(io::IReadFile* _file, const SAssetLoadParams& _params, IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u);

private:
	//! Verifies whether given file is of appropriate format. Also reads file version and assigns it to passed context object.
    //! Specialize if file header verification differs somehow from general template
    template<typename BAWFileT>
	bool verifyFile(SContext& _ctx, uint64_t _expectedVer) const;
    //! Add another case here when new version of .baw appears
    bool verifyFile(uint64_t _expectedVer, SContext& _ctx) const
    {
        switch (_expectedVer)
        {
        case 0ull: return verifyFile<asset::BAWFileV0>(_ctx, 0ull);
        case 1ull: return verifyFile<asset::BAWFileV1>(_ctx, 1ull);
        default: return false;
        }
    }
	//! Loads and checks correctness of offsets and headers. Also let us know blob count.
    //! Specialize if headers/offsets validation deffers somehow from general template
	/** @returns true if everythings ok, false otherwise. */
    template<typename BAWFileT, typename HeaderT>
	bool validateHeaders(uint32_t* _blobCnt, uint32_t** _offsets, void** _headers, SContext& _ctx);

	//! Reads `_size` bytes to `_buf` from `_file`, but previously checks whether file is big enough and returns true/false appropriately.
	bool safeRead(io::IReadFile* _file, void* _buf, size_t _size) const;

	//! Reads blob to memory on stack or allocates sufficient amount on heap if provided stack storage was not big enough.
	/** @returns `_stackPtr` if blob was read to it or pointer to malloc'd memory otherwise.*/
    template<typename HeaderT>
	void* tryReadBlobOnStack(const SBlobData_t<HeaderT>& _data, SContext& _ctx, const unsigned char pwd[16], void* _stackPtr=NULL, size_t _stackSize=0) const;

	bool decompressLzma(void* _dst, size_t _dstSize, const void* _src, size_t _srcSize) const;
	bool decompressLz4(void* _dst, size_t _dstSize, const void* _src, size_t _srcSize) const;

    inline std::string genSubAssetCacheKey(const std::string& _rootKey, uint64_t _handle) const { return _rootKey + "?" + std::to_string(_handle); }

    static inline void* toAddrUsedByBlobsLoadingMgr(asset::IAsset* _assetAddr, uint32_t _blobType)
    {
        // add here when more asset types will be available
        switch (_blobType)
        {
        case asset::Blob::EBT_MESH:
        case asset::Blob::EBT_SKINNED_MESH:
            assert(_assetAddr->getAssetType()==asset::IAsset::ET_MESH);
            return static_cast<asset::ICPUMesh*>(_assetAddr);
        case asset::Blob::EBT_MESH_BUFFER:
            assert(_assetAddr->getAssetType()==asset::IAsset::ET_SUB_MESH);
            return static_cast<asset::ICPUMeshBuffer*>(_assetAddr);
        case asset::Blob::EBT_SKINNED_MESH_BUFFER:
            assert(_assetAddr->getAssetType()==asset::IAsset::ET_SUB_MESH);
            return static_cast<asset::ICPUSkinnedMeshBuffer*>(_assetAddr);
        case asset::Blob::EBT_RAW_DATA_BUFFER:
            assert(_assetAddr->getAssetType()==asset::IAsset::ET_BUFFER);
            return static_cast<asset::ICPUBuffer*>(_assetAddr);
        case asset::Blob::EBT_TEXTURE_PATH:
            assert(_assetAddr->getAssetType()==asset::IAsset::ET_IMAGE);
            return static_cast<asset::ICPUTexture*>(_assetAddr);
        default: return nullptr;
        }
    }
    static inline void insertAssetIntoCache(const SContext& _ctx, asset::IAssetLoader::IAssetLoaderOverride* _override, void* _asset, uint32_t _blobType, uint32_t _hierLvl, const std::string& _cacheKey)
    {
        // add here when more asset types will be available
        asset::IAsset* asset = nullptr;
        switch (_blobType)
        {
        case asset::Blob::EBT_MESH:
        case asset::Blob::EBT_SKINNED_MESH:
            asset = reinterpret_cast<asset::ICPUMesh*>(_asset);
            break;
        case asset::Blob::EBT_MESH_BUFFER:
            asset = reinterpret_cast<asset::ICPUMeshBuffer*>(_asset);
            break;
        case asset::Blob::EBT_SKINNED_MESH_BUFFER:
            asset = reinterpret_cast<asset::ICPUSkinnedMeshBuffer*>(_asset);
            break;
        case asset::Blob::EBT_RAW_DATA_BUFFER:
            asset = reinterpret_cast<asset::ICPUBuffer*>(_asset);
            break;
        case asset::Blob::EBT_TEXTURE_PATH:
            asset = reinterpret_cast<asset::ICPUTexture*>(_asset);
            break;
        }
        if (asset && !asset->isInAResourceCache())
        {
            _override->insertAssetIntoCache(asset, _cacheKey, _ctx.inner, _hierLvl);
            asset->drop();
        }
    }

    // Compatibility functions:

    io::IReadFile* createConvertBAW0intoBAW1(io::IReadFile* _baw0file, asset::IAssetLoader::IAssetLoaderOverride* _override);

    io::IReadFile* tryCreateNewestFormatVersionFile(io::IReadFile* _originalFile, asset::IAssetLoader::IAssetLoaderOverride* _override);

private:
    IrrlichtDevice* m_device;
	scene::ISceneManager* m_sceneMgr;
	io::IFileSystem* m_fileSystem;
};

template<typename BAWFileT>
bool CBAWMeshFileLoader::verifyFile(SContext& _ctx, uint64_t _expectedVer) const
{
    char headerStr[sizeof(BAWFileT)];
    _ctx.inner.mainFile->seek(0);
    if (!safeRead(_ctx.inner.mainFile, headerStr, sizeof(headerStr)))
        return false;

    const char * const headerStrPattern = "IrrlichtBaW BinaryFile";
    if (strcmp(headerStr, headerStrPattern) != 0)
        return false;

    _ctx.fileVersion = ((uint64_t*)headerStr)[3];
    if (_ctx.fileVersion != _expectedVer)
        return false;

    return true;
}

template<typename BAWFileT, typename HeaderT>
bool CBAWMeshFileLoader::validateHeaders(uint32_t* _blobCnt, uint32_t** _offsets, void** _headers, SContext& _ctx)
{
    if (!_blobCnt)
        return false;

    _ctx.inner.mainFile->seek(sizeof(BAWFileT::fileHeader));
    if (!safeRead(_ctx.inner.mainFile, _blobCnt, sizeof(*_blobCnt)))
        return false;
    if (!safeRead(_ctx.inner.mainFile, _ctx.iv, 16))
        return false;
    uint32_t* const offsets = *_offsets = (uint32_t*)_IRR_ALIGNED_MALLOC(*_blobCnt*sizeof(uint32_t), _IRR_SIMD_ALIGNMENT);
    *_headers = _IRR_ALIGNED_MALLOC(*_blobCnt*sizeof(HeaderT), _IRR_SIMD_ALIGNMENT);
    HeaderT* const headers = (HeaderT*)*_headers;

    bool nope = false;

    if (!safeRead(_ctx.inner.mainFile, offsets, *_blobCnt*sizeof(uint32_t)))
        nope = true;
    if (!safeRead(_ctx.inner.mainFile, headers, *_blobCnt*sizeof(HeaderT)))
        nope = true;

    const uint32_t offsetRelByte = BAWFileT{{},*_blobCnt}.calcBlobsOffset(); // num of byte to which offsets are relative
    for (uint32_t i = 0; i < *_blobCnt-1; ++i) // whether offsets are in ascending order none of them points past the end of file
        if (offsets[i] >= offsets[i + 1] || offsetRelByte + offsets[i] >= _ctx.inner.mainFile->getSize())
            nope = true;

    if (offsetRelByte + offsets[*_blobCnt-1] >= _ctx.inner.mainFile->getSize()) // check the last offset
        nope = true;

    for (uint32_t i = 0; i < *_blobCnt-1; ++i) // whether blobs are tightly packed (do not overlays each other and there's no space bewteen any pair)
        if (offsets[i] + headers[i].effectiveSize() != offsets[i+1])
            nope = true;

    if (offsets[*_blobCnt-1] + headers[*_blobCnt-1].effectiveSize() >= _ctx.inner.mainFile->getSize()) // whether last blob doesn't "go out of file"
        nope = true;

    if (nope)
    {
        _IRR_ALIGNED_FREE(offsets);
        _IRR_ALIGNED_FREE(*_headers);
        return false;
    }
    return true;
}

template<typename HeaderT>
void* CBAWMeshFileLoader::tryReadBlobOnStack(const SBlobData_t<HeaderT> & _data, SContext & _ctx, const unsigned char _pwd[16], void * _stackPtr, size_t _stackSize) const
{
    void* dst;
    if (_stackPtr && _data.header->blobSizeDecompr <= _stackSize && _data.header->effectiveSize() <= _stackSize)
        dst = _stackPtr;
    else
        dst = _IRR_ALIGNED_MALLOC(asset::BlobHeaderV1::calcEncSize(_data.header->blobSizeDecompr), _IRR_SIMD_ALIGNMENT);

    const bool encrypted = (_data.header->compressionType & asset::Blob::EBCT_AES128_GCM);
    const bool compressed = (_data.header->compressionType & asset::Blob::EBCT_LZ4) || (_data.header->compressionType & asset::Blob::EBCT_LZMA);

    void* dstCompressed = dst; // ptr to mem to load possibly compressed data
    if (compressed)
        dstCompressed = _IRR_ALIGNED_MALLOC(_data.header->effectiveSize(), _IRR_SIMD_ALIGNMENT);

    _ctx.inner.mainFile->seek(_data.absOffset);
    _ctx.inner.mainFile->read(dstCompressed, _data.header->effectiveSize());

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
        void* out = _IRR_ALIGNED_MALLOC(size, _IRR_SIMD_ALIGNMENT);
        const bool ok = asset::decAes128gcm(dstCompressed, size, out, size, _pwd, _ctx.iv, _data.header->gcmTag);
        if (dstCompressed != _stackPtr)
            _IRR_ALIGNED_FREE(dstCompressed);
        if (!ok)
        {
            if (dst != _stackPtr && dstCompressed != dst)
                _IRR_ALIGNED_FREE(dst);
            _IRR_ALIGNED_FREE(out);
#ifdef _DEBUG
            os::Printer::log("Blob decryption failed!", ELL_ERROR);
#       endif
            return nullptr;
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

        if (comprType & asset::Blob::EBCT_LZ4)
            res = decompressLz4(dst, _data.header->blobSizeDecompr, dstCompressed, _data.header->blobSize);
        else if (comprType & asset::Blob::EBCT_LZMA)
            res = decompressLzma(dst, _data.header->blobSizeDecompr, dstCompressed, _data.header->blobSize);

        _IRR_ALIGNED_FREE(dstCompressed);
        if (!res)
        {
            if (dst != _stackPtr && dst != dstCompressed)
                _IRR_ALIGNED_FREE(dst);
#ifdef _DEBUG
            os::Printer::log("Blob decompression failed!", ELL_ERROR);
#endif
            return nullptr;
        }
    }

    return dst;
}

}} // irr::scene

#endif
