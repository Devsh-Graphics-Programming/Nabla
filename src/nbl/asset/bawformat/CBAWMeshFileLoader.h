// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_BAW_MESH_FILE_LOADER_H_INCLUDED__
#define __NBL_ASSET_C_BAW_MESH_FILE_LOADER_H_INCLUDED__

#include "nbl/asset/compile_config.h"

#include "nbl/asset/IAssetManager.h" /*
#include "nbl/asset/IAssetLoader.h"
#include "IFileSystem.h"
#include "nbl/asset/ICPUMesh.h"*/
#include "nbl/asset/bawformat/CBAWFile.h"
//#include "nbl/asset/bawformat/legacy/CBAWLegacy.h"
#include "nbl/asset/bawformat/CBlobsLoadingManager.h"

#include "nbl/system/ISystem.h"

namespace nbl
{
namespace asset
{
class CBAWMeshFileLoader : public asset::IAssetLoader
{
#ifdef OLD_SHADERS
    friend struct TypedBlob<TexturePathBlobV3, asset::ICPUTexture>;  // needed for loading textures
#endif

private:
    template<typename HeaderT>
    struct SBlobData_t
    {
        HeaderT* header;
        size_t absOffset;  // absolute
        void* heapBlob = nullptr;
        mutable bool validated = false;
        uint32_t hierarchyLvl = 0u;

        SBlobData_t(HeaderT* _hd = nullptr, size_t _offset = 0xdeadbeefdeadbeefu)
            : header(_hd), absOffset(_offset) {}
        SBlobData_t(const SBlobData_t<HeaderT>&) = delete;
        SBlobData_t(SBlobData_t<HeaderT>&& _other)
        {
            std::swap(heapBlob, _other.heapBlob);
            header = _other.header;
            absOffset = _other.absOffset;
            validated = _other.validated;
            hierarchyLvl = _other.hierarchyLvl;
        }
        SBlobData_t<HeaderT>& operator=(const SBlobData_t<HeaderT>&) = delete;
        ~SBlobData_t()
        {
            if(heapBlob)
                _NBL_ALIGNED_FREE(heapBlob);
        }

        bool validate() const
        {
            validated = false;
            return validated ? true : (validated = (heapBlob && header->validate(heapBlob)));
        }
    };
    using SBlobData = SBlobData_t<asset::BlobHeaderVn<_NBL_FORMAT_VERSION>>;

    struct SContext
    {
        void releaseLoadedObjects()
        {
            for(auto it = createdObjs.begin(); it != createdObjs.end(); ++it)
                loadingMgr.releaseObj(blobs[it->first].header->blobType, it->second);
            createdObjs.clear();
        }
        void releaseAllButThisOne(core::unordered_map<uint64_t, SBlobData>::iterator _thisIt)
        {
            const uint64_t theHandle = _thisIt != blobs.end() ? _thisIt->second.header->handle : 0;
            for(auto it = createdObjs.begin(); it != createdObjs.end(); ++it)
            {
                if(it->first != theHandle)
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
    CBAWMeshFileLoader(IAssetManager* _manager);

    virtual bool isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const override
    {
        SContext ctx{
            asset::IAssetLoader::SAssetLoadContext{
                asset::IAssetLoader::SAssetLoadParams{},
                _file},
            // following should shut up GCC
            0xdeadbeefu,
            {},
            {},
            asset::CBlobsLoadingManager(),
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

        bool res = false;
        for(uint32_t i = 0u; i <= _NBL_FORMAT_VERSION; ++i)
            res |= verifyFile(_NBL_FORMAT_VERSION - i, ctx);

        return res;
    }

    virtual const char** getAssociatedFileExtensions() const override
    {
        static const char* ext[]{"baw", nullptr};
        return ext;
    }

    virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MESH; }

    virtual asset::SAssetBundle loadAsset(system::IFile* _file, const SAssetLoadParams& _params, IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u);

private:
    //! Verifies whether given file is of appropriate format. Also reads file version and assigns it to passed context object.
    //! Specialize if file header verification differs somehow from general template
    template<typename BAWFileT>
    bool verifyFile(SContext& _ctx) const;
    //! Add another case here when new version of .baw appears
    bool verifyFile(uint64_t _expectedVer, SContext& _ctx) const
    {
        switch(_expectedVer)
        {
            case 0ull: return verifyFile<BAWFileVn<0>>(_ctx);
            case 1ull: return verifyFile<BAWFileVn<1>>(_ctx);
            case 2ull: return verifyFile<BAWFileVn<2>>(_ctx);
            case 3ull: return verifyFile<asset::BAWFileV3>(_ctx);
            default: return false;
        }
    }
    //! Loads and checks correctness of offsets and headers. Also let us know blob count.
    //! Specialize if headers/offsets validation deffers somehow from general template
    /** @returns true if everythings ok, false otherwise. */
    template<typename BAWFileT, typename HeaderT>
    bool validateHeaders(uint32_t* _blobCnt, uint32_t** _offsets, void** _headers, SContext& _ctx);

    //! Reads `_size` bytes to `_buf` from `_file`, but previously checks whether file is big enough and returns true/false appropriately.
    bool safeRead(system::IFile* _file, void* _buf, size_t _size) const;

    //! Reads blob to memory on stack or allocates sufficient amount on heap if provided stack storage was not big enough.
    /** @returns `_stackPtr` if blob was read to it or pointer to malloc'd memory otherwise.*/
    template<typename HeaderT>
    void* tryReadBlobOnStack(const SBlobData_t<HeaderT>& _data, SContext& _ctx, const unsigned char pwd[16], void* _stackPtr = NULL, size_t _stackSize = 0) const;

    bool decompressLzma(void* _dst, size_t _dstSize, const void* _src, size_t _srcSize) const;
    bool decompressLz4(void* _dst, size_t _dstSize, const void* _src, size_t _srcSize) const;

    inline std::string genSubAssetCacheKey(const std::string& _rootKey, uint64_t _handle) const { return _rootKey + "?" + std::to_string(_handle); }

    static inline void* toAddrUsedByBlobsLoadingMgr(asset::IAsset* _assetAddr, uint32_t _blobType)
    {
        // add here when more asset types will be available
        switch(_blobType)
        {
#ifdef OLD_MESHES
            case asset::Blob::EBT_MESH:
            case asset::Blob::EBT_SKINNED_MESH:
                assert(_assetAddr->getAssetType() == asset::IAsset::ET_MESH);
                return static_cast<asset::ICPUMesh*>(_assetAddr);
            case asset::Blob::EBT_MESH_BUFFER:
                assert(_assetAddr->getAssetType() == asset::IAsset::ET_SUB_MESH);
                return static_cast<asset::ICPUMeshBuffer*>(_assetAddr);
            case asset::Blob::EBT_SKINNED_MESH_BUFFER:
                assert(_assetAddr->getAssetType() == asset::IAsset::ET_SUB_MESH);
                return static_cast<asset::ICPUSkinnedMeshBuffer*>(_assetAddr);
            case asset::Blob::EBT_RAW_DATA_BUFFER:
                assert(_assetAddr->getAssetType() == asset::IAsset::ET_BUFFER);
                return static_cast<asset::ICPUBuffer*>(_assetAddr);
            case asset::Blob::EBT_TEXTURE_PATH:
                assert(_assetAddr->getAssetType() == asset::IAsset::ET_IMAGE);
                return static_cast<asset::ICPUTexture*>(_assetAddr);
#endif
            default: return nullptr;
        }
    }
    static inline void insertAssetIntoCache(const SContext& _ctx, asset::IAssetLoader::IAssetLoaderOverride* _override, void* _asset, uint32_t _blobType, uint32_t _hierLvl, const std::string& _cacheKey)
    {
        // add here when more asset types will be available
        asset::IAsset* asset = nullptr;
        switch(_blobType)
        {
#ifdef OLD_MESHES
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
#endif
            case asset::Blob::EBT_RAW_DATA_BUFFER:
                asset = reinterpret_cast<asset::ICPUBuffer*>(_asset);
                break;
            default:
                assert(false);  // unhandled type
                break;
        }
        if(asset)
        {
            // drop shouldn't be performed here at all; it's done in main loading function by ctx.releaseAllButThisOne(meshBlobDataIter);
            // this is quite different from other loaders so explenation is probably wellcome
            SAssetBundle bundle(nullptr, {core::smart_refctd_ptr<asset::IAsset>(asset)});  // yes we want the extra grab
            _override->insertAssetIntoCache(bundle, _cacheKey, _ctx.inner, _hierLvl);
        }
    }

    // Compatibility functions:

    template<uint64_t IntoVersion>
    std::enable_if_t<IntoVersion /*cannot convert into version 0*/, bool> formatConversionProlog(
        SContext& _ctx,
        uint32_t& _outBlobCnt,
        BlobHeaderVn<IntoVersion - 1ull>*& _outHeaders,
        uint32_t*& _outOffsets,
        uint32_t& _outBaseOffset_from,
        uint32_t& _outBaseOffset_into)
    {
        constexpr uint64_t FromVersion = IntoVersion - 1ull;

        if(!verifyFile<BAWFileVn<FromVersion>>(_ctx))
        {
            return false;
        }
        if(!validateHeaders<BAWFileVn<FromVersion>, BlobHeaderVn<FromVersion>>(&_outBlobCnt, &_outOffsets, reinterpret_cast<void**>(&_outHeaders), _ctx))
        {
            return false;
        }

        _outBaseOffset_from = BAWFileVn<FromVersion>{{}, _outBlobCnt}.calcBlobsOffset();
        _outBaseOffset_into = BAWFileVn<IntoVersion>{{}, _outBlobCnt}.calcBlobsOffset();

        return true;
    }

    //! tuple consists of: blobCnt, from-version headers, from-version offsets, from-version baseOffset, into-version baseOffset
    template<uint64_t FromVersion>
    using CommonDataTuple = std::tuple<uint32_t, BlobHeaderVn<FromVersion>*, uint32_t*, uint32_t, uint32_t>;

    //! Converts file from format version (IntoVersion-1) into version IntoVersion (i.e. specialization with IntoVersion == 0 would be invalid).
    //! Must return _original if _original's version is IntoVersion.
    //! If new format version comes up, just increment _NBL_FORMAT_VERSION and specialize this template. All the other code will take care of itself.
    template<uint64_t IntoVersion>
    system::IFile* createConvertIntoVer_spec(SContext& _ctx, system::IFile* _original, asset::IAssetLoader::IAssetLoaderOverride* _override, const CommonDataTuple<IntoVersion - 1ull>& _common);  // here goes unpack tuple
    template<uint64_t IntoVersion>
    system::IFile* createConvertIntoVer(system::IFile* _original, asset::IAssetLoader::IAssetLoaderOverride* _override)
    {
        constexpr uint64_t FromVersion = IntoVersion - 1ull;

        SContext ctx{
            asset::IAssetLoader::SAssetLoadContext{
                asset::IAssetLoader::SAssetLoadParams{},
                _original}};
        uint32_t blobCnt{};
        BlobHeaderVn<FromVersion>* headers = nullptr;
        uint32_t* offsets = nullptr;
        uint32_t baseOffsetv_from{};
        uint32_t baseOffsetv_to{};
        if(!formatConversionProlog<IntoVersion>(ctx, blobCnt, headers, offsets, baseOffsetv_from, baseOffsetv_to))
            return nullptr;

        return createConvertIntoVer_spec<IntoVersion>(ctx, _original, _override, std::make_tuple(blobCnt, headers, offsets, baseOffsetv_from, baseOffsetv_to));
    }

    template<uint64_t... Versions>
    system::IFile* tryCreateNewestFormatVersionFile(system::IFile* _originalFile, asset::IAssetLoader::IAssetLoaderOverride* _override, std::integer_sequence<uint64_t, Versions...>)
    {
        static_assert(sizeof...(Versions) == _NBL_FORMAT_VERSION, "sizeof...(Versions) must be equal to _NBL_FORMAT_VERSION");

        using convertFuncT = system::IFile* (CBAWMeshFileLoader::*)(system::IFile*, asset::IAssetLoader::IAssetLoaderOverride*);
        convertFuncT convertFunc[_NBL_FORMAT_VERSION]{&CBAWMeshFileLoader::createConvertIntoVer<Versions + 1ull>...};

        if(!_originalFile)
            return nullptr;

        _originalFile->grab();

        uint64_t version{};
        _originalFile->seek(24u);
        _originalFile->read(&version, 8u);
        _originalFile->seek(0u);

        system::IFile* newestFormatFile = _originalFile;
        while(version != _NBL_FORMAT_VERSION)
        {
#define CALL_MEMBER_FN(object, ptrToMember) ((object).*(ptrToMember))
            system::IFile* tmp = CALL_MEMBER_FN(*this, convertFunc[version])(newestFormatFile, _override);
            newestFormatFile->drop();
            newestFormatFile = tmp;
            ++version;
            if(!newestFormatFile)
                return nullptr;
        }
        if(newestFormatFile == _originalFile)
            newestFormatFile->drop();

        return newestFormatFile;
#undef CALL_MEMBER_FN
    }

private:
    IAssetManager* m_manager;
    system::ISystem* m_system;
};

template<typename BAWFileT>
bool CBAWMeshFileLoader::verifyFile(SContext& _ctx) const
{
    char headerStr[sizeof(BAWFileT)];
    system::ISystem::future_t<uint32_t> future;
    char firstChar = 0;
    bool valid = m_system->readFile(future, _ctx.inner.mainFile, headerStr, 0, sizeof(headerStr));
    if(!valid)
        return false;
    future.get();

    const char* const headerStrPattern = BAWFileT::HEADER_STRING;
    if(strcmp(headerStr, headerStrPattern) != 0)
        return false;

    _ctx.fileVersion = ((uint64_t*)headerStr)[3];
    if(_ctx.fileVersion != BAWFileT::version)
        return false;

    return true;
}

template<typename BAWFileT, typename HeaderT>
bool CBAWMeshFileLoader::validateHeaders(uint32_t* _blobCnt, uint32_t** _offsets, void** _headers, SContext& _ctx)
{
    if(!_blobCnt)
        return false;

    _ctx.inner.mainFile->seek(sizeof(BAWFileT::fileHeader));
    if(!safeRead(_ctx.inner.mainFile, _blobCnt, sizeof(*_blobCnt)))
        return false;
    if(!safeRead(_ctx.inner.mainFile, _ctx.iv, 16))
        return false;
    uint32_t* const offsets = *_offsets = (uint32_t*)_NBL_ALIGNED_MALLOC(*_blobCnt * sizeof(uint32_t), _NBL_SIMD_ALIGNMENT);
    *_headers = _NBL_ALIGNED_MALLOC(*_blobCnt * sizeof(HeaderT), _NBL_SIMD_ALIGNMENT);
    HeaderT* const headers = (HeaderT*)*_headers;

    bool nope = false;

    if(!safeRead(_ctx.inner.mainFile, offsets, *_blobCnt * sizeof(uint32_t)))
        nope = true;
    if(!safeRead(_ctx.inner.mainFile, headers, *_blobCnt * sizeof(HeaderT)))
        nope = true;

    const uint32_t offsetRelByte = BAWFileT{{}, *_blobCnt}.calcBlobsOffset();  // num of byte to which offsets are relative
    for(uint32_t i = 0; i < *_blobCnt - 1; ++i)  // whether offsets are in ascending order none of them points past the end of file
        if(offsets[i] >= offsets[i + 1] || offsetRelByte + offsets[i] >= _ctx.inner.mainFile->getSize())
            nope = true;

    if(offsetRelByte + offsets[*_blobCnt - 1] >= _ctx.inner.mainFile->getSize())  // check the last offset
        nope = true;

    for(uint32_t i = 0; i < *_blobCnt - 1; ++i)  // whether blobs are tightly packed (do not overlays each other and there's no space bewteen any pair)
        if(offsets[i] + headers[i].effectiveSize() != offsets[i + 1])
            nope = true;

    if(offsets[*_blobCnt - 1] + headers[*_blobCnt - 1].effectiveSize() >= _ctx.inner.mainFile->getSize())  // whether last blob doesn't "go out of file"
        nope = true;

    if(nope)
    {
        _NBL_ALIGNED_FREE(offsets);
        _NBL_ALIGNED_FREE(*_headers);
        return false;
    }
    return true;
}

template<typename HeaderT>
void* CBAWMeshFileLoader::tryReadBlobOnStack(const SBlobData_t<HeaderT>& _data, SContext& _ctx, const unsigned char _pwd[16], void* _stackPtr, size_t _stackSize) const
{
    void* dst;
    if(_stackPtr && _data.header->blobSizeDecompr <= _stackSize && _data.header->effectiveSize() <= _stackSize)
        dst = _stackPtr;
    else
        dst = _NBL_ALIGNED_MALLOC(BlobHeaderVn<_NBL_FORMAT_VERSION>::calcEncSize(_data.header->blobSizeDecompr), _NBL_SIMD_ALIGNMENT);

    const bool encrypted = (_data.header->compressionType & asset::Blob::EBCT_AES128_GCM);
    const bool compressed = (_data.header->compressionType & asset::Blob::EBCT_LZ4) || (_data.header->compressionType & asset::Blob::EBCT_LZMA);

    void* dstCompressed = dst;  // ptr to mem to load possibly compressed data
    if(compressed)
        dstCompressed = _NBL_ALIGNED_MALLOC(_data.header->effectiveSize(), _NBL_SIMD_ALIGNMENT);

    _ctx.inner.mainFile->seek(_data.absOffset);
    _ctx.inner.mainFile->read(dstCompressed, _data.header->effectiveSize());

    if(!_data.header->validate(dstCompressed))
    {
#ifdef _NBL_DEBUG
        os::Printer::log("Blob validation failed!", ELL_ERROR);
#endif
        if(compressed)
        {
            _NBL_ALIGNED_FREE(dstCompressed);
            if(dst != _stackPtr)
                _NBL_ALIGNED_FREE(dst);
        }
        else if(dst != _stackPtr)
            _NBL_ALIGNED_FREE(dstCompressed);
        return NULL;
    }

    if(encrypted)
    {
#ifdef _NBL_COMPILE_WITH_OPENSSL_
        const size_t size = _data.header->effectiveSize();
        void* out = _NBL_ALIGNED_MALLOC(size, _NBL_SIMD_ALIGNMENT);
        const bool ok = asset::decAes128gcm(dstCompressed, size, out, size, _pwd, _ctx.iv, _data.header->gcmTag);
        if(dstCompressed != _stackPtr)
            _NBL_ALIGNED_FREE(dstCompressed);
        if(!ok)
        {
            if(dst != _stackPtr && dstCompressed != dst)
                _NBL_ALIGNED_FREE(dst);
            _NBL_ALIGNED_FREE(out);
#ifdef _NBL_DEBUG
            os::Printer::log("Blob decryption failed!", ELL_ERROR);
#endif
            return nullptr;
        }
        dstCompressed = out;
        if(!compressed)
            dst = dstCompressed;
#else
        if(compressed)
        {
            _NBL_ALIGNED_FREE(dstCompressed);
            if(dst != _stackPtr)
                _NBL_ALIGNED_FREE(dst);
        }
        else if(dst != _stackPtr)
            _NBL_ALIGNED_FREE(dstCompressed);
        return NULL;
#endif
    }

    if(compressed)
    {
        const uint8_t comprType = _data.header->compressionType;
        bool res = false;

        if(comprType & asset::Blob::EBCT_LZ4)
            res = decompressLz4(dst, _data.header->blobSizeDecompr, dstCompressed, _data.header->blobSize);
        else if(comprType & asset::Blob::EBCT_LZMA)
            res = decompressLzma(dst, _data.header->blobSizeDecompr, dstCompressed, _data.header->blobSize);

        _NBL_ALIGNED_FREE(dstCompressed);
        if(!res)
        {
            if(dst != _stackPtr && dst != dstCompressed)
                _NBL_ALIGNED_FREE(dst);
#ifdef _NBL_DEBUG
            os::Printer::log("Blob decompression failed!", ELL_ERROR);
#endif
            return nullptr;
        }
    }

    return dst;
}

}
}

#endif
