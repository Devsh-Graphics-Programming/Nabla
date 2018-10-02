// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __C_BAW_MESH_FILE_LOADER_H_INCLUDED__
#define __C_BAW_MESH_FILE_LOADER_H_INCLUDED__


#include "IAssetLoader.h"
#include "ISceneManager.h"
#include "IFileSystem.h"
#include "IMesh.h"
#include "CBAWFile.h"
#include "CBlobsLoadingManager.h"
#include "SSkinMeshBuffer.h"

namespace irr { namespace scene
{

class CBAWMeshFileLoader : public asset::IAssetLoader
{
private:
	struct SBlobData
	{
		core::BlobHeaderV0* header;
		size_t absOffset; // absolute
		void* heapBlob;
		mutable bool validated;
        uint32_t hierarchyLvl;

		SBlobData(core::BlobHeaderV0* _hd=NULL, size_t _offset=0xdeadbeefdeadbeef) : header(_hd), absOffset(_offset), heapBlob(NULL), validated(false) {}
		~SBlobData() { free(heapBlob); }
		bool validate() const {
			validated = false;
			return validated ? true : (validated = (heapBlob && header->validate(heapBlob)));
		}
	private:
		// a bit dangerous to leave it copyable but until c++11 I have to to be able to store it in unordered_map
		// SBlobData(const SBlobData&) {}
		SBlobData& operator=(const SBlobData&) = delete;
	};

	struct SContext
	{
		void releaseLoadedObjects()
		{
			for (auto it = createdObjs.begin(); it != createdObjs.end(); ++it)
				loadingMgr.releaseObj(blobs[it->first].header->blobType, it->second);
		}
		void releaseAllButThisOne(core::unordered_map<uint64_t, SBlobData>::iterator _thisIt)
		{
			const uint64_t theHandle = _thisIt != blobs.end() ? _thisIt->second.header->handle : 0;
			for (auto it = createdObjs.begin(); it != createdObjs.end(); ++it)
			{
				if (it->first != theHandle)
					loadingMgr.releaseObj(blobs[it->first].header->blobType, it->second);
			}
		}

        asset::IAssetLoader::SAssetLoadContext inner;
		uint64_t fileVersion;
		core::unordered_map<uint64_t, SBlobData> blobs;
		core::unordered_map<uint64_t, void*> createdObjs;
		core::CBlobsLoadingManager loadingMgr;
		unsigned char iv[16];
	};

protected:
	//! Destructor
	virtual ~CBAWMeshFileLoader();

public:
	//! Constructor
	CBAWMeshFileLoader(scene::ISceneManager* _sm, io::IFileSystem* _fs);


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
        const bool res = verifyFile(ctx);
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
	bool verifyFile(SContext& _ctx) const;
	//! Loads and checks correctness of offsets and headers. Also let us know blob count.
	/** @returns true if everythings ok, false otherwise. */
	bool validateHeaders(uint32_t* _blobCnt, uint32_t** _offsets, void** _headers, SContext& _ctx);

	//! Reads `_size` bytes to `_buf` from `_file`, but previously checks whether file is big enough and returns true/false appropriately.
	bool safeRead(io::IReadFile* _file, void* _buf, size_t _size) const;

	//! Reads blob to memory on stack or allocates sufficient amount on heap if provided stack storage was not big enough.
	/** @returns `_stackPtr` if blob was read to it or pointer to malloc'd memory otherwise.*/
	void* tryReadBlobOnStack(const SBlobData& _data, SContext& _ctx, const unsigned char pwd[16], void* _stackPtr=NULL, size_t _stackSize=0) const;

	bool decompressLzma(void* _dst, size_t _dstSize, const void* _src, size_t _srcSize) const;
	bool decompressLz4(void* _dst, size_t _dstSize, const void* _src, size_t _srcSize) const;

    inline std::string genSubAssetCacheKey(const std::string& _rootKey, uint64_t _handle) const { return _rootKey + "?" + std::to_string(_handle); }

    static inline void* toAddrUsedByBlobsLoadingMgr(asset::IAsset* _assetAddr, uint32_t _blobType)
    {
        // add here when more asset types will be available
        switch (_blobType)
        {
        case core::Blob::EBT_MESH:
        case core::Blob::EBT_SKINNED_MESH:
            assert(_assetAddr->getAssetType()==asset::IAsset::ET_MESH);
            return static_cast<ICPUMesh*>(_assetAddr);
        case core::Blob::EBT_MESH_BUFFER:
            assert(_assetAddr->getAssetType()==asset::IAsset::ET_SUB_MESH);
            return static_cast<ICPUMeshBuffer*>(_assetAddr);
        case core::Blob::EBT_SKINNED_MESH_BUFFER:
            assert(_assetAddr->getAssetType()==asset::IAsset::ET_SUB_MESH);
            return static_cast<SCPUSkinMeshBuffer*>(_assetAddr);
        case core::Blob::EBT_RAW_DATA_BUFFER:
            assert(_assetAddr->getAssetType()==asset::IAsset::ET_BUFFER);
            return static_cast<core::ICPUBuffer*>(_assetAddr);
        case core::Blob::EBT_TEXTURE_PATH:
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
        case core::Blob::EBT_MESH:
        case core::Blob::EBT_SKINNED_MESH:
            asset = reinterpret_cast<ICPUMesh*>(_asset);
            break;
        case core::Blob::EBT_MESH_BUFFER:
            asset = reinterpret_cast<ICPUMeshBuffer*>(_asset);
            break;
        case core::Blob::EBT_SKINNED_MESH_BUFFER:
            asset = reinterpret_cast<SCPUSkinMeshBuffer*>(_asset);
            break;
        case core::Blob::EBT_RAW_DATA_BUFFER:
            asset = reinterpret_cast<core::ICPUBuffer*>(_asset);
            break;
        case core::Blob::EBT_TEXTURE_PATH:
            asset = reinterpret_cast<asset::ICPUTexture*>(_asset);
            break;
        }
        if (asset)
        {
            _override->setAssetCacheKey(asset, _cacheKey, _ctx.inner, _hierLvl);
            _override->insertAssetIntoCache(asset, _ctx.inner, _hierLvl);
            asset->drop();
        }
    }

private:
	scene::ISceneManager* m_sceneMgr;
	io::IFileSystem* m_fileSystem;
};

}} // irr::scene

#endif
