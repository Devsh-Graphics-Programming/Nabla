// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_I_ASSET_MANAGER_H_INCLUDED__
#define __IRR_I_ASSET_MANAGER_H_INCLUDED__

#include "IFileSystem.h"
#include "CConcurrentObjectCache.h"
#include "IReadFile.h"
#include "IWriteFile.h"
#include "IAssetLoader.h"
#include "IAssetWriter.h"
#include "irr/core/Types.h"

#include <array>
#include <ostream>

#define USE_MAPS_FOR_PATH_BASED_CACHE //benchmark and choose, paths can be full system paths

namespace irr
{
namespace asset
{
    class IGeometryCreator;
    class IMeshManipulator;

    std::function<void(IAsset*)> makeAssetGreetFunc(const IAssetManager* const _mgr);
    std::function<void(IAsset*)> makeAssetDisposeFunc(const IAssetManager* const _mgr);

	class IAssetManager
	{
        // the point of those functions is that lambdas returned by them "inherits" friendship
        friend std::function<void(IAsset*)> makeAssetGreetFunc(const IAssetManager* const _mgr);
        friend std::function<void(IAsset*)> makeAssetDisposeFunc(const IAssetManager* const _mgr);

    public:
#ifdef USE_MAPS_FOR_PATH_BASED_CACHE
        using AssetCacheType = core::CConcurrentMultiObjectCache<std::string, IAsset, std::multimap>;
#else
        using AssetCacheType = core::CConcurrentMultiObjectCache<std::string, IAsset, std::vector>;
#endif //USE_MAPS_FOR_PATH_BASED_CACHE

        using CpuGpuCacheType = core::CConcurrentObjectCache<const IAsset*, core::IReferenceCounted>;

    private:
        struct WriterKey
        {
            IAsset::E_TYPE first;
            std::string second;

            inline bool operator<(const WriterKey& _rhs) const
            {
                if (first != _rhs.first)
                    return first < _rhs.first;
                return second < _rhs.second;
            }

            inline friend std::ostream& operator<<(std::ostream& _outs, const WriterKey& _item)
            {
                return _outs << "{ " << static_cast<uint64_t>(_item.first) << ", " << _item.second << " }";
            }
        };

        template<typename T>
        static void refCtdGreet(T* _asset) { _asset->grab(); }
        template<typename T>
        static void refCtdDispose(T* _asset) { _asset->drop(); }

        io::IFileSystem* m_fileSystem;
        IAssetLoader::IAssetLoaderOverride m_defaultLoaderOverride;

        std::array<AssetCacheType*, IAsset::ET_STANDARD_TYPES_COUNT> m_assetCache;
        std::array<CpuGpuCacheType*, IAsset::ET_STANDARD_TYPES_COUNT> m_cpuGpuCache;

        struct Loaders {
            Loaders() : perFileExt{&refCtdGreet<IAssetLoader>, &refCtdDispose<IAssetLoader>} {}

            core::vector<IAssetLoader*> vector;
            //! The key is file extension
            core::CMultiObjectCache<std::string, IAssetLoader, std::vector> perFileExt;

            void pushToVector(IAssetLoader* _loader) {
                _loader->grab();
                vector.push_back(_loader);
            }
            void eraseFromVector(decltype(vector)::const_iterator _loaderItr) {
                (*_loaderItr)->drop();
                vector.erase(_loaderItr);
            }
        } m_loaders;

        struct Writers {
            Writers() : perTypeAndFileExt{&refCtdGreet<IAssetWriter>, &refCtdDispose<IAssetWriter>}, perType{&refCtdGreet<IAssetWriter>, &refCtdDispose<IAssetWriter>} {}

            core::CMultiObjectCache<WriterKey, IAssetWriter, std::vector> perTypeAndFileExt;
            core::CMultiObjectCache<IAsset::E_TYPE, IAssetWriter, std::vector> perType;
        } m_writers;

        friend class IAssetLoader;
        friend class IAssetLoader::IAssetLoaderOverride; // for access to non-const findAssets

        IGeometryCreator* m_geometryCreator;
        IMeshManipulator* m_meshManipulator;
        // called as a part of constructor only
        void initializeMeshTools();
        void dropMeshTools();

    public:
        //! Constructor
        explicit IAssetManager(io::IFileSystem* _fs) :
            m_fileSystem{_fs},
            m_defaultLoaderOverride{nullptr}
        {
            initializeMeshTools();

            for (size_t i = 0u; i < m_assetCache.size(); ++i)
                m_assetCache[i] = new AssetCacheType(asset::makeAssetGreetFunc(this), asset::makeAssetDisposeFunc(this));
            for (size_t i = 0u; i < m_cpuGpuCache.size(); ++i)
                m_cpuGpuCache[i] = new CpuGpuCacheType(&refCtdGreet<core::IReferenceCounted>, &refCtdDispose<core::IReferenceCounted>);
            m_fileSystem->grab();
            m_defaultLoaderOverride = IAssetLoader::IAssetLoaderOverride{this};
        }
        /*
        IAssetManager(const IAssetManager&) = delete;
        IAssetManager(IAssetManager&&) = delete;
        IAssetManager& operator=(const IAssetManager&) = delete;
        IAssetManager& operator=(IAssetManager&& _other)
        {
            std::swap(m_loaders, _other.m_loaders);
            std::swap(m_writers, _other.m_writers);
            std::swap(m_assetCache, _other.m_assetCache);
            std::swap(m_fileSystem, _other.m_fileSystem);
        }
        */
        virtual ~IAssetManager()
        {
            for (size_t i = 0u; i < m_assetCache.size(); ++i)
                if (m_assetCache[i])
                    delete m_assetCache[i];

            core::vector<typename CpuGpuCacheType::MutablePairType> buf;
            for (size_t i = 0u; i < m_cpuGpuCache.size(); ++i)
            {
                if (m_cpuGpuCache[i])
                {
                    size_t sizeToReserve{};
                    m_cpuGpuCache[i]->outputAll(sizeToReserve, nullptr);
                    buf.resize(sizeToReserve);
                    m_cpuGpuCache[i]->outputAll(sizeToReserve, buf.data());
                    for (auto& pair : buf)
                        pair.first->drop(); // drop keys (CPU "empty cache handles")
                    delete m_cpuGpuCache[i]; // drop on values (GPU objects) will be done by cache's destructor
                }
            }
            for (auto ldr : m_loaders.vector)
                ldr->drop();
            if (m_fileSystem)
                m_fileSystem->drop();
            dropMeshTools();
        }

        const IGeometryCreator* getGeometryCreator() const;
        const IMeshManipulator* getMeshManipulator() const;

    protected:
        IAsset* getAssetInHierarchy(io::IReadFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override)
        {
            IAssetLoader::SAssetLoadContext ctx{_params, _file};

            std::string filename = _file ? _file->getFileName().c_str() : _supposedFilename;
            io::IReadFile* file = _override->getLoadFile(_file, filename, ctx, _hierarchyLevel);
            filename = file ? file->getFileName().c_str() : _supposedFilename;

            const uint64_t levelFlags = _params.cacheFlags >> ((uint64_t)_hierarchyLevel * 2ull);

            IAsset* asset = nullptr;
            if ((levelFlags & IAssetLoader::ECF_DUPLICATE_TOP_LEVEL) != IAssetLoader::ECF_DUPLICATE_TOP_LEVEL)
            {
                core::vector<IAsset*> found = findAssets(filename);
                if (found.size())
                    return _override->chooseRelevantFromFound(found, ctx, _hierarchyLevel);
                else if (asset = _override->handleSearchFail(filename, ctx, _hierarchyLevel))
                    return asset;
            }

            auto capableLoadersRng = m_loaders.perFileExt.findRange(getFileExt(filename.c_str()));

            for (auto loaderItr = capableLoadersRng.first; loaderItr != capableLoadersRng.second; ++loaderItr) // loaders associated with the file's extension tryout
            {
                if (loaderItr->second->isALoadableFileFormat(file) && (asset = loaderItr->second->loadAsset(file, _params, _override, _hierarchyLevel)))
                    break;
            }
            for (auto loaderItr = std::begin(m_loaders.vector); !asset && loaderItr != std::end(m_loaders.vector); ++loaderItr) // all loaders tryout
            {
                if ((*loaderItr)->isALoadableFileFormat(file) && (asset = (*loaderItr)->loadAsset(file, _params, _override, _hierarchyLevel)))
                    break;
            }

            if (asset && !(levelFlags & IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL))
            {
                _override->insertAssetIntoCache(asset, filename, ctx, _hierarchyLevel);
                asset->drop(); // drop ownership after transfering it to cache container
            }
            else if (!asset)
            {
                bool addToCache;
                asset = _override->handleLoadFail(addToCache, file, filename, filename, ctx, _hierarchyLevel);
                if (asset && addToCache)
                    insertAssetIntoCache(asset);
            }

            return asset;
        }
        IAsset* getAssetInHierarchy(const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override)
        {
            IAssetLoader::SAssetLoadContext ctx{_params, nullptr};

            std::string filename = _filename;
            _override->getLoadFilename(filename, ctx, _hierarchyLevel);
            io::IReadFile* file = m_fileSystem->createAndOpenFile(filename.c_str());

            IAsset* asset = getAssetInHierarchy(file, _filename, _params, _hierarchyLevel, _override);

            if (file)
                file->drop();

            return asset;
        }

        IAsset* getAssetInHierarchy(io::IReadFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel)
        {
            return getAssetInHierarchy(_file, _supposedFilename, _params, _hierarchyLevel, &m_defaultLoaderOverride);
        }

        IAsset* getAssetInHierarchy(const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel)
        {
            return getAssetInHierarchy(_filename, _params, _hierarchyLevel, &m_defaultLoaderOverride);
        }

    public:
        //! These can be grabbed and dropped, but you must not use drop() to try to unload/release memory of a cached IAsset (which is cached if IAsset::isInAResourceCache() returns true). See IAsset::E_CACHING_FLAGS
        /** Instead for a cached asset you call IAsset::removeSelfFromCache() instead of IAsset::drop() as the cache has an internal grab of the IAsset and it will drop it on removal from cache, which will result in deletion if nothing else is holding onto the IAsset through grabs (in that sense the last drop will delete the object). */
        IAsset* getAsset(const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override)
        {
            return getAssetInHierarchy(_filename, _params, 0u, _override);
        }
        IAsset* getAsset(io::IReadFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override)
        {
            return getAssetInHierarchy(_file, _supposedFilename, _params,  0u, _override);
        }

        IAsset* getAsset(const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params)
        {
            return getAsset(_filename, _params, &m_defaultLoaderOverride);
        }

        IAsset* getAsset(io::IReadFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params)
        {
            return getAsset(_file, _supposedFilename, _params, &m_defaultLoaderOverride);
        }

        inline bool findAssets(size_t& _inOutStorageSize, IAsset** _out, const std::string& _key, const IAsset::E_TYPE* _types = nullptr) const
        {
            size_t availableSize = _inOutStorageSize;
            _inOutStorageSize = 0u;
            bool res = true;
            if (_types)
            {
                uint32_t i = 0u;
                while ((_types[i] != (IAsset::E_TYPE)0u) && (availableSize > 0u))
                {
                    uint32_t typeIx = IAsset::typeFlagToIndex(_types[i]);
                    size_t readCnt = availableSize;
                    res = m_assetCache[typeIx]->findAndStoreRange(_key, readCnt, _out);
                    availableSize -= readCnt;
                    _inOutStorageSize += readCnt;
                    _out += readCnt;
                    ++i;
                }
            }
            else
            {
                for (uint32_t typeIx = 0u; typeIx < IAsset::ET_STANDARD_TYPES_COUNT; ++typeIx)
                {
                    size_t readCnt = availableSize;
                    res = m_assetCache[typeIx]->findAndStoreRange(_key, readCnt, _out);
                    availableSize -= readCnt;
                    _inOutStorageSize += readCnt;
                    _out += readCnt;
                }
            }
            return res;
        }
        inline core::vector<IAsset*> findAssets(const std::string& _key, const IAsset::E_TYPE* _types = nullptr) const
        {
            size_t reqSz = 0u;
            if (_types)
            {
                uint32_t i = 0u;
                while ((_types[i] != (IAsset::E_TYPE)0u))
                {
                    const uint32_t typeIx = IAsset::typeFlagToIndex(_types[i]);
                    reqSz += m_assetCache[typeIx]->getSize();
                    ++i;
                }
            }
            else
            {
                for (const auto& cache : m_assetCache)
                    reqSz += cache->getSize();
            }
            core::vector<IAsset*> res(reqSz);
            findAssets(reqSz, res.data(), _key, _types);
            res.resize(reqSz);
            return res;
        }

        //! Changes the lookup key
        inline void changeAssetKey(IAsset* _asset, const std::string& _newKey)
        {
            if (!_asset->isCached)
                _asset->setNewCacheKey(_newKey);
            else
            {
                if (m_assetCache[IAsset::typeFlagToIndex(_asset->getAssetType())]->changeObjectKey(_asset, _asset->cacheKey, _newKey))
                    _asset->setNewCacheKey(_newKey);
            }
        }

        //! Insert an asset into the cache (calls the private methods of IAsset behind the scenes)
        /** \return boolean if was added into cache (no duplicate under same key found) and grab() was called on the asset. */
        bool insertAssetIntoCache(IAsset* _asset)
        {
            const uint32_t ix = IAsset::typeFlagToIndex(_asset->getAssetType());
            return m_assetCache[ix]->insert(_asset->cacheKey, _asset);
        }

        //! Remove an asset from cache (calls the private methods of IAsset behind the scenes)
        bool removeAssetFromCache(IAsset* _asset) //will actually look up by asset’s key instead
        {
            const uint32_t ix = IAsset::typeFlagToIndex(_asset->getAssetType());
            return m_assetCache[ix]->removeObject(_asset, _asset->cacheKey);
        }

        //! Removes all assets from the specified caches, all caches by default
        void clearAllAssetCache(const uint64_t& _assetTypeBitFlags = 0xffffffffffffffffull)
        {
            for (size_t i = 0u; i < IAsset::ET_STANDARD_TYPES_COUNT; ++i)
                if ((_assetTypeBitFlags>>i) & 1ull)
                    m_assetCache[i]->clear();
        }

        //! This function frees most of the memory consumed by IAssets, but not destroying them.
        /** Keeping assets around (by their pointers) helps a lot by letting the loaders retrieve them from the cache and not load cpu objects which have been loaded, converted to gpu resources and then would have been disposed of. However each dummy object needs to have a GPU object associated with it in yet-another-cache for use when we convert CPU objects to GPU objects.*/
        void convertAssetToEmptyCacheHandle(IAsset* _asset, core::IReferenceCounted* _gpuObject)
        {
			const uint32_t ix = IAsset::typeFlagToIndex(_asset->getAssetType());
            _asset->grab();
            _asset->convertToDummyObject();
            m_cpuGpuCache[ix]->insert(_asset, _gpuObject);
        }

        core::IReferenceCounted* findGPUObject(const IAsset* _asset)
        {
			const uint32_t ix = IAsset::typeFlagToIndex(_asset->getAssetType());
            core::IReferenceCounted* storage[1];
            size_t storageSz = 1u;
            m_cpuGpuCache[ix]->findAndStoreRange(_asset, storageSz, storage);
            if (storageSz > 0u)
                return storage[0];
            return nullptr;
        }

		//! Removes one GPU object matched to an IAsset.
        bool removeCachedGPUObject(const IAsset* _asset, core::IReferenceCounted* _gpuObject)
        {
			const uint32_t ix = IAsset::typeFlagToIndex(_asset->getAssetType());
			bool success = m_cpuGpuCache[ix]->removeObject(_gpuObject,_asset);
			_asset->drop();
			return success;
        }

		// we need a removeCachedGPUObjects(const IAsset* _asset) but CObjectCache.h needs a `removeAllAssociatedObjects(const Key& _key)`

        //! Writing an asset
        /** Compression level is a number between 0 and 1 to signify how much storage we are trading for writing time or quality, this is a non-linear scale and has different meanings and results with different asset types and writers. */
        bool writeAsset(const std::string& _filename, const IAssetWriter::SAssetWriteParams& _params, IAssetWriter::IAssetWriterOverride* _override)
        {
            IAssetWriter::IAssetWriterOverride defOverride;
            if (!_override)
                _override = &defOverride;

            io::IWriteFile* file = m_fileSystem->createAndWriteFile(_filename.c_str());
            bool res = writeAsset(file, _params, _override);
            file->drop();
            return res;
        }
        bool writeAsset(io::IWriteFile* _file, const IAssetWriter::SAssetWriteParams& _params, IAssetWriter::IAssetWriterOverride* _override)
        {
			if (!_file)
				return false;

            IAssetWriter::IAssetWriterOverride defOverride;
            if (!_override)
                _override = &defOverride;

            auto capableWritersRng = m_writers.perTypeAndFileExt.findRange({_params.rootAsset->getAssetType(), getFileExt(_file->getFileName())});

            for (auto it = capableWritersRng.first; it != capableWritersRng.second; ++it)
                if (it->second->writeAsset(_file, _params, _override))
                    return true;
            return false;
        }
        bool writeAsset(const std::string& _filename, const IAssetWriter::SAssetWriteParams& _params)
        {
            return writeAsset(_filename, _params, nullptr);
        }
        bool writeAsset(io::IWriteFile* _file, const IAssetWriter::SAssetWriteParams& _params)
        {
            return writeAsset(_file, _params, nullptr);
        }

        // Asset Loaders [FOLLOWING ARE NOT THREAD SAFE]
        uint32_t getAssetLoaderCount() { return m_loaders.vector.size(); }

        //! @returns 0xdeadbeefu on failure or 0-based index on success.
        uint32_t addAssetLoader(IAssetLoader* _loader)
        {
            // there's no way it ever fails, so no 0xdeadbeef return
            const char** exts = _loader->getAssociatedFileExtensions();
            size_t extIx = 0u;
            while (const char* ext = exts[extIx++])
                m_loaders.perFileExt.insert(ext, _loader);
            m_loaders.pushToVector(_loader);
            return m_loaders.vector.size()-1u;
        }
        void removeAssetLoader(IAssetLoader* _loader)
        {
            m_loaders.eraseFromVector(
                std::find(std::begin(m_loaders.vector), std::end(m_loaders.vector), _loader)
            );
            const char** exts = _loader->getAssociatedFileExtensions();
            size_t extIx = 0u;
            while (const char* ext = exts[extIx++])
                m_loaders.perFileExt.removeObject(_loader, ext);
        }

        // Asset Writers [FOLLOWING ARE NOT THREAD SAFE]
        uint32_t getAssetWriterCount() { return m_writers.perType.getSize(); } // todo.. well, it's not really writer count.. but rather type<->writer association count

        void addAssetWriter(IAssetWriter* _writer)
        {
            const uint64_t suppTypes = _writer->getSupportedAssetTypesBitfield();
            const char** exts = _writer->getAssociatedFileExtensions();
            for (uint32_t i = 0u; i < IAsset::ET_STANDARD_TYPES_COUNT; ++i)
            {
                const IAsset::E_TYPE type = IAsset::E_TYPE(1u << i);
                if ((suppTypes>>i) & 1u)
                {
                    m_writers.perType.insert(type, _writer);
                    size_t extIx = 0u;
                    while (const char* ext = exts[extIx++])
                        m_writers.perTypeAndFileExt.insert({type, ext}, _writer);
                }
            }
        }
        void removeAssetWriter(IAssetWriter* _writer)
        {
            const uint64_t suppTypes = _writer->getSupportedAssetTypesBitfield();
            const char** exts = _writer->getAssociatedFileExtensions();
            size_t extIx = 0u;
            for (uint32_t i = 0u; i < IAsset::ET_STANDARD_TYPES_COUNT; ++i)
            {
                if ((suppTypes >> i) & 1u)
                {
                    const IAsset::E_TYPE type = IAsset::E_TYPE(1u << i);
                    m_writers.perType.removeObject(_writer, type);
                    while (const char* ext = exts[extIx++])
                        m_writers.perTypeAndFileExt.removeObject(_writer, {type, ext});
                }
            }
        }

        void dumpDebug(std::ostream& _outs) const
        {
            for (uint32_t i = 0u; i < IAsset::ET_STANDARD_TYPES_COUNT; ++i)
            {
                _outs << "Asset cache (asset type " << (1u << i) << "):\n";
                size_t sz = m_assetCache[i]->getSize();
                typename AssetCacheType::MutablePairType* storage = new typename AssetCacheType::MutablePairType[sz];
                m_assetCache[i]->outputAll(sz, storage);
                for (uint32_t j = 0u; j < sz; ++j)
                    _outs << "\tKey: " << storage[j].first << ", Value: " << static_cast<void*>(storage[j].second) << '\n';
                delete[] storage;
            }
            _outs << "Loaders vector:\n";
            for (const auto& ldr : m_loaders.vector)
                _outs << '\t' << static_cast<void*>(ldr) << '\n';
            _outs << "Loaders per-file-ext cache:\n";
            for (const auto& ldr : m_loaders.perFileExt)
                _outs << "\tKey: " << ldr.first << ", Value: " << ldr.second << '\n';
            _outs << "Writers per-asset-type cache:\n";
            for (const auto& wtr : m_writers.perType)
                _outs << "\tKey: " << static_cast<uint64_t>(wtr.first) << ", Value: " << static_cast<void*>(wtr.second) << '\n';
            _outs << "Writers per-asset-type-and-file-ext cache:\n";
            for (const auto& wtr : m_writers.perTypeAndFileExt)
                _outs << "\tKey: " << wtr.first << ", Value: " << static_cast<void*>(wtr.second) << '\n';
        }

    private:
        static inline std::string getFileExt(const io::path& _filename)
        {
            int32_t dot = _filename.findLast('.');
            return _filename
                .subString(dot+1, _filename.size()-dot-1)
                .make_lower()
                .c_str();
        }

        // for greet/dispose lambdas for asset caches so we don't have to make another friend decl.
        inline void setAssetCached(IAsset* _asset, bool _val) const { _asset->isCached = _val; }
	};
}
}

#endif
