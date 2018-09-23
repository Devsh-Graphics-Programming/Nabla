// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_I_ASSET_MANAGER_H_INCLUDED__
#define __IRR_I_ASSET_MANAGER_H_INCLUDED__

#include "IFileSystem.h"
#include "CConcurrentObjectCache.h"
#include "IReadFile.h"
#include "IWriteFile.h"
#include "CGeometryCreator.h"
#include "IAssetLoader.h"
#include "IAssetWriter.h"

#include <array>
#include <ostream>

#define USE_MAPS_FOR_PATH_BASED_CACHE //benchmark and choose, paths can be full system paths

namespace irr
{
namespace asset
{
	typedef scene::ICPUMesh ICPUMesh;
    typedef scene::IGPUMesh IGPUMesh;

    // (Criss) I see there's no virtuals in it, so maybe we don't need an interface/base class
    // and also there's template member function which cannot be virtual so...
	class IAssetManager
	{
        // the point of those functions is that lambdas returned by them "inherits" friendship
        friend std::function<void(IAsset*)> makeAssetGreetFunc(const IAssetManager* const _mgr);
        friend std::function<void(IAsset*)> makeAssetDisposeFunc(const IAssetManager* const _mgr);

    public:
#ifdef USE_MAPS_FOR_PATH_BASED_CACHE
        using AssetCacheType = core::CConcurrentMultiObjectCache<std::string, IAsset, std::multimap>;
#else
        using AssetCacheType = core::CConcurrentObjectCache<std::string, IAsset, std::vector>;
#endif //USE_MAPS_FOR_PATH_BASED_CACHE

    private:
        template<typename T>
        static void refCtdGreet(T* _asset) { _asset->grab(); }
        template<typename T>
        static void refCtdDispose(T* _asset) { _asset->drop(); }

        io::IFileSystem* m_fileSystem;

        //AssetCacheType m_assetCache[IAsset::ET_STANDARD_TYPES_COUNT];
        core::array<AssetCacheType, IAsset::ET_STANDARD_TYPES_COUNT> m_assetCache;

        struct {
            core::vector<IAssetLoader*> vector;
            //! The key is file extension
            core::CMultiObjectCache<std::string, IAssetLoader, std::vector> assoc{ &refCtdGreet<IAssetLoader>, &refCtdDispose<IAssetLoader> };

            void pushToVector(IAssetLoader* _loader) { 
                _loader->grab();
                vector.push_back(_loader); 
            }
            void eraseFromVector(decltype(vector)::const_iterator _loaderItr) {
                (*_loaderItr)->drop();
                vector.erase(_loaderItr);
            }
        } m_loaders;

        struct {
            struct WriterKey : public std::pair<IAsset::E_TYPE, std::string>
            {
                using Base = std::pair<IAsset::E_TYPE, std::string>;
                using Base::Base; // inherit std::pair's ctors

                bool operator<(const WriterKey& _rhs) const
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

            core::CMultiObjectCache<WriterKey, IAssetWriter, std::vector> perTypeAndFileExt{ &refCtdGreet<IAssetWriter>, &refCtdDispose<IAssetWriter> };
            core::CMultiObjectCache<IAsset::E_TYPE, IAssetWriter, std::vector> perType{ &refCtdGreet<IAssetWriter>, &refCtdDispose<IAssetWriter> };
        } m_writers;

        friend IAssetLoader::IAssetLoaderOverride; // for access to non-const findAssets
    protected:
        // (Criss) What does it do? And why return value is single pair (range i assume) if we can search multiple types at once.
        // (Criss) And assume that by IAsset::iterator it's meant to be iterators to asset cache (typename decltype(m_meshCache)::IteratorType)
        inline core::array<AssetCacheType::RangeType, IAsset::ET_STANDARD_TYPES_COUNT> findAssets(const std::string& _key, const IAsset::E_TYPE* _types = nullptr)
        {
            core::CConcurrentMultiObjectCache<std::string, int>::RangeType;
            core::array<AssetCacheType::RangeType, IAsset::ET_STANDARD_TYPES_COUNT> res(m_assetCache[0].findRange("\\")); // filling `res` with zero-ranges ('\' won't ever be a filename)
            //auto hasFoundAnything = [](const decltype(res)& _res) {
            //    for (const AssetCacheType::RangeType& range : _res)
            //        if (AssetCacheType::isNonZeroRange(range))
            //            return true;
            //    return false;
            //};

            if (_types)
            {
                uint32_t i = 0u;
                while (_types[i] != (IAsset::E_TYPE)0u)
                {
                    uint32_t typeIx = IAsset::typeFlagToIndex(_types[i]);
                    res[typeIx] = m_assetCache[typeIx].findRange(_key);
                    ++i;
                }
            }
            else
            {
                for (uint32_t i = 0u; i < IAsset::ET_STANDARD_TYPES_COUNT; ++i)
                    res[i] = m_assetCache[i].findRange(_key);
            }

            return res;
        }
    public:
        //! Constructor
        explicit IAssetManager(io::IFileSystem* _fs) :
            m_assetCache(
                asset::makeAssetGreetFunc(this),
                asset::makeAssetDisposeFunc(this)
            ),
            m_fileSystem{_fs} 
        {
            m_fileSystem->grab();
        }
        virtual ~IAssetManager()
        {
            m_fileSystem->drop();
        }

        //! Default Asset Creators (more creators can follow)
        //IMeshCreator* getDefaultMeshCreator(); //old IGeometryCreator

    public:
        IAsset* getAssetInHierarchy(io::IReadFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
        {
            const uint64_t levelFlags = _params.cacheFlags >> ((uint64_t)_hierarchyLevel * 2ull);

            if ((levelFlags & IAssetLoader::ECF_DUPLICATE_TOP_LEVEL) != IAssetLoader::ECF_DUPLICATE_TOP_LEVEL)
            {
                core::array<AssetCacheType::RangeType, IAsset::ET_STANDARD_TYPES_COUNT> found = findAssets(_file->getFileName().c_str());
                for (const auto& rng : found)
                    if (AssetCacheType::isNonZeroRange(rng))
                        return rng.first->second; // return asset in the beginning of first valid range
            }

            IAsset* asset = nullptr;
            auto capableLoadersRng = m_loaders.assoc.findRange(getFileExt(_file->getFileName()));

            for (auto loaderItr = capableLoadersRng.first; loaderItr != capableLoadersRng.second; ++loaderItr) // loaders associated with the file's extension tryout
            {
                if (loaderItr->second->isALoadableFileFormat(_file) && (asset = loaderItr->second->loadAsset(_file, _params, _override, _hierarchyLevel)))
                    break;
            }
            for (auto loaderItr = std::begin(m_loaders.vector); loaderItr != std::end(m_loaders.vector); ++loaderItr) // all loaders tryout
            {
                if ((*loaderItr)->isALoadableFileFormat(_file) && (asset = (*loaderItr)->loadAsset(_file, _params, _override, _hierarchyLevel)))
                    break;
            }

            if (asset && !(levelFlags & IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL))
            {
                asset->setNewCacheKey(_file->getFileName().c_str());
                insertAssetIntoCache(asset);
            }

            return asset;
        }
        IAsset* getAssetInHierarchy(const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
        {
            io::IReadFile* file = m_fileSystem->createAndOpenFile(_filename.c_str());
            if (!file)
                return nullptr;

            IAsset* asset = getAssetInHierarchy(file, _params, _override, _hierarchyLevel);
            file->drop();

            return asset;
        }

        //! These can be grabbed and dropped, but you must not use drop() to try to unload/release memory of a cached IAsset (which is cached if IAsset::isInAResourceCache() returns true). See IAsset::E_CACHING_FLAGS
        /** Instead for a cached asset you call IAsset::removeSelfFromCache() instead of IAsset::drop() as the cache has an internal grab of the IAsset and it will drop it on removal from cache, which will result in deletion if nothing else is holding onto the IAsset through grabs (in that sense the last drop will delete the object). */
        IAsset* getAsset(const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override = nullptr)
        {
            return getAssetInHierarchy(_filename, _params, _override, 0u);
        }
        IAsset* getAsset(io::IReadFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override = nullptr)
        {
            return getAssetInHierarchy(_file, _params, _override, 0u);
        }

        //! Does not do any loading, but tries to retrieve the asset by key if already loaded
        /** \param types if not a nullptr, then it must be a 0-terminated list. Then we only retrieve specific types, i.e. when we are only looking for texture. */
        // (Criss) Same problem as with non-const overload. Single pair and many types. What's the idea behind it?
        inline core::array<AssetCacheType::ConstRangeType, IAsset::ET_STANDARD_TYPES_COUNT> findAssets(const std::string& _key, const IAsset::E_TYPE* _types = nullptr, IAssetLoader::IAssetLoaderOverride* _override = nullptr) const
        {
            // This isn't right. TODO.
            // Problem is, the right solution would be the opposite approach (call const function from non-const one), but that would require converting const iterator into non-const ones which is impossible.
            const core::array<AssetCacheType::RangeType, IAsset::ET_STANDARD_TYPES_COUNT> nonConst = const_cast<IAssetManager*>(this)->findAssets(_key, _types);

            // Here i would probably have to call _override->handleSearchFail ? But seems like not (why handleSearchFail has SAssetLoadContext arg?). There probably should be another handleSearchFail designed to be called from here?
            // Or why is _override arg here?
        }

        //! Changes the lookup key
        inline void changeAssetKey(IAsset* _asset, const std::string& _newKey)
        {
            if (!_asset->isCached)
                _asset->setNewCacheKey(_newKey);
            else
            {
                if (m_assetCache[IAsset::typeFlagToIndex(_asset->getAssetType())].changeObjectKey(_asset, _asset->cacheKey, _newKey))
                    _asset->setNewCacheKey(_newKey);
            }
        }

        //! Insert an asset into the cache (calls the private methods of IAsset behind the scenes)
        /** \return boolean if was added into cache (no duplicate under same key found) and grab() was called on the asset. */
        bool insertAssetIntoCache(IAsset* _asset)
        {
            const uint32_t ix = IAsset::typeFlagToIndex(_asset->getAssetType());
            if (!m_assetCache[ix].insert(_asset->cacheKey, _asset))
                return false;
            return true;
        }

        //! Remove an asset from cache (calls the private methods of IAsset behind the scenes)
        bool removeAssetFromCache(IAsset* _asset) //will actually look up by asset’s key instead
        {
            const uint32_t ix = IAsset::typeFlagToIndex(_asset->getAssetType());
            return m_assetCache[ix].removeObject(_asset, _asset->cacheKey);
        }

        //! Removes all assets from the specified caches, all caches by default
        void clearAllAssetCache(const uint64_t& _assetTypeBitFlags = 0xffffffffffffffffull)
        {
            for (size_t i = 0u; i < IAsset::ET_STANDARD_TYPES_COUNT; ++i)
                if ((_assetTypeBitFlags>>i) & 1ull)
                    m_assetCache[i].clear();
        }

        //! This function frees most of the memory consumed by IAssets, but not destroying them.
        /** Keeping assets around (by their pointers) helps a lot by letting the loaders retrieve them from the cache and not load cpu objects which have been loaded, converted to gpu resources and then would have been disposed of. However each dummy object needs to have a GPU object associated with it in yet-another-cache for use when we convert CPU objects to GPU objects.*/
        // (Criss) This function doesnt create gpu objects from cpu ones, right?
        // Also: why **ToEmptyCacheHandle**? Next one: cpu_t must be asset type (derivative of IAsset) and gpu_t is whatever corresponding GPU type?
        // How i understand it as for now: 
        //  we're passing asset as `object` parameter and as `objectToAssociate` we pass corresponding gpu object created from the cpu one completely outside asset-pipeline
        //  the cpu object (asset) frees all memory but object itself remains as key for cpu->gpu assoc cache.
        //Also what if we want to cache gpu stuff but don't want to touch cpu stuff? (e.g. to change one meshbuffer in mesh)
        template<cpu_t, gpu_t>
        void convertCPUObjectToEmptyCacheHandle(cpu_t* object, const gpu_t* objectToAssociate);

        //! Writing an asset
        /** Compression level is a number between 0 and 1 to signify how much storage we are trading for writing time or quality, this is a non-linear scale and has different meanings and results with different asset types and writers. */
        bool writeAsset(const std::string& _filename, const IAssetWriter::SAssetWriteParams& _params, IAssetWriter::IAssetWriterOverride* _override = nullptr)
        {
            io::IWriteFile* file = m_fileSystem->createAndWriteFile(_filename.c_str());
            bool res = writeAsset(file, _params, _override);
            file->drop();
            return res;
        }
        bool writeAsset(io::IWriteFile* _file, const IAssetWriter::SAssetWriteParams& _params, IAssetWriter::IAssetWriterOverride* _override = nullptr)
        {
            auto capableWritersRng = m_writers.perTypeAndFileExt.findRange({_params.rootAsset->getAssetType(), getFileExt(_file->getFileName())});

            for (auto it = capableWritersRng.first; it != capableWritersRng.second; ++it)
                if (it->second->writeAsset(_file, _params, _override))
                    return true;
            return false;
        }

        // Asset Loaders [FOLLOWING ARE NOT THREAD SAFE]
        uint32_t getAssetLoaderCount() { return m_loaders.vector.size(); }
        IAssetLoader* getAssetLoader(const uint32_t& _idx)
        {
            return m_loaders.vector[_idx];
        }

        //! @returns 0xdeadbeefu on failure or 0-based index on success.
        uint32_t addAssetLoader(IAssetLoader* _loader)
        {
            // there's no way it ever fails, so no 0xdeadbeef return
            const char** exts = _loader->getAssociatedFileExtensions();
            size_t extIx = 0u;
            while (const char* ext = exts[extIx++])
                m_loaders.assoc.insert(ext, _loader);
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
                m_loaders.assoc.removeObject(_loader, ext);
        }
        void removeAssetLoader(const uint32_t& _idx)
        {
            m_loaders.eraseFromVector(std::begin(m_loaders.vector)+_idx); // todo, i don't see a way to remove from assoc cache knowing only index in vector
        }

        // Asset Writers [FOLLOWING ARE NOT THREAD SAFE]
        uint32_t getAssetWriterCount() { return m_writers.perType.getSize(); } // todo.. well, it's not really writer count.. but rather type<->writer association count

        IAssetWriter* getAssetWriter(const uint32_t& _idx); //todo: how can index be understood here?

        uint32_t addAssetWriter(IAssetWriter* _writer)
        {
            const uint64_t suppTypes = _writer->getSupportedAssetTypesBitfield();
            const char** exts = _writer->getAssociatedFileExtensions();
            for (uint32_t i = 0u; i < IAsset::ET_STANDARD_TYPES_COUNT; ++i)
            {
                const IAsset::E_TYPE type = IAsset::E_TYPE(1u << i);
                if ((suppTypes>>i) & 1u)
                    m_writers.perType.insert(type, _writer);
                size_t extIx = 0u;
                while (const char* ext = exts[extIx++])
                    m_writers.perTypeAndFileExt.insert({type, ext}, _writer);
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
        void removeAssetWriter(const uint32_t& _idx); // TODO what is _idx here?

        void dumpCachesDebug(std::ostream& _outs) const
        {
            for (uint32_t i = 0u; i < IAsset::ET_STANDARD_TYPES_COUNT; ++i)
            {
                _outs << "Asset cache (asset type " << (1u << i) << "):\n";
                const size_t sz = m_assetCache[i].getSize();
                typename AssetCacheType::MutablePairType* storage = new typename AssetCacheType::MutablePairType[sz];
                m_assetCache[i].outputAll(sz, storage, nullptr);
                for (uint32_t j = 0u; j < sz; ++j)
                    _outs << "\tKey: " << storage[j].first << ", Value: " << static_cast<void*>(storage[j].second) << '\n';
            }
            _outs << "Loaders vector:\n";
            for (const auto& ldr : m_loaders.vector)
                _outs << '\t' << static_cast<void*>(ldr) << '\n';
            _outs << "Loaders assoc cache:\n";
            for (const auto& ldr : m_loaders.assoc)
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
