// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_ASSET_MANAGER_H_INCLUDED__
#define __NBL_ASSET_I_ASSET_MANAGER_H_INCLUDED__

#include <array>
#include <ostream>

#include "nbl/core/declarations.h"
#include "nbl/system/path.h"
#include "CConcurrentObjectCache.h"

#include "nbl/system/ISystem.h"
#include "nbl/system/IFile.h"
#include "nbl/asset/interchange/IAssetLoader.h"
#include "nbl/asset/interchange/IAssetWriter.h"

#include "nbl/asset/utils/CCompilerSet.h"
#include "nbl/asset/utils/IGeometryCreator.h"


#define USE_MAPS_FOR_PATH_BASED_CACHE //benchmark and choose, paths can be full system paths

namespace nbl::asset
{

class IAssetManager;


std::function<void(SAssetBundle&)> makeAssetGreetFunc(const IAssetManager* const _mgr);
std::function<void(SAssetBundle&)> makeAssetDisposeFunc(const IAssetManager* const _mgr);


//! Class responsible for handling loading of assets from file system or other resources
/**
	It provides a loading, writing and creation functionality that is almost thread-safe.
	There is one issue with threading, starting loading the same asset at the exact same time 
	may end up with two copies in the cache.

	IAssetManager performs caching of CPU assets associated with resource handles such as names, 
	filenames, UUIDs. However there are separate caches for each asset type.

	@see IAsset

*/
class NBL_API2 IAssetManager : public core::IReferenceCounted
{
        // the point of those functions is that lambdas returned by them "inherits" friendship
        friend std::function<void(SAssetBundle&)> makeAssetGreetFunc(const IAssetManager* const _mgr);
        friend std::function<void(SAssetBundle&)> makeAssetDisposeFunc(const IAssetManager* const _mgr);

    public:
#ifdef USE_MAPS_FOR_PATH_BASED_CACHE
        using AssetCacheType = core::CConcurrentMultiObjectCache<std::string, SAssetBundle, std::multimap>;
#else
        using AssetCacheType = core::CConcurrentMultiObjectCache<std::string, IAssetBundle, std::vector>;
#endif //USE_MAPS_FOR_PATH_BASED_CACHE

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

        core::smart_refctd_ptr<system::ISystem> m_system;
        IAssetLoader::IAssetLoaderOverride m_defaultLoaderOverride;

        std::array<AssetCacheType*, IAsset::ET_STANDARD_TYPES_COUNT> m_assetCache;

        struct Loaders {
            Loaders() : perFileExt{&refCtdGreet<IAssetLoader>, &refCtdDispose<IAssetLoader>} {}

            core::vector<core::smart_refctd_ptr<IAssetLoader> > vector;
            //! The key is file extension
            core::CMultiObjectCache<std::string, IAssetLoader*, std::vector> perFileExt;

            void pushToVector(core::smart_refctd_ptr<IAssetLoader>&& _loader)
			{
                vector.push_back(std::move(_loader));
            }
            void eraseFromVector(decltype(vector)::const_iterator _loaderItr)
			{
                vector.erase(_loaderItr);
            }
        } m_loaders;

        struct Writers {
            Writers() : perTypeAndFileExt{&refCtdGreet<IAssetWriter>, &refCtdDispose<IAssetWriter>}, perType{&refCtdGreet<IAssetWriter>, &refCtdDispose<IAssetWriter>} {}

            core::CMultiObjectCache<WriterKey, IAssetWriter*, std::vector> perTypeAndFileExt;
            core::CMultiObjectCache<IAsset::E_TYPE, IAssetWriter*, std::vector> perType;
        } m_writers;

        friend class IAssetLoader;
        friend class IAssetLoader::IAssetLoaderOverride; // for access to non-const findAssets

        core::smart_refctd_ptr<IGeometryCreator> m_geometryCreator;
        core::smart_refctd_ptr<IMeshManipulator> m_meshManipulator;
        core::smart_refctd_ptr<CCompilerSet> m_compilerSet;
        // called as a part of constructor only
        void initializeMeshTools();

    public:
        //! Constructor
        explicit IAssetManager(core::smart_refctd_ptr<system::ISystem>&& system, core::smart_refctd_ptr<CCompilerSet>&& compilerSet = nullptr) :
            m_system(std::move(system)),
            m_compilerSet(std::move(compilerSet)),
            m_defaultLoaderOverride(this)
        {
            initializeMeshTools();

            for (size_t i = 0u; i < m_assetCache.size(); ++i)
                m_assetCache[i] = new AssetCacheType(asset::makeAssetGreetFunc(this), asset::makeAssetDisposeFunc(this));

            insertBuiltinAssets();
			addLoadersAndWriters();
        }

		inline system::ISystem* getSystem() const { return m_system.get(); }

        const IGeometryCreator* getGeometryCreator() const;
        IMeshManipulator* getMeshManipulator();
        CCompilerSet* getCompilerSet() const { return m_compilerSet.get(); }

    protected:
		virtual ~IAssetManager()
		{
			for (size_t i = 0u; i < m_assetCache.size(); ++i)
				if (m_assetCache[i])
					delete m_assetCache[i];
		}

		//TODO change name (its multiple assets not just one)
        //! _supposedFilename is filename as it was, not touched by loader override with _override->getLoadFilename()
		/**
			Attempts to fetch an Asset Bundle. Valid loaders for certain extension are searched. So if we were to handle .gltf file extension, specified
			loaders dealing with it would be fetched. If none of those loaders can't deal with the file, then one more tryout is performed, so a function
			iterates through all available loaders regardless of supported file extensions they deal with, trying them all one by one on the file, counting on that one might just work.

            If (_params.cacheFlags & ECF_DONT_CACHE_TOP_LEVEL)==ECF_DONT_CACHE_TOP_LEVEL, returned bundle is not being cached.
            (_params.cacheFlags & ECF_DUPLICATE_TOP_LEVEL)==ECF_DUPLICATE_TOP_LEVEL implies behaviour with ECF_DONT_CACHE_TOP_LEVEL, but Asset is not searched for in the cache (loaders tryout always happen).
			With no flags (for top hierarchy level) given, Asset is looked for in the cache (whether it is already loaded), or - if not found - added to the cache just after getting loaded.
            Empty bundle is returned if no loader could load the Asset.

			Take a look on @param _hierarchyLevel.
            Hierarchy level is distance in such chain/tree from Root Asset (the one you asked for by calling IAssetManager::getAsset()) and the currently loaded Asset (needed by Root Asset).
            Calling getAssetInHierarchy()  with _hierarchyLevel=0 is identical to calling getAsset().

			For more details about hierarchy levels see IAssetLoader.

			There is a term in reference to above - \bDowngrade\b. 
			You can find Downgrades in Assets definitions. For instance ICPUMesh::IMAGEVIEW_HIERARCHYLEVELS_BELOW.
			The syntax is simple - \b(Current level + Downgrade)\b. You ought to pass just like that a expression to _hierarchyLevel.
			For instance you can pass (topHierarchyLevel + ICPUMesh::IMAGEVIEW_HIERARCHYLEVELS_BELOW), when expecting to load a ICPUImageView from a file that is needed by the ICPUMesh currently being loaded by an IAssetLoader.

            @see IAssetLoader::SAssetLoadParams
			@see IAssetLoader
			@see SAssetBundle
		*/
        SAssetBundle getAssetInHierarchy_impl(system::IFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override);

        //TODO change name
        SAssetBundle getAssetInHierarchy_impl(const std::string& _filePath, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override)
        {
            IAssetLoader::SAssetLoadContext ctx(_params, nullptr);

            system::path filePath = _filePath;
            _override->getLoadFilename(filePath, m_system.get(), ctx, _hierarchyLevel);
            if (!m_system->exists(filePath,system::IFile::ECF_READ))
            {
                filePath = _params.workingDirectory/filePath;
                _override->getLoadFilename(filePath, m_system.get(), ctx, _hierarchyLevel);
            }
            
            system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
            m_system->createFile(future, filePath, system::IFile::ECF_READ);
            if (auto file=future.acquire())
                return getAssetInHierarchy_impl(file->get(), filePath.string(), ctx.params, _hierarchyLevel, _override);
            return SAssetBundle(0);
        }


        inline SAssetBundle getAssetInHierarchy(system::IFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override)
        {
            if (!_override)
                _override = &m_defaultLoaderOverride;
            return getAssetInHierarchy_impl(_file, _supposedFilename, _params, _hierarchyLevel, _override);
        }
        inline SAssetBundle getAssetInHierarchy(const std::string& _filePath, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override)
        {
            if (!_override)
                _override = &m_defaultLoaderOverride;
            return getAssetInHierarchy_impl(_filePath, _params, _hierarchyLevel, _override);
        }

    public:
        //! These can be grabbed and dropped, but you must not use drop() to try to unload/release memory of a cached IAsset (which is cached if IAsset::isInAResourceCache() returns true). See IAsset::E_CACHING_FLAGS
        /** Instead for a cached asset you call IAsset::removeSelfFromCache() instead of IAsset::drop() as the cache has an internal grab of the IAsset and it will drop it on removal from cache, which will result in deletion if nothing else is holding onto the IAsset through grabs (in that sense the last drop will delete the object). */
        //TODO change name
        inline SAssetBundle getAsset(const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override=nullptr)
        {
            return getAssetInHierarchy(_filename, _params, 0u, _override);
        }
        //TODO change name
        inline SAssetBundle getAsset(system::IFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override=nullptr)
        {
            return getAssetInHierarchy(_file, _supposedFilename, _params,  0u, _override);
        }

        //TODO change name
		//! Check whether Assets exist in cache using a key and optionally their types
		/*
			\param _inOutStorageSize holds beginning size of Assets. Note that it changes,
			but if we expect 5 objects, it will hold finally (5 * sizeOfAsset) or \bless\b since 
			operation may fail.
			\param _out is a pointer that specifies an adress that new SAssetBundle will be copied to.
			\param _key stores a key used for Assets searching.
			\param _types stores null-terminated Asset types for better performance while searching.

			If types of Assets are specified, task is easier and more performance is ensured, 
			because Assets are being searched only in certain cache using a key (cache that matches
			with Assets type). If there aren't any types specified, Assets are being search in whole cache.

			Found Assets are being copied to _out.

			If Assets exist, true is returned - otherwise false.

			@see SAssetBundle
			@see IAsset::E_TYPE
		*/
        inline bool findAssets(size_t& _inOutStorageSize, SAssetBundle* _out, const std::string& _key, const IAsset::E_TYPE* _types = nullptr) const
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
        
		//! It finds Assets and returnes all found. 
        inline core::smart_refctd_dynamic_array<SAssetBundle> findAssets(const std::string& _key, const IAsset::E_TYPE* _types = nullptr) const
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
			auto res = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<SAssetBundle> >(reqSz);
            findAssets(reqSz, res->data(), _key, _types);
            auto retval = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<SAssetBundle> >(reqSz);
            std::move(res->begin(), res->begin()+reqSz, retval->begin());
			return retval;
        }

        //! Changes the lookup key
        //TODO change name
        inline void changeAssetKey(SAssetBundle& _asset, const std::string& _newKey)
        {
            _asset.setNewCacheKey(_newKey);
            m_assetCache[IAsset::typeFlagToIndex(_asset.getAssetType())]->changeObjectKey(_asset, _asset.getCacheKey(), _newKey);
        }

        //! Insert an asset into the cache (calls the private methods of IAsset behind the scenes)
        /** Keeping assets around and caching them (by their name-like keys) helps a lot by letting the loaders
        retrieve them from the cache and not load cpu objects again, which have been loaded before.
        \return boolean if was added into cache (no duplicate under same key found) and grab() was called on the asset. */
        //TODO change name (its not just one asset)
        bool insertAssetIntoCache(SAssetBundle& _asset, const bool _mutability=true)
        {
            const uint32_t ix = IAsset::typeFlagToIndex(_asset.getAssetType());
            for (auto ass : _asset.getContents())
                setAssetMutability(ass.get(), _mutability);
            return m_assetCache[ix]->insert(_asset.getCacheKey(), _asset);
        }

        //! Remove an asset from cache (calls the private methods of IAsset behind the scenes)
        //TODO change key
        bool removeAssetFromCache(SAssetBundle& _asset) //will actually look up by asset's key instead
        {
            const uint32_t ix = IAsset::typeFlagToIndex(_asset.getAssetType());
            return m_assetCache[ix]->removeObject(_asset, _asset.getCacheKey());
        }

        //! Removes all assets from the specified caches, all caches by default
        void clearAllAssetCache(const uint64_t& _assetTypeBitFlags = 0xffffffffffffffffull)
        {
            for (size_t i = 0u; i < IAsset::ET_STANDARD_TYPES_COUNT; ++i)
                if ((_assetTypeBitFlags>>i) & 1ull)
                    m_assetCache[i]->clear();
        }

        //! Writing an asset
        /** Compression level is a number between 0 and 1 to signify how much storage we are trading for writing time or quality, this is a non-linear 
		scale and has different meanings and results with different asset types and writers. */
        bool writeAsset(const std::string& _filename, const IAssetWriter::SAssetWriteParams& _params, IAssetWriter::IAssetWriterOverride* _override)
        {
            IAssetWriter::IAssetWriterOverride defOverride;
            if (!_override)
                _override = &defOverride;

            system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
            m_system->createFile(future, (_params.workingDirectory.generic_string()+_filename).c_str(), system::IFile::ECF_WRITE);
            if (auto file=future.acquire())
                return writeAsset(file->get(), _params, _override);
            return false;
        }
        bool writeAsset(system::IFile* _file, const IAssetWriter::SAssetWriteParams& _params, IAssetWriter::IAssetWriterOverride* _override)
        {
			if (!_file || !_params.rootAsset)
				return false;

            IAssetWriter::IAssetWriterOverride defOverride;
            if (!_override)
                _override = &defOverride;
            auto ext = system::extension_wo_dot(_file->getFileName());
            auto capableWritersRng = m_writers.perTypeAndFileExt.findRange({_params.rootAsset->getAssetType(), ext});

            for (auto& writer : capableWritersRng)
            if (writer.second->writeAsset(_file, _params, _override))
                return true;
            return false;
        }
        bool writeAsset(const std::string& _filename, const IAssetWriter::SAssetWriteParams& _params)
        {
            return writeAsset(_filename, _params, nullptr);
        }
        bool writeAsset(system::IFile* _file, const IAssetWriter::SAssetWriteParams& _params)
        {
            return writeAsset(_file, _params, nullptr);
        }

        // Asset Loaders [FOLLOWING ARE NOT THREAD SAFE]
        uint32_t getAssetLoaderCount() { return static_cast<uint32_t>(m_loaders.vector.size()); }

        //! @returns 0xdeadbeefu on failure or 0-based index on success.
        uint32_t addAssetLoader(core::smart_refctd_ptr<IAssetLoader>&& _loader)
        {
            // there's no way it ever fails, so no 0xdeadbeef return
            const char** exts = _loader->getAssociatedFileExtensions();
            size_t extIx = 0u;
            while (const char* ext = exts[extIx++])
                m_loaders.perFileExt.insert(ext, _loader.get());
            m_loaders.pushToVector(std::move(_loader));
            return static_cast<uint32_t>(m_loaders.vector.size())-1u;
        }
        void removeAssetLoader(IAssetLoader* _loader)
        {
            m_loaders.eraseFromVector(
                std::find_if(std::begin(m_loaders.vector), std::end(m_loaders.vector), [_loader](const core::smart_refctd_ptr<IAssetLoader>& a)->bool { return a.get()==_loader; })
            );
            const char** exts = _loader->getAssociatedFileExtensions();
            size_t extIx = 0u;
            while (const char* ext = exts[extIx++])
                m_loaders.perFileExt.removeObject(_loader, ext);
        }

        // Asset Writers [FOLLOWING ARE NOT THREAD SAFE]
        uint32_t getAssetWriterCount() { return static_cast<uint32_t>(m_writers.perType.getSize()); } // todo.. well, it's not really writer count.. but rather type<->writer association count

        void addAssetWriter(core::smart_refctd_ptr<IAssetWriter>&& _writer)
        {
            const uint64_t suppTypes = _writer->getSupportedAssetTypesBitfield();
            const char** exts = _writer->getAssociatedFileExtensions();
            for (uint32_t i = 0u; i < IAsset::ET_STANDARD_TYPES_COUNT; ++i)
            {
                const IAsset::E_TYPE type = IAsset::E_TYPE(1u << i);
                if ((suppTypes>>i) & 1u)
                {
                    m_writers.perType.insert(type, _writer.get());
                    size_t extIx = 0u;
                    while (const char* ext = exts[extIx++])
                        m_writers.perTypeAndFileExt.insert({type, ext}, _writer.get());
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
            /*
            TODO rework these prints
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
            */
            _outs << "Loaders vector:\n";
            for (const auto& ldr : m_loaders.vector)
                _outs << '\t' << static_cast<void*>(ldr.get()) << '\n';
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

    protected:
        bool insertBuiltinAssetIntoCache(SAssetBundle& _asset)
        {
            return insertAssetIntoCache(_asset,false);
        }


        // for greet/dispose lambdas for asset caches so we don't have to make another friend decl.
        //TODO change name
        inline void setAssetCached(SAssetBundle& _asset, bool _val) const { _asset.setCached(_val); }

        inline void setAssetMutability(IAsset* _asset, const bool _val) const { _asset->m_mutable = _val; }

		//
		void addLoadersAndWriters();

        void insertBuiltinAssets();
};


}

#endif
