// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef _I_ASSET_MANAGER_H_INCLUDED_
#define _I_ASSET_MANAGER_H_INCLUDED_

// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IFileSystem.h"
#include "CConcurrentObjectCache.h"
#include "IReadFile.h"
#include "IWriteFile.h"
#include "CGeometryCreator.h"
#include "IAssetLoader.h"
#include "IAssetWriter.h"

#define USE_MAPS_FOR_PATH_BASED_CACHE //benchmark and choose, paths can be full system paths

namespace irr
{
namespace asset
{
    // (Criss) Do we need those typedefs?
	typedef scene::ICPUMesh ICPUMesh;
    typedef scene::IGPUMesh IGPUMesh;

    // (Criss) I see there's no virtuals in it, so maybe we don't need an interface/base class
    // and also there's template member function which cannot be virtual so...
	class IAssetManager
	{
        // (Criss) And same caches for another asset types. Could be array of caches if ET_IMPLEMENTATION_SPECIFIC_METADATA wasn't =64 (and so we could have ET_COUNT)
        // cached by filename/custom string key
        // should it be multi-cache?
#ifdef USE_MAPS_FOR_PATH_BASED_CACHE
        core::CConcurrentObjectCache<std::string, IAsset, core::map> m_meshCache;
#else
        core::CConcurrentObjectCache<std::string, IAsset, core::vector> m_meshCache;
#endif //USE_MAPS_FOR_PATH_BASED_CACHE
        //...

        struct {
            std::vector<IAssetLoader*> vector;
            core::CMultiObjectCache<std::string, IAssetLoader, core::multimap> assoc;
        } m_loaders;

        struct {
            struct WriterKey : public std::pair<E_TYPE, std::string>
            {
                bool operator<(const WriterKey& _rhs)
                {
                    if (first != _rhs.first)
                        return first < _rhs.first;
                    return second < _rhs.second;
                }
            };

            core::CMultiObjectCache<WriterKey, IAssetWriter, core::multimap> perTypeAndFileExt;
            core::CMultiObjectCache<E_TYPE, IAssetWriter, core::multimap> perType;
        } m_writers;

        friend IAssetLoader::IAssetLoaderOverride; // for access to non-const findAssets
    protected:
        // (Criss) What does it do? And why return value is single pair (range i assume) if we can search multiple types at once.
        // (Criss) And assume that by IAsset::iterator it's meant to be iterators to asset cache (typename decltype(m_meshCache)::IteratorType)
        std::pair<IAsset::iterator, IAsset::iterator> findAssets(const std::string& _key, const E_TYPE* types = nullptr);
    public:
        //! Default Asset Creators (more creators can follow)
        IMeshCreator* getDefaultMeshCreator(); //old IGeometryCreator

        //! These can be grabbed and dropped, but you must not use drop() to try to unload/release memory of a cached IAsset (which is cached if IAsset::isInAResourceCache() returns true). See IAsset::E_CACHING_FLAGS
        /** Instead for a cached asset you call IAsset::removeSelfFromCache() instead of IAsset::drop() as the cache has an internal grab of the IAsset and it will drop it on removal from cache, which will result in deletion if nothing else is holding onto the IAsset through grabs (in that sense the last drop will delete the object). */
        IAsset* getAsset(const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override = nullptr);
        IAsset* getAsset(io::IReadFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override = nullptr);

        //! Does not do any loading, but tries to retrieve the asset by key if already loaded
        /** \param types if not a nullptr, then it must be a null terminated list. Then we only retrieve specific types, i.e. when we are only looking for texture. */
        // (Criss) Same problem as with non-const overload. Single pair and many types. What's the idea behind it?
        std::pair<IAsset::const_iterator, IAsset::const_iterator> findAssets(const std::string& _key, const E_TYPE* types = nullptr, IAssetLoader::IAssetLoaderOverride* _override = nullptr);

        //! Changes the lookup key
        void changeAssetKey(IAsset* asset, std::string& newKey);

        //! Insert an asset into the cache (calls the private methods of IAsset behind the scenes)
        /** \return boolean if was added into cache (no duplicate under same key found) and grab() was called on the asset. */
        bool insertAssetIntoCache(IAsset* asset);

        //! Remove an asset from cache (calls the private methods of IAsset behind the scenes)
        void removeAssetFromCache(IAsset* asset); //will actually look up by asset’s key instead

                                                  //! Removes all assets from the specified caches, all caches by default
        void clearAllAssetCache(const uint64_t& assetTypeBitFlags = 0xffffffffffffffffull);

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
        bool writeAsset(IAsset* _asset, const std::string& _filename, const E_WRITER_FLAGS& _flags = EWF_NONE, const float& compressionLevel = 0.f, const size_t& decryptionKeyLen = 0, const uint8_t* decryptionKey = nullptr, IAssetWriter::IAssetWriterOverride* _override = nullptr);
        bool writeAsset(IAsset* _asset, io::IWriteFile* _file, const E_WRITER_FLAGS& _flags = EWF_NONE, const float& compressionLevel = 0.f, const size_t& decryptionKeyLen = 0, const uint8_t* decryptionKey = nullptr, IAssetWriter::IAssetWriterOverride* _override = nullptr);

        //! Asset Loaders [FOLLOWING ARE NOT THREAD SAFE]
        uint32_t getAssetLoaderCount();
        IAssetLoader* getAssetLoader(const uint32_t& _idx);

        uint32_t addAssetLoader(IAssetLoader* _loader); //returns 0xdeadbeefu on failure
        void removeAssetLoader(IAssetLoader* _loader);
        void removeAssetLoader(const uint32_t& _idx);

        //! Asset Writers [FOLLOWING ARE NOT THREAD SAFE]
        uint32_t getAssetWriterCount();
        IAssetWriter* getAssetWriter(const uint32_t& _idx);
        uint32_t addAssetWriter(IAssetWriter* _writer); //returns 0xdeadbeefu on failure
        void removeAssetWriter(IAssetWriter* _writer);
        void removeAssetWriter(const uint32_t& _idx);
	};
}
}

#endif
