#ifndef __IRR_I_ASSET_LOADER_H_INCLUDED__
#define __IRR_I_ASSET_LOADER_H_INCLUDED__

#include "IAsset.h"
#include "IReadFile.h"

namespace irr { namespace asset
{

class IAssetLoader : public virtual core::IReferenceCounted
{
public:
    enum E_CACHING_FLAGS : uint64_t
    {
        ECF_CACHE_EVERYTHING = 0,
        //! master/parent is searched for in the caches, but not added to the cache if not found and loaded
        ECF_DONT_CACHE_TOP_LEVEL = 0x1ull,
        //! master/parent object is loaded without searching for it in the cache, nor adding it to the cache after the load   
        ECF_DUPLICATE_TOP_LEVEL = 0x3ull,
        //! this concerns any asset that the top level asset refers to, such as a texture
        ECF_DONT_CACHE_REFERENCES = 0x5555555555555555ull,
        //! meaning identical as to ECF_DUPLICATE_TOP_LEVEL but for any asset in the chain
        ECF_DUPLICATE_REFERENCES = 0xffffffffffffffffull
    };

    struct SAssetLoadParams
    {
        SAssetLoadParams(const size_t& _decryptionKeyLen = 0u, const uint8_t* _decryptionKey = nullptr, const E_CACHING_FLAGS& _cacheFlags = ECF_CACHE_EVERYTHING)
            : decryptionKeyLen(_decryptionKeyLen), decryptionKey(_decryptionKey), cacheFlags(_cacheFlags)
        {
        }
        size_t decryptionKeyLen;
        const uint8_t* decryptionKey;
        const E_CACHING_FLAGS cacheFlags;
    };

    //! Struct for keeping the state of the current loadoperation for safe threading
    struct SAssetLoadContext
    {
        const SAssetLoadParams params;
        io::IReadFile* mainFile;
    };

    // following could be inlined
    static E_CACHING_FLAGS ECF_DONT_CACHE_LEVEL(uint64_t N)
    {
        N *= 2ull;
        return (E_CACHING_FLAGS)(ECF_DONT_CACHE_TOP_LEVEL << N);
    }
    static E_CACHING_FLAGS ECF_DUPLICATE_LEVEL(uint64_t N)
    {
        N *= 2ull;
        return (E_CACHING_FLAGS)(ECF_DUPLICATE_TOP_LEVEL << N);
    }
    static E_CACHING_FLAGS ECF_DONT_CACHE_FROM_LEVEL(uint64_t N)
    {
        // (Criss) Shouldn't be set all DONT_CACHE bits from hierarchy numbers N-1 to 32 (64==2*32) ? Same for ECF_DUPLICATE_FROM_LEVEL below
        N *= 2ull;
        return (E_CACHING_FLAGS)(ECF_DONT_CACHE_REFERENCES << N);
    }
    static E_CACHING_FLAGS ECF_DUPLICATE_FROM_LEVEL(uint64_t N)
    {
        N *= 2ull;
        return (E_CACHING_FLAGS)(ECF_DUPLICATE_REFERENCES << N);
    }
    static E_CACHING_FLAGS ECF_DONT_CACHE_UNTIL_LEVEL(uint64_t N)
    {
        // (Criss) is this ok? Shouldn't be set all DONT_CACHE bits from hierarchy numbers 0 to N-1? Same for ECF_DUPLICATE_UNTIL_LEVEL below
        N = 64ull - N * 2ull;
        return (E_CACHING_FLAGS)(ECF_DONT_CACHE_REFERENCES >> N);
    }
    static E_CACHING_FLAGS ECF_DUPLICATE_UNTIL_LEVEL(uint64_t N)
    {
        N = 64ull - N * 2ull;
        return (E_CACHING_FLAGS)(ECF_DUPLICATE_REFERENCES >> N);
    }

    //! Override class to facilitate changing how assets are loaded
    // (Criss) Rename to IAssetLoaderCallback ?
    // (Criss) While using IAssetLoaderOverride how can i know in which hierarchy level i am?
    class IAssetLoaderOverride
    {
    protected:
        IAssetManager* m_manager;
    public:
        IAssetLoaderOverride(IAssetManager* _manager) : m_manager(_manager) {}

        // The only reason these functions are not declared static is to allow stateful overrides

        //! The most imporant overrides are the ones for caching
        virtual IAsset* findCachedAsset(const std::string& inSearchKey, const IAsset::E_TYPE* inAssetTypes, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel);

        //! Since more then one asset of the same key of the same type can exist, this function is called right after search for cached assets (if anything was found) and decides which of them is relevant.
        //! Note: this function can assume that `found` is never empty.
        inline virtual IAsset* chooseRelevantFromFound(const core::vector<IAsset*>& found, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel)
        {
            return found.front();
        }

        //! Only called when the asset was searched for, no correct asset was found
        /** Any non-nullptr asset returned here will not be added to cache,
        since the overload operates “as if” the asset was found. */
        inline virtual IAsset* handleSearchFail(const std::string& keyUsed, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel)
        {
            return nullptr;
        }

        //! Called before loading a file
        // (Criss) Whats does this one?
        inline virtual void getLoadFilename(std::string& inOutFilename, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel) {} //default do nothing

        // (Criss) Also what does this one?
        inline virtual io::IReadFile* getLoadFile(io::IReadFile* inFile, const std::string& supposedFilename, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel)
        {
            return inFile;
        }
        // I would really like to merge getLoadFilename and getLoadFile into one function!

        //! When you sometimes have different passwords for different assets
        /** \param inOutDecrKeyLen expects length of buffer `outDecrKey`, then function writes into it length of actual key.
                Write to `outDecrKey` happens only if output value of `inOutDecrKeyLen` is less or equal to input value of `inOutDecrKeyLen`.
        \param supposedFilename is the string after modification by getLoadFilename.
        \param attempt if decryption or validation algorithm supports reporting failure, you can try different key*/
        inline virtual bool getDecryptionKey(uint8_t* outDecrKey, size_t& inOutDecrKeyLen, const uint32_t& attempt, const io::IReadFile* assetsFile, const std::string& supposedFilename, const std::string& cacheKey, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel)
        {
            if (ctx.params.decryptionKeyLen <= inOutDecrKeyLen)
                memcpy(outDecrKey, ctx.params.decryptionKey, ctx.params.decryptionKeyLen);
            inOutDecrKeyLen = ctx.params.decryptionKeyLen;
            return attempt == 0u; // no failed attempts
        }

        //! Only called when the was unable to be loaded
        inline virtual IAsset* handleLoadFail(bool& outAddToCache, const io::IReadFile* assetsFile, const std::string& supposedFilename, const std::string& cacheKey, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel)
        {
            outAddToCache = false; // if you want to return a “default error asset”
            return nullptr;
        }

        //! After a successful load of an asset or sub-asset
        virtual void insertAssetIntoCache(IAsset* asset, const std::string& supposedKey, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel);
    };

public:
    //! Check if the file might be loaded by this class
    /** Check might look into the file.
    \param file File handle to check.
    \return True if file seems to be loadable. */
    virtual bool isALoadableFileFormat(io::IReadFile* _file) const = 0;

    //! Returns an array of string literals terminated by nullptr
    virtual const char** getAssociatedFileExtensions() const = 0;

    //! Returns the assets loaded by the loader
    /** Bits of the returned value correspond to each IAsset::E_TYPE
    enumeration member, and the return value cannot be 0. */
    virtual uint64_t getSupportedAssetTypesBitfield() const { return 0; }

    //! Loads an asset from an opened file, returns nullptr in case of failure.
    virtual IAsset* loadAsset(io::IReadFile* _file, const SAssetLoadParams& _params, IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) = 0;

protected:
    // accessors for loaders
    IAsset* interm_getAssetInHierarchy(IAssetManager& _mgr, io::IReadFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override);
    IAsset* interm_getAssetInHierarchy(IAssetManager& _mgr, const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override);
    IAsset* interm_getAssetInHierarchy(IAssetManager& _mgr, io::IReadFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel);
    IAsset* interm_getAssetInHierarchy(IAssetManager& _mgr, const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel);
};

}}

#endif
