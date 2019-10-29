#ifndef __IRR_I_ASSET_LOADER_H_INCLUDED__
#define __IRR_I_ASSET_LOADER_H_INCLUDED__

#include "irr/core/core.h"
#include "IFileSystem.h"

#include "IAsset.h"
#include "IReadFile.h"

namespace irr { namespace asset
{

//! A class automating process of loading Assets from resources, eg. files
/**
	Every Asset must be loaded by a particular class derived from IAssetLoader.
	These classes must be registered with IAssetManager::addAssetLoader() which will 
	add it to the list of loaders (grab return 0-based index) or just not register 
	the loader upon failure (don’t grab and return 0xdeadbeefu).

	The loading is impacted by caching and resource duplication flags, defined as IAssetLoader::E_CACHING_FLAGS.

	The flag having an impact on loading an Asset is a bitfield with 2 bits per level,
	so the enums provide are some useful constants. Different combinations are valid as well, so
	
	\code{.cpp}
    IAssetLoader::SAssetLoadParams params;
	params.cacheFlags = static_cast<E_CACHING_FLAGS>(ECF_DONT_CACHE_TOP_LEVEL << 4ull);
    //synonymous to:
    params.cacheFlags = ECF_DONT_CACHE_LEVEL(2);
    //where ECF_DONT_CACHE_LEVEL() is a utility function.
	\endcode

	Means that anything on level 2 will not get cached (top is 0, but we have shifted for 4 bits,
	where 2 bits represent one single level, so we've been on second level).
    Notice that loading process can be seen as a chain. When you're loading a mesh, it can references a submesh.
    Submesh can reference graphics pipeline and descriptor set. Descriptor set can reference, for example, textures.
    Hierarchy level is distance in such chain/tree from Root Asset (the one you asked for by calling IAssetManager::getAsset()) and the currently loaded Asset (needed by Root Asset).
    
	When the class derived from IAssetLoader is added, its put once on an 
	vector<IAssetLoader*> and once on an multimap<std::string,IAssetLoader*> 
	inside the IAssetManager for every associated file extension it reports.

	The loaders are tried in the order they were registered per file extensions, 
	and later in the global order in case of needing to fallback to examining files.

	An IAssetLoader can only be removed/deregistered by its original pointer or global loader index.

    @see IAssetLoader::SAssetLoadParams
	@see IReferenceCounted::grab()
	@see IAsset
	@see IAssetManager
	@see IAssetWriter
*/

class IAssetLoader : public virtual core::IReferenceCounted
{
public:

	//! Caching and resource duplication flags 
	/**
		They have an impact on loading an Asset.

		E_CACHING_FLAGS::ECF_CACHE_EVERYTHING is default, means that asset you can find in previously cached asset will be just returned.
		There may occour that you can't, so it will be loaded and added to the cache before returning
		E_CACHING_FLAGS::ECF_DONT_CACHE_TOP_LEVEL means that master/parent is searched for in the caches, 
		but not added to the cache if not found and loaded
		E_CACHING_FLAGS::ECF_DUPLICATE_TOP_LEVEL means that master/parent object is loaded without searching
		for it in the cache, nor adding it to the cache after the load
		E_CACHING_FLAGS::ECF_DONT_CACHE_REFERENCES means that it concerns any asset that the top level asset refers to, such as a texture
		E_CACHING_FLAGS::ECF_DUPLICATE_REFERENCES means almost the same as E_CACHING_FLAGS::ECF_DUPLICATE_TOP_LEVEL but for any asset in the chain
	*/
    enum E_CACHING_FLAGS : uint64_t
    {
        ECF_CACHE_EVERYTHING = 0,
        ECF_DUPLICATE_REFERENCES = 0xffffffffffffffffull
    };


	//! Parameter flags for a loader
	/**
		These are extra flags for an `IAssetLoader` to take into account when loading the asset.

		E_LOADER_PARAMETER_FLAGS::ELPF_NONE is default and means that there is nothing to perform.
		E_LOADER_PARAMETER_FLAGS::ELPF_RIGHT_HANDED_MESHES specifies that a mesh will be flipped in such
		a way that it'll look correctly in right-handed camera system. If it isn't set, compatibility with 
		left-handed coordinate camera is assumed.
		E_LOADER_PARAMETER_FLAGS::ELPF_DONT_COMPILE_GLSL means that GLSL won't be compiled to SPIR-V if it is loaded or generated.
	*/

	enum E_LOADER_PARAMETER_FLAGS : uint64_t
	{
		ELPF_NONE = 0,											//!< default value, it doesn't do anything
		ELPF_RIGHT_HANDED_MESHES = 0x1,							//!< specifies that a mesh will be flipped in such a way that it'll look correctly in right-handed camera system
		ELPF_DONT_COMPILE_GLSL = 0x2							//!< it states that GLSL won't be compiled to SPIR-V if it is loaded or generated						
	};

	//! Struct storing important data used for Asset loading process
	/**
		Struct stores a key decryptionKey for potentially encrypted files, it is used to decrypt them. You can find an usage
		example in CBAWMeshFileLoader .cpp file. Since decryptionKey is a pointer, size must be specified 
		for iterating through key properly and decryptionKeyLen stores it.
		Current flags set by user that defines rules during loading process are stored in cacheFlags.

		@see CBAWMeshFileLoader
		@see E_CACHING_FLAGS
		@see E_LOADER_PARAMETER_FLAGS
	*/
    struct SAssetLoadParams
    {
        SAssetLoadParams(	size_t _decryptionKeyLen = 0u, const uint8_t* _decryptionKey = nullptr,
							E_CACHING_FLAGS _cacheFlags = ECF_CACHE_EVERYTHING,
							const char* _relativeDir = nullptr, const E_LOADER_PARAMETER_FLAGS& _loaderFlags = ELPF_NONE) :
				decryptionKeyLen(_decryptionKeyLen), decryptionKey(_decryptionKey),
				cacheFlags(_cacheFlags), relativeDir(_relativeDir), loaderFlags(_loaderFlags)
        {
        }

        	size_t decryptionKeyLen;			 		//!< The size of decryptionKey
        	const uint8_t* decryptionKey;			 	//!< The key is used to decrypt potentially encrypted files
        	const E_CACHING_FLAGS cacheFlags;	 		//!< Flags defining rules during loading process
		    const char* relativeDir;
		    const E_LOADER_PARAMETER_FLAGS loaderFlags;	//!< Flags having an impact on extraordinary tasks during loading process
    };

    //! Struct for keeping the state of the current loadoperation for safe threading
	/**
		Important data used for Asset loading process is stored by params.
		Also a path to Asset data file is specified, stored by mainFile. You can store path
		to file as an absolute path or a relative path, flexibility is provided.

		@see SAssetLoadParams
	*/
    struct SAssetLoadContext
    {
		SAssetLoadContext(const SAssetLoadParams& _params, io::IReadFile* _mainFile) : params(_params), mainFile(_mainFile) {}

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
        N = 64ull - N * 2ull;
        return (E_CACHING_FLAGS)(ECF_DONT_CACHE_REFERENCES >> N);
    }
    static E_CACHING_FLAGS ECF_DUPLICATE_UNTIL_LEVEL(uint64_t N)
    {
        N = 64ull - N * 2ull;
        return (E_CACHING_FLAGS)(ECF_DUPLICATE_REFERENCES >> N);
    }

	//! Class for user to override functions to facilitate changing the way assets are loaded
	/**
		User may override those functions to get more control on some process, but default implementations are provided.
		All asset loaders must defer to the override, especially when loading dependant assets via other loaders.
		It handles such operations like finding Assets cached so far, inserting them to cache, getting path to
		file with Asset data, handling predefined opeartions if Assets searching fails or loading them, etc.
	*/

    class IAssetLoaderOverride
    {
    protected:
        IAssetManager* m_manager;
		io::IFileSystem* m_filesystem;
    public:
		IAssetLoaderOverride(IAssetManager* _manager);

        // The only reason these functions are not declared static is to allow stateful overrides

        //! The most imporant overrides are the ones for caching
        virtual SAssetBundle findCachedAsset(const std::string& inSearchKey, const IAsset::E_TYPE* inAssetTypes, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel);

        //! Since more then one asset of the same key of the same type can exist, this function is called right after search for cached assets (if anything was found) and decides which of them is relevant.
        //! Note: this function can assume that `found` is never empty.
        inline virtual SAssetBundle chooseRelevantFromFound(const core::vector<SAssetBundle>& found, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel)
        {
            return found.front();
        }

        //! Only called when the asset was searched for, no correct asset was found
        /** Any non-nullptr asset returned here will not be added to cache,
        since the overload operates “as if” the asset was found. */
        inline virtual SAssetBundle handleSearchFail(const std::string& keyUsed, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel)
        {
            return {};
        }

        //! Called before loading a file 
		/**
			\param inOutFilename is a path to file Asset data needs to correspond with.
			A path could be relative to a different folder than initially as we go down the hierarchy level when loading each dependent resource.
			The override decides how to resolve a local path or even a URL into a "proper" filename.
			It could even be used to return a fake path to i.e. trick the IAssetManager into thinking file has a specific extension (and tries the correct loader first). 
			\param ctx provides data required for loading process.
			\param hierarchyLevel specifies how deep are we being in some referenced-struct-data in a file,
			but it is more like a stack counter.

			Expected behaviour is that Asset loading will get called recursively. For instance mesh needs material, material needs texture, etc.
			GetLoadFilename() could be called separately for each dependent resource from deeper recursions in the loading stack.
			So inOutFilename is provided, and if path to a texture needed by a material is required - inOutFilename will store it after calling the function.

			More information about hierarchy level you can find at IAssetLoader description.

			@see IAssetLoader
			@see SAssetLoadContext
		*/
        inline virtual void getLoadFilename(std::string& inOutFilename, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel)
		{
			if (!ctx.params.relativeDir)
				return;
			// try compute absolute path
			std::string relative = ctx.params.relativeDir+inOutFilename;
			if (m_filesystem->existFile(relative.c_str()))
			{
				inOutFilename = relative;
				return;
			}
			// otherwise it was already absolute
		}

        //! This function can be used to swap out the actually opened (or unknown unopened file if `inFile` is nullptr) file for a different one.
		/** Especially useful if you've used some sort of a fake path and the file won't load from that path just via `io::IFileSystem` . */
        inline virtual io::IReadFile* getLoadFile(io::IReadFile* inFile, const std::string& supposedFilename, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel)
        {
            return inFile;
        }

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
        inline virtual SAssetBundle handleLoadFail(bool& outAddToCache, const io::IReadFile* assetsFile, const std::string& supposedFilename, const std::string& cacheKey, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel)
        {
            outAddToCache = false; // if you want to return a “default error asset”
            return SAssetBundle();
        }

        //! After a successful load of an asset or sub-asset
        //TODO change name
        virtual void insertAssetIntoCache(SAssetBundle& asset, const std::string& supposedKey, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel);
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
    virtual SAssetBundle loadAsset(io::IReadFile* _file, const SAssetLoadParams& _params, IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) = 0;

protected:
    // accessors for loaders
    SAssetBundle interm_getAssetInHierarchy(IAssetManager* _mgr, io::IReadFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override);
    SAssetBundle interm_getAssetInHierarchy(IAssetManager* _mgr, const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override);
    SAssetBundle interm_getAssetInHierarchy(IAssetManager* _mgr, io::IReadFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel);
    SAssetBundle interm_getAssetInHierarchy(IAssetManager* _mgr, const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel);
};

}}

#endif
