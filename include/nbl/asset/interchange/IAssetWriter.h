// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_ASSET_WRITER_H_INCLUDED__
#define __NBL_ASSET_I_ASSET_WRITER_H_INCLUDED__

#include "nbl/system/IFile.h"
#include "nbl/asset/IAsset.h"

namespace nbl
{
namespace asset
{

//! Writing flags
/**
	They have an impact on writing (saving) an Asset.
	Take into account that a writer may not support all of those flags.
	For instance, if there is a PNG or JPG writer, it won't write encrypted images.

	@see IAssetWriter::getSupportedFlags()
	@see IAssetWriter::getForcedFlags()

	E_WRITER_FLAGS::EWF_NONE means that there aren't writer flags (default).
	E_WRITER_FLAGS::EWF_COMPRESSED means that it has to write in a way that consumes less disk space if possible.
	E_WRITER_FLAGS::EWF_ENCRYPTED means that it has to write in encrypted way if possible.
	E_WRITER_FLAGS::EWF_BINARY means that it has to write in binary format rather than text if possible.
*/
enum E_WRITER_FLAGS : uint32_t
{
    EWF_NONE = 0u,						//!< No writer flags (default writer settings)
    EWF_COMPRESSED = 1u<<0u,			//!< Write in a way that consumes less disk space if possible
    EWF_ENCRYPTED = 1u<<1u,				//!< Write encrypted if possible
    //! write in binary format rather than text if possible
    EWF_BINARY = 1u << 2u,

    //!< specifies the incoming orientation of loaded mesh we want to write. Flipping will be performed if needed in dependency of format extension orientation	
    EWF_MESH_IS_RIGHT_HANDED = 1u << 3u
};

//! A class that defines rules during Asset-writing (saving) process
/**
	Some assets can be saved to file (or memory file) by classes derived from IAssetWriter.
	These classes must be registered with IAssetManager::addAssetWriter() which will add it
	to the list of writers (grab return 0-based index) or just not register the writer upon 
	failure (don�t grab and return 0xdeadbeefu).

	The writing is impacted by writing flags, defined as E_WRITER_FLAGS.

	Remember that loaded Asset doesn't actually know how it was created in reference to certain file extension it was called from,
	so if you loaded an Asset from .baw, you can save it to another file with different extension if a valid writer is provided.

	When the class derived from IAssetWriter is added, its put once on a 
	std::multimap<std::pair<IAsset::E_TYPE,std::string>,IAssetWriter*> for every 
	asset type and file extension it supports, and once on a std::multimap<IAsset::E_TYPE,IAssetWriter*> 
	for every asset type it supports.

	The writers are tried in the order they were registered per-asset-type-and-file-extension, 
	or in the per-asset-type order in the case of writing straight to file without a name.

	An IAssetWriter can only be removed/deregistered by its original pointer or global loader index.

	@see IAsset
	@see IAssetManager
	@see IAssetLoader
	@see E_WRITER_FLAGS
*/
class IAssetWriter : public virtual core::IReferenceCounted
{
public:
	//! Struct storing important data used for Asset writing process
	/**
		Struct stores an Asset on which entire writing process is based. It also stores decryptionKey for file encryption. 
		You can find an usage example in CBAWMeshFileLoader .cpp file. Since decryptionKey is a pointer, size must be specified 
		for iterating through key properly and encryptionKeyLen stores it.
		Current flags set by user that defines rules during writing process are stored in flags.
		Compression level dependent on entire Asset size reserved for writing is stored in compressionLevel.
		The more size it has, the more compression level is. Indeed user data is specified in userData and
		it holds writer-dependets parameters. It is usually a struct provided by a writer author.

		@see CBAWMeshFileLoader
		@see E_WRITER_FLAGS
	*/
    struct SAssetWriteParams
    {
        SAssetWriteParams(IAsset* _asset, const E_WRITER_FLAGS& _flags = EWF_NONE, const float& _compressionLevel = 0.f, const size_t& _encryptionKeyLen = 0, const uint8_t* _encryptionKey = nullptr, const void* _userData = nullptr, const system::logger_opt_ptr _logger = nullptr) :
            rootAsset(_asset), flags(_flags), compressionLevel(_compressionLevel),
            encryptionKeyLen(_encryptionKeyLen), encryptionKey(_encryptionKey),
            userData(_userData), logger(_logger)
        {
        }

        const IAsset* rootAsset;			//!< An Asset on which entire writing process is based.
        E_WRITER_FLAGS flags;				//!< Flags set by user that defines rules during writing process.
        float compressionLevel;				//!< The more compression level, the more expensive (slower) compression algorithm is launched. @see IAsset::conservativeSizeEstimate().
        size_t encryptionKeyLen;			//!< Stores a size of data in encryptionKey pointer for correct iteration.
        const uint8_t* encryptionKey;		//!< Stores an encryption key used for encryption process.
        const void* userData;				//!< Stores writer-dependets parameters. It is usually a struct provided by a writer author.
        system::logger_opt_ptr logger;
    };

    //! Struct for keeping the state of the current write operation for safe threading
	/**
		Important data used for Asset writing process is stored by params.
		Also a path to Asset data file to write is specified, stored by outputFile.
		You can store path to file as an absolute path or a relative path, flexibility is provided.

		@see SAssetWriteParams
	*/

    struct SAssetWriteContext
    {
        const SAssetWriteParams params;
        system::IFile* outputFile;
    };

public:

    //! Returns an array of string literals terminated by nullptr
    virtual const char** getAssociatedFileExtensions() const = 0;

    //! Returns the assets which can be written out by the loader
    /** 
		Bits of the returned value correspond to each IAsset::E_TYPE
		enumeration member, and the return value cannot be 0.
	*/
    virtual uint64_t getSupportedAssetTypesBitfield() const { return 0; }

    //! Returns which flags are supported for writing modes
    virtual uint32_t getSupportedFlags() = 0;

    //! Returns which flags are forced for writing modes, i.e. a writer can only support binary
    virtual uint32_t getForcedFlags() = 0;

    //! Override class to facilitate changing how assets are written, especially the sub-assets
	/*
		Each writer may override those functions to get more control on some process, but default implementations are provided.
		It handles getter-functions (eg. getting writing flags, compression level, encryption key or extra file paths). It
		also has a function for handling writing errors.
	*/
    class IAssetWriterOverride
    {
        //! The only reason these functions are not declared static is to allow stateful overrides
    public:
        //! To allow the asset writer to write different sub-assets with different flags
        inline virtual E_WRITER_FLAGS getAssetWritingFlags(const SAssetWriteContext& ctx, const IAsset* assetToWrite, const uint32_t& hierarchyLevel)
        {
            return ctx.params.flags;
        }

        //! For altering the compression level for individual assets, i.e. images, etc.
        inline virtual float getAssetCompressionLevel(const SAssetWriteContext& ctx, const IAsset* assetToWrite, const uint32_t& hierarchyLevel)
        {
            return ctx.params.compressionLevel;
        }

        //! For writing different sub-assets with different encryption keys (if supported)
        // if not supported then will never get called
        inline virtual size_t getEncryptionKey(const uint8_t* outEncryptionKey, const SAssetWriteContext& ctx, const IAsset* assetToWrite, const uint32_t& hierarchyLevel)
        {
            outEncryptionKey = ctx.params.encryptionKey;
            return ctx.params.encryptionKeyLen;
        }

        //! If the writer has to output multiple files (e.g. write out textures)
        inline virtual void getExtraFilePaths(std::string& inOutAbsoluteFileWritePath, std::string& inOutPathToRecord, const SAssetWriteContext& ctx, std::pair<const IAsset*, uint32_t> assetsToWriteAndTheirLevel) {} // do absolutely nothing, no changes to paths

        inline virtual system::IFile* getOutputFile(system::IFile* origIntendedOutput, const SAssetWriteContext& ctx, std::pair<const IAsset*, uint32_t> assetsToWriteAndTheirLeve)
        {
            // if you want to return something else, better drop origIntendedOutput
            return origIntendedOutput;
        }

        //!This function is supposed to give an already seeked file the IAssetWriter can write to 
        inline virtual system::IFile* handleWriteError(system::IFile* failingFile, const uint32_t& failedPos, const SAssetWriteContext& ctx, const IAsset* assetToWrite, const uint32_t& hierarchyLevel)
        {
            return nullptr; // no handling of fail
        }
    };

    //! Writes asset to a file (can be a memory write file)
    virtual bool writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) = 0;

private:
    static IAssetWriterOverride s_defaultOverride;

protected:
    static void getDefaultOverride(IAssetWriterOverride*& _out) { _out = &s_defaultOverride; }
};

}} //nbl::asset

#endif