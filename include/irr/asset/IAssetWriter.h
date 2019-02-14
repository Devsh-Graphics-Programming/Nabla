#ifndef __IRR_I_ASSET_WRITER_H_INCLUDED__
#define __IRR_I_ASSET_WRITER_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "irr/core/IReferenceCounted.h"
#include "IWriteFile.h"
#include "IAsset.h"

namespace irr { namespace asset
{

enum E_WRITER_FLAGS : uint32_t
{
    //! no writer flags (default writer settings)
    EWF_NONE = 0u,

    //! write in a way that consumes less disk space if possible
    EWF_COMPRESSED = 1u<<0u,

    //! write encrypted if possible
    EWF_ENCRYPTED = 1u<<1u,

    //! write in binary format rather than text if possible
    EWF_BINARY = 1u<<2u
};

class IAssetWriter : public virtual core::IReferenceCounted
{
public:
    struct SAssetWriteParams
    {
        SAssetWriteParams(IAsset* _asset, const E_WRITER_FLAGS& _flags = EWF_NONE, const float& _compressionLevel = 0.f, const size_t& _encryptionKeyLen = 0, const uint8_t* _encryptionKey = nullptr, const void* _userData = nullptr) :
            rootAsset(_asset), flags(_flags), compressionLevel(_compressionLevel),
            encryptionKeyLen(_encryptionKeyLen), encryptionKey(_encryptionKey),
            userData(_userData)
        {
        }

        const IAsset* rootAsset;
        E_WRITER_FLAGS flags;
        float compressionLevel;
        size_t encryptionKeyLen;
        const uint8_t* encryptionKey;
        const void* userData;
    };

    //! Struct for keeping the state of the current write operation for safe threading
    struct SAssetWriteContext
    {
        const SAssetWriteParams params;
        io::IWriteFile* outputFile;
    };

public:
    //! Returns an array of string literals terminated by nullptr
    virtual const char** getAssociatedFileExtensions() const = 0;

    //! Returns the assets which can be written out by the loader
    /** Bits of the returned value correspond to each IAsset::E_TYPE
    enumeration member, and the return value cannot be 0. */
    virtual uint64_t getSupportedAssetTypesBitfield() const { return 0; }

    //! Returns which flags are supported for writing modes
    virtual uint32_t getSupportedFlags() = 0;

    //! Returns which flags are forced for writing modes, i.e. a writer can only support binary
    virtual uint32_t getForcedFlags() = 0;

    //! Override class to facilitate changing how assets are written, especially the sub-assets
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

        inline virtual io::IWriteFile* getOutputFile(io::IWriteFile* origIntendedOutput, const SAssetWriteContext& ctx, std::pair<const IAsset*, uint32_t> assetsToWriteAndTheirLeve)
        {
            // if you want to return something else, better drop origIntendedOutput
            return origIntendedOutput;
        }

        //!This function is supposed to give an already seeked file the IAssetWriter can write to 
        inline virtual io::IWriteFile* handleWriteError(io::IWriteFile* failingFile, const uint32_t& failedPos, const SAssetWriteContext& ctx, const IAsset* assetToWrite, const uint32_t& hierarchyLevel)
        {
            return nullptr; // no handling of fail
        }
    };

    //! Writes asset to a file (can be a memory write file)
    virtual bool writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) = 0;

private:
    static IAssetWriterOverride s_defaultOverride;

protected:
    static void getDefaultOverride(IAssetWriterOverride* _out) { _out = &s_defaultOverride; }
};

}} //irr::asset

#endif //__IRR_I_ASSET_WRITER_H_INCLUDED__