// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __IRR_BAW_MESH_WRITER_H_INCLUDED__
#define __IRR_BAW_MESH_WRITER_H_INCLUDED__

#include "irr/asset/IAssetWriter.h"
#include "irr/asset/ICPUMesh.h"
#include "irr/asset/bawformat/CBAWFile.h"

namespace irr {

namespace io { class IFileSystem; }

namespace asset
{
	class ISceneManager;

	class CBAWMeshWriter : public asset::IAssetWriter
	{
	public:
		//! Flags deciding what will be encrypted.
		/** @see @ref WriteProperties::encryptBlobBitField
		*/
		enum E_ENCRYPTION_TARGETS
		{
			EET_NOTHING = 0x00,
			EET_RAW_BUFFERS = 0x01,
			EET_TEXTURES = 0x02,
			EET_TEXTURE_PATHS = 0x04,
			EET_MESH_BUFFERS = 0x08,
			EET_DATA_FORMAT_DESC = 0x10,
			EET_ANIMATION_DATA = 0x20,
			EET_MESHES = 0x40,
			EET_EVERYTHING = 0xffffffffu
		};

		//! Settings struct for mesh export
		struct WriteProperties
		{
			//! Initialization vector for GCM encryption
			unsigned char initializationVector[16];
			//! Directory to which texture paths will be relative in output mesh file
			io::path relPath;
		};

	private:
		struct SContext
		{
			asset::IAssetWriter::SAssetWriteContext inner;
            asset::IAssetWriter::IAssetWriterOverride* writerOverride;
			core::vector<asset::BlobHeaderV1> headers;
			core::vector<uint32_t> offsets;
		};

        class CBAWOverride : public IAssetWriterOverride
        {
            inline float getAssetCompressionLevel(const SAssetWriteContext& ctx, const asset::IAsset* assetToWrite, const uint32_t& hierarchyLevel) override
            {
                const size_t est = assetToWrite->conservativeSizeEstimate();
                if (est >= 32768u) // lzma threshold
                    return 0.5f;
                else if (est >= 4096u) // lz4 threshold
                    return 0.3f;
				
				return 0.f;
            }
        };

	protected:
		~CBAWMeshWriter() {}

	public:
		explicit CBAWMeshWriter(io::IFileSystem*);

        //! Returns an array of string literals terminated by nullptr
        virtual const char** getAssociatedFileExtensions() const override
        {
            static const char* ext[]{ "baw" };
            return ext;
        }

        //! Returns the assets which can be written out by the loader
        /** Bits of the returned value correspond to each IAsset::E_TYPE
        enumeration member, and the return value cannot be 0. */
        virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_MESH; }

        //! Returns which flags are supported for writing modes
        virtual uint32_t getSupportedFlags() override { return asset::EWF_COMPRESSED | asset::EWF_ENCRYPTED; }

        //! Returns which flags are forced for writing modes, i.e. a writer can only support binary
        virtual uint32_t getForcedFlags() override { return asset::EWF_BINARY; }

        virtual bool writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;

	private:
		//! Takes object and exports (writes to file) its data as another blob.
		/** @param _obj Pointer to object which is to be exported.
		@param _headersIdx Corresponding index of headers array.
		@param _file Target file.*/
		template<typename T>
		void exportAsBlob(T* _obj, uint32_t _headerIdx, io::IWriteFile* _file, SContext& _ctx);

		//! Generates header of blobs from mesh object and pushes them to `SContext::headers`.
		/** After calling this method headers are NOT ready yet. Hashes (and also size in case of texture path blob) are calculated while writing blob data.
		@param _mesh Pointer to the mesh object.
		@return Amount of generated headers.*/
		uint32_t genHeaders(const asset::ICPUMesh* _mesh, SContext& _ctx);

		//! Pushes new offset value to `SContext::offsets` array.
		/** @param _blobSize Byte-distance from previous blob's first byte (i.e. size of previous blob).
		*/
		void calcAndPushNextOffset(uint32_t _blobSize, SContext& _ctx) const;

		//! Pushes corrupted offset so that, while loading resulting .baw file, it will be easy to find out something went wrong.
		void pushCorruptedOffset(SContext& _ctx) const { _ctx.offsets.push_back(0xffffffff); }

		//! Tries to write given data to file. If not possible (i.e. _data is NULL) - pushes "corrupted offset" and does not call .finalize() on blob-header.
		void tryWrite(void* _data, io::IWriteFile* _file, SContext& _ctx, size_t _size, uint32_t _headerIdx, asset::E_WRITER_FLAGS _flags, const uint8_t* _encrPwd = nullptr, float _comprLvl = 0.f) const;
		
		//! Uint32_t because lzma doesn't support compressing more than 4GB
		void* compressWithLz4AndTryOnStack(const void* _input, uint32_t _inputSize, void* _stack, uint32_t _stackSize, size_t& _outComprSize) const;
		void* compressWithLzma(const void* _input, size_t _inputSize, size_t& _outComprSize) const;

	private:
		io::IFileSystem* m_fileSystem;

		static const char * const BAW_FILE_HEADER;
	};

}
} // end of ns irr:scene

#endif
