// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#ifdef NEW_FILESYSTEM
#ifndef __NBL_C_ZIP_READER_H_INCLUDED__
#define __NBL_C_ZIP_READER_H_INCLUDED__

#include "nbl/asset/compile_config.h"

#ifdef __NBL_COMPILE_WITH_ZIP_ARCHIVE_LOADER_

#include "nbl/core/Types.h"
#include "nbl/system/IFile.h"
#include "IFileSystem.h"

namespace nbl
{
namespace io
{
	// set if the file is encrypted
	const int16_t ZIP_FILE_ENCRYPTED =		0x0001;
	// the fields crc-32, compressed size and uncompressed size are set to
	// zero in the local header
	const int16_t ZIP_INFO_IN_DATA_DESCRIPTOR =	0x0008;

// byte-align structures
#include "nbl/nblpack.h"

	struct SZIPFileDataDescriptor
	{
		uint32_t CRC32;
		uint32_t CompressedSize;
		uint32_t UncompressedSize;
	} PACK_STRUCT;

	struct SZIPFileHeader
	{
		uint32_t Sig;				// 'PK0304' little endian (0x04034b50)
		int16_t VersionToExtract;
		int16_t GeneralBitFlag;
		int16_t CompressionMethod;
		int16_t LastModFileTime;
		int16_t LastModFileDate;
		SZIPFileDataDescriptor DataDescriptor;
		int16_t FilenameLength;
		int16_t ExtraFieldLength;
		// filename (variable size)
		// extra field (variable size )
	} PACK_STRUCT;

	struct SZIPFileCentralDirFileHeader
	{
		uint32_t Sig;	// 'PK0102' (0x02014b50)
		uint16_t VersionMadeBy;
		uint16_t VersionToExtract;
		uint16_t GeneralBitFlag;
		uint16_t CompressionMethod;
		uint16_t LastModFileTime;
		uint16_t LastModFileDate;
		uint32_t CRC32;
		uint32_t CompressedSize;
		uint32_t UncompressedSize;
		uint16_t FilenameLength;
		uint16_t ExtraFieldLength;
		uint16_t FileCommentLength;
		uint16_t DiskNumberStart;
		uint16_t InternalFileAttributes;
		uint32_t ExternalFileAttributes;
		uint32_t RelativeOffsetOfLocalHeader;

		// filename (variable size)
		// extra field (variable size)
		// file comment (variable size)

	} PACK_STRUCT;

	struct SZIPFileCentralDirEnd
	{
		uint32_t Sig;			// 'PK0506' end_of central dir signature			// (0x06054b50)
		uint16_t NumberDisk;		// number of this disk
		uint16_t NumberStart;	// number of the disk with the start of the central directory
		uint16_t TotalDisk;		// total number of entries in the central dir on this disk
		uint16_t TotalEntries;	// total number of entries in the central dir
		uint32_t Size;			// size of the central directory
		uint32_t Offset;			// offset of start of centraldirectory with respect to the starting disk number
		uint16_t CommentLength;	// zipfile comment length
		// zipfile comment (variable size)
	} PACK_STRUCT;

	struct SZipFileExtraHeader
	{
		int16_t ID;
		int16_t Size;
	} PACK_STRUCT;

	struct SZipFileAESExtraData
	{
		int16_t Version;
		uint8_t Vendor[2];
		uint8_t EncryptionStrength;
		int16_t CompressionMode;
	} PACK_STRUCT;

	enum E_GZIP_FLAGS
	{
		EGZF_TEXT_DAT      = 1,
		EGZF_CRC16         = 2,
		EGZF_EXTRA_FIELDS  = 4,
		EGZF_FILE_NAME     = 8,
		EGZF_COMMENT       = 16
	};

	struct SGZIPMemberHeader
	{
		uint16_t sig; // 0x8b1f
		uint8_t  compressionMethod; // 8 = deflate
		uint8_t  flags;
		uint32_t time;
		uint8_t  extraFlags; // slow compress = 2, fast compress = 4
		uint8_t  operatingSystem;
	} PACK_STRUCT;

// Default alignment
#include "nbl/nblunpack.h"

	//! Contains extended info about zip files in the archive
	struct SZipFileEntry
	{
		//! Position of data in the archive file
		int32_t Offset;

		//! The header for this file containing compression info etc
		SZIPFileHeader header;
	};

	//! Archiveloader capable of loading ZIP Archives
	class CArchiveLoaderZIP : public IArchiveLoader
	{
	public:

		//! Constructor
		CArchiveLoaderZIP(io::IFileSystem* fs);

		//! returns true if the file maybe is able to be loaded by this class
		//! based on the file extension (e.g. ".zip")
		virtual bool isALoadableFileFormat(const io::path& filename) const;

		//! Check if the file might be loaded by this class
		/** Check might look into the file.
		\param file File handle to check.
		\return True if file seems to be loadable. */
		virtual bool isALoadableFileFormat(io::IReadFile* file) const;

		//! Check to see if the loader can create archives of this type.
		/** Check based on the archive type.
		\param fileType The archive type to check.
		\return True if the archile loader supports this type, false if not */
		virtual bool isALoadableFileFormat(E_FILE_ARCHIVE_TYPE fileType) const;

		//! Creates an archive from the filename
		/** \param file File handle to check.
		\return Pointer to newly created archive, or 0 upon error. */
		virtual IFileArchive* createArchive(const io::path& filename) const;

		//! creates/loads an archive from the file.
		//! \return Pointer to the created archive. Returns 0 if loading failed.
		virtual io::IFileArchive* createArchive(io::IReadFile* file) const;

	private:
		io::IFileSystem* FileSystem;
	};

/*!
	Zip file Reader written April 2002 by N.Gebhardt.
*/
	class CZipReader : public virtual IFileArchive, virtual CFileList
	{
        protected:
            //! destructor
            virtual ~CZipReader();

        public:
            //! constructor
            CZipReader(IReadFile* file, bool isGZip=false);

            //! opens a file by file name
            virtual IReadFile* createAndOpenFile(const io::path& filename);

            //! returns the list of files
            virtual const IFileList* getFileList() const;

            //! get the archive type
            virtual E_FILE_ARCHIVE_TYPE getType() const;

        protected:

            //! reads the next file header from a ZIP file, returns false if there are no more headers.
            /* if ignoreGPBits is set, the item will be read despite missing
            file information. This is used when reading items from the central
            directory. */
            bool scanZipHeader(bool ignoreGPBits=false);

            //! the same but for gzip files
            bool scanGZipHeader();

            bool scanCentralDirectoryHeader();

            IReadFile* File;

            // holds extended info about files
            core::vector<SZipFileEntry> FileInfo;

            bool IsGZip;
	};


} // end namespace io
} // end namespace nbl

#endif // __NBL_COMPILE_WITH_ZIP_ARCHIVE_LOADER_
#endif

#endif