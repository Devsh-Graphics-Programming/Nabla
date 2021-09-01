#ifndef __NBL_SYSTEM_C_ARCHIVE_LOADER_ZIP_H_INCLUDED__
#define __NBL_SYSTEM_C_ARCHIVE_LOADER_ZIP_H_INCLUDED__
#include "IFileArchive.h"
#include "nbl/system/path.h"
#include "nbl/system/ISystem.h"

namespace nbl::system
{
// set if the file is encrypted
constexpr int16_t ZIP_FILE_ENCRYPTED = 0x0001;
// the fields crc-32, compressed size and uncompressed size are set to
// zero in the local header
constexpr int16_t ZIP_INFO_IN_DATA_DESCRIPTOR = 0x0008;

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
	EGZF_TEXT_DAT = 1,
	EGZF_CRC16 = 2,
	EGZF_EXTRA_FIELDS = 4,
	EGZF_FILE_NAME = 8,
	EGZF_COMMENT = 16
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

class CFileArchiveZip : public IFileArchive 
{
private:
	bool m_isGZip;
	core::vector<SZipFileEntry> m_fileInfo;
	std::string m_password = ""; // TODO password
public:
	CFileArchiveZip(core::smart_refctd_ptr<system::IFile>&& file, core::smart_refctd_ptr<ISystem>&& system, bool isGZip, const std::string_view& password, system::logger_opt_smart_ptr&& logger = nullptr) :
		IFileArchive(std::move(file), std::move(system), std::move(logger)), m_isGZip(isGZip), m_password(password)
	{
		size_t offset = 0;
		if (m_file.get())
		{
			// load file entries
			if (m_isGZip)
				while (scanGZipHeader(offset)) {}
			else
				while (scanZipHeader(offset)) {}
			
			setFlagsVectorSize(m_files.size());
		}
	}
	virtual core::smart_refctd_ptr<IFile> readFile_impl(const SOpenFileParams& params) override;
private:
	bool scanGZipHeader(size_t& offset);
	bool scanZipHeader(size_t& offset, bool ignoreGPBits = false);
	bool scanCentralDirectoryHeader(size_t& offset);
	E_ALLOCATOR_TYPE getAllocatorType(uint32_t compressionType)
	{
		switch (compressionType)
		{
		case 0: return EAT_NULL;
		default: return EAT_VIRTUAL_ALLOC;
		}
	}
};

class CArchiveLoaderZip : public IArchiveLoader
{
public:
	CArchiveLoaderZip(core::smart_refctd_ptr<ISystem>&& system, system::logger_opt_smart_ptr&& logger) : IArchiveLoader(std::move(system), std::move(logger)) {}
	virtual bool isALoadableFileFormat(IFile* file) const override
	{
		SZIPFileHeader header;
		system::future<size_t> fut;
		file->read(fut, &header.Sig, 0, 4);
		fut.get();

		return header.Sig == 0x04034b50 || // ZIP
			(header.Sig & 0xffff) == 0x8b1f; // gzip
	}

	virtual const char** getAssociatedFileExtensions() const override
	{
		static const char* ext[]{ "zip", "pk3", "tgz", "gz", nullptr};
		return ext;
	}

private:
	virtual core::smart_refctd_ptr<IFileArchive> createArchive_impl(core::smart_refctd_ptr<system::IFile>&& file, const std::string_view& password) const override
	{
		core::smart_refctd_ptr<IFileArchive> archive = nullptr;
		if (file.get())
		{
			uint16_t sig;
			{
				system::future<size_t> fut;
				file->read(fut, &sig, 0, 2);
				fut.get();
			}
			bool isGZip = (sig == 0x8b1f);

			archive = core::make_smart_refctd_ptr<CFileArchiveZip>(std::move(file), core::smart_refctd_ptr<ISystem>(m_system), isGZip, password);
		}
		return archive;
	}
};

}

#endif