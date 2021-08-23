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
	core::smart_refctd_ptr<system::IFile> m_file;
	bool m_isGZip;
public:
	CFileArchiveZip(core::smart_refctd_ptr<system::IFile>&& file, bool isGZip) : m_file(std::move(file)), m_isGZip(isGZip)
	{
		if (m_file.get())
		{
			// load file entries
			if (m_isGZip)
				while (scanGZipHeader()) {}
			else
				while (scanZipHeader()) {}
		}
	}
	virtual core::smart_refctd_ptr<IFile> readFile(const SOpenFileParams& params) override
	{
		
	}
private:
	bool scanGZipHeader()
	{
		size_t read_offset = 0;
		SZipFileEntry entry;
		entry.Offset = 0;
		memset(&entry.header, 0, sizeof(SZIPFileHeader));

		SGZIPMemberHeader header;
		system::future<size_t> headerFuture;
		m_file->read(headerFuture, &header, read_offset, sizeof(SGZIPMemberHeader));
		read_offset += sizeof(SGZIPMemberHeader);
		if (headerFuture.get() == sizeof(SGZIPMemberHeader))
		{
			// check header value
			if (header.sig != 0x8b1f)
				return false;
			// now get the file info
			if (header.flags & EGZF_EXTRA_FIELDS)
			{
				// read lenth of extra data
				uint16_t dataLen;

				system::future<size_t> lenFuture;
				m_file->read(lenFuture, &dataLen, read_offset, 2);
				read_offset += 2;
			}
			std::filesystem::path zipFileName = "";
			if (header.flags & EGZF_FILE_NAME)
			{
				char c;
				{
					system::future<size_t> fut;
					m_file->read(fut, &c, read_offset++, 1);
				}
				while (c)
				{
					zipFileName += c;
					system::future<size_t> fut;
					m_file->read(fut, &c, read_offset++, 1);
				}
			}
			else
			{
				// no file name?
				zipFileName = m_file->getFileName().filename();
			
				// rename tgz to tar or remove gz extension
				if (zipFileName.extension() == ".tgz")
				{
					zipFileName.string()[zipFileName.string().size() - 2] = 'a';
					zipFileName.string()[zipFileName.string().size() - 1] = 'r';
				}
				else if (zipFileName.extension() == ".gz")
				{
					zipFileName.string()[zipFileName.string().size() - 3] = 0;
				}
			}
			if (header.flags & EGZF_COMMENT)
			{
				char c = 'a';
				while (c)
				{
					system::future<size_t> fut;
					m_file->read(fut, &c, read_offset++, 1);
				}
			}
			if (header.flags & EGZF_CRC16)
				read_offset += 2;

			entry.Offset = read_offset;
			entry.header.FilenameLength = zipFileName.native().length();
			entry.header.CompressionMethod = header.compressionMethod;
			entry.header.DataDescriptor.CompressedSize = (m_file->getSize() - 8) - read_offset;

			read_offset += entry.header.DataDescriptor.CompressedSize;

			// read CRC
			{
				system::future<size_t> fut;
				m_file->read(fut, &entry.header.DataDescriptor.CRC32, read_offset, 4);
				read_offset += 4;
			}
			// read uncompressed size
			{
				system::future<size_t> fut;
				m_file->read(fut, &entry.header.DataDescriptor.UncompressedSize, read_offset, 4);
				read_offset += 4;
			}

			addItem(ZipFileName, entry.Offset, entry.header.DataDescriptor.UncompressedSize, false, 0);
			FileInfo.push_back(entry);
		}
		return false;
	}
	bool scanZipHeader()
	{

	}
};

class CArchiveLoaderZip : public IArchiveLoader
{
public:
	virtual bool isALoadableFileFormat(IFile* file) const override
	{
		std::string ext = system::extension_wo_dot(file->getFileName().extension());
		auto available_exts = getAssociatedFileExtensions();
		const char* current_ext = available_exts[0];
		while (current_ext)
		{
			if (ext == current_ext)
			{
				return true;
			}
			++current_ext;
		}
		return false;
	}

	virtual const char** getAssociatedFileExtensions() const override
	{
		static const char* ext[]{ "zip", "pk3", "tgz", "gz", nullptr};
	}

	virtual bool isMountable() const override
	{
		return true;
	}

	virtual bool mount(const IFile* file, const std::string_view& passphrase) override
	{

		return true;
	}

	virtual bool unmount(const IFile* file) override
	{

		return true;
	}

private:
	virtual core::smart_refctd_ptr<IFileArchive> createArchive_impl(IFile* file) const override
	{

	}
};

}

#endif