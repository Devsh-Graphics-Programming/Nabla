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
	core::vector<SZipFileEntry> m_fileInfo;
	size_t m_readOffset = 0;
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
		SZipFileEntry entry;
		entry.Offset = 0;
		memset(&entry.header, 0, sizeof(SZIPFileHeader));

		SGZIPMemberHeader header;
		system::future<size_t> headerFuture;
		m_file->read(headerFuture, &header, m_readOffset, sizeof(SGZIPMemberHeader));
		headerFuture.get();
		m_readOffset += sizeof(SGZIPMemberHeader);
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
				m_file->read(lenFuture, &dataLen, m_readOffset, 2);
				lenFuture.get();
				m_readOffset += 2;
			}
			std::filesystem::path zipFileName = "";
			if (header.flags & EGZF_FILE_NAME)
			{
				char c;
				{
					system::future<size_t> fut;
					m_file->read(fut, &c, m_readOffset++, 1);
					fut.get();
				}
				while (c)
				{
					zipFileName += c;
					system::future<size_t> fut;
					m_file->read(fut, &c, m_readOffset++, 1);
					fut.get();
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
					m_file->read(fut, &c, m_readOffset++, 1);
					fut.get();
				}
			}
			if (header.flags & EGZF_CRC16)
				m_readOffset += 2;

			entry.Offset = m_readOffset;
			entry.header.FilenameLength = zipFileName.native().length();
			entry.header.CompressionMethod = header.compressionMethod;
			entry.header.DataDescriptor.CompressedSize = (m_file->getSize() - 8) - m_readOffset;

			m_readOffset += entry.header.DataDescriptor.CompressedSize;

			// read CRC
			{
				system::future<size_t> fut;
				m_file->read(fut, &entry.header.DataDescriptor.CRC32, m_readOffset, 4);
				fut.get();
				m_readOffset += 4;
			}
			// read uncompressed size
			{
				system::future<size_t> fut;
				m_file->read(fut, &entry.header.DataDescriptor.UncompressedSize, m_readOffset, 4);
				fut.get();
				m_readOffset += 4;
			}

			addItem(zipFileName, entry.Offset, entry.header.DataDescriptor.UncompressedSize, false, 0);
			m_fileInfo.push_back(entry);
		}
		return false;
	}
	bool scanZipHeader(bool ignoreGPBits = false)
	{
		std::filesystem::path ZipFileName = "";
		SZipFileEntry entry;
		entry.Offset = 0;
		memset(&entry.header, 0, sizeof(SZIPFileHeader));

		{
			system::future<size_t> fut;
			m_file->read(fut, &entry.header, m_readOffset, sizeof(SZIPFileHeader));
			fut.get();
			m_readOffset += sizeof(SZIPFileHeader);
		}

		if (entry.header.Sig != 0x04034b50)
			return false; // local file headers end here.

		// read filename
		{
			char* tmp = new char[entry.header.FilenameLength + 2];
			{
				system::future<size_t> fut;
				m_file->read(fut, tmp, m_readOffset, entry.header.FilenameLength);
				fut.get();
				m_readOffset += entry.header.FilenameLength;
			}
			tmp[entry.header.FilenameLength] = 0;
			ZipFileName = tmp;
			delete[] tmp;
		}

//#ifdef _NBL_COMPILE_WITH_ZIP_ENCRYPTION_
#if 1
		// AES encryption
		if ((entry.header.GeneralBitFlag & ZIP_FILE_ENCRYPTED) && (entry.header.CompressionMethod == 99))
		{
			int16_t restSize = entry.header.ExtraFieldLength;
			SZipFileExtraHeader extraHeader;
			while (restSize)
			{
				{
					system::future<size_t> fut;
					m_file->read(fut, &extraHeader, m_readOffset, sizeof(extraHeader));
					fut.get();
					m_readOffset += sizeof(extraHeader);
				}
				restSize -= sizeof(extraHeader);
				if (extraHeader.ID == (int16_t)0x9901)
				{
					SZipFileAESExtraData data;
					{
						system::future<size_t> fut;
						m_file->read(fut, &data, m_readOffset, sizeof(data));
						fut.get();
						m_readOffset += sizeof(data);
					}

					restSize -= sizeof(data);
					if (data.Vendor[0] == 'A' && data.Vendor[1] == 'E')
					{
						// encode values into Sig
						// AE-Version | Strength | ActualMode
						entry.header.Sig =
							((data.Version & 0xff) << 24) |
							(data.EncryptionStrength << 16) |
							(data.CompressionMode);
						m_readOffset += restSize;
						break;
					}
				}
			}
		}
		// move forward length of extra field.
		else
#endif
			if (entry.header.ExtraFieldLength)
				m_readOffset += entry.header.ExtraFieldLength;

		// if bit 3 was set, use CentralDirectory for setup
		if (!ignoreGPBits && entry.header.GeneralBitFlag & ZIP_INFO_IN_DATA_DESCRIPTOR)
		{
			SZIPFileCentralDirEnd dirEnd;
			m_fileInfo.clear();
			m_files.clear();
			// First place where the end record could be stored
			m_readOffset = m_file->getSize() - 22;
			const char endID[] = { 0x50, 0x4b, 0x05, 0x06, 0x0 };
			char tmp[5] = { '\0' };
			bool found = false;
			// search for the end record ID
			while (!found && m_readOffset > 0)
			{
				int seek = 8;
				{
					system::future<size_t> fut;
					m_file->read(fut, tmp, m_readOffset, 4);
					fut.get();
					m_readOffset += 4;
				}
				switch (tmp[0])
				{
				case 0x50:
					if (!strcmp(endID, tmp))
					{
						seek = 4;
						found = true;
					}
					break;
				case 0x4b:
					seek = 5;
					break;
				case 0x05:
					seek = 6;
					break;
				case 0x06:
					seek = 7;
					break;
				}
				m_readOffset -= seek;
			}
			{
				system::future<size_t> fut;
				m_file->read(fut, &dirEnd, m_readOffset, sizeof(dirEnd));
				fut.get();
				m_readOffset += sizeof(dirEnd);
			}
			m_fileInfo.reserve(dirEnd.TotalEntries);
			m_readOffset = dirEnd.Offset;
			while (scanCentralDirectoryHeader()) {}
			return false;
		}
	}

	bool scanCentralDirectoryHeader()
	{
		std::filesystem::path ZipFileName = "";
		SZIPFileCentralDirFileHeader entry;
		{
			system::future<size_t> fut;
			m_file->read(fut, &entry, m_readOffset, sizeof(SZIPFileCentralDirFileHeader));
			fut.get();
			m_readOffset += sizeof(SZIPFileCentralDirFileHeader);
		}

		if (entry.Sig != 0x02014b50)
			return false; // central dir headers end here.

		const long pos = m_readOffset;
		m_readOffset = entry.RelativeOffsetOfLocalHeader;
		scanZipHeader(true);
		m_readOffset = pos + entry.FilenameLength + entry.ExtraFieldLength + entry.FileCommentLength;
		m_fileInfo.back().header.DataDescriptor.CompressedSize = entry.CompressedSize;
		m_fileInfo.back().header.DataDescriptor.UncompressedSize = entry.UncompressedSize;
		m_fileInfo.back().header.DataDescriptor.CRC32 = entry.CRC32;
		m_files.back().size = entry.UncompressedSize;
		return true;
	}
};

class CArchiveLoaderZip : public IArchiveLoader
{
public:
	virtual bool isALoadableFileFormat(IFile* file) const override
	{
		SZIPFileHeader header;
		system::future<size_t> fut;
		file->read(fut, &header.Sig, 0, 4);

		return header.Sig == 0x04034b50 || // ZIP
			(header.Sig & 0xffff) == 0x8b1f; // gzip
	}

	virtual const char** getAssociatedFileExtensions() const override
	{
		static const char* ext[]{ "zip", "pk3", "tgz", "gz", nullptr};
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