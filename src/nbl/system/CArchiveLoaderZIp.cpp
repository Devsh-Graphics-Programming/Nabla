#include "nbl/system/CArchiveLoaderZip.h"

namespace nbl::system
{
	bool CFileArchiveZip::scanGZipHeader()
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

			addItem(zipFileName, entry.Offset, entry.header.DataDescriptor.UncompressedSize, 0);
			m_fileInfo.push_back(entry);
		}
		return false;
	}

	bool CFileArchiveZip::scanZipHeader(bool ignoreGPBits)
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

#ifdef _NBL_COMPILE_WITH_ZIP_ENCRYPTION_
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

	bool CFileArchiveZip::scanCentralDirectoryHeader()
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
}