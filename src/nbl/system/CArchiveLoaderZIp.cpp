#include "nbl/system/CArchiveLoaderZip.h"
#include "nbl/system/CFileViewVirtualAllocatorWin32.h"
#include <aesGladman/fileenc.h>
#include <zconf.h>
#include <zlib/zlib.h>
#include <bzip2/bzlib.h>

namespace nbl::system
{
	bool CFileArchiveZip::scanGZipHeader(size_t& offset)
	{
		SZipFileEntry entry;
		entry.Offset = 0;
		memset(&entry.header, 0, sizeof(SZIPFileHeader));

		SGZIPMemberHeader header;
		system::future<size_t> headerFuture;
		m_file->read(headerFuture, &header, offset, sizeof(SGZIPMemberHeader));
		headerFuture.get();
		offset += sizeof(SGZIPMemberHeader);
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
				m_file->read(lenFuture, &dataLen, offset, 2);
				lenFuture.get();
				offset += 2;
			}
			std::filesystem::path zipFileName = "";
			if (header.flags & EGZF_FILE_NAME)
			{
				char c;
				{
					system::future<size_t> fut;
					m_file->read(fut, &c, offset++, 1);
					fut.get();
				}
				while (c)
				{
					zipFileName += c;
					system::future<size_t> fut;
					m_file->read(fut, &c, offset++, 1);
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
					m_file->read(fut, &c, offset++, 1);
					fut.get();
				}
			}
			if (header.flags & EGZF_CRC16)
				offset += 2;

			entry.Offset = offset;
			entry.header.FilenameLength = zipFileName.native().length();
			entry.header.CompressionMethod = header.compressionMethod;
			entry.header.DataDescriptor.CompressedSize = (m_file->getSize() - 8) - offset;

			offset += entry.header.DataDescriptor.CompressedSize;

			// read CRC
			{
				system::future<size_t> fut;
				m_file->read(fut, &entry.header.DataDescriptor.CRC32, offset, 4);
				fut.get();
				offset += 4;
			}
			// read uncompressed size
			{
				system::future<size_t> fut;
				m_file->read(fut, &entry.header.DataDescriptor.UncompressedSize, offset, 4);
				fut.get();
				offset += 4;
			}

			addItem(zipFileName, entry.Offset, entry.header.DataDescriptor.UncompressedSize, false, 0);
			m_fileInfo.push_back(entry);
		}
		return false;
	}

	bool CFileArchiveZip::scanZipHeader(size_t& offset, bool ignoreGPBits)
	{
		std::filesystem::path ZipFileName = "";
		SZipFileEntry entry;
		entry.Offset = 0;
		memset(&entry.header, 0, sizeof(SZIPFileHeader));

		{
			system::future<size_t> fut;
			m_file->read(fut, &entry.header, offset, sizeof(SZIPFileHeader));
			fut.get();
			offset += sizeof(SZIPFileHeader);
		}

		if (entry.header.Sig != 0x04034b50)
			return false; // local file headers end here.

		// read filename
		{
			char* tmp = new char[entry.header.FilenameLength + 2];
			{
				system::future<size_t> fut;
				m_file->read(fut, tmp, offset, entry.header.FilenameLength);
				fut.get();
				offset += entry.header.FilenameLength;
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
					m_file->read(fut, &extraHeader, offset, sizeof(extraHeader));
					fut.get();
					offset += sizeof(extraHeader);
				}
				restSize -= sizeof(extraHeader);
				if (extraHeader.ID == (int16_t)0x9901)
				{
					SZipFileAESExtraData data;
					{
						system::future<size_t> fut;
						m_file->read(fut, &data, offset, sizeof(data));
						fut.get();
						offset += sizeof(data);
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
						offset += restSize;
						break;
					}
				}
			}
		}
		// move forward length of extra field.
		else
#endif
			if (entry.header.ExtraFieldLength)
				offset += entry.header.ExtraFieldLength;

		// if bit 3 was set, use CentralDirectory for setup
		if (!ignoreGPBits && entry.header.GeneralBitFlag & ZIP_INFO_IN_DATA_DESCRIPTOR)
		{
			SZIPFileCentralDirEnd dirEnd;
			m_fileInfo.clear();
			m_files.clear();
			// First place where the end record could be stored
			offset = m_file->getSize() - 22;
			const char endID[] = { 0x50, 0x4b, 0x05, 0x06, 0x0 };
			char tmp[5] = { '\0' };
			bool found = false;
			// search for the end record ID
			while (!found && offset > 0)
			{
				int seek = 8;
				{
					system::future<size_t> fut;
					m_file->read(fut, tmp, offset, 4);
					fut.get();
					offset += 4;
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
				offset -= seek;
			}
			{
				system::future<size_t> fut;
				m_file->read(fut, &dirEnd, offset, sizeof(dirEnd));
				fut.get();
				offset += sizeof(dirEnd);
			}
			m_fileInfo.reserve(dirEnd.TotalEntries);
			offset = dirEnd.Offset;
			while (scanCentralDirectoryHeader(offset)) {}
			return false;
		}
		entry.Offset = offset;
		// move forward length of data
		offset += entry.header.DataDescriptor.CompressedSize;

		addItem(ZipFileName, entry.Offset, entry.header.DataDescriptor.UncompressedSize, *ZipFileName.string().rbegin() == '/', m_fileInfo.size());
		m_fileInfo.push_back(entry);
	}

	bool CFileArchiveZip::scanCentralDirectoryHeader(size_t& offset)
	{
		std::filesystem::path ZipFileName = "";
		SZIPFileCentralDirFileHeader entry;
		{
			system::future<size_t> fut;
			m_file->read(fut, &entry, offset, sizeof(SZIPFileCentralDirFileHeader));
			fut.get();
			offset += sizeof(SZIPFileCentralDirFileHeader);
		}

		if (entry.Sig != 0x02014b50)
			return false; // central dir headers end here.

		const long pos = offset;
		offset = entry.RelativeOffsetOfLocalHeader;
		scanZipHeader(offset, true);
		offset = pos + entry.FilenameLength + entry.ExtraFieldLength + entry.FileCommentLength;
		m_fileInfo.back().header.DataDescriptor.CompressedSize = entry.CompressedSize;
		m_fileInfo.back().header.DataDescriptor.UncompressedSize = entry.UncompressedSize;
		m_fileInfo.back().header.DataDescriptor.CRC32 = entry.CRC32;
		m_files.back().size = entry.UncompressedSize;
		return true;
	}

	core::smart_refctd_ptr<IFile> CFileArchiveZip::readFile(const SOpenFileParams& params)
	{
		size_t readOffset;
		auto found = std::find_if(m_files.begin(), m_files.end(), [&params](const SFileListEntry& entry) { return params.filename == entry.fullName; });

		const SZipFileEntry& e = m_fileInfo[found->ID];
		wchar_t buf[64];
		int16_t actualCompressionMethod = e.header.CompressionMethod;
		//TODO: CFileView factory
		// CFileViewVirtualAllocatorWin32
		core::smart_refctd_ptr<CFileView<CFileViewVirtualAllocatorWin32>> decrypted = nullptr;
		uint8_t* decryptedBuf = 0;
		uint32_t decryptedSize = e.header.DataDescriptor.CompressedSize;
#ifdef _NBL_COMPILE_WITH_ZIP_ENCRYPTION_
		if ((e.header.GeneralBitFlag & ZIP_FILE_ENCRYPTED) && (e.header.CompressionMethod == 99))
		{
			uint8_t salt[16] = { 0 };
			const uint16_t saltSize = (((e.header.Sig & 0x00ff0000) >> 16) + 1) * 4;
			{
				readOffset = e.Offset;
				read_blocking(m_file.get(), salt, readOffset, saltSize);
				readOffset += saltSize;
			}
			char pwVerification[2];
			char pwVerificationFile[2];
			{
				read_blocking(m_file.get(), pwVerification, readOffset, 2);
				readOffset += 2;
			}
			fcrypt_ctx zctx; // the encryption context
			int rc = fcrypt_init(
				(e.header.Sig & 0x00ff0000) >> 16,
				(const unsigned char*)m_password.c_str(), // the password
				m_password.size(), // number of bytes in password
				salt, // the salt
				(unsigned char*)pwVerificationFile, // on return contains password verifier
				&zctx); // encryption context
			if (strncmp(pwVerificationFile, pwVerification, 2))
			{
				m_logger.log("Wrong password", ILogger::ELL_ERROR);
				return 0;
			}
			decryptedSize = e.header.DataDescriptor.CompressedSize - saltSize - 12;
			decryptedBuf = new uint8_t[decryptedSize];
			uint32_t c = 0;
			while ((c + 32768) <= decryptedSize)
			{
				{
					read_blocking(m_file.get(), decryptedBuf + c, readOffset, 32768);
					readOffset += 32768;
				}
				fcrypt_decrypt(
					decryptedBuf + c, // pointer to the data to decrypt
					32768,   // how many bytes to decrypt
					&zctx); // decryption context
				c += 32768;
			}
			{
				read_blocking(m_file.get(), decryptedBuf + c, readOffset, decryptedSize - c);
				readOffset += decryptedSize - c;
			}
			fcrypt_decrypt(
				decryptedBuf + c, // pointer to the data to decrypt
				decryptedSize - c,   // how many bytes to decrypt
				&zctx); // decryption context

			char fileMAC[10];
			char resMAC[10];
			rc = fcrypt_end(
				(unsigned char*)resMAC, // on return contains the authentication code
				&zctx); // encryption context
			if (rc != 10)
			{
				m_logger.log("Error on encryption closing", ILogger::ELL_ERROR);
				delete[] decryptedBuf;
				return 0;
			}
			{
				read_blocking(m_file.get(), fileMAC, readOffset, 10);
				readOffset += 10;
			}
			if (strncmp(fileMAC, resMAC, 10))
			{
				m_logger.log("Error on encryption check", ILogger::ELL_ERROR);
				delete[] decryptedBuf;
				return 0;
			}
			decrypted = core::make_smart_refctd_ptr<CFileView<CFileViewVirtualAllocatorWin32>>(core::smart_refctd_ptr<ISystem>(m_system), found->fullName, IFile::ECF_READ_WRITE, decryptedSize);//new io::CMemoryReadFile(decryptedBuf, decryptedSize, found->FullName);
			{
				write_blocking(decrypted.get(), decryptedBuf, 0, decryptedSize);
			}
			actualCompressionMethod = (e.header.Sig & 0xffff);
#if 0
			if ((e.header.Sig & 0xff000000) == 0x01000000)
			{
			}
			else if ((e.header.Sig & 0xff000000) == 0x02000000)
			{
			}
			else
			{
				m_logger.log("Unknown encryption method", ILogger::ELL_ERROR);
				return 0;
			}
#endif
		}
#endif
		switch (actualCompressionMethod)
		{
		case 0: // no compression
		{
			delete[] decryptedBuf;
			if (decrypted)
				return decrypted;
			else
			{
				uint8_t* buff = (uint8_t*)m_file->getMappedPointer() + e.Offset;
				auto a = core::make_smart_refctd_ptr<CFileView<CNullAllocator>>(
					core::smart_refctd_ptr<ISystem>(m_system),
					found->fullName, 
					IFile::ECF_READ_WRITE, 
					buff, 
					decryptedSize);
				return a;
			}
		}
		case 8:
		{
#ifdef _NBL_COMPILE_WITH_ZLIB_

			const uint32_t uncompressedSize = e.header.DataDescriptor.UncompressedSize;
			char* pBuf = new char[uncompressedSize];
			if (!pBuf)
			{
				delete[] decryptedBuf;
				if (decrypted)
					decrypted->drop();
				return 0;
			}

			uint8_t* pcData = decryptedBuf;
			if (!pcData)
			{
				pcData = new uint8_t[decryptedSize];
				if (!pcData)
				{
					delete[] decryptedBuf;
					delete[] pBuf;
					return 0;
				}

				//memset(pcData, 0, decryptedSize);
				readOffset = e.Offset;
				{
					read_blocking(m_file.get(), pcData, readOffset, decryptedSize);
					readOffset += decryptedSize;
				}
			}

			// Setup the inflate stream.
			z_stream stream;
			int32_t err;

			stream.next_in = (Bytef*)pcData;
			stream.avail_in = (uInt)decryptedSize;
			stream.next_out = (Bytef*)pBuf;
			stream.avail_out = uncompressedSize;
			stream.zalloc = (alloc_func)0;
			stream.zfree = (free_func)0;

			// Perform inflation. wbits < 0 indicates no zlib header inside the data.
			err = inflateInit2(&stream, -MAX_WBITS);
			if (err == Z_OK)
			{
				err = inflate(&stream, Z_FINISH);
				inflateEnd(&stream);
				if (err == Z_STREAM_END)
					err = Z_OK;
				err = Z_OK;
				inflateEnd(&stream);
			}

			if (decrypted)
				decrypted->drop();
			else
				delete[] pcData;

			delete[] decryptedBuf;
			if (err != Z_OK)
			{
				delete[] pBuf;
				return 0;
			}
			else
			{
				auto ret = core::make_smart_refctd_ptr<CFileView<CFileViewVirtualAllocatorWin32>>(
					core::smart_refctd_ptr<ISystem>(m_system),
					found->fullName, 
					IFile::ECF_READ_WRITE, 
					uncompressedSize);
				{
					write_blocking(ret.get(), pBuf, 0, uncompressedSize);
				}
				delete[] pBuf;
				return ret;
			}

#else
			return 0; // zlib not compiled, we cannot decompress the data.
#endif
		}
		case 12:
		{
#ifdef _NBL_COMPILE_WITH_BZIP2_

			const uint32_t uncompressedSize = e.header.DataDescriptor.UncompressedSize;
			char* pBuf = new char[uncompressedSize];
			if (!pBuf)
			{
				m_logger.log("Not enough memory for decompressing %s", ILogger::ELL_ERROR, found->fullName.string().c_str());
				delete[] decryptedBuf;
				if (decrypted)
					decrypted->drop();
				return 0;
			}

			uint8_t* pcData = decryptedBuf;
			if (!pcData)
			{
				pcData = new uint8_t[decryptedSize];
				if (!pcData)
				{
					m_logger.log("Not enough memory for decompressing %s", ILogger::ELL_ERROR, found->fullName.string().c_str());
					delete[] pBuf;
					delete[] decryptedBuf;
					return 0;
				}

				{
					readOffset = e.Offset;
					read_blocking(m_file.get(), pcData, readOffset, decryptedSize);
					readOffset += decryptedSize;
				}
			}

			bz_stream bz_ctx = { 0 };
			/* use BZIP2's default memory allocation
			bz_ctx->bzalloc = NULL;
			bz_ctx->bzfree  = NULL;
			bz_ctx->opaque  = NULL;
			*/
			int err = BZ2_bzDecompressInit(&bz_ctx, 0, 0); /* decompression */
			if (err != BZ_OK)
			{
				m_logger.log("bzip2 decompression failed. File cannot be read.", ILogger::ELL_ERROR);
				delete[] decryptedBuf;
				return 0;
			}
			bz_ctx.next_in = (char*)pcData;
			bz_ctx.avail_in = decryptedSize;
			/* pass all input to decompressor */
			bz_ctx.next_out = pBuf;
			bz_ctx.avail_out = uncompressedSize;
			err = BZ2_bzDecompress(&bz_ctx);
			err = BZ2_bzDecompressEnd(&bz_ctx);

			if (decrypted)
				decrypted->drop();
			else
				delete[] pcData;

			if (err != BZ_OK)
			{
				m_logger.log("Error decompressing %s", ILogger::ELL_ERROR, found->fullName.string().c_str());
				delete[] pBuf;
				delete[] decryptedBuf;
				return 0;
			}
			else
			{
				auto ret = core::make_smart_refctd_ptr<CFileView<CFileViewVirtualAllocatorWin32>>(std::move(m_system), found->fullName, IFile::ECF_READ_WRITE, uncompressedSize);
				{
					write_blocking(decrypted.get(), pBuf, 0, uncompressedSize);
				}
				delete[] pBuf;
				return ret;
			}

#else
			delete[] decryptedBuf;
			m_logger.log("bzip2 decompression not supported. File cannot be read.", ILogger::ELL_ERROR);
			return 0;
#endif
		}
		case 14:
		{
#ifdef _NBL_COMPILE_WITH_LZMA_

			uint32_t uncompressedSize = e.header.DataDescriptor.UncompressedSize;
			char* pBuf = new char[uncompressedSize];
			if (!pBuf)
			{
				m_logger.log("Not enough memory for decompressing %s", ILogger::ELL_ERROR, found->FullName.c_str());
				delete[] decryptedBuf;
				if (decrypted)
					decrypted->drop();
				return 0;
			}

			uint8_t* pcData = decryptedBuf;
			if (!pcData)
			{
				pcData = new uint8_t[decryptedSize];
				if (!pcData)
				{
					m_logger.log("Not enough memory for decompressing %s", ILogger::ELL_ERROR, found->FullName.string().c_str());
					delete[] pBuf;
					return 0;
				}

				//memset(pcData, 0, decryptedSize);
				readOffset = e.Offset;
				{
					read_blocking(m_file.get(), pcData, readOffset, decryptedSize);
					readOffset += decryptedSize;
				}
			}

			ELzmaStatus status;
			SizeT tmpDstSize = uncompressedSize;
			SizeT tmpSrcSize = decryptedSize;

			unsigned int propSize = (pcData[3] << 8) + pcData[2];
			int err = LzmaDecode((Byte*)pBuf, &tmpDstSize,
				pcData + 4 + propSize, &tmpSrcSize,
				pcData + 4, propSize,
				e.header.GeneralBitFlag & 0x1 ? LZMA_FINISH_END : LZMA_FINISH_ANY, &status,
				&lzmaAlloc);
			uncompressedSize = tmpDstSize; // may be different to expected value

			if (decrypted)
				decrypted->drop();
			else
				delete[] pcData;

			delete[] decryptedBuf;
			if (err != SZ_OK)
			{
				m_logger.log("Error decompressing %s", ELL_ERROR, found->FullName.string().c_str());
				delete[] pBuf;
				return 0;
			}
			else
				return io::createMemoryReadFile(pBuf, uncompressedSize, found->FullName, true);

#else
			delete[] decryptedBuf;
			m_logger.log("lzma decompression not supported. File cannot be read.", ILogger::ELL_ERROR);
			return 0;
#endif
		}
		case 99:
			// If we come here with an encrypted file, decryption support is missing
			m_logger.log("Decryption support not enabled. File cannot be read.", ILogger::ELL_ERROR);
			delete[] decryptedBuf;
			return 0;
		default:
			m_logger.log("file has unsupported compression method. %s", ILogger::ELL_ERROR, found->fullName.string().c_str());
			delete[] decryptedBuf;
			return 0;
		};
	}
}