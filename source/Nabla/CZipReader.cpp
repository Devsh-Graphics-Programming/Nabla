// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#ifdef NEW_FILESYSTEM
#include "CZipReader.h"
#include "CMemoryFile.h"
#include "CLimitReadFile.h"

#include "nbl_os.h"
#include <sstream>

#ifdef __NBL_COMPILE_WITH_ZIP_ARCHIVE_LOADER_

#include "CFileList.h"
#include "CReadFile.h"

#ifdef _NBL_COMPILE_WITH_ZLIB_
	#include "zlib/zlib.h"

	#ifdef _NBL_COMPILE_WITH_ZIP_ENCRYPTION_
	#include "aesGladman/fileenc.h"
	#endif
	#ifdef _NBL_COMPILE_WITH_BZIP2_
	#include "bzip2/bzlib.h"
	#endif
	#ifdef _NBL_COMPILE_WITH_LZMA_
	#include "lzma/LzmaDec.h"
	#endif
#endif

namespace nbl
{
namespace io
{


// -----------------------------------------------------------------------------
// zip loader
// -----------------------------------------------------------------------------

//! Constructor
CArchiveLoaderZIP::CArchiveLoaderZIP(io::IFileSystem* fs)
: FileSystem(fs)
{
	#ifdef _NBL_DEBUG
	setDebugName("CArchiveLoaderZIP");
	#endif
}

//! returns true if the file maybe is able to be loaded by this class
bool CArchiveLoaderZIP::isALoadableFileFormat(const std::filesystem::path& filename) const
{
	return core::hasFileExtension(filename, "zip", "pk3") ||
	       core::hasFileExtension(filename, "gz", "tgz");
}

//! Check to see if the loader can create archives of this type.
bool CArchiveLoaderZIP::isALoadableFileFormat(E_FILE_ARCHIVE_TYPE fileType) const
{
	return (fileType == EFAT_ZIP || fileType == EFAT_GZIP);
}


//! Creates an archive from the filename
/** \param file File handle to check.
\return Pointer to newly created archive, or 0 upon error. */
IFileArchive* CArchiveLoaderZIP::createArchive(const std::filesystem::path& filename) const
{
	IFileArchive *archive = 0;
	io::IReadFile* file = FileSystem->createAndOpenFile(filename);

	if (file)
	{
		archive = createArchive(file);
		file->drop();
	}

	return archive;
}

//! creates/loads an archive from the file.
//! \return Pointer to the created archive. Returns 0 if loading failed.
IFileArchive* CArchiveLoaderZIP::createArchive(io::IReadFile* file) const
{
	IFileArchive *archive = 0;
	if (file)
	{
		file->seek(0);

		uint16_t sig;
		file->read(&sig, 2);

		file->seek(0);

		bool isGZip = (sig == 0x8b1f);

		archive = new CZipReader(file, isGZip);
	}
	return archive;
}

//! Check if the file might be loaded by this class
/** Check might look into the file.
\param file File handle to check.
\return True if file seems to be loadable. */
bool CArchiveLoaderZIP::isALoadableFileFormat(io::IReadFile* file) const
{
	const size_t prevPos = file->getPos();
	file->seek(0u);
	SZIPFileHeader header;
	file->read( &header.Sig, 4 );
	file->seek(prevPos);

	return header.Sig == 0x04034b50 || // ZIP
		   (header.Sig&0xffff) == 0x8b1f; // gzip
}

// -----------------------------------------------------------------------------
// zip archive
// -----------------------------------------------------------------------------

CZipReader::CZipReader(IReadFile* file, bool isGZip) : CFileList(file ? file->getFileName() : std::filesystem::path("")), File(file), IsGZip(isGZip)
{
	#ifdef _NBL_DEBUG
	setDebugName("CZipReader");
	#endif

	if (File)
	{
		File->grab();

		// load file entries
		if (IsGZip)
			while (scanGZipHeader()) { }
		else
			while (scanZipHeader()) { }
	}
}

CZipReader::~CZipReader()
{
	if (File)
		File->drop();
}


//! get the archive type
E_FILE_ARCHIVE_TYPE CZipReader::getType() const
{
	return IsGZip ? EFAT_GZIP : EFAT_ZIP;
}

const IFileList* CZipReader::getFileList() const
{
	return this;
}


//! scans for a local header, returns false if there is no more local file header.
//! The gzip file format seems to think that there can be multiple files in a gzip file
//! but none
bool CZipReader::scanGZipHeader()
{
	SZipFileEntry entry;
	entry.Offset = 0;
	memset(&entry.header, 0, sizeof(SZIPFileHeader));

	// read header
	SGZIPMemberHeader header;
	if (File->read(&header, sizeof(SGZIPMemberHeader)) == sizeof(SGZIPMemberHeader))
	{
		// check header value
		if (header.sig != 0x8b1f)
			return false;

		// now get the file info
		if (header.flags & EGZF_EXTRA_FIELDS)
		{
			// read lenth of extra data
			uint16_t dataLen;

			File->read(&dataLen, 2);

			// skip it
			File->seek(dataLen, true);
		}

		std::filesystem::path ZipFileName = "";

		if (header.flags & EGZF_FILE_NAME)
		{
			char c;
			File->read(&c, 1);
			while (c)
			{
				ZipFileName += c;
				File->read(&c, 1);
			}
		}
		else
		{
			// no file name?
			ZipFileName = Path;
			core::deletePathFromFilename(ZipFileName);

			// rename tgz to tar or remove gz extension
			if (core::hasFileExtension(ZipFileName, "tgz"))
			{
				ZipFileName.string()[ ZipFileName.string().size() - 2] = 'a';
				ZipFileName.string()[ ZipFileName.string().size() - 1] = 'r';
			}
			else if (core::hasFileExtension(ZipFileName, "gz"))
			{
				ZipFileName.string()[ ZipFileName.string().size() - 3] = 0;
			}
		}

		if (header.flags & EGZF_COMMENT)
		{
			char c='a';
			while (c)
				File->read(&c, 1);
		}

		if (header.flags & EGZF_CRC16)
			File->seek(2, true);

		// we are now at the start of the data blocks
		entry.Offset = File->getPos();

		entry.header.FilenameLength = ZipFileName.string().length();

		entry.header.CompressionMethod = header.compressionMethod;
		entry.header.DataDescriptor.CompressedSize = (File->getSize() - 8) - File->getPos();

		// seek to file end
		File->seek(entry.header.DataDescriptor.CompressedSize, true);

		// read CRC
		File->read(&entry.header.DataDescriptor.CRC32, 4);
		// read uncompressed size
		File->read(&entry.header.DataDescriptor.UncompressedSize, 4);

		// now we've filled all the fields, this is just a standard deflate block
		addItem(ZipFileName, entry.Offset, entry.header.DataDescriptor.UncompressedSize, false, 0);
		FileInfo.push_back(entry);
	}

	// there's only one block of data in a gzip file
	return false;
}

//! scans for a local header, returns false if there is no more local file header.
bool CZipReader::scanZipHeader(bool ignoreGPBits)
{
	std::filesystem::path ZipFileName = "";
	SZipFileEntry entry;
	entry.Offset = 0;
	memset(&entry.header, 0, sizeof(SZIPFileHeader));

	File->read(&entry.header, sizeof(SZIPFileHeader));


	if (entry.header.Sig != 0x04034b50)
		return false; // local file headers end here.

	// read filename
	{
		char *tmp = new char [ entry.header.FilenameLength + 2 ];
		File->read(tmp, entry.header.FilenameLength);
		tmp[entry.header.FilenameLength] = 0;
		ZipFileName = tmp;
		delete [] tmp;
	}

#ifdef _NBL_COMPILE_WITH_ZIP_ENCRYPTION_
	// AES encryption
	if ((entry.header.GeneralBitFlag & ZIP_FILE_ENCRYPTED) && (entry.header.CompressionMethod == 99))
	{
		int16_t restSize = entry.header.ExtraFieldLength;
		SZipFileExtraHeader extraHeader;
		while (restSize)
		{
			File->read(&extraHeader, sizeof(extraHeader));
			restSize -= sizeof(extraHeader);
			if (extraHeader.ID==(int16_t)0x9901)
			{
				SZipFileAESExtraData data;
				File->read(&data, sizeof(data));

				restSize -= sizeof(data);
				if (data.Vendor[0]=='A' && data.Vendor[1]=='E')
				{
					// encode values into Sig
					// AE-Version | Strength | ActualMode
					entry.header.Sig =
						((data.Version & 0xff) << 24) |
						(data.EncryptionStrength << 16) |
						(data.CompressionMode);
					File->seek(restSize, true);
					break;
				}
			}
		}
	}
	// move forward length of extra field.
	else
#endif
	if (entry.header.ExtraFieldLength)
		File->seek(entry.header.ExtraFieldLength, true);

	// if bit 3 was set, use CentralDirectory for setup
	if (!ignoreGPBits && entry.header.GeneralBitFlag & ZIP_INFO_IN_DATA_DESCRIPTOR)
	{
		SZIPFileCentralDirEnd dirEnd;
		FileInfo.clear();
		Files.clear();
		// First place where the end record could be stored
		File->seek(File->getSize()-22);
		const char endID[] = {0x50, 0x4b, 0x05, 0x06, 0x0};
		char tmp[5]={'\0'};
		bool found=false;
		// search for the end record ID
		while (!found && File->getPos()>0)
		{
			int seek=8;
			File->read(tmp, 4);
			switch (tmp[0])
			{
			case 0x50:
				if (!strcmp(endID, tmp))
				{
					seek=4;
					found=true;
				}
				break;
			case 0x4b:
				seek=5;
				break;
			case 0x05:
				seek=6;
				break;
			case 0x06:
				seek=7;
				break;
			}
			File->seek(-seek, true);
		}
		File->read(&dirEnd, sizeof(dirEnd));
		FileInfo.reserve(dirEnd.TotalEntries);
		File->seek(dirEnd.Offset);
		while (scanCentralDirectoryHeader()) { }
		return false;
	}

	// store position in file
	entry.Offset = File->getPos();
	// move forward length of data
	File->seek(entry.header.DataDescriptor.CompressedSize, true);

	#ifdef _NBL_DEBUG
	//os::Debuginfo::print("added file from archive", ZipFileName.c_str());
	#endif

	addItem(ZipFileName, entry.Offset, entry.header.DataDescriptor.UncompressedSize, *ZipFileName.string().rbegin()=='/', FileInfo.size());
	FileInfo.push_back(entry);

	return true;
}


//! scans for a local header, returns false if there is no more local file header.
bool CZipReader::scanCentralDirectoryHeader()
{
	std::filesystem::path ZipFileName = "";
	SZIPFileCentralDirFileHeader entry;
	File->read(&entry, sizeof(SZIPFileCentralDirFileHeader));

	if (entry.Sig != 0x02014b50)
		return false; // central dir headers end here.

	const long pos = File->getPos();
	File->seek(entry.RelativeOffsetOfLocalHeader);
	scanZipHeader(true);
	File->seek(pos+entry.FilenameLength+entry.ExtraFieldLength+entry.FileCommentLength);
	FileInfo.back().header.DataDescriptor.CompressedSize=entry.CompressedSize;
	FileInfo.back().header.DataDescriptor.UncompressedSize=entry.UncompressedSize;
	FileInfo.back().header.DataDescriptor.CRC32=entry.CRC32;
	Files.back().Size=entry.UncompressedSize;
	return true;
}


//! opens a file by file name
IReadFile* CZipReader::createAndOpenFile(const std::filesystem::path& filename)
{
    auto found = findFile(Files.begin(),Files.end(),io::IFileSystem::flattenFilename(filename),false);
	if (found==Files.end())
        return nullptr;

	// Irrlicht supports 0, 8, 12, 14, 99
	//0 - The file is stored (no compression)
	//1 - The file is Shrunk
	//2 - The file is Reduced with compression factor 1
	//3 - The file is Reduced with compression factor 2
	//4 - The file is Reduced with compression factor 3
	//5 - The file is Reduced with compression factor 4
	//6 - The file is Imploded
	//7 - Reserved for Tokenizing compression algorithm
	//8 - The file is Deflated
	//9 - Reserved for enhanced Deflating
	//10 - PKWARE Date Compression Library Imploding
	//12 - bzip2 - Compression Method from libbz2, WinZip 10
	//14 - LZMA - Compression Method, WinZip 12
	//96 - Jpeg compression - Compression Method, WinZip 12
	//97 - WavPack - Compression Method, WinZip 11
	//98 - PPMd - Compression Method, WinZip 10
	//99 - AES encryption, WinZip 9

	const SZipFileEntry &e = FileInfo[found->ID];
	wchar_t buf[64];
	int16_t actualCompressionMethod=e.header.CompressionMethod;
	IReadFile* decrypted=0;
	uint8_t* decryptedBuf=0;
	uint32_t decryptedSize=e.header.DataDescriptor.CompressedSize;
#ifdef _NBL_COMPILE_WITH_ZIP_ENCRYPTION_
	if ((e.header.GeneralBitFlag & ZIP_FILE_ENCRYPTED) && (e.header.CompressionMethod == 99))
	{
		os::Printer::log("Reading encrypted file.");
		uint8_t salt[16]={0};
		const uint16_t saltSize = (((e.header.Sig & 0x00ff0000) >>16)+1)*4;
		File->seek(e.Offset);
		File->read(salt, saltSize);
		char pwVerification[2];
		char pwVerificationFile[2];
		File->read(pwVerification, 2);
		fcrypt_ctx zctx; // the encryption context
		int rc = fcrypt_init(
			(e.header.Sig & 0x00ff0000) >>16,
			(const unsigned char*)Password.c_str(), // the password
			Password.size(), // number of bytes in password
			salt, // the salt
			(unsigned char*)pwVerificationFile, // on return contains password verifier
			&zctx); // encryption context
		if (strncmp(pwVerificationFile, pwVerification, 2))
		{
			os::Printer::log("Wrong password");
			return 0;
		}
		decryptedSize= e.header.DataDescriptor.CompressedSize-saltSize-12;
		decryptedBuf= new uint8_t[decryptedSize];
		uint32_t c = 0;
		while ((c+32768)<=decryptedSize)
		{
			File->read(decryptedBuf+c, 32768);
			fcrypt_decrypt(
				decryptedBuf+c, // pointer to the data to decrypt
				32768,   // how many bytes to decrypt
				&zctx); // decryption context
			c+=32768;
		}
		File->read(decryptedBuf+c, decryptedSize-c);
		fcrypt_decrypt(
			decryptedBuf+c, // pointer to the data to decrypt
			decryptedSize-c,   // how many bytes to decrypt
			&zctx); // decryption context

		char fileMAC[10];
		char resMAC[10];
		rc = fcrypt_end(
			(unsigned char*)resMAC, // on return contains the authentication code
			&zctx); // encryption context
		if (rc != 10)
		{
			os::Printer::log("Error on encryption closing");
			delete [] decryptedBuf;
			return 0;
		}
		File->read(fileMAC, 10);
		if (strncmp(fileMAC, resMAC, 10))
		{
			os::Printer::log("Error on encryption check");
			delete [] decryptedBuf;
			return 0;
		}
        decrypted = new io::CMemoryReadFile(decryptedBuf, decryptedSize, found->FullName);
		actualCompressionMethod = (e.header.Sig & 0xffff);
#if 0
		if ((e.header.Sig & 0xff000000)==0x01000000)
		{
		}
		else if ((e.header.Sig & 0xff000000)==0x02000000)
		{
		}
		else
		{
			os::Printer::log("Unknown encryption method");
			return 0;
		}
#endif
	}
#endif
	switch(actualCompressionMethod)
	{
	case 0: // no compression
		{
            delete[] decryptedBuf;
			if (decrypted)
				return decrypted;
			else
                return new CLimitReadFile(File, e.Offset, decryptedSize, found->FullName);
		}
	case 8:
		{
  			#ifdef _NBL_COMPILE_WITH_ZLIB_

			const uint32_t uncompressedSize = e.header.DataDescriptor.UncompressedSize;
			char* pBuf = new char[ uncompressedSize ];
			if (!pBuf)
			{
				swprintf ( buf, 64, L"Not enough memory for decompressing %s", found->FullName.c_str() );
				os::Printer::log( buf, ELL_ERROR);
                delete[] decryptedBuf;
				if (decrypted)
					decrypted->drop();
				return 0;
			}

			uint8_t *pcData = decryptedBuf;
			if (!pcData)
			{
				pcData = new uint8_t[decryptedSize];
				if (!pcData)
				{
					swprintf ( buf, 64, L"Not enough memory for decompressing %s", found->FullName.c_str() );
					os::Printer::log( buf, ELL_ERROR);
                    delete[] decryptedBuf;
					delete [] pBuf;
					return 0;
				}

				//memset(pcData, 0, decryptedSize);
				File->seek(e.Offset);
				File->read(pcData, decryptedSize);
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
				swprintf ( buf, 64, L"Error decompressing %s", found->FullName.c_str() );
				os::Printer::log( buf, ELL_ERROR);
				delete [] pBuf;
				return 0;
			}
            else
            {
                auto ret = new io::CMemoryReadFile(pBuf, uncompressedSize, found->FullName);
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
			char* pBuf = new char[ uncompressedSize ];
			if (!pBuf)
			{
				swprintf ( buf, 64, L"Not enough memory for decompressing %s", found->FullName.c_str() );
				os::Printer::log( buf, ELL_ERROR);
                delete[] decryptedBuf;
				if (decrypted)
					decrypted->drop();
				return 0;
			}

			uint8_t *pcData = decryptedBuf;
			if (!pcData)
			{
				pcData = new uint8_t[decryptedSize];
				if (!pcData)
				{
					swprintf ( buf, 64, L"Not enough memory for decompressing %s", found->FullName.c_str() );
					os::Printer::log( buf, ELL_ERROR);
					delete [] pBuf;
                    delete[] decryptedBuf;
					return 0;
				}

				//memset(pcData, 0, decryptedSize);
				File->seek(e.Offset);
				File->read(pcData, decryptedSize);
			}

			bz_stream bz_ctx={0};
			/* use BZIP2's default memory allocation
			bz_ctx->bzalloc = NULL;
			bz_ctx->bzfree  = NULL;
			bz_ctx->opaque  = NULL;
			*/
			int err = BZ2_bzDecompressInit(&bz_ctx, 0, 0); /* decompression */
			if(err != BZ_OK)
			{
				os::Printer::log("bzip2 decompression failed. File cannot be read.", ELL_ERROR);
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
				swprintf ( buf, 64, L"Error decompressing %s", found->FullName.c_str() );
				os::Printer::log( buf, ELL_ERROR);
				delete [] pBuf;
                delete[] decryptedBuf;
				return 0;
			}
            else
            {
                auto ret = new io::CMemoryReadFile(pBuf, uncompressedSize, found->FullName);
                delete[] pBuf;
                return ret;
            }

			#else
            delete[] decryptedBuf;
			os::Printer::log("bzip2 decompression not supported. File cannot be read.", ELL_ERROR);
			return 0;
			#endif
		}
	case 14:
		{
  			#ifdef _NBL_COMPILE_WITH_LZMA_

			uint32_t uncompressedSize = e.header.DataDescriptor.UncompressedSize;
			char* pBuf = new char[ uncompressedSize ];
			if (!pBuf)
			{
				swprintf ( buf, 64, L"Not enough memory for decompressing %s", found->FullName.c_str() );
				os::Printer::log( buf, ELL_ERROR);
                delete[] decryptedBuf;
				if (decrypted)
					decrypted->drop();
				return 0;
			}

			uint8_t *pcData = decryptedBuf;
			if (!pcData)
			{
				pcData = new uint8_t[decryptedSize];
				if (!pcData)
				{
					swprintf ( buf, 64, L"Not enough memory for decompressing %s", found->FullName.c_str() );
					os::Printer::log( buf, ELL_ERROR);
					delete [] pBuf;
					return 0;
				}

				//memset(pcData, 0, decryptedSize);
				File->seek(e.Offset);
				File->read(pcData, decryptedSize);
			}

			ELzmaStatus status;
			SizeT tmpDstSize = uncompressedSize;
			SizeT tmpSrcSize = decryptedSize;

			unsigned int propSize = (pcData[3]<<8)+pcData[2];
			int err = LzmaDecode((Byte*)pBuf, &tmpDstSize,
					pcData+4+propSize, &tmpSrcSize,
					pcData+4, propSize,
					e.header.GeneralBitFlag&0x1?LZMA_FINISH_END:LZMA_FINISH_ANY, &status,
					&lzmaAlloc);
			uncompressedSize = tmpDstSize; // may be different to expected value

			if (decrypted)
				decrypted->drop();
			else
				delete[] pcData;

            delete[] decryptedBuf;
			if (err != SZ_OK)
			{
				os::Printer::log( "Error decompressing", found->FullName, ELL_ERROR);
				delete [] pBuf;
				return 0;
			}
			else
				return io::createMemoryReadFile(pBuf, uncompressedSize, found->FullName, true);

			#else
            delete[] decryptedBuf;
			os::Printer::log("lzma decompression not supported. File cannot be read.", ELL_ERROR);
			return 0;
			#endif
		}
	case 99:
		// If we come here with an encrypted file, decryption support is missing
		os::Printer::log("Decryption support not enabled. File cannot be read.", ELL_ERROR);
        delete[] decryptedBuf;
		return 0;
	default:
		swprintf ( buf, 64, L"file has unsupported compression method. %s", found->FullName.c_str() );
		os::Printer::log( buf, ELL_ERROR);
        delete[] decryptedBuf;
		return 0;
	};
}

#ifdef _NBL_COMPILE_WITH_LZMA_
//! Used for LZMA decompression. The lib has no default memory management
namespace
{
	void *SzAlloc(void *p, size_t size) { p = p; return _NBL_ALIGNED_MALLOC(size,_NBL_SIMD_ALIGNMENT); }
	void SzFree(void *p, void *address) { p = p; _NBL_ALIGNED_FREE(address); }
	ISzAlloc lzmaAlloc = { SzAlloc, SzFree };
}
#endif

} // end namespace io
} // end namespace nbl

#endif // __NBL_COMPILE_WITH_ZIP_ARCHIVE_LOADER_
#endif